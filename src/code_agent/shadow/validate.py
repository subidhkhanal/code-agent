"""Validate a candidate change set in the shadow worktree (ADR 0006).

For the files the change touches:
  ruff      lint errors                         static: always runs
  pyright   type errors (severity "error")      static: always runs
  pytest    targeted tests                      executes repo code: needs a test/lint approval

**Only new problems fail validation.** Before writing the change, the same checks run on the
unmodified shadow (the baseline). A diagnostic present in both is pre-existing; a test failing
in both is "still failing". Validation fails on new lint/type errors and on regressions (tests
that passed before and fail after). Diagnostics are compared by (file, rule, message), without
line numbers, which shift when code is edited.

Targeted tests: test files named after a changed module (`tokens.py` -> `test_tokens.py`) or
that import it, plus any changed test file. With no targeted tests, the full suite runs only if
it is small (`full_suite_max_test_files`).

`PYTHONPATH` is set to the shadow (and `src/` if present) so tests import the shadow's copy of
the code even when the project is installed in editable mode; `import_check` verifies that.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from code_agent.edits.apply import ApplyPlan
from code_agent.llm.types import CancelToken
from code_agent.security.runner import CommandResult, run_command, scrubbed_env
from code_agent.shadow.worktree import ShadowWorktree

MAX_FEEDBACK_CHARS = 6_000


@dataclass(frozen=True)
class Diagnostic:
    tool: str  # ruff | pyright
    file: str  # workspace-relative
    line: int
    rule: str
    message: str

    @property
    def key(self) -> tuple[str, str, str, str]:
        return (self.tool, self.file, self.rule, self.message)

    def render(self) -> str:
        return f"{self.file}:{self.line}: [{self.tool} {self.rule}] {self.message}"


@dataclass
class ValidationReport:
    attempted: list[str] = field(default_factory=list)  # checks that ran
    skipped: list[str] = field(default_factory=list)  # checks that could not run, and why
    new_diagnostics: list[Diagnostic] = field(default_factory=list)
    regressions: list[str] = field(default_factory=list)  # test ids: passed before, fail now
    still_failing: list[str] = field(default_factory=list)  # failing before and after
    fixed: list[str] = field(default_factory=list)  # failing before, passing now
    tests_run: list[str] = field(default_factory=list)  # test files selected
    test_output: str = ""  # tail of pytest output after the change (for feedback)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.new_diagnostics and not self.regressions

    def summary(self) -> str:
        parts = []
        for check in ("ruff", "pyright", "pytest"):
            if check in self.attempted:
                bad = (
                    len(self.regressions) if check == "pytest"
                    else sum(1 for d in self.new_diagnostics if d.tool == check)
                )  # fmt: skip
                parts.append(f"{check} {'ok' if bad == 0 else f'{bad} new problem(s)'}")
        for reason in self.skipped:
            parts.append(f"skipped: {reason}")
        if self.fixed:
            parts.append(f"{len(self.fixed)} failing test(s) now pass")
        if self.still_failing:
            parts.append(f"{len(self.still_failing)} test(s) were failing before and still fail")
        return " | ".join(parts) or "nothing to validate"

    def feedback(self) -> str:
        """Diagnostics for the model, truncated."""
        lines = ["Validation of your edits in a sandbox copy of the repository failed:"]
        lines += [f"- {d.render()}" for d in self.new_diagnostics]
        if self.regressions:
            lines.append("Tests that passed before your change and fail now:")
            lines += [f"- {t}" for t in self.regressions]
            if self.test_output:
                lines.append("pytest output (tail):\n" + self.test_output)
        text = "\n".join(lines)
        return text[:MAX_FEEDBACK_CHARS] + (
            "\n[... truncated]" if len(text) > MAX_FEEDBACK_CHARS else ""
        )


# -- helpers -------------------------------------------------------------------------------------


def detect_python(root: Path) -> str:
    """The project's own interpreter if it has a local venv, else the agent's."""
    for venv in (".venv", "venv", "env"):
        for candidate in (root / venv / "Scripts" / "python.exe", root / venv / "bin" / "python"):
            if candidate.is_file():
                return str(candidate)
    return sys.executable


def module_names(rel_path: str) -> list[str]:
    """`src/pkg/auth/tokens.py` -> ['pkg.auth.tokens', 'auth.tokens', 'tokens'] (import forms)."""
    parts = Path(rel_path).with_suffix("").parts
    if parts and parts[0] == "src":
        parts = parts[1:]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return [".".join(parts[i:]) for i in range(len(parts))] if parts else []


def _is_test_file(rel_path: str) -> bool:
    name = Path(rel_path).name
    return name.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py"))


_SKIP_DIRS = frozenset({"node_modules", "site-packages", "venv", "env", "build", "dist"})


def all_test_files(root: Path) -> list[str]:
    """Test files in the tree, skipping hidden dirs (.venv, .git, .agent) and dependencies."""
    found = []
    for p in root.rglob("*.py"):
        parts = p.relative_to(root).parts
        if not _is_test_file(p.name):
            continue
        if any(part.startswith(".") or part in _SKIP_DIRS for part in parts[:-1]):
            continue
        found.append(p.relative_to(root).as_posix())
    return sorted(found)


def select_tests(root: Path, changed: Iterable[str]) -> list[str]:
    changed = [c for c in changed if c.endswith(".py")]
    test_files = all_test_files(root)
    selected = {c for c in changed if _is_test_file(c)}
    for rel in changed:
        if _is_test_file(rel):
            continue
        stem = Path(rel).stem
        mods = module_names(rel)
        pattern = re.compile(
            r"^\s*(?:from\s+(?:" + "|".join(re.escape(m) for m in mods) + r")\b"
            r"|import\s+(?:" + "|".join(re.escape(m) for m in mods) + r")\b"
            r"|from\s+[\w.]+\s+import\s+(?:.*\b" + re.escape(stem) + r"\b))",
            re.MULTILINE,
        ) if mods else None  # fmt: skip
        for test in test_files:
            name = Path(test).name
            if name in (f"test_{stem}.py", f"{stem}_test.py"):
                selected.add(test)
            elif pattern is not None:
                try:
                    text = (root / test).read_text("utf-8", "replace")
                except OSError:
                    continue
                if pattern.search(text):
                    selected.add(test)
    return sorted(selected)


# -- the validator -------------------------------------------------------------------------------

# Decides whether tests may run (they execute repository code). Returns a reason if not.
TestGate = Callable[[list[str]], str | None]


def _always_allow(_: list[str]) -> str | None:
    return None


class ShadowValidator:
    def __init__(
        self,
        shadow: ShadowWorktree,
        *,
        python: str | None = None,
        test_gate: TestGate = _always_allow,
        lint: bool = True,
        type_check: bool = True,
        tests: bool = True,
        test_timeout_s: float = 300,
        full_suite_max_test_files: int = 30,
        cancel: CancelToken | None = None,
    ) -> None:
        self.shadow = shadow
        self.python = python or detect_python(shadow.workspace.root)
        self.test_gate = test_gate
        self.lint, self.type_check, self.tests = lint, type_check, tests
        self.test_timeout_s = test_timeout_s
        self.full_suite_max_test_files = full_suite_max_test_files
        self.cancel = cancel or CancelToken()
        # Baselines are per task: the unmodified state does not change between fix rounds.
        self._lint_baseline: dict[str, set[tuple]] = {}
        self._test_baseline: dict[str, dict[str, bool]] = {}

    # -- running tools ------------------------------------------------------------------------

    def _env(self) -> dict[str, str]:
        root = self.shadow.root
        paths = [str(root / "src"), str(root)] if (root / "src").is_dir() else [str(root)]
        env = scrubbed_env()
        env["PYTHONPATH"] = ";".join(paths) if sys.platform == "win32" else ":".join(paths)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        return env

    def _run(self, argv: list[str], timeout: float = 180) -> CommandResult:
        return run_command(argv, self.shadow.root, timeout_s=timeout, cancel=self.cancel,
                           isolate_network=True, env=self._env())  # fmt: skip

    def _ruff(self, files: list[str]) -> list[Diagnostic]:
        result = self._run([sys.executable, "-m", "ruff", "check", "--output-format=json",
                            "--no-cache", "--exit-zero", *files])  # fmt: skip
        out: list[Diagnostic] = []
        for item in json.loads(result.output or "[]"):
            rel = (
                Path(item["filename"]).resolve().relative_to(self.shadow.root.resolve()).as_posix()
            )
            out.append(Diagnostic("ruff", rel, item["location"]["row"], item.get("code") or "",
                                  item["message"]))  # fmt: skip
        return out

    def _pyright(self, files: list[str]) -> list[Diagnostic]:
        result = self._run([sys.executable, "-m", "pyright", "--outputjson",
                            "--pythonpath", self.python, *files], timeout=300)  # fmt: skip
        start = result.output.find("{")
        if start < 0:
            return []
        data = json.loads(result.output[start:])
        out: list[Diagnostic] = []
        for d in data.get("generalDiagnostics", []):
            if d.get("severity") != "error":
                continue
            try:
                rel = Path(d["file"]).resolve().relative_to(self.shadow.root.resolve()).as_posix()
            except ValueError:
                continue
            out.append(Diagnostic("pyright", rel, d["range"]["start"]["line"] + 1,
                                  d.get("rule", ""), d["message"].split("\n")[0]))  # fmt: skip
        return out

    def _pytest(self, test_files: list[str]) -> tuple[dict[str, bool], str]:
        """test id -> passed, plus the tail of the output."""
        with tempfile.TemporaryDirectory() as tmp:
            junit = Path(tmp) / "junit.xml"
            result = self._run(
                [self.python, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider",
                 f"--junitxml={junit}", *test_files], timeout=self.test_timeout_s,
            )  # fmt: skip
            outcomes: dict[str, bool] = {}
            if junit.exists():
                for case in ET.parse(junit).getroot().iter("testcase"):
                    test_id = f"{case.get('classname', '')}::{case.get('name', '')}".strip(":")
                    failed = any(child.tag in ("failure", "error") for child in case)
                    skipped = any(child.tag == "skipped" for child in case)
                    if not skipped:
                        outcomes[test_id] = not failed
        tail = "\n".join(result.output.strip().splitlines()[-40:])
        if result.timed_out:
            tail += "\n[pytest timed out]"
        return outcomes, tail

    # -- validation ---------------------------------------------------------------------------

    def validate(self, plan: ApplyPlan) -> ValidationReport:
        report = ValidationReport()
        self.shadow.reset()
        changed = sorted(plan.changes)
        py_files = [f for f in changed if f.endswith((".py", ".pyi"))]
        existing = [f for f in py_files if plan.changes[f].before is not None]

        # Baselines on the unmodified shadow (computed once per file per task).
        missing_lint = [f for f in existing if f not in self._lint_baseline]
        if missing_lint:
            for f in missing_lint:
                self._lint_baseline[f] = set()
            for d in self._static(missing_lint, report, record=False):
                self._lint_baseline[d.file].add(d.key)

        tests: list[str] = []
        if self.tests:
            tests = select_tests(self.shadow.root, changed)
            if not tests:
                every = all_test_files(self.shadow.root)
                if 0 < len(every) <= self.full_suite_max_test_files:
                    tests = every
                    report.warnings.append("no targeted tests found; ran the full (small) suite")
            if not tests:
                report.skipped.append("pytest (no tests found for the changed files)")
            elif reason := self.test_gate(tests):
                report.skipped.append(f"pytest ({reason})")
                tests = []
        pre_existing_tests = [t for t in tests if (self.shadow.root / t).exists()]
        missing_tests = [t for t in pre_existing_tests if t not in self._test_baseline]
        if missing_tests:
            baseline, _ = self._pytest(missing_tests)
            for t in missing_tests:
                self._test_baseline[t] = {k: v for k, v in baseline.items()
                                          if k.startswith(_test_prefix(t))}  # fmt: skip

        # Apply the candidate change and re-run everything.
        for rel, change in plan.changes.items():
            self.shadow.write(rel, change.after)
        after = self._static(py_files, report, record=True)
        for d in after:
            if d.key not in self._lint_baseline.get(d.file, set()):
                report.new_diagnostics.append(d)

        if tests:
            report.attempted.append("pytest")
            report.tests_run = tests
            outcomes, report.test_output = self._pytest(tests)
            before: dict[str, bool] = {}
            for t in tests:
                before.update(self._test_baseline.get(t, {}))
            for test_id, passed in sorted(outcomes.items()):
                was = before.get(test_id)
                if not passed and was is not False:
                    report.regressions.append(test_id)  # passed before, or a new test that fails
                elif not passed:
                    report.still_failing.append(test_id)
                elif was is False:
                    report.fixed.append(test_id)
        return report

    def _static(
        self, files: list[str], report: ValidationReport, *, record: bool
    ) -> list[Diagnostic]:
        if not files:
            return []
        diagnostics: list[Diagnostic] = []
        if self.lint:
            diagnostics += self._ruff(files)
            if record:
                report.attempted.append("ruff")
        if self.type_check:
            diagnostics += self._pyright(files)
            if record:
                report.attempted.append("pyright")
        return diagnostics


def _test_prefix(test_file: str) -> str:
    """JUnit classnames look like `tests.test_tokens` (dotted path without `.py`)."""
    return Path(test_file).with_suffix("").as_posix().replace("/", ".")
