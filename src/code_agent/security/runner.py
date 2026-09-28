"""Run one approved command: no shell, allowlisted environment, bounded time and output, and a
process-tree kill on timeout or cancel.

What is enforced, per OS (ADR 0007 has the full table):
* both   no shell unless the user approved a shell command; env allowlist; cwd inside the
         workspace; timeout; output cap; kill the whole process tree on cancel/timeout
* Linux  optional network isolation via `unshare --net` (used for shadow validation; skipped
         with a note if unprivileged user namespaces are unavailable)
* Windows no network isolation: interactive mode relies on classification + approval, and
         headless mode runs in a Docker container with `--network none`
"""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import subprocess
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from code_agent.llm.types import CancelToken

MAX_OUTPUT_BYTES = 200_000
DEFAULT_TIMEOUT_S = 300.0

# Only these variables reach child processes. Allowlisting (rather than stripping names that
# look secret) means a credential under an unexpected name still doesn't leak.
ENV_ALLOWLIST = frozenset({
    "PATH", "PATHEXT", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "TEMP", "TMP", "TMPDIR",
    "HOME", "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "APPDATA", "LOCALAPPDATA", "LANG", "LC_ALL",
    "LC_CTYPE", "TERM", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE", "OS", "VIRTUAL_ENV",
    "PYTHONIOENCODING", "PYTHONUTF8",
})  # fmt: skip


def scrubbed_env(base: dict[str, str] | None = None) -> dict[str, str]:
    source = os.environ if base is None else base
    env = {k: v for k, v in source.items() if k.upper() in ENV_ALLOWLIST}
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env["NO_COLOR"] = "1"  # plain output is easier for the model to read
    return env


@dataclass(frozen=True)
class CommandResult:
    argv: tuple[str, ...]
    exit_code: int | None  # None if killed
    output: str  # stdout and stderr interleaved
    truncated: bool
    timed_out: bool
    cancelled: bool
    seconds: float
    network_isolated: bool

    def render(self) -> str:
        status = (
            "cancelled" if self.cancelled
            else "timed out" if self.timed_out
            else f"exit code {self.exit_code}"
        )  # fmt: skip
        note = "\n[output truncated]" if self.truncated else ""
        return f"$ {' '.join(self.argv)}\n[{status}, {self.seconds:.1f}s]\n{self.output}{note}"


def _kill_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        # taskkill /T walks the child tree; proc.kill() alone would orphan grandchildren.
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       capture_output=True, check=False)  # fmt: skip
    else:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)  # the child leads its own process group
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def network_isolation_available() -> bool:
    if os.name != "posix" or shutil.which("unshare") is None:
        return False
    probe = subprocess.run(["unshare", "--user", "--map-root-user", "--net", "true"],
                           capture_output=True, check=False)  # fmt: skip
    return probe.returncode == 0


def run_command(
    argv: Sequence[str],
    cwd: Path,
    *,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    cancel: CancelToken | None = None,
    isolate_network: bool = False,
    max_output_bytes: int = MAX_OUTPUT_BYTES,
    env: dict[str, str] | None = None,
) -> CommandResult:
    cancel = cancel or CancelToken()
    full = list(argv)
    isolated = False
    if isolate_network and network_isolation_available():
        full = ["unshare", "--user", "--map-root-user", "--net", "--", *full]
        isolated = True

    kwargs: dict = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True

    started = time.monotonic()
    proc = subprocess.Popen(
        full,
        cwd=cwd,
        env=env if env is not None else scrubbed_env(),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        **kwargs,
    )
    chunks: list[bytes] = []
    size = [0]
    truncated = [False]

    def pump() -> None:
        assert proc.stdout is not None
        for block in iter(lambda: proc.stdout.read(8192), b""):  # type: ignore[union-attr]
            if size[0] < max_output_bytes:
                chunks.append(block[: max_output_bytes - size[0]])
            if size[0] + len(block) > max_output_bytes:
                truncated[0] = True
            size[0] += len(block)

    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    timed_out = cancelled = False
    while proc.poll() is None:
        if cancel.cancelled:
            cancelled = True
            _kill_tree(proc)
            break
        if time.monotonic() - started > timeout_s:
            timed_out = True
            _kill_tree(proc)
            break
        time.sleep(0.05)
    reader.join(timeout=5)
    output = b"".join(chunks).decode("utf-8", "replace")
    return CommandResult(
        tuple(argv),
        None if (timed_out or cancelled) else proc.returncode,
        output,
        truncated[0],
        timed_out,
        cancelled,
        time.monotonic() - started,
        isolated,
    )
