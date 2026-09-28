"""Wires the index, retrieval, gateway, tools and loop together for one workspace.

Used by `agent chat` now and by headless `agent run` (M4). It owns no UI: events go to a
callback, and applying a plan is a separate, explicit call.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from code_agent.agent.loop import (
    AgentEvent,
    AgentLoop,
    RetrievalDone,
    TaskResult,
    Validator,
)
from code_agent.agent.tasks import TaskStatus, TaskStore
from code_agent.agent.tools import Capability, ToolContext, ToolRegistry
from code_agent.config import AgentConfig
from code_agent.context.assembly import ContextAssembler
from code_agent.context.log import TaskLog
from code_agent.edits.apply import ApplyPlan, Planner
from code_agent.edits.changesets import ChangeSetStore
from code_agent.index.embeddings import Embedder
from code_agent.index.indexer import Indexer, IndexStats
from code_agent.index.store import open_index
from code_agent.llm.factory import build_gateway
from code_agent.llm.gateway import Gateway, ModelUnavailableError
from code_agent.llm.types import CancelToken, ProviderError
from code_agent.retrieval.search import Searcher
from code_agent.security.approvals import ApprovalManager, Approver
from code_agent.security.audit import AuditLog
from code_agent.security.commands import classify
from code_agent.security.paths import SensitivePathPolicy
from code_agent.security.secrets import OutboundRedactor
from code_agent.shadow.validate import ShadowValidator, ValidationReport
from code_agent.shadow.worktree import ShadowUnavailableError, ShadowWorktree
from code_agent.workspace import Workspace


@dataclass(frozen=True)
class TokenBudgets:
    context: int  # retrieved code in the first message
    prompt: int  # whole conversation (history is elided above this)
    output: int


class AgentSession:
    def __init__(
        self,
        workspace: Workspace,
        cfg: AgentConfig,
        embedder: Embedder | None,
        *,
        gateway: Gateway | None = None,
        approver: Approver | None = None,
    ) -> None:
        """`approver` is the UI that asks the user about commands. Without one, the model gets no
        command tool at all (not merely a tool that always says no)."""
        self.workspace = workspace
        self.cfg = cfg
        self.conn, self.repairs = open_index(workspace)
        self.embedder = embedder
        self.indexer = Indexer(workspace, self.conn, cfg, embedder)
        self.searcher = Searcher(workspace.index_path, workspace.repo_id, cfg.retrieval, embedder)
        self.gateway = gateway or build_gateway(cfg.llm)
        self.log = TaskLog(workspace.ensure_state_dir())
        self.redactor = OutboundRedactor()
        # Order matters: redact first, then log, so the log holds exactly what was sent and
        # neither the provider nor our own logs ever see a detected secret.
        self.gateway.outbound_filter = lambda request: self.log(self.redactor(request))
        self.tasks = TaskStore(self.conn, workspace.repo_id)
        self.changes = ChangeSetStore(self.conn, workspace.root)
        self.sensitive = SensitivePathPolicy(cfg.index.extra_sensitive)
        self.audit = AuditLog(self.conn)
        self.approvals = ApprovalManager(self.conn, approver) if approver else None
        self._shadow: ShadowWorktree | None = None
        self.shadow_unavailable: str | None = None

    def close(self) -> None:
        if self._shadow is not None:
            self._shadow.close()
        self.conn.close()

    # -- validation ---------------------------------------------------------------------------

    def _validator(self, request_id: str, cancel: CancelToken) -> Validator | None:
        """A shadow validator for one task, or None (disabled / not a git repo)."""
        vcfg = self.cfg.validation
        if not vcfg.enabled:
            return None
        if self._shadow is None:
            try:
                self._shadow = ShadowWorktree(self.workspace)
                self._shadow.create()
            except ShadowUnavailableError as exc:
                self.shadow_unavailable = str(exc)
                self._shadow = None
                return None

        def test_gate(tests: list[str]) -> str | None:
            # Running tests executes repository code (including the model's edits), so it goes
            # through the same approval as any test/lint command.
            if self.approvals is None:
                return "no approval prompt available"
            command = f"python -m pytest {' '.join(tests)}"
            approval = self.approvals.authorize(
                request_id, classify(command, root=self.workspace.root, sensitive=self.sensitive)
            )
            if approval is None:
                return "the user declined running tests"
            return self.approvals.recheck(approval, request_id, cancel)

        validator = ShadowValidator(
            self._shadow,
            python=str(vcfg.python) if vcfg.python else None,
            test_gate=test_gate,
            lint=vcfg.lint,
            type_check=vcfg.type_check,
            tests=vcfg.tests,
            test_timeout_s=vcfg.test_timeout_s,
            full_suite_max_test_files=vcfg.full_suite_max_test_files,
            cancel=cancel,
        )

        def validate(plan: ApplyPlan) -> ValidationReport:
            report = validator.validate(plan)
            self.audit.record(
                tool_name="shadow_validation",
                actor="system",
                request_id=request_id,
                args={"files": sorted(plan.changes), "tests": report.tests_run},
                output=report.summary(),
                status="ok" if report.ok else "failed",
            )
            return report

        return validate

    # -- setup --------------------------------------------------------------------------------

    def check_generation(self) -> str | None:
        """None if the configured models are reachable and exist, else a user-facing reason."""
        if not self.cfg.llm.routes:
            return "no models configured: add [llm.routes] to your config (see `agent models`)"
        missing = {"cheap", "strong"} - set(self.cfg.llm.routes)
        if "strong" in missing:
            return "no 'strong' route configured in [llm.routes]"
        try:
            self.gateway.verify_models()
        except (ModelUnavailableError, ProviderError) as exc:
            return str(exc)
        return None

    def refresh_index(self) -> IndexStats:
        """Bring the index up to date so the model sees what is on disk right now."""
        return self.indexer.sync()

    def token_budgets(self) -> TokenBudgets:
        llm = self.cfg.llm
        prompt = llm.max_prompt_tokens
        window = self.gateway.context_window("strong")
        if window:
            prompt = min(prompt, int(window * llm.context_fraction) - llm.max_output_tokens)
        context = min(self.cfg.retrieval.max_context_tokens, prompt // 2)
        return TokenBudgets(context=context, prompt=prompt, output=llm.max_output_tokens)

    # -- tasks --------------------------------------------------------------------------------

    def run_task(
        self,
        task: str,
        cancel: CancelToken | None = None,
        on_event: Callable[[AgentEvent], None] = lambda _: None,
    ) -> tuple[str, TaskResult]:
        _, revision = self.workspace.git_revision()
        request_id = self.tasks.create(task, self.cfg.budgets, base_revision=revision)
        self.log.start(request_id)

        def on_event_logged(event: AgentEvent) -> None:
            if isinstance(event, RetrievalDone):
                self.log.retrieval(event.bundle)
            on_event(event)

        budgets = self.token_budgets()
        cancel = cancel or CancelToken()
        ctx = ToolContext(
            self.workspace.root, self.sensitive, self.searcher, self.conn, self.workspace.repo_id,
            request_id=request_id, audit=self.audit, approvals=self.approvals, cancel=cancel,
        )  # fmt: skip
        allowed = {Capability.READ} | ({Capability.EXECUTE} if self.approvals else set())
        redactions_before = self.redactor.redactions.copy()
        loop = AgentLoop(
            self.gateway,
            ToolRegistry(ctx, allowed=frozenset(allowed)),
            ContextAssembler(self.searcher, self.conn, self.workspace.repo_id,
                             top_k=self.cfg.retrieval.top_k),
            Planner(self.workspace.root, self.sensitive),
            self.cfg.budgets,
            context_budget_tokens=budgets.context,
            history_budget_tokens=budgets.prompt,
            max_output_tokens=budgets.output,
            on_event=on_event_logged,
            validator=self._validator(request_id, cancel),
        )  # fmt: skip
        result = loop.run(task, cancel)
        status = TaskStatus.WAITING_FOR_APPROVAL if result.plan else result.status
        self.tasks.update(request_id, status, result.usage)
        new_redactions = self.redactor.redactions - redactions_before
        if new_redactions:
            # Kinds and counts only: the audit log must never hold the secret itself.
            self.audit.record(
                tool_name="outbound_redaction",
                actor="system",
                args={"kinds": dict(new_redactions)},
                request_id=request_id,
            )
        return request_id, result

    def apply(self, request_id: str, result: TaskResult, files: list[str] | None = None) -> str:
        """Apply an approved plan, or only the approved `files` of it. Returns the change-set id
        (for `agent undo`)."""
        plan: ApplyPlan | None = result.plan
        if plan is None:
            raise ValueError("task has no edits to apply")
        if files is not None:
            plan = ApplyPlan(plan.results, {f: plan.changes[f] for f in files})
        change_set_id = self.changes.apply(plan, request_id=request_id)
        self.tasks.update(request_id, TaskStatus.SUCCEEDED, result.usage)
        self.audit.record(
            tool_name="apply_change_set",
            actor="user",
            request_id=request_id,
            args={"change_set_id": change_set_id, "files": sorted(plan.changes)},
        )
        self.indexer.update_paths(plan.changes)  # keep the index in step with the edit
        return change_set_id

    def decline(self, request_id: str, result: TaskResult) -> None:
        self.tasks.update(request_id, TaskStatus.CANCELLED, result.usage)
        files = sorted(result.plan.changes) if result.plan else []
        self.audit.record(tool_name="decline_change_set", actor="user", request_id=request_id,
                          args={"files": files})  # fmt: skip
