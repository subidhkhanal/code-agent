"""Wires the index, retrieval, gateway, tools and loop together for one workspace.

Used by `agent chat` now and by headless `agent run` (M4). It owns no UI: events go to a
callback, and applying a plan is a separate, explicit call.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from code_agent.agent.loop import AgentEvent, AgentLoop, TaskResult
from code_agent.agent.tasks import TaskStatus, TaskStore
from code_agent.agent.tools import ToolContext, ToolRegistry
from code_agent.config import AgentConfig
from code_agent.context.assembly import ContextAssembler
from code_agent.edits.apply import ApplyPlan, Planner
from code_agent.edits.changesets import ChangeSetStore
from code_agent.index.embeddings import Embedder
from code_agent.index.indexer import Indexer, IndexStats
from code_agent.index.store import open_index
from code_agent.llm.factory import build_gateway
from code_agent.llm.gateway import Gateway, ModelUnavailableError
from code_agent.llm.types import CancelToken, ProviderError
from code_agent.retrieval.search import Searcher
from code_agent.security.paths import SensitivePathPolicy
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
    ) -> None:
        self.workspace = workspace
        self.cfg = cfg
        self.conn, self.repairs = open_index(workspace)
        self.embedder = embedder
        self.indexer = Indexer(workspace, self.conn, cfg, embedder)
        self.searcher = Searcher(workspace.index_path, workspace.repo_id, cfg.retrieval, embedder)
        self.gateway = gateway or build_gateway(cfg.llm)
        self.tasks = TaskStore(self.conn, workspace.repo_id)
        self.changes = ChangeSetStore(self.conn, workspace.root)
        self.sensitive = SensitivePathPolicy(cfg.index.extra_sensitive)

    def close(self) -> None:
        self.conn.close()

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
        budgets = self.token_budgets()
        ctx = ToolContext(
            self.workspace.root, self.sensitive, self.searcher, self.conn, self.workspace.repo_id
        )
        loop = AgentLoop(
            self.gateway,
            ToolRegistry(ctx),
            ContextAssembler(self.searcher, self.conn, self.workspace.repo_id,
                             top_k=self.cfg.retrieval.top_k),
            Planner(self.workspace.root, self.sensitive),
            self.cfg.budgets,
            context_budget_tokens=budgets.context,
            history_budget_tokens=budgets.prompt,
            max_output_tokens=budgets.output,
            on_event=on_event,
        )  # fmt: skip
        result = loop.run(task, cancel)
        status = TaskStatus.WAITING_FOR_APPROVAL if result.plan else result.status
        self.tasks.update(request_id, status, result.usage)
        return request_id, result

    def apply(self, request_id: str, result: TaskResult) -> str:
        """Apply an approved plan. Returns the change-set id (for `agent undo`)."""
        plan: ApplyPlan | None = result.plan
        if plan is None:
            raise ValueError("task has no edits to apply")
        change_set_id = self.changes.apply(plan, request_id=request_id)
        self.tasks.update(request_id, TaskStatus.SUCCEEDED, result.usage)
        self.indexer.update_paths(plan.changes)  # keep the index in step with the edit
        return change_set_id

    def decline(self, request_id: str, result: TaskResult) -> None:
        self.tasks.update(request_id, TaskStatus.CANCELLED, result.usage)
