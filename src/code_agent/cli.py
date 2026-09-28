"""`agent` command-line entry point."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.syntax import Syntax
from rich.table import Table

from code_agent.agent.loop import (
    AgentEvent,
    AssistantText,
    EditProposed,
    EditsRejected,
    RetrievalDone,
    Status,
    TaskResult,
    ToolFinished,
    ToolStarted,
)
from code_agent.agent.session import AgentSession
from code_agent.agent.tasks import TaskStatus
from code_agent.config import AgentConfig, load_config
from code_agent.edits.atomic import WriteConflictError
from code_agent.edits.changesets import ChangeSetStore, UndoConflictError
from code_agent.edits.diff import diff_stats, plan_diff
from code_agent.index.embeddings import Embedder, FastEmbedEmbedder
from code_agent.index.indexer import Indexer, IndexStats
from code_agent.index.store import IndexStore, open_index
from code_agent.index.watcher import watch
from code_agent.llm.factory import build_providers
from code_agent.llm.types import CancelToken, ProviderError
from code_agent.retrieval.search import Mode, Searcher
from code_agent.workspace import Workspace

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="A terminal coding agent with a local hybrid code index.",
)
console = Console()
err = Console(stderr=True)

PathOpt = Annotated[
    Path, typer.Option("--path", "-p", help="Any directory inside the repo.", show_default=False)
]


@app.callback()
def main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


def _embedder(cfg: AgentConfig) -> Embedder:
    return FastEmbedEmbedder(
        cfg.index.embedding_model, cfg.model_cache_dir, batch_size=cfg.index.embedding_batch_size
    )


def _print_stats(ws: Workspace, stats: IndexStats, counts: dict[str, int]) -> None:
    for note in stats.repairs:
        err.print(f"[yellow]repair:[/] {note}")
    for error in stats.errors:
        err.print(f"[red]error:[/] {error}")
    console.print(
        f"[bold]{ws.root}[/]  {stats.files_seen} files | "
        f"[green]+{stats.added}[/] [cyan]~{stats.updated}[/] [red]-{stats.removed}[/] "
        f"={stats.unchanged} | {stats.chunks_written} chunks written | "
        f"{stats.vectors_embedded} embedded, {stats.vectors_reused} reused | "
        f"{stats.seconds:.2f}s"
    )
    console.print(
        f"index: {counts['files']} files, {counts['chunks']} chunks, "
        f"{counts['vectors']}/{counts['chunks']} with vectors"
    )
    if stats.embedding_error:
        err.print(
            f"[yellow]embeddings unavailable ({stats.embedding_error}); "
            "keyword and symbol search still work.[/]"
        )


@app.command()
def index(
    path: PathOpt = Path("."),
    watch_changes: Annotated[
        bool, typer.Option("--watch", help="Keep running and re-index files as they change.")
    ] = False,
    no_embed: Annotated[
        bool, typer.Option("--no-embed", help="Skip embeddings (BM25 + symbol search only).")
    ] = False,
    rebuild: Annotated[
        bool, typer.Option("--rebuild", help="Discard the index first (keeps undo history).")
    ] = False,
) -> None:
    """Build or incrementally refresh the index for the current repository."""
    cfg = load_config()
    ws = Workspace.discover(path)
    conn, repairs = open_index(ws)
    if rebuild:
        IndexStore(conn, ws.repo_id).clear()
    indexer = Indexer(ws, conn, cfg, None if no_embed else _embedder(cfg))

    with Progress(
        TextColumn("embedding"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(),
        console=console, transient=True,
    ) as progress:  # fmt: skip
        task = None

        def on_progress(done: int, total: int) -> None:
            nonlocal task
            if task is None:
                task = progress.add_task("embed", total=total)
            progress.update(task, completed=done)

        stats = indexer.sync(embed=not no_embed, progress=on_progress)
    stats.repairs[:0] = repairs
    store = IndexStore(conn, ws.repo_id)
    _print_stats(ws, stats, store.counts())

    if watch_changes:
        console.print("[dim]watching for changes (Ctrl+C to stop)...[/]")
        try:
            watch(
                indexer,
                on_batch=lambda s: console.print(
                    f"[dim]re-indexed[/] +{s.added} ~{s.updated} -{s.removed} "
                    f"({s.chunks_written} chunks, {s.vectors_embedded} embedded) "
                    f"in {s.seconds * 1000:.0f} ms"
                ),
            )
        except KeyboardInterrupt:
            console.print("[dim]stopped[/]")
    conn.close()


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="Natural language or identifiers.")],
    k: Annotated[int | None, typer.Option("-k", help="Number of results.")] = None,
    mode: Annotated[Mode, typer.Option("--mode", "-m")] = Mode.HYBRID,
    show_code: Annotated[bool, typer.Option("--code", help="Print each chunk.")] = False,
    as_json: Annotated[bool, typer.Option("--json", help="Machine-readable output.")] = False,
    path: PathOpt = Path("."),
) -> None:
    """Search the index (works offline; vector search degrades gracefully)."""
    cfg = load_config()
    ws = Workspace.discover(path)
    if not ws.index_path.exists():
        err.print("[red]No index found.[/] Run [bold]agent index[/] first.")
        raise typer.Exit(1)
    searcher = Searcher(ws.index_path, ws.repo_id, cfg.retrieval, _embedder(cfg))
    result = searcher.search(query, k=k, mode=mode)

    if as_json:
        payload = {
            "query": result.query,
            "mode": result.mode.value,
            "notes": result.notes,
            "timings_ms": result.timings_ms,
            "hits": [
                {
                    "file_path": h.file_path, "symbol": h.symbol, "kind": h.kind,
                    "start_line": h.start_line, "end_line": h.end_line,
                    "score": h.score, "ranks": h.ranks,
                }
                for h in result.hits
            ],
        }  # fmt: skip
        typer.echo(json.dumps(payload, indent=2))
        return

    for note in result.notes:
        err.print(f"[yellow]{note}[/]")
    if not result.hits:
        console.print("No results.")
        return
    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("#", justify="right")
    table.add_column("location", no_wrap=True)  # never truncate: it's what you open next
    table.add_column("symbol", overflow="ellipsis", max_width=40)
    table.add_column("kind", no_wrap=True)
    table.add_column("hits", style="dim", overflow="fold")
    for i, hit in enumerate(result.hits, 1):
        table.add_row(
            str(i),
            f"{hit.file_path}:{hit.start_line}-{hit.end_line}",
            hit.symbol or "",
            hit.kind,
            " ".join(f"{name}#{rank}" for name, rank in hit.ranks.items()),
        )
    console.print(table)
    timings = ", ".join(f"{name} {ms:.0f} ms" for name, ms in result.timings_ms.items())
    console.print(f"[dim]{timings}[/]")
    if show_code:
        for hit in result.hits:
            lexer = "python" if hit.file_path.endswith((".py", ".pyi")) else "text"
            console.rule(f"{hit.file_path}:{hit.start_line}")
            console.print(Syntax(hit.content, lexer, line_numbers=True, start_line=hit.start_line))


@app.command()
def undo(
    change_set: Annotated[
        str | None, typer.Option("--id", help="Change set to undo (default: the latest).")
    ] = None,
    path: PathOpt = Path("."),
) -> None:
    """Revert the last applied change set (works offline, no git needed)."""
    ws = Workspace.discover(path)
    if not ws.index_path.exists():
        console.print("Nothing to undo.")
        raise typer.Exit(1)
    conn, _ = open_index(ws)
    try:
        result = ChangeSetStore(conn, ws.root).undo(change_set)
    except LookupError as exc:
        console.print(f"Nothing to undo ({exc}).")
        raise typer.Exit(1) from exc
    except UndoConflictError as exc:
        err.print(f"[red]Undo refused:[/] {exc}")
        raise typer.Exit(1) from exc
    finally:
        conn.close()
    for file_path in result.restored:
        console.print(f"[green]restored[/] {file_path}")
    for file_path in result.already_original:
        console.print(f"[dim]unchanged[/] {file_path}")
    console.print(f"Undid change set {result.change_set_id[:12]}.")


@app.command()
def models() -> None:
    """List models available from each configured provider and check the configured routes."""
    cfg = load_config()
    configured = {spec for specs in cfg.llm.routes.values() for spec in specs}
    ok = True
    for name, provider in build_providers(cfg.llm).items():
        try:
            available = sorted(provider.list_models(), key=lambda m: m.name)
        except ProviderError as exc:
            err.print(f"[red]{name}:[/] {exc}")
            ok = False
            continue
        table = Table(title=f"{name} ({len(available)} models)", box=None, header_style="bold")
        table.add_column("model")
        table.add_column("context", justify="right")
        table.add_column("max output", justify="right")
        table.add_column("routed")
        for m in available:
            spec = f"{name}:{m.name}"
            table.add_row(
                m.name,
                f"{m.input_token_limit:,}" if m.input_token_limit else "?",
                f"{m.output_token_limit:,}" if m.output_token_limit else "?",
                "yes" if spec in configured else "",
            )
        console.print(table)
        names = {m.name for m in available}
        for spec in sorted(s for s in configured if s.startswith(f"{name}:")):
            if spec.partition(":")[2] not in names:
                err.print(f"[red]configured route {spec} is not available[/]")
                ok = False
    if not cfg.llm.routes:
        err.print("[yellow]No routes configured yet: set [llm.routes] in your config.[/]")
    if not ok:
        raise typer.Exit(1)


# -- chat -----------------------------------------------------------------------------------------


def _label(style: str, label: str, text: str) -> None:
    """Print `[label] text`. Labels and text are escaped: tool output, paths and feedback come
    from the model or the repo, and a `[...]` in them must never be parsed as Rich markup."""
    console.print(f"[{style}]{escape(f'[{label}]')}[/] {escape(text)}", highlight=False)


def _render_event(event: AgentEvent) -> None:
    if isinstance(event, RetrievalDone):
        _label("dim", "retrieval", event.bundle.summary())
    elif isinstance(event, AssistantText):
        console.print(event.text, end="", markup=False, highlight=False, soft_wrap=True)
    elif isinstance(event, EditProposed):
        _label("cyan", "edit", f"{event.block.path} (block {event.block.index + 1})")
    elif isinstance(event, ToolStarted):
        args = ", ".join(f"{k}={v!r}" for k, v in event.call.arguments.items())
        _label("dim", "tool", f"{event.call.name}({args})")
    elif isinstance(event, ToolFinished) and event.is_error:
        _label("yellow", "tool error", event.summary)
    elif isinstance(event, EditsRejected):
        first = event.feedback.strip().splitlines()[0] if event.feedback.strip() else ""
        _label("yellow", f"edits rejected, attempt {event.attempt}", first)
    elif isinstance(event, Status):
        console.print(escape(event.message), style="dim", highlight=False)


def _run_cancellable[T](fn: Callable[[CancelToken], T]) -> T:
    """Run `fn` on a worker thread; Ctrl+C sets the cancel token instead of killing mid-write."""
    cancel = CancelToken()
    box: dict[str, T] = {}
    errors: list[BaseException] = []

    def target() -> None:
        try:
            box["value"] = fn(cancel)
        except BaseException as exc:  # re-raised on the main thread
            errors.append(exc)

    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    try:
        while worker.is_alive():
            worker.join(0.1)
    except KeyboardInterrupt:
        cancel.cancel()
        console.print("\n[yellow]cancelling...[/]")
        worker.join()
    if errors:
        raise errors[0]
    return box["value"]


def _status_line(result: TaskResult, seconds: float) -> str:
    u = result.usage
    cost = "unknown (no price configured)" if u.cost_usd is None else f"${u.cost_usd:.4f}"
    return (
        f"{result.status.value.lower()} | {u.input_tokens:,} in + {u.output_tokens:,} out tokens | "
        f"cost {cost} | {u.llm_calls} model calls, {u.tool_calls} tool calls | {seconds:.1f}s"
    )


def _handle_task(session: AgentSession, task: str) -> TaskResult:
    started = time.monotonic()
    request_id, result = _run_cancellable(
        lambda cancel: session.run_task(task, cancel, _render_event)
    )
    console.print()
    if result.plan is None:
        style = "green" if result.status is TaskStatus.SUCCEEDED else "yellow"
        console.print(f"[{style}]{result.message}[/]")
    else:
        added, removed = diff_stats(result.plan)
        console.rule(f"diff: {len(result.plan.changes)} file(s), +{added} -{removed}")
        console.print(Syntax(plan_diff(result.plan), "diff", theme="ansi_dark", word_wrap=True))
        if typer.confirm(f"Apply changes to {len(result.plan.changes)} file(s)?", default=False):
            try:
                change_set = session.apply(request_id, result)
            except WriteConflictError as exc:
                err.print(f"[red]Not applied:[/] {exc}. Nothing was written; ask again.")
            else:
                console.print(f"[green]Applied[/] (change set {change_set[:12]}). "
                              "Revert with `agent undo`.")  # fmt: skip
        else:
            session.decline(request_id, result)
            console.print("Discarded; no files were changed.")
    console.print(f"[dim]{_status_line(result, time.monotonic() - started)}[/]")
    return result


@app.command()
def chat(
    message: Annotated[
        str | None, typer.Option("--message", "-m", help="Run one task, then exit.")
    ] = None,
    path: PathOpt = Path("."),
) -> None:
    """Interactive session: describe a change, review the diff, approve it."""
    cfg = load_config()
    ws = Workspace.discover(path)
    session = AgentSession(ws, cfg, _embedder(cfg))
    try:
        problem = session.check_generation()
        if problem:
            err.print(f"[yellow]Generation unavailable:[/] {problem}")
            err.print("`agent index`, `agent search` and `agent undo` still work.")
            raise typer.Exit(1)
        stats = session.refresh_index()
        if stats.changed:
            console.print(
                f"[dim]index refreshed: +{stats.added} ~{stats.updated} -{stats.removed}[/]"
            )
        if message is not None:
            result = _handle_task(session, message)
            raise typer.Exit(0 if result.status is not TaskStatus.FAILED else 1)

        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory

        prompt = PromptSession(history=FileHistory(str(ws.ensure_state_dir() / "chat_history")))
        console.print(f"[bold]{ws.root}[/]  (Ctrl+C cancels a running task, Ctrl+D exits)")
        while True:
            try:
                task = prompt.prompt("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if task in ("exit", "quit"):
                break
            if task:
                session.refresh_index()
                _handle_task(session, task)
    finally:
        session.close()
