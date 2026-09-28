"""`agent` command-line entry point."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.syntax import Syntax
from rich.table import Table

from code_agent.config import AgentConfig, load_config
from code_agent.index.embeddings import Embedder, FastEmbedEmbedder
from code_agent.index.indexer import Indexer, IndexStats
from code_agent.index.store import IndexStore, open_index
from code_agent.index.watcher import watch
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
    rebuild: Annotated[bool, typer.Option("--rebuild", help="Discard the index first.")] = False,
) -> None:
    """Build or incrementally refresh the index for the current repository."""
    cfg = load_config()
    ws = Workspace.discover(path)
    if rebuild:
        for suffix in ("", "-wal", "-shm"):
            Path(str(ws.index_path) + suffix).unlink(missing_ok=True)
    conn, repairs = open_index(ws)
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
