"""The playground web app: a static UI plus a small JSON/SSE API in front of `engine.run_task`.

    GET  /                       the UI
    GET  /api/samples            sample repos and suggested tasks
    GET  /api/samples/{id}/files a sample's source, for the code browser
    GET  /api/status             runs left for this visitor, budget open, busy, model
    POST /api/run                {sample, task} -> text/event-stream of run events

Every byte sent to the browser passes through `scrub`: the server's own secrets are replaced by
exact match, then the outbound secret scanner runs over the rest. Closing the browser tab
cancels the run.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from code_agent.config import AgentConfig
from code_agent.index.embeddings import Embedder
from code_agent.llm.providers import Provider
from code_agent.llm.types import CancelToken
from code_agent.playground.engine import RunLimits, Sample, load_samples, run_task
from code_agent.playground.guard import RunRefusedError, SpendGuard
from code_agent.security.secrets import redact

log = logging.getLogger("code_agent.playground")
STATIC_DIR = Path(__file__).parent / "static"


def _env(name: str, default: str) -> str:
    return os.environ.get(f"PLAYGROUND_{name}", default)


@dataclass(frozen=True)
class Settings:
    daily_usd: float = 3.0
    runs_per_visitor: int = 3
    max_concurrent: int = 2
    run_max_usd: float = 0.5
    run_max_seconds: int = 240
    run_max_tool_calls: int = 25
    max_task_chars: int = 600
    # How many reverse proxies in front of the app append to X-Forwarded-For (0 = use the
    # socket peer). Only the per-visitor limit depends on this; the daily budget is global.
    proxy_hops: int = 0
    state_dir: Path = field(default_factory=lambda: Path.home() / ".cache" / "code-agent")

    @classmethod
    def from_env(cls) -> Settings:
        d = cls()
        return cls(
            daily_usd=float(_env("DAILY_USD", str(d.daily_usd))),
            runs_per_visitor=int(_env("RUNS_PER_VISITOR", str(d.runs_per_visitor))),
            max_concurrent=int(_env("MAX_CONCURRENT", str(d.max_concurrent))),
            run_max_usd=float(_env("RUN_MAX_USD", str(d.run_max_usd))),
            run_max_seconds=int(_env("RUN_MAX_SECONDS", str(d.run_max_seconds))),
            run_max_tool_calls=int(_env("RUN_MAX_TOOL_CALLS", str(d.run_max_tool_calls))),
            max_task_chars=int(_env("MAX_TASK_CHARS", str(d.max_task_chars))),
            proxy_hops=int(_env("PROXY_HOPS", str(d.proxy_hops))),
            state_dir=Path(_env("STATE_DIR", str(d.state_dir))),
        )

    @property
    def reserve_usd(self) -> float:
        # The per-run cap is checked before each model call, so one call can overshoot it.
        # Reserving twice the cap covers that overshoot at the configured output limit.
        return 2 * self.run_max_usd


def make_scrubber(server_secrets: list[str]) -> Callable[[str], str]:
    secrets_ = [s for s in server_secrets if s]

    def scrub(text: str) -> str:
        for value in secrets_:
            text = text.replace(value, "[REDACTED:server_secret]")
        return redact(text)[0]

    return scrub


class RunBody(BaseModel):
    sample: str
    task: str


def create_app(
    cfg: AgentConfig,
    providers: dict[str, Provider],
    embedder: Embedder | None,
    *,
    settings: Settings,
    server_secrets: list[str],
    model_error: str | None = None,
    guard: SpendGuard | None = None,
) -> FastAPI:
    app = FastAPI(title="code-agent playground", docs_url=None, redoc_url=None)
    app.state.workers = set()  # keeps run futures referenced until they finish
    samples: dict[str, Sample] = load_samples()
    scrub = make_scrubber(server_secrets)
    guard = guard or SpendGuard(
        settings.state_dir / "playground-spend.json",
        daily_usd=settings.daily_usd,
        runs_per_visitor=settings.runs_per_visitor,
        max_concurrent=settings.max_concurrent,
        reserve_usd=settings.reserve_usd,
    )
    limits = RunLimits(settings.run_max_usd, settings.run_max_seconds, settings.run_max_tool_calls)
    strong = cfg.llm.routes.get("strong", [""])[0]
    model = strong.partition(":")[2] or strong

    def visitor(request: Request) -> str:
        address = request.client.host if request.client else "unknown"
        if settings.proxy_hops > 0:
            chain = [a.strip() for a in request.headers.get("x-forwarded-for", "").split(",")]
            chain = [a for a in chain if a]
            if len(chain) >= settings.proxy_hops:
                address = chain[-settings.proxy_hops]
        return guard.visitor_id(address)

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/samples")
    def list_samples() -> list[dict]:
        return [
            {"id": s.id, "title": s.title, "blurb": s.blurb, "tasks": s.tasks}
            for s in samples.values()
        ]

    @app.get("/api/samples/{sample_id}/files")
    def sample_files(sample_id: str) -> dict[str, str]:
        sample = samples.get(sample_id)
        if sample is None:
            raise HTTPException(404, "unknown sample")
        return sample.files()

    @app.get("/api/status")
    def status(request: Request) -> dict:
        return {
            **guard.status(visitor(request)),
            "model": model,
            "available": model_error is None,
            "run_max_usd": settings.run_max_usd,
            "max_task_chars": settings.max_task_chars,
        }

    @app.post("/api/run")
    async def run(body: RunBody, request: Request):
        sample = samples.get(body.sample)
        if sample is None:
            raise HTTPException(404, "unknown sample")
        task = body.task.strip()
        if not 3 <= len(task) <= settings.max_task_chars:
            raise HTTPException(400, f"describe the task in 3-{settings.max_task_chars} characters")
        if model_error is not None:
            return JSONResponse({"error": "The model is unavailable right now."}, status_code=503)
        try:
            ticket = guard.admit(visitor(request))
        except RunRefusedError as exc:
            return JSONResponse({"error": str(exc)}, status_code=429)

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[dict | None] = asyncio.Queue()
        cancel = CancelToken()

        def emit(event: dict) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, event)

        def work() -> None:
            cost = None
            try:
                cost = run_task(sample, task, cfg=cfg, providers=providers, embedder=embedder,
                                limits=limits, cancel=cancel, emit=emit)  # fmt: skip
            except Exception:
                log.exception("playground run failed")
            finally:
                guard.settle(ticket, cost)
                loop.call_soon_threadsafe(queue.put_nowait, None)

        app.state.workers.add(worker := loop.run_in_executor(None, work))
        worker.add_done_callback(app.state.workers.discard)

        async def events():
            try:
                while True:
                    try:
                        item = await asyncio.wait_for(queue.get(), timeout=15)
                    except TimeoutError:
                        yield ": keep-alive\n\n"  # proxies drop idle streams
                        continue
                    if item is None:
                        break
                    yield f"data: {scrub(json.dumps(item))}\n\n"
            finally:
                # The visitor left (or the run ended): stop spending. The worker still settles
                # the ticket when it unwinds.
                cancel.cancel()

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app
