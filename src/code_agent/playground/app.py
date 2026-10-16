"""The playground web app: a static UI plus a small JSON/SSE API in front of `engine.run_task`.

    GET  /                       the UI
    GET  /api/samples            sample repos and suggested tasks
    GET  /api/samples/{id}/files a sample's source, for the code browser
    GET  /api/status             runs left for this visitor, budget open, busy
    POST /api/run                {sample, task} -> text/event-stream of run events

Every byte sent to the browser passes through `scrub`: the server's own secrets are replaced by
exact match, then the outbound secret scanner runs over the rest. Configured model ids are
replaced too: the page never says which model is behind it (error messages name routes).
Closing the browser tab cancels the run.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import string
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
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
    daily_usd: float = 3.0  # owner-paid (free) runs only
    free_runs_per_device: int = 5
    free_runs_per_ip: int = 15  # backstop: clearing browser storage makes a new device id
    max_concurrent: int = 2
    run_max_usd: float = 0.5
    byok_run_max_usd: float = 1.0  # cap per run on a visitor's own key, to protect them
    run_max_seconds: int = 240
    run_max_tool_calls: int = 25
    max_task_chars: int = 600
    # Where the visitor's address comes from. A header the edge proxy *overwrites* (e.g.
    # Cloudflare's cf-connecting-ip) is best: clients can't spoof it. Otherwise, how many
    # reverse proxies append to X-Forwarded-For (0 = use the socket peer). Only the
    # per-visitor limit depends on this; the daily budget is global.
    client_ip_header: str = ""
    proxy_hops: int = 0
    state_dir: Path = field(default_factory=lambda: Path.home() / ".cache" / "code-agent")
    # Other sites allowed to call the API from a browser (the UI hosted on GitHub Pages).
    cors_origins: tuple[str, ...] = ()

    @classmethod
    def from_env(cls) -> Settings:
        d = cls()
        return cls(
            daily_usd=float(_env("DAILY_USD", str(d.daily_usd))),
            free_runs_per_device=int(_env("FREE_RUNS_PER_DEVICE", str(d.free_runs_per_device))),
            free_runs_per_ip=int(_env("FREE_RUNS_PER_IP", str(d.free_runs_per_ip))),
            byok_run_max_usd=float(_env("BYOK_RUN_MAX_USD", str(d.byok_run_max_usd))),
            max_concurrent=int(_env("MAX_CONCURRENT", str(d.max_concurrent))),
            run_max_usd=float(_env("RUN_MAX_USD", str(d.run_max_usd))),
            run_max_seconds=int(_env("RUN_MAX_SECONDS", str(d.run_max_seconds))),
            run_max_tool_calls=int(_env("RUN_MAX_TOOL_CALLS", str(d.run_max_tool_calls))),
            max_task_chars=int(_env("MAX_TASK_CHARS", str(d.max_task_chars))),
            client_ip_header=_env("CLIENT_IP_HEADER", d.client_ip_header).lower(),
            proxy_hops=int(_env("PROXY_HOPS", str(d.proxy_hops))),
            state_dir=Path(_env("STATE_DIR", str(d.state_dir))),
            cors_origins=tuple(o.strip() for o in _env("CORS_ORIGINS", "").split(",") if o.strip()),
        )

    @property
    def reserve_usd(self) -> float:
        # The per-run cap is checked before each model call, so one call can overshoot it.
        # Reserving twice the cap covers that overshoot at the configured output limit.
        return 2 * self.run_max_usd


def make_scrubber(server_secrets: list[str], hidden: Sequence[str] = ()) -> Callable[[str], str]:
    secrets_ = [s for s in server_secrets if s]
    # Longest first, so "provider:model" is replaced before its "model" part.
    hidden_ = sorted({h for h in hidden if h}, key=len, reverse=True)

    def scrub(text: str) -> str:
        for value in secrets_:
            text = text.replace(value, "[REDACTED:server_secret]")
        for value in hidden_:
            text = text.replace(value, "the model")
        return redact(text)[0]

    return scrub


_TOKEN_CHARS = frozenset(string.ascii_letters + string.digits + "-_")


def _clean_token(value: str, *, prefix: str = "", min_len: int, max_len: int) -> str | None:
    """A header value made of [A-Za-z0-9_-] only, within length bounds, else None."""
    value = value.strip()
    if not (min_len <= len(value) <= max_len and value.startswith(prefix)):
        return None
    return value if set(value) <= _TOKEN_CHARS else None


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
    byok_providers: Callable[[str], dict[str, Provider]] | None = None,
) -> FastAPI:
    """`byok_providers(key)` builds providers that use a visitor's own key; None disables
    bring-your-own-key."""
    app = FastAPI(title="code-agent playground", docs_url=None, redoc_url=None)
    app.state.workers = set()  # keeps run futures referenced until they finish
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "X-Device-Id", "X-Anthropic-Key"],
        )
    samples: dict[str, Sample] = load_samples()
    routes = [spec for specs in cfg.llm.routes.values() for spec in specs]
    scrub = make_scrubber(server_secrets, routes + [r.partition(":")[2] for r in routes])
    guard = guard or SpendGuard(
        settings.state_dir / "playground-spend.json",
        daily_usd=settings.daily_usd,
        free_runs_per_device=settings.free_runs_per_device,
        free_runs_per_ip=settings.free_runs_per_ip,
        max_concurrent=settings.max_concurrent,
        reserve_usd=settings.reserve_usd,
        # Derived from the API key: secret, yet the same after a restart.
        salt=hashlib.sha256(b"playground-visitor:" + server_secrets[0].encode()).digest()
        if server_secrets
        else None,
    )
    limits = RunLimits(settings.run_max_usd, settings.run_max_seconds, settings.run_max_tool_calls)
    byok_limits = RunLimits(
        settings.byok_run_max_usd, settings.run_max_seconds, settings.run_max_tool_calls
    )

    def visitor(request: Request) -> str:
        address = request.client.host if request.client else "unknown"
        trusted = request.headers.get(settings.client_ip_header, "").strip()
        if settings.client_ip_header and trusted:
            return guard.ident(trusted)
        if settings.proxy_hops > 0:
            chain = [a.strip() for a in request.headers.get("x-forwarded-for", "").split(",")]
            chain = [a for a in chain if a]
            if len(chain) >= settings.proxy_hops:
                address = chain[-settings.proxy_hops]
        return guard.ident(address)

    def device(request: Request) -> str | None:
        raw = _clean_token(request.headers.get("x-device-id", ""), min_len=8, max_len=64)
        return guard.ident("device:" + raw) if raw else None

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
            **guard.status(device(request), visitor(request)),
            "available": model_error is None,
            "byok": byok_providers is not None,
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
        raw_key = request.headers.get("x-anthropic-key", "")
        user_key = None
        if raw_key:
            user_key = _clean_token(raw_key, prefix="sk-ant-", min_len=20, max_len=400)
            if user_key is None:
                return JSONResponse({"error": "That doesn't look like an Anthropic API key.",
                                     "code": "invalid_key"}, status_code=400)  # fmt: skip
            if byok_providers is None:
                return JSONResponse({"error": "Bring-your-own-key isn't enabled here."},
                                    status_code=503)  # fmt: skip
        elif model_error is not None:
            return JSONResponse({"error": "The free tier is unavailable right now. "
                                 "Add your own API key to run.", "code": "byok_required"},
                                status_code=503)  # fmt: skip
        try:
            ticket = (guard.admit_byok() if user_key
                      else guard.admit_free(device(request), visitor(request)))  # fmt: skip
        except RunRefusedError as exc:
            return JSONResponse({"error": str(exc), "code": exc.code}, status_code=429)
        run_providers = byok_providers(user_key) if user_key and byok_providers else providers
        run_limits = byok_limits if user_key else limits

        def run_scrub(text: str) -> str:
            # The visitor's key is scrubbed from their own stream too; it's never stored or logged.
            if user_key:
                text = text.replace(user_key, "[REDACTED:your_key]")
            return scrub(text)

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[dict | None] = asyncio.Queue()
        cancel = CancelToken()

        def emit(event: dict) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, event)

        def work() -> None:
            cost = None
            try:
                cost = run_task(sample, task, cfg=cfg, providers=run_providers,
                                embedder=embedder, limits=run_limits, cancel=cancel, emit=emit,
                                byok=user_key is not None)  # fmt: skip
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
                    yield f"data: {run_scrub(json.dumps(item))}\n\n"
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
