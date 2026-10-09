"""Start the playground: `python -m code_agent.playground [--host H] [--port P]`.

The playground runs model-written code (the sample's tests, after the model's edits) in the
same container as the server, which holds the API key. The key is protected in code:

1. Key hand-off (Linux): the launcher reads the key from the environment, writes it into a pipe
   and re-executes itself with an environment that no longer contains it. After `execve`,
   `/proc/<pid>/environ` shows only the new environment, so the key lives in memory alone.
2. The server marks itself non-dumpable (`prctl(PR_SET_DUMPABLE, 0)`): other processes of the
   same user, such as the tests it spawns, can no longer read its `/proc/<pid>/mem` or
   `environ`.
3. Commands the agent runs get an allowlisted environment (`security/runner.py`): no secrets.
4. Everything streamed to the browser is scrubbed of the key by exact match (`app.py`).

On other platforms (local development on Windows/macOS) step 1 and 2 are skipped and the key
is simply removed from `os.environ` after it is read.
"""

from __future__ import annotations

import argparse
import ctypes
import logging
import os
import sys
from pathlib import Path

KEY_ENV = "ANTHROPIC_API_KEY"
PR_SET_DUMPABLE = 4


def _hand_off_key(argv: list[str]) -> None:
    """Re-exec without the key in the environment; the child reads it from a pipe. Linux only."""
    key = os.environ[KEY_ENV]
    read_end, write_end = os.pipe()
    os.write(write_end, key.encode())
    os.close(write_end)
    os.set_inheritable(read_end, True)
    env = {k: v for k, v in os.environ.items() if k != KEY_ENV}
    args = [sys.executable, "-m", "code_agent.playground", *argv, "--key-fd", str(read_end)]
    os.execve(sys.executable, args, env)


def _read_key(fd: int | None) -> str | None:
    if fd is not None:
        with os.fdopen(fd, "rb") as fh:
            return fh.read().decode().strip() or None
    return os.environ.pop(KEY_ENV, None)


def _make_undumpable() -> None:
    if sys.platform.startswith("linux"):
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "prctl(PR_SET_DUMPABLE) failed")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m code_agent.playground")
    parser.add_argument("--host", default="127.0.0.1")
    # Hosts such as Render tell the app which port to bind through $PORT.
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "7860")))
    parser.add_argument("--config", type=Path, default=None, help="agent config TOML")
    parser.add_argument("--key-fd", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.key_fd is None and sys.platform.startswith("linux") and os.environ.get(KEY_ENV):
        _hand_off_key(sys.argv[1:])  # does not return
    key = _read_key(args.key_fd)
    _make_undumpable()

    # Heavy imports after the hand-off, so the first process stays small and short-lived.
    import uvicorn

    from code_agent.config import load_config
    from code_agent.index.embeddings import FastEmbedEmbedder
    from code_agent.llm.claude import AnthropicProvider
    from code_agent.llm.factory import build_gateway, build_providers
    from code_agent.llm.gateway import ModelUnavailableError
    from code_agent.llm.providers import Provider
    from code_agent.llm.types import ProviderError
    from code_agent.playground.app import Settings, create_app
    from code_agent.playground.engine import CachedModels, LockedEmbedder

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    log = logging.getLogger("code_agent.playground")

    cfg = load_config(args.config)
    raw = build_providers(cfg.llm)
    for provider in raw.values():
        if isinstance(provider, AnthropicProvider):
            provider.api_key = key
    providers: dict[str, Provider] = {name: CachedModels(p) for name, p in raw.items()}

    model_error = None
    try:
        build_gateway(cfg.llm, providers=dict(providers)).verify_models()
    except (ModelUnavailableError, ProviderError) as exc:
        model_error = str(exc)
        log.error("model check failed; runs are disabled: %s", exc)

    embedder = None
    if cfg.index.embeddings:
        embedder = LockedEmbedder(FastEmbedEmbedder(
            cfg.index.embedding_model, cfg.model_cache_dir,
            batch_size=cfg.index.embedding_batch_size,
        ))  # fmt: skip

    app = create_app(
        cfg,
        dict(providers),
        embedder,
        settings=Settings.from_env(),
        server_secrets=[key] if key else [],
        model_error=model_error,
    )
    uvicorn.run(app, host=args.host, port=args.port, proxy_headers=False, log_level="info")


if __name__ == "__main__":
    main()
