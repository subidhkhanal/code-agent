# Agent runtime for SWE-bench: everything under /opt/agent, mounted into each task's own image
# (so the task image is used unmodified, with its Python and dependencies intact).
#   docker build -f docker/runtime.Dockerfile -t code-agent-runtime:latest .
FROM ubuntu:22.04

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates git libatomic1 \
 && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# A self-contained CPython 3.12 (python-build-standalone) and a venv, both under /opt/agent, so
# the whole runtime is relocatable as one directory into images running any Ubuntu 22.04 userland.
ENV UV_PYTHON_INSTALL_DIR=/opt/agent/python UV_NO_CACHE=1
RUN uv python install 3.12 && uv venv /opt/agent/venv --python 3.12
COPY pyproject.toml README.md /src/
COPY src /src/src
RUN uv pip install --python /opt/agent/venv/bin/python /src

# Fetch at build time what would otherwise be downloaded on first use (no internet at run time).
ENV HOME=/opt/agent/home
ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RUN mkdir -p /opt/agent/home \
 && /opt/agent/venv/bin/python -c "from pathlib import Path; from fastembed import TextEmbedding; \
TextEmbedding('${EMBEDDING_MODEL}', cache_dir=str(Path.home() / '.cache/code-agent/models'))" \
 && /opt/agent/venv/bin/python -m pyright --version \
 && chmod -R a+rwX /opt/agent/home && chmod -R a+rX /opt/agent

COPY docker/swebench-entrypoint.sh /opt/agent/entrypoint.sh
RUN chmod a+rx /opt/agent/entrypoint.sh
VOLUME /opt/agent
