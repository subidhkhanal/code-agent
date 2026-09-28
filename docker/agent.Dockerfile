# Headless agent image (ADR 0009). Build from the repo root:
#   docker build -f docker/agent.Dockerfile -t code-agent:latest .
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends git libatomic1 \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 1000 agent

WORKDIR /opt/code-agent
COPY pyproject.toml README.md ./
COPY src ./src
# pytest is included so that projects without their own environment can still run their tests
# during shadow validation (SWE-bench images bring their own interpreter and pytest instead).
RUN pip install --no-cache-dir . pytest

USER agent
# The container has no internet at run time, so everything that is normally downloaded on first
# use is fetched now: the embedding model weights and pyright's Node.js runtime.
ARG EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
RUN python -c "from pathlib import Path; from fastembed import TextEmbedding; \
TextEmbedding('${EMBEDDING_MODEL}', cache_dir=str(Path.home() / '.cache/code-agent/models'))" \
 && python -m pyright --version \
 && git config --global --add safe.directory '*'

ENV HF_HUB_OFFLINE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /work
ENTRYPOINT ["agent"]
