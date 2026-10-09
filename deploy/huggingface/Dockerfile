# Playground image (docs/playground.md). The build context is the folder made by
# deploy/huggingface/assemble.sh: pyproject.toml, README.md, src/, deploy/playground.toml.
#   deploy/huggingface/assemble.sh /tmp/space && docker build -t code-agent-playground /tmp/space
FROM python:3.12-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends git libatomic1 \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 1000 user

# Installed as root, so the code and the sample repos are read-only to the app user, and so to
# any model-written code the playground runs.
WORKDIR /opt/code-agent
COPY pyproject.toml README.md ./
COPY src ./src
COPY deploy/playground.toml ./deploy/playground.toml
RUN pip install --no-cache-dir ".[playground]" pytest

USER user
# Fetched at build time instead of on first request: the embedding model and pyright's Node.js.
RUN python -c "from pathlib import Path; from fastembed import TextEmbedding; \
TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir=str(Path.home() / '.cache/code-agent/models'))" \
 && python -m pyright --version \
 && git config --global --add safe.directory '*'

ENV HF_HUB_OFFLINE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CODE_AGENT_CONFIG=/opt/code-agent/deploy/playground.toml \
    PLAYGROUND_STATE_DIR=/home/user/state \
    PLAYGROUND_PROXY_HOPS=1
WORKDIR /home/user
EXPOSE 7860
CMD ["python", "-m", "code_agent.playground", "--host", "0.0.0.0", "--port", "7860"]
