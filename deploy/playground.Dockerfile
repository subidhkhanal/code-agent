# Playground image (docs/playground.md). Build from the repo root:
#   docker build -f deploy/playground.Dockerfile -t code-agent-playground .
# Render builds it the same way (render.yaml).
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
RUN git config --global --add safe.directory '*'

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CODE_AGENT_CONFIG=/opt/code-agent/deploy/playground.toml \
    PLAYGROUND_STATE_DIR=/home/user/state \
    PLAYGROUND_PROXY_HOPS=1
WORKDIR /home/user
EXPOSE 7860
# The port comes from $PORT when the host sets it (Render does), else 7860.
CMD ["python", "-m", "code_agent.playground", "--host", "0.0.0.0"]
