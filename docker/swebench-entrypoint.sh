#!/bin/bash
# Runs inside a SWE-bench task image with the agent runtime mounted at /opt/agent.
# Starts as root only to prepare a writable copy of the repo, then drops to uid 1000.
set -euo pipefail

cp -a /testbed /work
chown -R 1000:1000 /work
chmod -R a+rwX /out
git config --system --add safe.directory '*'

exec setpriv --reuid=1000 --regid=1000 --clear-groups \
  env HOME=/opt/agent/home \
      PATH="/opt/agent/venv/bin:${PATH}" \
      CODE_AGENT_CONFIG=/config/agent.toml \
      PYTHONDONTWRITEBYTECODE=1 \
  agent run --task "$(cat /task/problem.txt)" --headless --auto-approve -p /work --out /out
