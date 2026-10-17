# Hosted playground

A web page where anyone can run the agent on a small sample repo: pick a repo, describe a fix,
watch the events stream in (retrieval, tool calls, blocked commands, edits, validation), and get
a validated diff. Design and threat model: [ADR 0011](adr/0011-hosted-playground.md).

## Run it locally

```bash
pip install -e ".[playground]"
export ANTHROPIC_API_KEY=...         # PowerShell: $env:ANTHROPIC_API_KEY = "..."
export CODE_AGENT_CONFIG=deploy/playground.toml
python -m code_agent.playground      # http://127.0.0.1:7860
```

Or the exact image that gets deployed:

```bash
docker build -f deploy/playground.Dockerfile -t code-agent-playground .
docker run --rm -p 7860:7860 -e ANTHROPIC_API_KEY code-agent-playground
```

## Limits

Environment variables, read at startup:

| Variable | Default | |
|---|---|---|
| `PLAYGROUND_DAILY_USD` | `3.0` | Global spend per UTC day |
| `PLAYGROUND_RUNS_PER_VISITOR` | `3` | Runs per visitor per day |
| `PLAYGROUND_RUN_MAX_USD` | `0.5` | Cost cap per run (each run reserves twice this) |
| `PLAYGROUND_RUN_MAX_SECONDS` | `240` | Time cap per run |
| `PLAYGROUND_RUN_MAX_TOOL_CALLS` | `25` | Tool-call cap per run |
| `PLAYGROUND_MAX_CONCURRENT` | `2` | Runs at the same time |
| `PLAYGROUND_MAX_TASK_CHARS` | `600` | Longest task description |
| `PLAYGROUND_CLIENT_IP_HEADER` | unset (`cf-connecting-ip` on Render) | Header the edge proxy overwrites with the client address; preferred over `X-Forwarded-For` |
| `PLAYGROUND_PROXY_HOPS` | `0` (`1` in the image) | Reverse proxies that append to `X-Forwarded-For` |
| `PLAYGROUND_STATE_DIR` | `~/.cache/code-agent` | Where the day's spend is persisted |

The model, effort level and prices are in [`deploy/playground.toml`](../deploy/playground.toml).

## Deploy to Render (free)

[`render.yaml`](../render.yaml) describes the service: Docker, free plan (no card; 0.1 CPU,
512 MB, sleeps after 15 idle minutes), health check on `/api/status`, and automatic redeploys of
`main` once CI passes. The config in `deploy/playground.toml` is sized for that box; measured
under the same limits, a run takes about 65 s and peaks at about 140 MB.

1. **API key.** In the Claude Console, create a key for the playground and set a monthly spend
   limit on its workspace (for example USD 20). This is the hard stop if anything else fails.
2. **Deploy.** Open
   <https://render.com/deploy?repo=https://github.com/subidhkhanal/code-agent>, sign in with
   GitHub, and paste the key when Render asks for `ANTHROPIC_API_KEY`. The first build takes a
   few minutes.

After the first deploy, check that per-visitor limits see real addresses: the runs-left counter
must drop for you after a run, and must not drop when you send a request with a made-up
`X-Forwarded-For` header from a different address.

Hugging Face Spaces was the first choice, but Docker Spaces now need a paid PRO subscription.
