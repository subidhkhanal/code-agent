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
deploy/huggingface/assemble.sh /tmp/space
docker build -t code-agent-playground /tmp/space
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
| `PLAYGROUND_PROXY_HOPS` | `0` (`1` in the image) | Reverse proxies that append to `X-Forwarded-For` |
| `PLAYGROUND_STATE_DIR` | `~/.cache/code-agent` | Where the day's spend is persisted |

The model, effort level and prices are in [`deploy/playground.toml`](../deploy/playground.toml).

## Deploy to Hugging Face Spaces

One-time setup:

1. **API key.** In the Claude Console, create a key for the playground and set a monthly spend
   limit on its workspace (for example USD 20). This is the hard stop if anything else fails.
2. **Deploy access.** Create a Hugging Face access token with *write* access. In the GitHub
   repo's *Settings → Secrets and variables → Actions*, add the secret `HF_TOKEN` (the token)
   and the variable `HF_SPACE` (for example `your-name/code-agent`).
3. **First deploy.** Run the *Playground* workflow from the Actions tab (or push to `main`). It
   creates the Space (Docker, free CPU tier) if it doesn't exist and uploads the build folder.
4. **Secret.** In the Space's *Settings → Variables and secrets*, add a **secret** named
   `ANTHROPIC_API_KEY`. Paste the key there and nowhere else. The Space restarts with it.

From then on, every push to `main` that passes CI redeploys the Space
([`.github/workflows/playground.yml`](../.github/workflows/playground.yml)). A build takes
about five minutes.

After the first deploy, check that per-visitor limits see real addresses: the runs-left counter
must drop for you after a run, and must not drop when you send a request with a made-up
`X-Forwarded-For` header from a different address.
