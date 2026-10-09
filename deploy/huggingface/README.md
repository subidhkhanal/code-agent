---
title: code-agent playground
emoji: 🛠️
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: Run a coding agent that validates its own edits
---

# code-agent playground

Pick a small buggy Python repo, describe a fix, and watch the agent search the code, propose an
edit, and validate it with lint, type checks and the repo's tests before it shows you the diff.

Source, architecture and evaluation: https://github.com/subidhkhanal/code-agent

This Space is deployed automatically from that repository (`deploy/huggingface/`); edits made
here are overwritten on the next deploy.
