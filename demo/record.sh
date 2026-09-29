#!/bin/bash
# Records both demos into /out (asciinema casts + GIFs). No API key or network needed.
set -euo pipefail
export HOME=/home/agent NO_COLOR= COLUMNS=108 LINES=34 HF_HUB_OFFLINE=1
setup_repo() {  # $1 = fixture dir, $2 = target
  rm -rf "$2"; cp -r "$1" "$2"
  git -C "$2" init -q -b main && git -C "$2" add -A
  git -C "$2" -c user.email=demo@example.com -c user.name=demo commit -qm init
}
config() {  # $1 = script, $2 = extra toml
  cat > /tmp/agent.toml <<TOML
[llm.providers.fake]
kind = "fake"
script = "$1"
[llm.routes]
cheap = ["fake:fake-cheap"]
strong = ["fake:fake-strong"]
$2
TOML
  export CODE_AGENT_CONFIG=/tmp/agent.toml
}
record() {  # $1 = name, $2 = expect script, $3 = repo dir
  (cd "$3" && asciinema rec --overwrite -q --cols 108 --rows 34 \
      -c "expect $2" "/out/$1.cast")
  agg --theme monokai --font-family "DejaVu Sans Mono" --font-size 14 --speed 1.4 "/out/$1.cast" "/out/$1.gif"
}

setup_repo /demo/fixtures/sample_repo /tmp/tokens
config /demo/fix-turns.json ""
record fix /demo/fix.exp /tmp/tokens

setup_repo /demo/fixtures/injection_repo /tmp/payments
echo "OPENAI_API_KEY=sk-proj-demoFakeKey000000000000" > /tmp/payments/.env
echo "outside the workspace" > /tmp/outside.txt
config /demo/attack-turns.json $'[budgets]\nmax_edit_attempts = 1\n[validation]\nenabled = false'
record attack /demo/attack.exp /tmp/payments
ls -la /out
