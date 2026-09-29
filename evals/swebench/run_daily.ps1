# One command per day on the Gemini free tier (20 requests/day per model):
# runs pending pilot tasks until the daily quota is used up, then scores finished tasks.
#   $env:GEMINI_API_KEY = "..."; .\evals\swebench\run_daily.ps1
$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path "$PSScriptRoot\..\..")
if (-not $env:GEMINI_API_KEY) { throw "Set GEMINI_API_KEY first." }
$env:PATH = "C:\Program Files\Docker\Docker\resources\bin;$env:PATH"
$env:PYTHONIOENCODING = "utf-8"
.\.venv\Scripts\python evals\swebench\run_agent.py --run pilot --pilot
if (Test-Path evals\swebench\runs\pilot\predictions.jsonl) {
    .\.venv-eval\Scripts\python evals\swebench\evaluate.py --run pilot
}
