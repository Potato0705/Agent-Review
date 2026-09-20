$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

python -m compileall -q agent_audit tests
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python -m agent_audit audit `
  --input examples/demo_scores.csv `
  --report outputs/review_report.md `
  --json outputs/review_result.json `
  --score-min 0 `
  --score-max 10 `
  --data-provenance synthetic
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

git diff --check
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

git diff --cached --check
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "All local quality gates passed."
