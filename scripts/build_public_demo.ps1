$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

python -m agent_audit audit `
  --input examples/demo_scores.csv `
  --report outputs/public_demo_report.md `
  --json outputs/public_demo_result.json `
  --html docs/examples/example_audit_report.html `
  --score-min 0 `
  --score-max 10 `
  --data-provenance synthetic
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Public HTML demo rebuilt successfully."
