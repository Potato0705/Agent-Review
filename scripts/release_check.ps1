param(
  [switch]$RequireClean
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $ProjectRoot

$ExpectedRemote = "https://github.com/Potato0705/Agent-Review.git"
$ExpectedName = "Potato0705"
$ExpectedEmail = "yaoshuowu75@gmail.com"

function Assert-LastExitCode([string]$Message) {
  if ($LASTEXITCODE -ne 0) {
    throw $Message
  }
}

& powershell -ExecutionPolicy Bypass -File scripts\review.ps1
Assert-LastExitCode "Local quality gates failed."

python scripts/mutation_check.py
Assert-LastExitCode "Mutation gate failed; a defect in the decision logic would ship undetected."

$remote = git remote get-url origin
Assert-LastExitCode "Cannot read origin remote."
if ($remote.Trim() -ne $ExpectedRemote) {
  throw "Unexpected origin remote: $remote"
}

$candidateFiles = @(git ls-files --cached --others --exclude-standard)
Assert-LastExitCode "Cannot list repository files."

$credentialNames = @(
  $candidateFiles | Where-Object {
    $_ -match '(^|/|\\)(\.env($|\.)|.*(secret|credential|private[_-]?key|api[_-]?key).*)' -or
    $_ -match '\.(pem|p12|pfx|key)$'
  }
)
if ($credentialNames.Count -gt 0) {
  throw "Credential-like filenames are present: $($credentialNames -join ', ')"
}

$largeFiles = @(
  foreach ($path in $candidateFiles) {
    if (Test-Path -LiteralPath $path -PathType Leaf) {
      $item = Get-Item -LiteralPath $path
      if ($item.Length -gt 1MB) {
        "$path ($($item.Length) bytes)"
      }
    }
  }
)
if ($largeFiles.Count -gt 0) {
  throw "Files larger than 1 MiB require explicit review: $($largeFiles -join ', ')"
}

$trackedOutputs = @(git ls-files outputs)
Assert-LastExitCode "Cannot inspect tracked outputs."
if ($trackedOutputs.Count -ne 1 -or $trackedOutputs[0] -ne "outputs/.gitkeep") {
  throw "Only outputs/.gitkeep may be tracked under outputs/."
}

$expectedIdentity = "$ExpectedName|$ExpectedEmail"
$authors = @(git log --all --format='%an|%ae' | Sort-Object -Unique)
Assert-LastExitCode "Cannot inspect commit authors."
if ($authors.Count -ne 1 -or $authors[0] -ne $expectedIdentity) {
  throw "Commit authors are not limited to $expectedIdentity. Found: $($authors -join ', ')"
}

$committers = @(git log --all --format='%cn|%ce' | Sort-Object -Unique)
Assert-LastExitCode "Cannot inspect committers."
if ($committers.Count -ne 1 -or $committers[0] -ne $expectedIdentity) {
  throw "Committers are not limited to $expectedIdentity. Found: $($committers -join ', ')"
}

$trailers = @(git log --all --format='%B' | Select-String -Pattern '^(Co-authored-by|Signed-off-by):' -CaseSensitive:$false)
if ($trailers.Count -gt 0) {
  throw "Contributor trailers are not allowed in this repository."
}

$expectedTagger = "$ExpectedName|<$ExpectedEmail>"
$taggers = @(git tag --format='%(taggername)|%(taggeremail)' | Where-Object { $_ -ne '|' } | Sort-Object -Unique)
Assert-LastExitCode "Cannot inspect taggers."
if ($taggers.Count -gt 0 -and ($taggers.Count -ne 1 -or $taggers[0] -ne $expectedTagger)) {
  throw "Taggers are not limited to $expectedTagger. Found: $($taggers -join ', ')"
}

git diff --check
Assert-LastExitCode "Working-tree whitespace check failed."
git diff --cached --check
Assert-LastExitCode "Staged whitespace check failed."

if ($RequireClean) {
  $status = @(git status --porcelain)
  Assert-LastExitCode "Cannot inspect working-tree status."
  if ($status.Count -gt 0) {
    throw "Working tree must be clean before push."
  }
}

Write-Host "Release checks passed for $($candidateFiles.Count) repository files."
Write-Host "Git authors, committers, and taggers are limited to $ExpectedName."
