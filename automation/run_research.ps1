param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("check", "smoke", "research", "resume", "report", "record-score")]
    [string]$Mode,
    [int]$MaxExperiments = 0,
    [double]$MaxHours = 0,
    [string]$ExperimentId = "",
    [Nullable[double]]$LeaderboardScore = $null,
    [ValidateSet("public", "private", "unknown")]
    [string]$ScoreType = "unknown",
    [string]$Notes = "",
    [string]$CandidateId = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonCandidates = [System.Collections.Generic.List[string]]::new()
$pythonCandidates.Add((Join-Path $projectRoot ".venv\Scripts\python.exe"))
if ($env:VIRTUAL_ENV) {
    $pythonCandidates.Add((Join-Path $env:VIRTUAL_ENV "Scripts\python.exe"))
}
$pathPython = Get-Command python -ErrorAction SilentlyContinue
if ($pathPython) { $pythonCandidates.Add($pathPython.Source) }
$pyLauncher = Get-Command py -ErrorAction SilentlyContinue

$python = $null
foreach ($candidate in $pythonCandidates) {
    if ($candidate -and (Test-Path -LiteralPath $candidate -PathType Leaf)) {
        $python = (Resolve-Path -LiteralPath $candidate).Path
        break
    }
}
if (-not $python -and $pyLauncher) {
    $probe = & $pyLauncher.Source -3 -c "import sys; print(sys.executable)"
    if ($LASTEXITCODE -eq 0 -and (Test-Path -LiteralPath $probe -PathType Leaf)) { $python = $probe }
}
if (-not $python) {
    Write-Error "사용 가능한 Python 3 실행 파일을 찾지 못했습니다."
    exit 2
}

$arguments = @((Join-Path $PSScriptRoot "src\research.py"), "--mode", $Mode)
if ($MaxExperiments -gt 0) { $arguments += @("--max-experiments", "$MaxExperiments") }
if ($MaxHours -gt 0) { $arguments += @("--max-hours", "$MaxHours") }
if ($ExperimentId) { $arguments += @("--experiment-id", $ExperimentId) }
if ($null -ne $LeaderboardScore) { $arguments += @("--leaderboard-score", "$LeaderboardScore") }
if ($ScoreType) { $arguments += @("--score-type", $ScoreType) }
if ($Notes) { $arguments += @("--notes", $Notes) }
if ($CandidateId) { $arguments += @("--candidate-id", $CandidateId) }

Write-Host "Python: $python"
Write-Host "Mode: $Mode"
& $python @arguments
exit $LASTEXITCODE
