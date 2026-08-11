[CmdletBinding()]
param(
    [ValidateRange(0.001, 168.0)][double]$MaxHours = 6,
    [ValidateRange(1, 100)][int]$MaxIterations = 8,
    [ValidateRange(1, 100)][int]$MaxFailures = 3,
    [ValidateRange(0.001, 1440.0)][double]$IterationTimeoutMinutes = 60,
    [string]$CodexCommand = "codex",
    [string]$CodexPath = "",
    [ValidateSet("", "success", "failed", "timeout", "malformed", "needs_human")]
    [string]$MockScenario = "",
    [string]$AgentRoot = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Write-JsonAtomic([string]$Path, [object]$Value) {
    $parent = Split-Path -Parent $Path
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    $temporary = "$Path.$PID.tmp"
    $Value | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $Path -Force
}

function Read-StructuredResult([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "structured output missing: $Path" }
    $value = Get-Content -LiteralPath $Path -Raw -Encoding utf8 | ConvertFrom-Json
    $required = @("status", "hypothesis_id", "config_hash", "hypothesis", "rationale", "files_changed",
        "tests_passed", "smoke_run_id", "smoke_metrics", "leakage_checks",
        "failure_reason", "next_recommendation")
    foreach ($field in $required) {
        if (-not ($value.PSObject.Properties.Name -contains $field)) { throw "structured output field missing: $field" }
    }
    foreach ($field in $value.PSObject.Properties.Name) {
        if ($required -notcontains $field) { throw "structured output field is not allowed: $field" }
    }
    if ($value.status -notin @("success", "rejected", "failed", "needs_human")) {
        throw "invalid status: $($value.status)"
    }
    $metrics = if ($null -eq $value.smoke_metrics) { @() } else { @($value.smoke_metrics) }
    foreach ($metric in $metrics) {
        $metricFields = @("name", "value", "split")
        foreach ($field in $metricFields) {
            if (-not ($metric.PSObject.Properties.Name -contains $field)) { throw "smoke metric field missing: $field" }
        }
        foreach ($field in $metric.PSObject.Properties.Name) {
            if ($metricFields -notcontains $field) { throw "smoke metric field is not allowed: $field" }
        }
        if ($metric.name -isnot [string]) { throw "smoke metric name must be a string" }
        if ($null -ne $metric.value -and $metric.value -isnot [ValueType]) { throw "smoke metric value must be a number or null" }
        if ($null -ne $metric.split -and $metric.split -isnot [string]) { throw "smoke metric split must be a string or null" }
    }
    $leakageFields = @("passed", "details")
    foreach ($field in $leakageFields) {
        if (-not ($value.leakage_checks.PSObject.Properties.Name -contains $field)) { throw "leakage_checks field missing: $field" }
    }
    foreach ($field in $value.leakage_checks.PSObject.Properties.Name) {
        if ($leakageFields -notcontains $field) { throw "leakage_checks field is not allowed: $field" }
    }
    return $value
}

function Assert-StrictJsonSchema([object]$Node, [string]$Context = '$') {
    if ($null -eq $Node) { throw "JSON schema node is null: context=$Context" }
    $nodeFields = @($Node.PSObject.Properties.Name)
    $types = if ($nodeFields -contains "type") { @($Node.type) } else { @() }
    if ($types -contains "object") {
        if ($nodeFields -notcontains "additionalProperties" -or $Node.additionalProperties -ne $false) {
            throw "strict JSON schema requires additionalProperties=false: context=$Context"
        }
        if ($nodeFields -notcontains "properties") { throw "object schema properties missing: context=$Context" }
        if ($nodeFields -notcontains "required") { throw "object schema required missing: context=$Context" }
        $required = @($Node.required)
        foreach ($property in $Node.properties.PSObject.Properties) {
            if ($required -notcontains $property.Name) {
                throw "all object properties must be required: context=$Context; property=$($property.Name)"
            }
            Assert-StrictJsonSchema -Node $property.Value -Context "$Context.properties.$($property.Name)"
        }
    }
    if ($types -contains "array") {
        if ($nodeFields -notcontains "items") { throw "array schema items missing: context=$Context" }
        Assert-StrictJsonSchema -Node $Node.items -Context "$Context.items"
    }
}

function Get-ProtectedSnapshot([string]$ProjectRoot, [string]$AutomationRoot) {
    $snapshot = [ordered]@{}
    # 기존 결과는 덮어쓰기 탐지만 수행한다. agent 자체 경로는 의도된 쓰기 영역이다.
    foreach ($root in @((Join-Path $AutomationRoot "runs"))) {
        if (Test-Path -LiteralPath $root) {
            Get-ChildItem -LiteralPath $root -Recurse -File | ForEach-Object {
                $snapshot[$_.FullName] = "$($_.Length):$($_.LastWriteTimeUtc.Ticks)"
            }
        }
    }
    foreach ($path in @(
        (Join-Path $ProjectRoot "train.csv"),
        (Join-Path $ProjectRoot "test.csv"),
        (Join-Path $AutomationRoot "registry\CHAMPION.json")
    )) {
        if (Test-Path -LiteralPath $path -PathType Leaf) {
            $snapshot[$path] = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash
        }
    }
    return $snapshot
}

function Compare-ProtectedSnapshot([object]$Before, [object]$After) {
    $changes = @()
    foreach ($key in $Before.Keys) {
        if (-not $After.Contains($key) -or $After[$key] -ne $Before[$key]) { $changes += $key }
    }
    return $changes
}

function ConvertTo-ProcessArgument([string]$Argument) {
    if ($Argument.Length -eq 0) { return '""' }
    if ($Argument -notmatch '[\s"]') { return $Argument }
    # CommandLineToArgvW 규칙: 따옴표 앞 backslash와 마지막 backslash를 두 배로 만든다.
    $escaped = [regex]::Replace($Argument, '(\\*)"', '$1$1\"')
    $escaped = [regex]::Replace($escaped, '(\\+)$', '$1$1')
    return '"' + $escaped + '"'
}

function Write-LauncherDiagnostic([string]$Path, [string]$Event, [hashtable]$Details) {
    $record = [ordered]@{ timestamp = [DateTimeOffset]::Now.ToString("o"); event = $Event }
    foreach ($key in $Details.Keys) { $record[$key] = $Details[$key] }
    ($record | ConvertTo-Json -Compress -Depth 10) | Add-Content -LiteralPath $Path -Encoding utf8
}

function Get-SafeLoggedArguments([string[]]$Arguments) {
    return @($Arguments | ForEach-Object {
        if ($_ -match '(?i)(token|secret|password|api[_-]?key)') { "<redacted>" } else { $_ }
    })
}

function Resolve-CodexLauncher(
    [string]$RequestedCommand,
    [string]$RequestedPath,
    [string]$DiagnosticPath
) {
    $commandInfo = $null
    $selectedPath = $null
    if (-not [string]::IsNullOrWhiteSpace($RequestedPath)) {
        if (-not [IO.Path]::IsPathRooted($RequestedPath)) {
            throw "CodexPath는 절대 경로여야 합니다. variable=CodexPath; value='$RequestedPath'"
        }
        $selectedPath = Resolve-CheckedFullPath -VariableName 'CodexPath' -Value $RequestedPath
        if (-not (Test-Path -LiteralPath $selectedPath -PathType Leaf)) {
            throw "지정한 CodexPath 파일이 없습니다. variable=CodexPath; value='$selectedPath'"
        }
        $commandInfo = Get-Command -Name $selectedPath -ErrorAction Stop
    } else {
        try {
            $commandInfo = Get-Command $RequestedCommand -ErrorAction Stop
        } catch {
            $pathValue = [Environment]::GetEnvironmentVariable("PATH")
            throw "Codex CLI를 찾지 못했습니다. command='$RequestedCommand'; PATH='$pathValue'. 확인: Get-Command codex -All; where.exe codex; npm prefix -g"
        }
        $selectedPath = if ($commandInfo.Path) { $commandInfo.Path } else { $commandInfo.Source }
        Write-LauncherDiagnostic $DiagnosticPath "get_command" @{
            name = $commandInfo.Name; command_type = [string]$commandInfo.CommandType
            source = [string]$commandInfo.Source; path = [string]$commandInfo.Path
        }
        # npm 설치에서는 PowerShell이 codex.ps1을 우선 반환한다. Process.Start로 ps1을
        # 직접 실행하지 않고 같은 npm shim의 cmd 파일을 선택한다.
        if ([IO.Path]::GetExtension($selectedPath) -ieq ".ps1") {
            $siblingCmd = [IO.Path]::ChangeExtension($selectedPath, ".cmd")
            if (Test-Path -LiteralPath $siblingCmd -PathType Leaf) {
                $selectedPath = Resolve-CheckedFullPath -VariableName 'codex.cmd sibling' -Value $siblingCmd
                $commandInfo = Get-Command -Name $selectedPath -ErrorAction Stop
            } else {
                throw "Codex가 PowerShell shim으로 해석됐지만 실행 가능한 codex.cmd가 없습니다. ps1='$selectedPath'; expected_cmd='$siblingCmd'; 확인: where.exe codex"
            }
        }
    }

    $selectedPath = Resolve-CheckedFullPath -VariableName 'resolved Codex path' -Value $selectedPath
    $extension = [IO.Path]::GetExtension($selectedPath).ToLowerInvariant()
    if ($extension -notin @(".exe", ".cmd", ".bat")) {
        throw "지원하지 않는 Codex 실행 형식입니다. path='$selectedPath'; extension='$extension'; 허용=.exe,.cmd,.bat"
    }
    Write-LauncherDiagnostic $DiagnosticPath "resolved_codex" @{
        name = $commandInfo.Name; command_type = [string]$commandInfo.CommandType
        source = [string]$commandInfo.Source; path = [string]$commandInfo.Path
        selected_path = $selectedPath; extension = $extension
    }
    return [pscustomobject]@{ Path = $selectedPath; Extension = $extension }
}

function Resolve-CheckedFullPath([string]$VariableName, [AllowNull()][string]$Value, [AllowNull()][string]$BasePath = $null) {
    $shownValue = if ($null -eq $Value) { "<null>" } else { $Value }
    $shownBase = if ($null -eq $BasePath) { "<null>" } else { $BasePath }
    if ([string]::IsNullOrWhiteSpace($Value)) {
        throw "경로 값이 null 또는 빈 문자열입니다. variable=$VariableName; value='$shownValue'; base='$shownBase'"
    }
    $candidate = $Value
    if (-not [IO.Path]::IsPathRooted($candidate)) {
        if ([string]::IsNullOrWhiteSpace($BasePath)) {
            throw "상대 경로의 기준 경로가 비어 있습니다. variable=$VariableName; value='$shownValue'; base='$shownBase'"
        }
        $candidate = Join-Path $BasePath $candidate
    }
    try {
        return [IO.Path]::GetFullPath($candidate)
    } catch {
        throw "경로 정규화에 실패했습니다. variable=$VariableName; value='$shownValue'; base='$shownBase'; candidate='$candidate'; error=$($_.Exception.Message)"
    }
}

# Windows PowerShell 5.1에서는 param 기본값 평가 시 $PSScriptRoot가 비어 있을 수 있다.
# param 처리 후의 실제 스크립트 위치만 신뢰하여 모든 루트를 계산한다.
$scriptDirectory = Resolve-CheckedFullPath -VariableName 'PSScriptRoot' -Value $PSScriptRoot
$automationRoot = Resolve-CheckedFullPath -VariableName 'automationRoot' -Value '..' -BasePath $scriptDirectory
$projectRoot = Resolve-CheckedFullPath -VariableName 'projectRoot' -Value '..' -BasePath $automationRoot
if ([string]::IsNullOrWhiteSpace($AgentRoot)) {
    $AgentRoot = $scriptDirectory
}
$agentRootPath = Resolve-CheckedFullPath -VariableName 'AgentRoot' -Value $AgentRoot -BasePath $scriptDirectory
if (-not (Test-Path -LiteralPath $projectRoot -PathType Container)) {
    throw "계산된 저장소 루트가 존재하지 않습니다. variable=projectRoot; value='$projectRoot'; scriptDirectory='$scriptDirectory'"
}
$statePath = Join-Path $agentRootPath "agent_state.json"
$runsRoot = Join-Path $agentRootPath "runs"
$schemaPath = Join-Path $scriptDirectory "output_schema.json"
$templatePath = Join-Path $scriptDirectory "agent_prompt.md"
if (-not (Test-Path -LiteralPath $schemaPath -PathType Leaf)) { throw "output schema 파일이 없습니다: $schemaPath" }
$outputSchema = Get-Content -LiteralPath $schemaPath -Raw -Encoding utf8 | ConvertFrom-Json
Assert-StrictJsonSchema -Node $outputSchema
New-Item -ItemType Directory -Force -Path $runsRoot | Out-Null
$launcherDiagnosticPath = Join-Path $agentRootPath "launcher_diagnostics.jsonl"

$codexLauncher = $null
if (-not $MockScenario) {
    $pathEnvironment = [Environment]::GetEnvironmentVariable("PATH")
    if ([string]::IsNullOrWhiteSpace($pathEnvironment) -and -not [IO.Path]::IsPathRooted($CodexCommand)) {
        throw "환경변수 PATH가 null 또는 빈 문자열이라 Codex CLI를 탐색할 수 없습니다. variable=PATH; value='$pathEnvironment'; CodexCommand='$CodexCommand'"
    }
    $codexLauncher = Resolve-CodexLauncher -RequestedCommand $CodexCommand -RequestedPath $CodexPath -DiagnosticPath $launcherDiagnosticPath
}

$started = [DateTimeOffset]::Now
$deadline = $started.AddHours($MaxHours)
$state = [ordered]@{
    started_at = $started.ToString("o")
    deadline = $deadline.ToString("o")
    completed_iterations = 0
    consecutive_failures = 0
    attempted_hypotheses = @()
    successful_hypotheses = @()
    rejected_hypotheses = @()
    stop_reason = $null
    last_run_id = $null
}
Write-JsonAtomic $statePath $state

$usedConfigHashes = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
$registryPath = Join-Path $automationRoot "registry\EXPERIMENT_REGISTRY.csv"
if (Test-Path -LiteralPath $registryPath) {
    Import-Csv -LiteralPath $registryPath | Where-Object { $_.status -eq "completed" -and $_.config_hash } |
        ForEach-Object { [void]$usedConfigHashes.Add($_.config_hash) }
}

for ($number = 1; $number -le $MaxIterations; $number++) {
    if ([DateTimeOffset]::Now -ge $deadline) { $state.stop_reason = "max_hours"; break }
    if ($state.consecutive_failures -ge $MaxFailures) { $state.stop_reason = "max_failures"; break }

    $iterationId = "{0}_{1:d3}" -f ([DateTimeOffset]::Now.ToString("yyyyMMdd_HHmmss")), $number
    $iterationDir = Join-Path $runsRoot $iterationId
    New-Item -ItemType Directory -Force -Path $iterationDir | Out-Null
    $state.last_run_id = $iterationId

    $attemptedJson = @($state.attempted_hypotheses) | ConvertTo-Json -Compress
    $prompt = Get-Content -LiteralPath $templatePath -Raw -Encoding utf8
    $prompt = $prompt.Replace("{{ITERATION_ID}}", $iterationId).Replace("{{DEADLINE}}", $deadline.ToString("o"))
    $prompt = $prompt.Replace("{{ATTEMPTED_HYPOTHESES}}", $attemptedJson).Replace("{{ITERATION_DIR}}", $iterationDir)
    $promptPath = Join-Path $iterationDir "prompt.txt"
    $prompt | Set-Content -LiteralPath $promptPath -Encoding utf8

    $stdoutPath = Join-Path $iterationDir "codex_stdout.jsonl"
    $stderrPath = Join-Path $iterationDir "codex_stderr.txt"
    $summaryPath = Join-Path $iterationDir "summary.json"
    $testOutputPath = Join-Path $iterationDir "test_output.txt"
    $diffPath = Join-Path $iterationDir "git_diff.patch"
    $smokeReferencePath = Join-Path $iterationDir "smoke_result_reference.json"
    $errorPath = Join-Path $iterationDir "error.json"
    $protectedBefore = Get-ProtectedSnapshot $projectRoot $automationRoot
    $registryRowsBefore = @()
    if (Test-Path -LiteralPath $registryPath) { $registryRowsBefore = @(Import-Csv -LiteralPath $registryPath) }

    if ($MockScenario) {
        $python = Join-Path $projectRoot ".venv\Scripts\python.exe"
        if (-not (Test-Path -LiteralPath $python)) { $python = "python" }
        $mockScript = Join-Path $automationRoot "tests\mock_codex.py"
        $processArgs = @($mockScript, "--scenario", $MockScenario, "--output", $summaryPath)
        $actualCommand = $python
    } else {
        $actualCommand = $codexLauncher.Path
        $processArgs = @("exec", "--sandbox", "workspace-write", "-c", 'approval_policy="never"',
            "-c", "sandbox_workspace_write.network_access=false",
            "--json", "--output-schema", $schemaPath, "--output-last-message", $summaryPath, "-")
    }

    $startInfo = [Diagnostics.ProcessStartInfo]::new()
    if (-not $MockScenario -and $codexLauncher.Extension -in @(".cmd", ".bat")) {
        $comSpecValue = [Environment]::GetEnvironmentVariable("ComSpec")
        if ([string]::IsNullOrWhiteSpace($comSpecValue)) {
            throw "환경변수 ComSpec이 null 또는 빈 문자열입니다. variable=ComSpec; value='$comSpecValue'"
        }
        $comSpecPath = Resolve-CheckedFullPath -VariableName 'ComSpec' -Value $comSpecValue
        $childArguments = (($processArgs | ForEach-Object { ConvertTo-ProcessArgument $_ }) -join " ")
        $cmdLine = '"' + $actualCommand + '" ' + $childArguments
        $startInfo.FileName = $comSpecPath
        $startInfo.Arguments = '/d /s /c "' + $cmdLine + '"'
    } else {
        $startInfo.FileName = $actualCommand
        $startInfo.Arguments = (($processArgs | ForEach-Object { ConvertTo-ProcessArgument $_ }) -join " ")
    }
    $startInfo.WorkingDirectory = $projectRoot
    $startInfo.UseShellExecute = $false
    $startInfo.RedirectStandardInput = -not [bool]$MockScenario
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    Write-LauncherDiagnostic $launcherDiagnosticPath "process_start" @{
        file_name = $startInfo.FileName
        arguments = @(Get-SafeLoggedArguments $processArgs)
        working_directory = $startInfo.WorkingDirectory
    }
    $process = [Diagnostics.Process]::new()
    $process.StartInfo = $startInfo
    [void]$process.Start()
    if (-not $MockScenario) { $process.StandardInput.Write($prompt); $process.StandardInput.Close() }
    $stdoutTask = $process.StandardOutput.ReadToEndAsync()
    $stderrTask = $process.StandardError.ReadToEndAsync()
    $timeoutMs = [Math]::Max(1, [int]($IterationTimeoutMinutes * 60 * 1000))
    $finished = $process.WaitForExit($timeoutMs)
    if (-not $finished) {
        $process.Kill()
        $process.WaitForExit()
    }
    $stdoutTask.GetAwaiter().GetResult() | Set-Content -LiteralPath $stdoutPath -Encoding utf8
    $stderrTask.GetAwaiter().GetResult() | Set-Content -LiteralPath $stderrPath -Encoding utf8

    $iterationFailed = $false
    try {
        if (-not $finished) { throw "iteration_timeout" }
        if ($process.ExitCode -ne 0) { throw "codex_exit_code_$($process.ExitCode)" }
        $result = Read-StructuredResult $summaryPath
        foreach ($changedFile in @($result.files_changed)) {
            $normalized = ($changedFile -replace '\\', '/').TrimStart('./')
            if (-not $normalized.StartsWith("automation/")) { throw "file_change_outside_automation: $changedFile" }
        }
        if ($result.config_hash -and $usedConfigHashes.Contains([string]$result.config_hash)) {
            throw "duplicate_config_hash: $($result.config_hash)"
        }
        if ($result.hypothesis_id -and $state.attempted_hypotheses -notcontains $result.hypothesis_id) {
            $state.attempted_hypotheses += $result.hypothesis_id
        }
        if ($result.status -eq "success") {
            if (-not $result.tests_passed) { throw "success_without_passing_tests" }
            if (-not $result.config_hash) { throw "success_without_config_hash" }
            [void]$usedConfigHashes.Add([string]$result.config_hash)
            $state.successful_hypotheses += $result.hypothesis_id
            $state.consecutive_failures = 0
        } elseif ($result.status -eq "rejected") {
            $state.rejected_hypotheses += $result.hypothesis_id
            $state.consecutive_failures = 0
        } else {
            $iterationFailed = $true
            $state.consecutive_failures++
        }
        if ($result.smoke_run_id) {
            Write-JsonAtomic $smokeReferencePath ([ordered]@{ smoke_run_id = $result.smoke_run_id; smoke_metrics = $result.smoke_metrics })
        }
        if ($result.status -eq "needs_human") { $state.stop_reason = if ($result.failure_reason -eq "no_safe_hypothesis") { "no_safe_hypothesis" } else { "needs_human" } }
        if ($result.status -eq "failed" -and -not $result.tests_passed) { $state.stop_reason = "needs_human" }
    } catch {
        $iterationFailed = $true
        $state.consecutive_failures++
        Write-JsonAtomic $errorPath ([ordered]@{ type = $_.Exception.GetType().Name; message = $_.Exception.Message; occurred_at = [DateTimeOffset]::Now.ToString("o") })
    }

    $protectedAfter = Get-ProtectedSnapshot $projectRoot $automationRoot
    $protectedChanges = @(Compare-ProtectedSnapshot $protectedBefore $protectedAfter)
    $registryRowsAfter = @()
    if (Test-Path -LiteralPath $registryPath) { $registryRowsAfter = @(Import-Csv -LiteralPath $registryPath) }
    if ($registryRowsAfter.Count -lt $registryRowsBefore.Count) {
        $protectedChanges += $registryPath
    } else {
        for ($rowIndex = 0; $rowIndex -lt $registryRowsBefore.Count; $rowIndex++) {
            if (($registryRowsBefore[$rowIndex] | ConvertTo-Json -Compress) -ne ($registryRowsAfter[$rowIndex] | ConvertTo-Json -Compress)) {
                $protectedChanges += $registryPath
                break
            }
        }
    }
    if ($protectedChanges.Count -gt 0) {
        $iterationFailed = $true
        $state.consecutive_failures++
        $state.stop_reason = "needs_human"
        Write-JsonAtomic $errorPath ([ordered]@{
            type = "ProtectedFileChanged"; message = "보호 대상이 변경됨"
            paths = $protectedChanges; occurred_at = [DateTimeOffset]::Now.ToString("o")
        })
    }

    # Codex가 남기지 않은 경우에도 필수 감사 아티팩트를 보장한다.
    if (-not (Test-Path -LiteralPath $testOutputPath)) { "Codex summary tests_passed 필드를 참조하십시오." | Set-Content $testOutputPath -Encoding utf8 }
    $gitDiff = (& git -C $projectRoot diff -- automation | Out-String)
    $gitDiff | Set-Content -LiteralPath $diffPath -Encoding utf8
    $state.completed_iterations++
    Write-JsonAtomic $statePath $state

    if ($state.stop_reason) { break }
    if ($iterationFailed -and $state.consecutive_failures -ge $MaxFailures) { $state.stop_reason = "max_failures"; break }
}

if (-not $state.stop_reason) {
    if ([DateTimeOffset]::Now -ge $deadline) { $state.stop_reason = "max_hours" }
    elseif ($state.completed_iterations -ge $MaxIterations) { $state.stop_reason = "max_iterations" }
    elseif ($state.consecutive_failures -ge $MaxFailures) { $state.stop_reason = "max_failures" }
}
Write-JsonAtomic $statePath $state
Write-Host "completed_iterations=$($state.completed_iterations)"
Write-Host "consecutive_failures=$($state.consecutive_failures)"
Write-Host "stop_reason=$($state.stop_reason)"
if ($state.stop_reason -in @("needs_human", "max_failures")) { exit 2 }
exit 0
