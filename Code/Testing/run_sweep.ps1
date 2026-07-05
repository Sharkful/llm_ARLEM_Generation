<#
.SYNOPSIS
    Resumable, connectivity-guarded driver for the full formative benchmark sweep.

.DESCRIPTION
    Runs the first *full* formative sweep - all 11 registered models x 5 lab
    topics x L1/L3/L4 x 3 output formats (~385 runs, several hours). The matrix is
    chunked into 33 invocations of benchmark.py (one per model x level-group):

        L1 (x11):  --model <M> --all-labs --level L1 --spec json_lab
                       -> 5 runs each (format fan-out is L3/L4 only; L1 returns a
                          spec-agnostic LabOutline, so one spec label suffices)
        L3 (x11):  --model <M> --all-labs --level L3 --spec json_lab arlem arlem_simple
                       -> 15 runs each
        L4 (x11):  --model <M> --all-labs --level L4 --spec json_lab arlem arlem_simple
                       -> 15 runs each

    Ordering is level-major: all 11 L1 chunks, then L3, then L4 - so the
    cheap/fast L1 pass exercises every model early and surfaces config problems
    before the expensive L4 runs.

    Before every chunk it preflights connectivity (Test-NetConnection :443 to all
    three API hosts). The dev machine has an intermittent DNS failure that requires
    a reboot to clear, so on a failed preflight the driver HALTS (does not wait or
    retry - the fix is a reboot) after printing the last completed chunk and a
    resume message. Completed chunk ids are checkpointed to sweep_progress.log;
    re-running the script after a reboot skips finished chunks and resumes.

.PARAMETER Models
    Model ids to sweep (default = all 11). A cheap pilot is:
        .\Code\Testing\run_sweep.ps1 -Models gemini-2.5-flash-lite
    which runs only that model's L1/L3/L4 chunks (3 chunks).

.PARAMETER DryRun
    Enumerate the planned chunks and print each benchmark.py command without
    running the preflight, invoking python, or checkpointing. Use it to eyeball
    the matrix/ordering (a full run enumerates 33 chunks, level-major) safely.

.NOTES
    FAILURE SEMANTICS
    benchmark.py does NOT retry network/API errors: it swallows the exception,
    records success:false, and STILL EXITS 0. So a DNS drop that hits *after* a
    chunk's preflight passed silently turns the rest of that chunk into logged-but-
    lost failures - and because the exit code is always 0, this driver cannot tell
    a contaminated chunk from a clean one, so it checkpoints the chunk regardless.

    That is acceptable: at most one chunk (~15 runs) is lost per outage, because
    the NEXT chunk's preflight catches the dead network and halts the sweep. Those
    specific network-failures land in the report's "Failed runs" section. To retry
    a contaminated chunk, delete its line from sweep_progress.log and re-run - the
    companion report change (issue #36) collapses re-run duplicates via keep-latest
    dedup, so an earlier failure never double-counts against a later success.

    Run from repo root:  .\Code\Testing\run_sweep.ps1
#>

[CmdletBinding()]
param(
    [string[]] $Models,
    [switch]   $DryRun
)

$ErrorActionPreference = 'Stop'

# -- Paths (anchored to this script; benchmark.py resolves its own paths from
#    __file__, so the caller's working directory is irrelevant) --
$RepoRoot    = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Python      = Join-Path $RepoRoot '.venv\Scripts\python.exe'
$BenchScript = Join-Path $RepoRoot 'Code\Testing\benchmark.py'
$BenchDir    = Join-Path $RepoRoot 'Artifacts\Data\Benchmark'
$ProgressLog = Join-Path $BenchDir 'sweep_progress.log'
$LogDir      = Join-Path $BenchDir 'logs'

# API hosts the provider clients connect to (benchmark.py:100-149). Preflight
# passes only if ALL THREE resolve+connect - a strict "is my DNS alive at all"
# proxy, which is the actual failure mode. Repoint one at a bogus name to
# rehearse an outage.
$ApiHosts = @(
    'api.openai.com'
    'api.anthropic.com'
    'generativelanguage.googleapis.com'
)

# The 11 registered models (verify against: python "Code/Testing/benchmark.py" --list-models)
$AllModels = @(
    'gpt-5.5'
    'gpt-5.4'
    'gpt-5.4-mini'
    'gpt-5.4-nano'
    'claude-opus-4.8'
    'claude-sonnet-5'
    'claude-haiku-4.5'
    'gemini-3.1-pro'
    'gemini-3.5-flash'
    'gemini-3.1-flash-lite'
    'gemini-2.5-flash-lite'
)

# Level groups, in level-major order. Specs are the --spec fan-out per level;
# L1 runs a single spec label (LabOutline is spec-agnostic).
$LevelGroups = @(
    [pscustomobject]@{ Level = 'L1'; Specs = @('json_lab') }
    [pscustomobject]@{ Level = 'L3'; Specs = @('json_lab', 'arlem', 'arlem_simple') }
    [pscustomobject]@{ Level = 'L4'; Specs = @('json_lab', 'arlem', 'arlem_simple') }
)

function Test-Connectivity {
    <#
        Returns $true only if a TCP :443 connection succeeds to every API host.
        A single failure (the intermittent DNS outage takes them all down at once)
        returns $false so the caller halts and waits for a reboot.
    #>
    foreach ($h in $ApiHosts) {
        $ok = Test-NetConnection -ComputerName $h -Port 443 `
            -InformationLevel Quiet -WarningAction SilentlyContinue
        if (-not $ok) {
            Write-Host "  connectivity: $h :443 unreachable" -ForegroundColor DarkYellow
            return $false
        }
    }
    return $true
}

# -- Resolve & validate the model list --
if (-not $Models) {
    $Models = $AllModels
}
else {
    $unknown = $Models | Where-Object { $AllModels -notcontains $_ }
    if ($unknown) {
        Write-Host "Error: unknown model id(s): $($unknown -join ', ')" -ForegroundColor Red
        Write-Host "Known models: $($AllModels -join ', ')" -ForegroundColor Yellow
        exit 1
    }
}

if (-not (Test-Path $Python)) {
    Write-Host "Error: venv python not found at $Python" -ForegroundColor Red
    Write-Host "Run .\setup_windows.ps1 first." -ForegroundColor Yellow
    exit 1
}

# -- Build the chunk list (level-major: all L1, then all L3, then all L4) --
$chunks = foreach ($lg in $LevelGroups) {
    foreach ($m in $Models) {
        [pscustomobject]@{
            Id    = "${m}_$($lg.Level)"
            Model = $m
            Level = $lg.Level
            Specs = $lg.Specs
        }
    }
}

# -- Load checkpoint (ids already completed) --
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$done = @{}
if (Test-Path $ProgressLog) {
    foreach ($line in Get-Content $ProgressLog) {
        $id = $line.Trim()
        if ($id) { $done[$id] = $true }
    }
}

$total = $chunks.Count
Write-Host "Sweep: $($Models.Count) model(s) x 3 level-groups = $total chunk(s)." -ForegroundColor Green
if ($done.Count) {
    Write-Host "Resuming - $($done.Count) chunk id(s) already in sweep_progress.log will be skipped." -ForegroundColor Green
}

# -- Run --
# Switch off Stop for the run loop: benchmark.py is a native process, and under
# ErrorActionPreference='Stop' a single line it writes to stderr (httpx/instructor
# warnings, a caught traceback, logging) merged via *>&1 throws a terminating
# NativeCommandError -- which would crash the sweep mid-chunk *and* drop that line
# from the log. Under 'Continue' the stderr just flows into the per-chunk log and
# the sweep keeps going, which is the whole point of this driver.
$ErrorActionPreference = 'Continue'

$i = 0
$lastCompleted = '(none)'
foreach ($chunk in $chunks) {
    $i++

    if ($done.ContainsKey($chunk.Id)) {
        Write-Host "[$i/$total] $($chunk.Id) - already done, skipping" -ForegroundColor DarkGray
        $lastCompleted = $chunk.Id
        continue
    }

    # Preflight immediately before this chunk. Halt (don't wait) on an outage.
    if (-not $DryRun -and -not (Test-Connectivity)) {
        Write-Host ""
        Write-Host "Preflight connectivity check FAILED before chunk '$($chunk.Id)'." -ForegroundColor Red
        Write-Host "Last completed chunk: $lastCompleted" -ForegroundColor Yellow
        Write-Host "DNS looks down - reboot, then re-run this script to resume." -ForegroundColor Yellow
        exit 1
    }

    $runCount = $chunk.Specs.Count * 5
    Write-Host ("[$i/$total] $($chunk.Id) - level $($chunk.Level), specs [$($chunk.Specs -join ', ')], ~$runCount runs ...") -ForegroundColor Cyan

    $logFile = Join-Path $LogDir "$($chunk.Id).log"
    $pyArgs = @(
        $BenchScript
        '--model', $chunk.Model
        '--all-labs'
        '--level', $chunk.Level
        '--spec'
    ) + $chunk.Specs

    if ($DryRun) {
        Write-Host "        [dry-run] $Python $($pyArgs -join ' ')" -ForegroundColor DarkGray
        $lastCompleted = $chunk.Id
        continue
    }

    # Full output -> per-chunk log only; console stays quiet (Out-Null discards the
    # Tee pass-through). *>&1 folds every stream into the pipeline first.
    & $Python @pyArgs *>&1 | Tee-Object -FilePath $logFile | Out-Null

    # Always checkpoint: the runner exits 0 regardless, so we cannot distinguish a
    # clean chunk from one a post-preflight DNS drop contaminated (see .NOTES).
    Add-Content -Path $ProgressLog -Value $chunk.Id
    $lastCompleted = $chunk.Id
    Write-Host "        done -> logs/$($chunk.Id).log" -ForegroundColor DarkGreen
}

Write-Host ""
if ($DryRun) {
    Write-Host "Dry-run complete: $total chunk(s) enumerated. Nothing was run or checkpointed." -ForegroundColor Green
}
else {
    Write-Host "Sweep complete: $total/$total chunk(s) checkpointed in sweep_progress.log." -ForegroundColor Green
}
