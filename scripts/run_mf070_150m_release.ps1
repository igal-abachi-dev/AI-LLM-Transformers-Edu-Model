# MF-070's real 150M-Modern, 3B-token release run: 5-source data prep, then
# the real training launch. 350M/500M are a confirmed real NO-GO on this local
# 8GB card (see MF-070's backlog entry, 2026-09-14) -- this script trains the
# preset that IS viable here: 150M-Modern, head_dim=96 (the config's own
# default), max_seq_len=2048, with --activation-checkpointing, which is now a
# real, required flag at this context length (measured 2026-09-14: without it,
# peak VRAM hits 12.37GB against this card's 8.59GB real total -- 44% over
# physical capacity, ~239 tok/s; with it, 5.59GB peak, ~1,966 tok/s).
#
# Run from the repository root, in Windows PowerShell:
#   .\scripts\run_mf070_150m_release.ps1
#
# Safe to walk away from and re-run: each data-prep step is skipped if its
# real metadata.json already exists (AlreadyDoneMarker, same convention as
# run_overnight_mf083_mf095_mf097.ps1). The training step is skipped entirely
# if artifacts\mf070-150m-release\final\model.safetensors already exists, and
# auto-resumes from the latest checkpoint-* directory if training was
# interrupted partway (this run is real multi-day: ~17-18 days of continuous
# GPU time at the measured throughput above, so an interruption is a real,
# expected possibility, not a hypothetical). Matches AGENTS.md's "never report
# a run successful without command output or a persisted run record" rule:
# every step's real stdout/stderr goes to its own log file, and
# train/pretrain.py's/prepare_data.py's own real run.json/metadata.json remain
# the actual evidence -- this script only sequences, logs, and reports.
#
# Two real, disclosed caveats, not silently assumed:
#   1. Only FineWeb-Edu's --limit below is anchored to this project's own real
#      in-session measurement (3,000 docs -> 3,100,672 tokens at
#      sequence_length=2048, i.e. ~1,034 tokens/doc). The other four sources'
#      --limit values are reasoned estimates (typical tokens/doc for that kind
#      of content), not measured for this project's own corpus. The real
#      per-source token totals are checked automatically after data prep
#      (see the validation step below) and compared against each source's
#      target share of the real 3B-token budget; a real shortfall is reported,
#      not hidden, but the pipeline still proceeds (killable during the
#      30-second pause) since the user wants this to run unattended.
#   2. --accumulation-steps is deliberately left at 1, not MF-101's measured
#      "free" 8/32 win -- that win was measured at sequence_length=1024,
#      batch_size=2, WITHOUT --activation-checkpointing (MF-101's own recorded
#      gap 1/2), a materially different memory profile than this run's actual
#      real recipe (head_dim=96, seq_len=2048, activation-checkpointing). The
#      combination used here (batch_size=2, accumulation_steps=1) is the exact
#      configuration this project has real, persisted evidence for
#      (peak_reserved_vram_bytes=5.59GB / total_vram_bytes=8.59GB, ~1,966
#      tok/s) -- accumulation_steps=8/32 would very likely still fit (per-
#      microbatch memory shape is unchanged by accumulation_steps) but has not
#      been independently re-measured in this exact combination, so it is not
#      silently applied to an unattended ~17-day run.

$ErrorActionPreference = "Stop"
$Python = ".\.venv\Scripts\python.exe"
$LogDir = "logs\mf070-release"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Summary = Join-Path $LogDir "SUMMARY.txt"
Set-Content -Path $Summary -Value "" -Encoding utf8

# MF-070's own 1B-token pre-work run hit a real transient HF Hub read-timeout
# mid-prep once (see MF-070's backlog entry) -- fixed there by raising this
# same timeout. Set proactively here since this prep is larger still.
$env:HF_HUB_DOWNLOAD_TIMEOUT = "180"

$Status = @{}

function Write-Log {
    param([string]$Message)
    Write-Host $Message
    Add-Content -Path $Summary -Value $Message -Encoding utf8
}

function Invoke-Step {
    param(
        [string]$Name,
        [string[]]$ArgumentList,
        [string]$AlreadyDoneMarker = $null
    )
    if ($AlreadyDoneMarker -and (Test-Path $AlreadyDoneMarker)) {
        $Status[$Name] = "SUCCESS"
        Write-Log "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] ALREADY DONE $Name (found $AlreadyDoneMarker) -- skipping re-run"
        return
    }
    $logFile = Join-Path $LogDir "$Name.log"
    $startTs = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Log "[$startTs] START $Name"
    # Real bug found and fixed here (2026-09-15): with the script-level
    # $ErrorActionPreference = "Stop" active, Windows PowerShell 5.1
    # promotes *any* stderr line from a native command redirected via `*>`
    # into a terminating error -- even a normal, recoverable warning
    # prepare_data.py itself prints and handles correctly (e.g. skipping one
    # repo whose real Windows path exceeded MAX_PATH). This killed the
    # whole pipeline on the first such warning, even though the underlying
    # python.exe process was working correctly. Scoped down to "Continue"
    # for just this one native call so stderr flows through into the log
    # file normally; $LASTEXITCODE below (already the real, authoritative
    # pass/fail signal) is unaffected either way.
    $previousErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    & $Python @ArgumentList *> $logFile
    $ErrorActionPreference = $previousErrorActionPreference
    if ($LASTEXITCODE -eq 0) {
        $Status[$Name] = "SUCCESS"
        Write-Log "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] SUCCESS $Name (log: $logFile)"
    } else {
        $Status[$Name] = "FAILED"
        Write-Log "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] FAILED $Name (exit $LASTEXITCODE) -- see $logFile"
    }
}

Write-Log "=== MF-070 150M-Modern release run started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

# --- Preflight: fail fast with a clear message instead of mid-prep ---

if (-not (Test-Path ".\.venv\Scripts\python.exe")) {
    Write-Log "FATAL: .venv not found -- run 'uv sync' (or equivalent) first."
    exit 1
}
if (-not (Test-Path "data\tokenizer")) {
    Write-Log "FATAL: data\tokenizer not found -- the real MF-119 production tokenizer must exist first."
    exit 1
}
if (-not (Test-Path "configs\code-repo-allowlist.txt")) {
    Write-Log "FATAL: configs\code-repo-allowlist.txt not found -- required for the github-code source."
    exit 1
}

# --- Step 1: real data prep, 5 sources at the MF-095-adopted mixture shares,
# packed at seq_len=2048 (the config's real declared max_seq_len -- MF-070's
# own finding this session is that measuring/prepping at half that length
# was a real, corrected mistake). Target: 3B tokens total. Ordered smallest
# --limit first: no real measured docs/sec exists yet for this pipeline at
# this scale (only reasoned estimates, see header caveat 1), so running the
# smallest source first surfaces a real per-source wall-clock number (this
# step's own log start/end timestamps) within minutes rather than after
# waiting through the largest source first.
#   cosmopedia-v2   5% ->   150,000,000 tokens (110,000 docs)
#   finemath       15% ->   450,000,000 tokens (620,000 docs)
#   fineweb-edu    25% ->   750,000,000 tokens (800,000 docs)
#   dclm-edu       35% -> 1,050,000,000 tokens (1,150,000 docs)
#   github-code    20% ->   600,000,000 tokens (1,380,000 docs; curated
#                                                allowlist, least real
#                                                precedent for yield -- run
#                                                last, after 4 other real
#                                                successes already confirm
#                                                the pipeline itself works)

Invoke-Step "prep-cosmopedia-v2" @(
    "scripts/prepare_data.py", "--source", "cosmopedia-v2",
    "--limit", "110000", "--shuffle-seed", "42", "--tokenizer", "data/tokenizer",
    "--output", "data/shards/mf070-150m-3b-cosmopedia-v2", "--sequence-length", "2048",
    "--validation-fraction", "0.01",
    "--export-parquet-dir", "data/parquet/mf070-150m-3b-cosmopedia-v2"
) -AlreadyDoneMarker "data/shards/mf070-150m-3b-cosmopedia-v2/metadata.json"

Invoke-Step "prep-finemath" @(
    "scripts/prepare_data.py", "--source", "finemath", "--finemath-config", "finemath-4plus",
    "--limit", "620000", "--shuffle-seed", "42", "--tokenizer", "data/tokenizer",
    "--output", "data/shards/mf070-150m-3b-finemath", "--sequence-length", "2048",
    "--validation-fraction", "0.01",
    "--export-parquet-dir", "data/parquet/mf070-150m-3b-finemath"
) -AlreadyDoneMarker "data/shards/mf070-150m-3b-finemath/metadata.json"

Invoke-Step "prep-fineweb-edu" @(
    "scripts/prepare_data.py", "--source", "fineweb-edu",
    "--limit", "800000", "--shuffle-seed", "42", "--tokenizer", "data/tokenizer",
    "--output", "data/shards/mf070-150m-3b-fineweb-edu", "--sequence-length", "2048",
    "--validation-fraction", "0.01",
    "--export-parquet-dir", "data/parquet/mf070-150m-3b-fineweb-edu"
) -AlreadyDoneMarker "data/shards/mf070-150m-3b-fineweb-edu/metadata.json"

Invoke-Step "prep-dclm-edu" @(
    "scripts/prepare_data.py", "--source", "dclm-edu", "--dclm-min-score", "3",
    "--limit", "1150000", "--shuffle-seed", "42", "--tokenizer", "data/tokenizer",
    "--output", "data/shards/mf070-150m-3b-dclm-edu", "--sequence-length", "2048",
    "--validation-fraction", "0.01",
    "--export-parquet-dir", "data/parquet/mf070-150m-3b-dclm-edu"
) -AlreadyDoneMarker "data/shards/mf070-150m-3b-dclm-edu/metadata.json"

Invoke-Step "prep-github-code" @(
    "scripts/prepare_data.py", "--source", "github-code",
    "--github-repo-allowlist", "configs/code-repo-allowlist.txt",
    "--limit", "1380000", "--shuffle-seed", "42", "--tokenizer", "data/tokenizer",
    "--output", "data/shards/mf070-150m-3b-github-code", "--sequence-length", "2048",
    "--validation-fraction", "0.01",
    "--export-parquet-dir", "data/parquet/mf070-150m-3b-github-code"
) -AlreadyDoneMarker "data/shards/mf070-150m-3b-github-code/metadata.json"

$allPrepsOk = ($Status["prep-cosmopedia-v2"] -eq "SUCCESS") -and
              ($Status["prep-finemath"] -eq "SUCCESS") -and
              ($Status["prep-fineweb-edu"] -eq "SUCCESS") -and
              ($Status["prep-dclm-edu"] -eq "SUCCESS") -and
              ($Status["prep-github-code"] -eq "SUCCESS")

if (-not $allPrepsOk) {
    Write-Log ""
    Write-Log "One or more data-prep steps failed -- not starting the real training run on incomplete data."
    Write-Log "Fix the failure (see the per-step log above) and re-run this script; completed steps are skipped automatically."
    Write-Log ""
    Write-Log "Step                Status"
    Write-Log "-------------------- -------"
    foreach ($name in @("prep-cosmopedia-v2", "prep-finemath", "prep-fineweb-edu", "prep-dclm-edu", "prep-github-code")) {
        $value = "NOT RUN"
        if ($Status.ContainsKey($name)) { $value = $Status[$name] }
        Write-Log ("{0,-20} {1}" -f $name, $value)
    }
    exit 1
}

# --- Step 2: real, honest check of what data prep actually produced, since
# every --limit above except fineweb-edu's is an estimate (see header). Does
# not abort on a shortfall (the user wants this to run unattended) -- reports
# the real numbers and pauses 30s so a badly-off estimate can still be caught.

Write-Log ""
Write-Log "=== Real per-source token yield vs. target (from each source's own metadata.json) ==="
$targets = @{
    "prep-dclm-edu"      = @{ Path = "data/shards/mf070-150m-3b-dclm-edu/metadata.json"; Target = 1050000000 }
    "prep-fineweb-edu"   = @{ Path = "data/shards/mf070-150m-3b-fineweb-edu/metadata.json"; Target = 750000000 }
    "prep-github-code"   = @{ Path = "data/shards/mf070-150m-3b-github-code/metadata.json"; Target = 600000000 }
    "prep-finemath"      = @{ Path = "data/shards/mf070-150m-3b-finemath/metadata.json"; Target = 450000000 }
    "prep-cosmopedia-v2" = @{ Path = "data/shards/mf070-150m-3b-cosmopedia-v2/metadata.json"; Target = 150000000 }
}
$anyShortfall = $false
foreach ($name in $targets.Keys) {
    $info = $targets[$name]
    $metadata = Get-Content -Raw -Path $info.Path | ConvertFrom-Json
    $realTokens = [int64]$metadata.train.total_non_padding_tokens
    $pct = [math]::Round((100.0 * $realTokens / $info.Target), 1)
    Write-Log ("{0,-20} real={1,15:N0}  target={2,15:N0}  ({3}%)" -f $name, $realTokens, $info.Target, $pct)
    if ($pct -lt 70) {
        $anyShortfall = $true
        Write-Log ("  WARNING: {0} yielded only {1}% of its target token share -- the --limit estimate for this source may be too low." -f $name, $pct)
    }
}
if ($anyShortfall) {
    Write-Log ""
    Write-Log "Real shortfall found above. Proceeding in 30 seconds anyway (unattended run) -- Ctrl+C now to stop and fix --limit/--mixture first."
    Start-Sleep -Seconds 30
}

# --- Step 3: the real training launch, auto-resuming if interrupted ---
# Recipe: 150M-Modern, head_dim=96 (config default), seq_len=2048,
# --activation-checkpointing (required, see header), MTP heads on per the
# 2026-09-09 decision, WSD schedule (resumability safeguard for a multi-day
# run), batch_size=2/accumulation_steps=1 (the exact real-measured-safe
# combination, see header caveat 2). 732,422 updates x 4,096 tokens/update
# (batch_size=2 x seq_len=2048 x accumulation_steps=1) = 3,000,004,224 tokens,
# matching the frozen 3B-token V1 target (MF-063); 14,648 warmup updates (2%,
# matching this project's own established warmup convention).

$trainOutput = "artifacts\mf070-150m-release"
$resumeArgs = @()
if (Test-Path (Join-Path $trainOutput "final\model.safetensors")) {
    Write-Log ""
    Write-Log "Training already completed (found $trainOutput\final\model.safetensors) -- nothing more to do."
    exit 0
}
if (Test-Path $trainOutput) {
    $checkpoints = Get-ChildItem $trainOutput -Directory -Filter "checkpoint-*" -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending
    if ($checkpoints.Count -gt 0) {
        $latest = $checkpoints[0].FullName
        Write-Log ""
        Write-Log "Found an existing checkpoint -- resuming training from $latest"
        $resumeArgs = @("--resume", $latest)
    }
}

$mixtureArgs = @(
    "--mixture", "dclm-edu;data/shards/mf070-150m-3b-dclm-edu/train;0.35",
    "--mixture", "fineweb-edu;data/shards/mf070-150m-3b-fineweb-edu/train;0.25",
    "--mixture", "github-code;data/shards/mf070-150m-3b-github-code/train;0.20",
    "--mixture", "finemath;data/shards/mf070-150m-3b-finemath/train;0.15",
    "--mixture", "cosmopedia-v2;data/shards/mf070-150m-3b-cosmopedia-v2/train;0.05"
)

$trainArgs = @("train/pretrain.py") + $mixtureArgs + @(
    "--config", "configs/150m-modern.toml",
    "--output", $trainOutput,
    "--updates", "732422",
    "--warmup-updates", "14648",
    "--batch-size", "2",
    "--accumulation-steps", "1",
    "--schedule", "wsd",
    "--wsd-decay-fraction", "0.2",
    "--activation-checkpointing",
    "--mtp-extra-heads", "1",
    "--mtp-loss-weight", "0.3",
    "--seed", "42",
    "--device", "cuda",
    "--progress-interval", "100",
    "--checkpoint-interval", "500",
    "--keep-last-n-checkpoints", "3"
) + $resumeArgs

Write-Log ""
Write-Log "=== Starting the real 150M-Modern 3B-token training run (this takes real days, not hours) ==="
Invoke-Step "train-150m-release" $trainArgs

Write-Log ""
Write-Log "=== Pipeline finished $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
Write-Log ""
Write-Log "Step                  Status"
Write-Log "--------------------- -------"
foreach ($name in @("prep-cosmopedia-v2", "prep-finemath", "prep-fineweb-edu", "prep-dclm-edu", "prep-github-code", "train-150m-release")) {
    $value = "NOT RUN"
    if ($Status.ContainsKey($name)) { $value = $Status[$name] }
    Write-Log ("{0,-21} {1}" -f $name, $value)
}
