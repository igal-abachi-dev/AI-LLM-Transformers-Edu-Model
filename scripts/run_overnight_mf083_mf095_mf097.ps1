# One-off overnight orchestration for MF-083/MF-097/MF-095's real bounded
# comparisons (2026-09-10/11). Not part of the permanent CLI surface -- it
# exists to run tonight's exact, already-pressure-tested command set
# unattended and leave a clear record of what succeeded/failed/was skipped,
# matching AGENTS.md's "never report a run successful without command
# output or a persisted run record" rule: every step's real stdout/stderr
# is captured to its own log file, and train/pretrain.py's/
# prepare_data.py's own real run.json/metadata.json remain the actual
# evidence -- this script only sequences and logs, never fabricates a
# result.
#
# Run from the repository root, in Windows PowerShell:
#   .\scripts\run_overnight_mf083_mf095_mf097.ps1
#
# Safe to walk away from: each step's failure only skips the specific later
# steps that depend on it (e.g. a failed mf097-best-fit-train data prep
# skips only the best-fit training arm, not ribbon/bos-crop/MF-083/MF-095),
# never aborts the whole run. A final summary is printed and written to
# logs\mf-overnight\SUMMARY.txt.

$Python = ".\.venv\Scripts\python.exe"
$LogDir = "logs\mf-overnight"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Summary = Join-Path $LogDir "SUMMARY.txt"
Set-Content -Path $Summary -Value "" -Encoding utf8

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
    & $Python @ArgumentList *> $logFile
    if ($LASTEXITCODE -eq 0) {
        $Status[$Name] = "SUCCESS"
        Write-Log "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] SUCCESS $Name (log: $logFile)"
    } else {
        $Status[$Name] = "FAILED"
        Write-Log "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] FAILED $Name (exit $LASTEXITCODE) -- see $logFile"
    }
}

function Skip-Step {
    param([string]$Name, [string]$Reason)
    $Status[$Name] = "SKIPPED ($Reason)"
    Write-Log "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] SKIPPED $Name -- $Reason"
}

Write-Log "=== MF-083 / MF-097 / MF-095 overnight run started $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="

# --- Step 1: data prep (network/CPU-bound) ---

Invoke-Step "prep-mf097-best-fit" @(
    "scripts/prepare_data.py", "--source", "fineweb-edu", "--start", "5000", "--limit", "42000",
    "--shuffle-seed", "123", "--shuffle-buffer", "10000", "--tokenizer", "data/tokenizer",
    "--output", "data/shards/mf097-best-fit-train", "--sequence-length", "1024",
    "--validation-fraction", "0.01", "--packing", "best_fit"
)

Invoke-Step "prep-mf097-bos-crop" @(
    "scripts/prepare_data.py", "--source", "fineweb-edu", "--start", "5000", "--limit", "42000",
    "--shuffle-seed", "123", "--shuffle-buffer", "10000", "--tokenizer", "data/tokenizer",
    "--output", "data/shards/mf097-bos-crop-train", "--sequence-length", "1024",
    "--validation-fraction", "0.01", "--packing", "bos_crop"
)

Invoke-Step "prep-mf095-dclm-edu" @(
    "scripts/prepare_data.py", "--source", "dclm-edu", "--dclm-min-score", "3", "--limit", "2500",
    "--shuffle-seed", "123", "--tokenizer", "data/tokenizer",
    "--output", "data/shards/mf095-dclm-edu-train", "--sequence-length", "1024",
    "--validation-fraction", "0.01"
)

Invoke-Step "prep-mf095-finemath" @(
    "scripts/prepare_data.py", "--source", "finemath", "--finemath-config", "finemath-4plus",
    "--limit", "2500", "--shuffle-seed", "123", "--tokenizer", "data/tokenizer",
    "--output", "data/shards/mf095-finemath-train", "--sequence-length", "1024",
    "--validation-fraction", "0.01"
)

Invoke-Step "prep-mf095-cosmopedia-v2" @(
    "scripts/prepare_data.py", "--source", "cosmopedia-v2", "--limit", "2500",
    "--shuffle-seed", "123", "--tokenizer", "data/tokenizer",
    "--output", "data/shards/mf095-cosmopedia-v2-train", "--sequence-length", "1024",
    "--validation-fraction", "0.01"
)

Invoke-Step "prep-mf095-github-code" @(
    "scripts/prepare_data.py", "--source", "github-code",
    "--github-repo-allowlist", "configs/code-repo-allowlist.txt", "--limit", "3000",
    "--shuffle-seed", "123", "--tokenizer", "data/tokenizer",
    "--output", "data/shards/mf095-github-code-train", "--sequence-length", "1024",
    "--validation-fraction", "0.01"
)

# --- Step 2: GPU training arms, strictly one at a time ---
# --keep-last-n-checkpoints 2 on every arm: tonight's first real run showed
# what happens without it (93GB from one 5000-update arm, exhausting the
# entire drive and cascading into 7 failures). 2 gives a small safety
# margin over 1 in case a crash lands mid-write, at ~4-8GB/arm instead of
# ~93GB. Each Invoke-Step also gets an -AlreadyDoneMarker so re-running this
# script after a partial failure does not waste time/GPU redoing arms that
# already completed (e.g. tonight's real mf083-baseline).

# MF-083: no new data needed.
Invoke-Step "mf083-baseline" @(
    "train/pretrain.py", "--config", "configs/150m-modern.toml",
    "--train-shards", "data/shards/mf064-150m-train/train",
    "--output", "artifacts/mf083-cautious-adamw/baseline",
    "--updates", "5000", "--batch-size", "2", "--seed", "42", "--device", "cuda",
    "--keep-last-n-checkpoints", "2"
) -AlreadyDoneMarker "artifacts/mf083-cautious-adamw/baseline/final/model.safetensors"

Invoke-Step "mf083-cautious" @(
    "train/pretrain.py", "--config", "configs/150m-modern.toml",
    "--train-shards", "data/shards/mf064-150m-train/train",
    "--output", "artifacts/mf083-cautious-adamw/cautious",
    "--updates", "5000", "--batch-size", "2", "--seed", "42", "--device", "cuda",
    "--optimizer", "cautious_adamw", "--keep-last-n-checkpoints", "2"
) -AlreadyDoneMarker "artifacts/mf083-cautious-adamw/cautious/final/model.safetensors"

# MF-097: ribbon needs no new data; best-fit/bos-crop depend on their own prep step.
Invoke-Step "mf097-ribbon" @(
    "train/pretrain.py", "--config", "configs/150m-modern.toml",
    "--train-shards", "data/shards/mf064-150m-train/train",
    "--output", "artifacts/mf097-packing/ribbon",
    "--updates", "5000", "--batch-size", "2", "--seed", "42", "--device", "cuda",
    "--keep-last-n-checkpoints", "2"
) -AlreadyDoneMarker "artifacts/mf097-packing/ribbon/final/model.safetensors"

if ($Status["prep-mf097-best-fit"] -eq "SUCCESS") {
    Invoke-Step "mf097-best-fit" @(
        "train/pretrain.py", "--config", "configs/150m-modern.toml",
        "--train-shards", "data/shards/mf097-best-fit-train/train",
        "--output", "artifacts/mf097-packing/best-fit",
        "--updates", "5000", "--batch-size", "2", "--seed", "42", "--device", "cuda",
        "--keep-last-n-checkpoints", "2"
    ) -AlreadyDoneMarker "artifacts/mf097-packing/best-fit/final/model.safetensors"
} else {
    Skip-Step "mf097-best-fit" "prep-mf097-best-fit did not succeed"
}

if ($Status["prep-mf097-bos-crop"] -eq "SUCCESS") {
    Invoke-Step "mf097-bos-crop" @(
        "train/pretrain.py", "--config", "configs/150m-modern.toml",
        "--train-shards", "data/shards/mf097-bos-crop-train/train",
        "--output", "artifacts/mf097-packing/bos-crop",
        "--updates", "5000", "--batch-size", "2", "--seed", "42", "--device", "cuda",
        "--keep-last-n-checkpoints", "2"
    ) -AlreadyDoneMarker "artifacts/mf097-packing/bos-crop/final/model.safetensors"
} else {
    Skip-Step "mf097-bos-crop" "prep-mf097-bos-crop did not succeed"
}

# MF-095: the 5-source mixture needs all four new preps; the other two arms need none.
$mixturePrepsOk = ($Status["prep-mf095-dclm-edu"] -eq "SUCCESS") -and
                  ($Status["prep-mf095-finemath"] -eq "SUCCESS") -and
                  ($Status["prep-mf095-cosmopedia-v2"] -eq "SUCCESS") -and
                  ($Status["prep-mf095-github-code"] -eq "SUCCESS")

if ($mixturePrepsOk) {
    Invoke-Step "mf095-proposed-5-source" @(
        "train/pretrain.py", "--config", "configs/150m-modern.toml",
        "--output", "artifacts/mf095-mixture/proposed-5-source",
        "--updates", "5000", "--batch-size", "2", "--seed", "42", "--device", "cuda",
        "--keep-last-n-checkpoints", "2",
        "--mixture", "dclm;data/shards/mf095-dclm-edu-train/train;0.45",
        "--mixture", "web;data/shards/mf064-150m-train/train;0.30",
        "--mixture", "code;data/shards/mf095-github-code-train/train;0.15",
        "--mixture", "math;data/shards/mf095-finemath-train/train;0.05",
        "--mixture", "synthetic;data/shards/mf095-cosmopedia-v2-train/train;0.05"
    ) -AlreadyDoneMarker "artifacts/mf095-mixture/proposed-5-source/final/model.safetensors"

    # Real decay-phase curriculum test (MF-095's own core question, previously
    # unexercised): the same proposed mixture for the stable phase, reweighted
    # toward the highest-quality/most-curated sources (Cosmopedia-v2, FineMath)
    # once WSD's decay phase starts. --decay-mixture requires --schedule wsd,
    # so a matching wsd-schedule/fixed-mixture arm is run alongside it (not
    # cosine) -- comparing these two isolates the curriculum's own real effect
    # without also confounding it with a schedule-type change.
    Invoke-Step "mf095-wsd-fixed" @(
        "train/pretrain.py", "--config", "configs/150m-modern.toml",
        "--output", "artifacts/mf095-mixture/wsd-fixed",
        "--updates", "5000", "--batch-size", "2", "--seed", "42", "--device", "cuda",
        "--keep-last-n-checkpoints", "2", "--schedule", "wsd",
        "--mixture", "dclm;data/shards/mf095-dclm-edu-train/train;0.45",
        "--mixture", "web;data/shards/mf064-150m-train/train;0.30",
        "--mixture", "code;data/shards/mf095-github-code-train/train;0.15",
        "--mixture", "math;data/shards/mf095-finemath-train/train;0.05",
        "--mixture", "synthetic;data/shards/mf095-cosmopedia-v2-train/train;0.05"
    ) -AlreadyDoneMarker "artifacts/mf095-mixture/wsd-fixed/final/model.safetensors"

    Invoke-Step "mf095-wsd-decay-curriculum" @(
        "train/pretrain.py", "--config", "configs/150m-modern.toml",
        "--output", "artifacts/mf095-mixture/wsd-decay-curriculum",
        "--updates", "5000", "--batch-size", "2", "--seed", "42", "--device", "cuda",
        "--keep-last-n-checkpoints", "2", "--schedule", "wsd",
        "--mixture", "dclm;data/shards/mf095-dclm-edu-train/train;0.45",
        "--mixture", "web;data/shards/mf064-150m-train/train;0.30",
        "--mixture", "code;data/shards/mf095-github-code-train/train;0.15",
        "--mixture", "math;data/shards/mf095-finemath-train/train;0.05",
        "--mixture", "synthetic;data/shards/mf095-cosmopedia-v2-train/train;0.05",
        "--decay-mixture", "dclm;0.35",
        "--decay-mixture", "web;0.15",
        "--decay-mixture", "math;0.15",
        "--decay-mixture", "synthetic;0.20"
    ) -AlreadyDoneMarker "artifacts/mf095-mixture/wsd-decay-curriculum/final/model.safetensors"
} else {
    Skip-Step "mf095-proposed-5-source" "one or more of its four data-prep steps did not succeed"
    Skip-Step "mf095-wsd-fixed" "one or more of its four data-prep steps did not succeed"
    Skip-Step "mf095-wsd-decay-curriculum" "one or more of its four data-prep steps did not succeed"
}

Invoke-Step "mf095-fineweb-only" @(
    "train/pretrain.py", "--config", "configs/150m-modern.toml",
    "--train-shards", "data/shards/mf064-150m-train/train",
    "--output", "artifacts/mf095-mixture/fineweb-only",
    "--updates", "5000", "--batch-size", "2", "--seed", "42", "--device", "cuda",
    "--keep-last-n-checkpoints", "2"
) -AlreadyDoneMarker "artifacts/mf095-mixture/fineweb-only/final/model.safetensors"

Invoke-Step "mf095-web-code-70-30" @(
    "train/pretrain.py", "--config", "configs/150m-modern.toml",
    "--output", "artifacts/mf095-mixture/web-code-70-30",
    "--updates", "5000", "--batch-size", "2", "--seed", "42", "--device", "cuda",
    "--keep-last-n-checkpoints", "2",
    "--mixture", "web;data/shards/mf064-150m-train/train;0.7",
    "--mixture", "code;data/shards/mf094-code-train/train;0.3"
) -AlreadyDoneMarker "artifacts/mf095-mixture/web-code-70-30/final/model.safetensors"

# --- Final summary ---

Write-Log ""
Write-Log "=== Overnight run finished $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ==="
Write-Log ""
Write-Log "Step                          Status"
Write-Log "----------------------------- -------"

$order = @(
    "prep-mf097-best-fit", "prep-mf097-bos-crop", "prep-mf095-dclm-edu", "prep-mf095-finemath",
    "prep-mf095-cosmopedia-v2", "prep-mf095-github-code",
    "mf083-baseline", "mf083-cautious", "mf097-ribbon", "mf097-best-fit", "mf097-bos-crop",
    "mf095-proposed-5-source", "mf095-wsd-fixed", "mf095-wsd-decay-curriculum",
    "mf095-fineweb-only", "mf095-web-code-70-30"
)
foreach ($name in $order) {
    $value = "NOT RUN"
    if ($Status.ContainsKey($name)) { $value = $Status[$name] }
    Write-Log ("{0,-30} {1}" -f $name, $value)
}
