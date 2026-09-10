#!/usr/bin/env bash
# One-off overnight orchestration for MF-083/MF-097/MF-095's real bounded
# comparisons (2026-09-10/11). Not part of the permanent CLI surface -- it
# exists to run tonight's exact, already-pressure-tested command set
# unattended and leave a clear record of what succeeded/failed/was skipped,
# matching AGENTS.md's "never report a training run successful without
# command output or a persisted run record" rule: every step's real stdout/
# stderr is captured to its own log file, and train/pretrain.py's/
# prepare_data.py's own real run.json/metadata.json remain the actual
# evidence -- this script only sequences them and never fabricates a result.
#
# Run from the repository root:
#   bash scripts/run_overnight_mf083_mf095_mf097.sh
#
# Safe to walk away from: each step's failure only skips the specific later
# steps that depend on it (e.g. a failed mf097-best-fit-train data prep
# skips only the best-fit training arm, not ribbon/bos-crop/MF-083/MF-095),
# never aborts the whole run. A final summary is printed and written to
# logs/mf-overnight/SUMMARY.txt.

set -uo pipefail

PYTHON="./.venv/Scripts/python.exe"
LOG_DIR="logs/mf-overnight"
mkdir -p "$LOG_DIR"
SUMMARY="$LOG_DIR/SUMMARY.txt"
: > "$SUMMARY"

declare -A STATUS

log_line() {
    echo "$1" | tee -a "$SUMMARY"
}

# run_step <step_name> <command...>
# Skips (without running) if $2 is the literal string "SKIP" (used to
# propagate a failed prerequisite to whatever depends on it).
run_step() {
    local name="$1"
    shift
    local log_file="$LOG_DIR/$name.log"
    local start_ts
    start_ts=$(date '+%Y-%m-%d %H:%M:%S')
    log_line "[$start_ts] START $name"
    if "$@" > "$log_file" 2>&1; then
        STATUS["$name"]="SUCCESS"
        log_line "[$(date '+%Y-%m-%d %H:%M:%S')] SUCCESS $name (log: $log_file)"
    else
        STATUS["$name"]="FAILED"
        log_line "[$(date '+%Y-%m-%d %H:%M:%S')] FAILED $name -- see $log_file"
    fi
}

skip_step() {
    local name="$1"
    local reason="$2"
    STATUS["$name"]="SKIPPED ($reason)"
    log_line "[$(date '+%Y-%m-%d %H:%M:%S')] SKIPPED $name -- $reason"
}

log_line "=== MF-083 / MF-097 / MF-095 overnight run started $(date '+%Y-%m-%d %H:%M:%S') ==="

# --- Step 1: data prep (network/CPU-bound) ---

run_step "prep-mf097-best-fit" "$PYTHON" scripts/prepare_data.py \
    --source fineweb-edu --start 5000 --limit 42000 --shuffle-seed 123 --shuffle-buffer 10000 \
    --tokenizer data/tokenizer --output data/shards/mf097-best-fit-train \
    --sequence-length 1024 --validation-fraction 0.01 --packing best_fit

run_step "prep-mf097-bos-crop" "$PYTHON" scripts/prepare_data.py \
    --source fineweb-edu --start 5000 --limit 42000 --shuffle-seed 123 --shuffle-buffer 10000 \
    --tokenizer data/tokenizer --output data/shards/mf097-bos-crop-train \
    --sequence-length 1024 --validation-fraction 0.01 --packing bos_crop

run_step "prep-mf095-dclm-edu" "$PYTHON" scripts/prepare_data.py \
    --source dclm-edu --dclm-min-score 3 --limit 2500 --shuffle-seed 123 \
    --tokenizer data/tokenizer --output data/shards/mf095-dclm-edu-train \
    --sequence-length 1024 --validation-fraction 0.01

run_step "prep-mf095-finemath" "$PYTHON" scripts/prepare_data.py \
    --source finemath --finemath-config finemath-4plus --limit 2500 --shuffle-seed 123 \
    --tokenizer data/tokenizer --output data/shards/mf095-finemath-train \
    --sequence-length 1024 --validation-fraction 0.01

run_step "prep-mf095-cosmopedia-v2" "$PYTHON" scripts/prepare_data.py \
    --source cosmopedia-v2 --limit 2500 --shuffle-seed 123 \
    --tokenizer data/tokenizer --output data/shards/mf095-cosmopedia-v2-train \
    --sequence-length 1024 --validation-fraction 0.01

run_step "prep-mf095-github-code" "$PYTHON" scripts/prepare_data.py \
    --source github-code --github-repo-allowlist configs/code-repo-allowlist.txt \
    --limit 3000 --shuffle-seed 123 \
    --tokenizer data/tokenizer --output data/shards/mf095-github-code-train \
    --sequence-length 1024 --validation-fraction 0.01

# --- Step 2: GPU training arms, strictly one at a time ---

# MF-083: no new data needed.
run_step "mf083-baseline" "$PYTHON" train/pretrain.py \
    --config configs/150m-modern.toml --train-shards data/shards/mf064-150m-train/train \
    --output artifacts/mf083-cautious-adamw/baseline --updates 5000 --batch-size 2 --seed 42 --device cuda

run_step "mf083-cautious" "$PYTHON" train/pretrain.py \
    --config configs/150m-modern.toml --train-shards data/shards/mf064-150m-train/train \
    --output artifacts/mf083-cautious-adamw/cautious --updates 5000 --batch-size 2 --seed 42 --device cuda \
    --optimizer cautious_adamw

# MF-097: ribbon needs no new data; best-fit/bos-crop depend on their own prep step.
run_step "mf097-ribbon" "$PYTHON" train/pretrain.py \
    --config configs/150m-modern.toml --train-shards data/shards/mf064-150m-train/train \
    --output artifacts/mf097-packing/ribbon --updates 5000 --batch-size 2 --seed 42 --device cuda

if [[ "${STATUS[prep-mf097-best-fit]:-}" == "SUCCESS" ]]; then
    run_step "mf097-best-fit" "$PYTHON" train/pretrain.py \
        --config configs/150m-modern.toml --train-shards data/shards/mf097-best-fit-train/train \
        --output artifacts/mf097-packing/best-fit --updates 5000 --batch-size 2 --seed 42 --device cuda
else
    skip_step "mf097-best-fit" "prep-mf097-best-fit did not succeed"
fi

if [[ "${STATUS[prep-mf097-bos-crop]:-}" == "SUCCESS" ]]; then
    run_step "mf097-bos-crop" "$PYTHON" train/pretrain.py \
        --config configs/150m-modern.toml --train-shards data/shards/mf097-bos-crop-train/train \
        --output artifacts/mf097-packing/bos-crop --updates 5000 --batch-size 2 --seed 42 --device cuda
else
    skip_step "mf097-bos-crop" "prep-mf097-bos-crop did not succeed"
fi

# MF-095: the 5-source mixture needs all four new preps; the other two arms need none.
if [[ "${STATUS[prep-mf095-dclm-edu]:-}" == "SUCCESS" && "${STATUS[prep-mf095-finemath]:-}" == "SUCCESS" \
    && "${STATUS[prep-mf095-cosmopedia-v2]:-}" == "SUCCESS" && "${STATUS[prep-mf095-github-code]:-}" == "SUCCESS" ]]; then
    run_step "mf095-proposed-5-source" "$PYTHON" train/pretrain.py \
        --config configs/150m-modern.toml --output artifacts/mf095-mixture/proposed-5-source \
        --updates 5000 --batch-size 2 --seed 42 --device cuda \
        --mixture "dclm;data/shards/mf095-dclm-edu-train/train;0.45" \
        --mixture "web;data/shards/mf064-150m-train/train;0.30" \
        --mixture "code;data/shards/mf095-github-code-train/train;0.15" \
        --mixture "math;data/shards/mf095-finemath-train/train;0.05" \
        --mixture "synthetic;data/shards/mf095-cosmopedia-v2-train/train;0.05"
else
    skip_step "mf095-proposed-5-source" "one or more of its four data-prep steps did not succeed"
fi

run_step "mf095-fineweb-only" "$PYTHON" train/pretrain.py \
    --config configs/150m-modern.toml --train-shards data/shards/mf064-150m-train/train \
    --output artifacts/mf095-mixture/fineweb-only --updates 5000 --batch-size 2 --seed 42 --device cuda

run_step "mf095-web-code-70-30" "$PYTHON" train/pretrain.py \
    --config configs/150m-modern.toml --output artifacts/mf095-mixture/web-code-70-30 \
    --updates 5000 --batch-size 2 --seed 42 --device cuda \
    --mixture "web;data/shards/mf064-150m-train/train;0.7" \
    --mixture "code;data/shards/mf094-code-train/train;0.3"

# --- Final summary ---

log_line ""
log_line "=== Overnight run finished $(date '+%Y-%m-%d %H:%M:%S') ==="
log_line ""
log_line "Step                          Status"
log_line "----------------------------- -------"
for name in prep-mf097-best-fit prep-mf097-bos-crop prep-mf095-dclm-edu prep-mf095-finemath \
    prep-mf095-cosmopedia-v2 prep-mf095-github-code \
    mf083-baseline mf083-cautious mf097-ribbon mf097-best-fit mf097-bos-crop \
    mf095-proposed-5-source mf095-fineweb-only mf095-web-code-70-30; do
    printf "%-30s %s\n" "$name" "${STATUS[$name]:-NOT RUN}" | tee -a "$SUMMARY"
done
