#!/usr/bin/env bash
# run_full_pipeline.sh
#
# Full re-training pipeline after transferring the complete 1M RSCD dataset.
#
# Steps
#   1. Verify RSCD data is present
#   2. Generate COCO binary masks (all 118K images)
#   3. Generate ~1M synthetic images (resume-safe)
#   4. Train classification   — exp10-full  (GPU 3)
#   5. Train segmentation     — exp21 freeze5  (GPU 0)
#                             — exp22 freeze3  (GPU 1)
#                             — exp23 freeze0  (GPU 2)  ← main experiment
#
# Usage:
#   cd /home/talt_wireten_c/road-segmentation
#   bash run_full_pipeline.sh
#
#   To run only one step, set STEP_START / STEP_END:
#   STEP_START=3 STEP_END=3 bash run_full_pipeline.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO=/home/talt_wireten_c/road-segmentation
STEP_START=${STEP_START:-1}
STEP_END=${STEP_END:-5}

log() { echo "[$(date '+%H:%M:%S')] $*"; }
step_enabled() { [[ $1 -ge $STEP_START && $1 -le $STEP_END ]]; }

cd "$REPO"

# ── Step 1: verify RSCD data ──────────────────────────────────────────────────
if step_enabled 1; then
    log "=== STEP 1: Verify RSCD dataset ==="
    RSCD_TRAIN="$REPO/datasets/rscd/train"
    if [[ ! -d "$RSCD_TRAIN" ]]; then
        echo "[ERROR] RSCD train directory not found: $RSCD_TRAIN"
        echo "Transfer the full 1M RSCD dataset before running this pipeline."
        exit 1
    fi
    N_IMAGES=$(find "$RSCD_TRAIN" -maxdepth 2 -type f \( -name "*.jpg" -o -name "*.png" \) | wc -l)
    log "RSCD train images found: $N_IMAGES"
    if [[ $N_IMAGES -lt 500000 ]]; then
        echo "[WARNING] Only $N_IMAGES images found in RSCD train — expected ~1M."
        echo "          Transfer may be incomplete. Continue anyway? [y/N]"
        read -r REPLY
        [[ "$REPLY" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 1; }
    fi
    log "STEP 1 done."
fi

# ── Step 2: generate COCO binary masks (full 118K) ───────────────────────────
if step_enabled 2; then
    log "=== STEP 2: Generate COCO binary masks (full) ==="
    MASK_OUT="$REPO/datasets/coco_binary_masks_full"
    EXISTING_MASKS=$(find "$MASK_OUT" -name "*.png" 2>/dev/null | wc -l)
    log "Existing masks: $EXISTING_MASKS (will skip already-done)"
    python scripts/utils/generate_coco_binary_masks_full.py
    log "STEP 2 done."
fi

# ── Step 3: generate synthetic dataset (~1M images) ──────────────────────────
if step_enabled 3; then
    log "=== STEP 3: Generate synthetic dataset (SAMPLES_PER_MASK=${SAMPLES_PER_MASK:-9}) ==="
    SYNTH_OUT="$REPO/masked_rscd_dataset_coco_full/images"
    EXISTING_SYNTH=$(find "$SYNTH_OUT" -name "*.jpg" 2>/dev/null | wc -l)
    log "Existing synthetic images: $EXISTING_SYNTH (will skip already-done)"
    SAMPLES_PER_MASK=${SAMPLES_PER_MASK:-9} python scripts/utils/generate_synthetic_dataset_full.py
    log "STEP 3 done."
fi

# ── Step 4: classification retrain ───────────────────────────────────────────
if step_enabled 4; then
    log "=== STEP 4: Retrain classification (exp10-full, GPU 3) ==="
    python scripts/train/train_exp10_rscd_full.py 2>&1 | tee logs/exp10_rscd_full.log
    log "STEP 4 done."
fi

# ── Step 5: segmentation experiments (parallel on 3 GPUs) ───────────────────
if step_enabled 5; then
    log "=== STEP 5: Segmentation experiments exp21/22/23 ==="
    mkdir -p logs

    log "  Launching exp21 (freeze=5, GPU 0) ..."
    python scripts/train/train_exp21_full_freeze5.py 2>&1 | tee logs/exp21_full_freeze5.log &
    PID21=$!

    log "  Launching exp22 (freeze=3, GPU 1) ..."
    python scripts/train/train_exp22_full_freeze3.py 2>&1 | tee logs/exp22_full_freeze3.log &
    PID22=$!

    log "  Launching exp23 (freeze=0, GPU 2) ..."
    python scripts/train/train_exp23_full_freeze0.py 2>&1 | tee logs/exp23_full_freeze0.log &
    PID23=$!

    log "  Waiting for all three experiments to finish..."
    wait $PID21 && log "  exp21 finished." || log "  [WARN] exp21 exited with error — check logs/exp21_full_freeze5.log"
    wait $PID22 && log "  exp22 finished." || log "  [WARN] exp22 exited with error — check logs/exp22_full_freeze3.log"
    wait $PID23 && log "  exp23 finished." || log "  [WARN] exp23 exited with error — check logs/exp23_full_freeze0.log"

    log "STEP 5 done."
fi

log "=== Pipeline complete ==="
log "Results:"
log "  Classification : runs/classify/exp10_rscd_full/"
log "  Segmentation   : runs/segment/exp21_full_freeze5/"
log "                   runs/segment/exp22_full_freeze3/"
log "                   runs/segment/exp23_full_freeze0/  ← main"
