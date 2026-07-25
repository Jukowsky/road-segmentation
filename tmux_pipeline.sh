#!/usr/bin/env bash
# tmux_pipeline.sh — tüm adımları tmux oturumunda paralel başlatır
#
# Kullanım:
#   bash tmux_pipeline.sh          # tüm adımlar
#   bash tmux_pipeline.sh train    # sadece training (veri hazırsa)
#
# Oturumu izlemek için:
#   tmux attach -t rscd
#
# Pencereler arası geçiş: Ctrl+B, ardından pencere numarası
# ─────────────────────────────────────────────────────────────────────────────

SESSION="rscd"
REPO="/home/talt_wireten_c/road-segmentation"
MODE="${1:-full}"
PYTHON="/home/talt_wireten_c/miniconda3/envs/canenv/bin/python"

# Varolan oturumu kapat
tmux kill-session -t "$SESSION" 2>/dev/null || true

cd "$REPO"

# ── Oturum + pencereler ───────────────────────────────────────────────────────

# [0] gpu-monitor — sürekli açık kalır
tmux new-session  -d -s "$SESSION" -n "gpu-monitor"
tmux send-keys    -t "$SESSION:gpu-monitor" \
    "watch -n 2 nvidia-smi" Enter

if [[ "$MODE" == "full" ]]; then
    # [1] mask-gen — COCO binary maskeler (CPU, ~1-2 saat)
    tmux new-window -t "$SESSION" -n "mask-gen"
    tmux send-keys  -t "$SESSION:mask-gen" \
        "cd $REPO && $PYTHON scripts/utils/generate_coco_binary_masks_full.py && echo '=== MASK GEN DONE ===' && bash -i" Enter

    # [2] synth-gen — sentetik dataset (CPU + disk, ~4-8 saat)
    tmux new-window -t "$SESSION" -n "synth-gen"
    tmux send-keys  -t "$SESSION:synth-gen" \
        "cd $REPO && echo 'Mask gen bekleniyor...' && until [ -d datasets/coco_binary_masks_full ] && [ \$(find datasets/coco_binary_masks_full -name '*.png' 2>/dev/null | wc -l) -gt 50000 ]; do sleep 60; done && echo 'Basliyor...' && $PYTHON scripts/utils/generate_synthetic_dataset_full.py && echo '=== SYNTH GEN DONE ===' && bash -i" Enter
fi

# Segmentasyon deneyleri sentetik dataset hazır olana kadar bekler
WAIT_SYNTH="until [ \$(find $REPO/masked_rscd_dataset_coco_full/images -name '*.jpg' 2>/dev/null | wc -l) -gt 100000 ]; do echo 'Sentetik dataset bekleniyor...'; sleep 120; done && echo 'Dataset hazir, training basliyor...'"

# [3] exp21 — GPU 0, freeze=5
tmux new-window -t "$SESSION" -n "exp21-gpu0"
tmux send-keys  -t "$SESSION:exp21-gpu0" \
    "cd $REPO && $WAIT_SYNTH && $PYTHON scripts/train/train_exp21_full_freeze5.py 2>&1 | tee logs/exp21_full_freeze5.log" Enter

# [4] exp22 — GPU 1, freeze=3
tmux new-window -t "$SESSION" -n "exp22-gpu1"
tmux send-keys  -t "$SESSION:exp22-gpu1" \
    "cd $REPO && $WAIT_SYNTH && $PYTHON scripts/train/train_exp22_full_freeze3.py 2>&1 | tee logs/exp22_full_freeze3.log" Enter

# [5] exp23 — GPU 2, freeze=0  (ana deney)
tmux new-window -t "$SESSION" -n "exp23-gpu2"
tmux send-keys  -t "$SESSION:exp23-gpu2" \
    "cd $REPO && $WAIT_SYNTH && $PYTHON scripts/train/train_exp23_full_freeze0.py 2>&1 | tee logs/exp23_full_freeze0.log" Enter

# [6] exp10 — GPU 3, sınıflandırma
tmux new-window -t "$SESSION" -n "exp10-gpu3"
tmux send-keys  -t "$SESSION:exp10-gpu3" \
    "cd $REPO && $PYTHON scripts/train/train_exp10_rscd_full.py 2>&1 | tee logs/exp10_rscd_full.log" Enter

# [7] logs — tüm logları izle
tmux new-window -t "$SESSION" -n "logs"
tmux send-keys  -t "$SESSION:logs" \
    "cd $REPO/logs && tail -f exp23_full_freeze0.log" Enter

# gpu-monitor penceresine dön
tmux select-window -t "$SESSION:gpu-monitor"

echo ""
echo "tmux oturumu başlatıldı: '$SESSION'"
echo ""
echo "Bağlanmak için:"
echo "  tmux attach -t $SESSION"
echo ""
echo "Pencereler:"
echo "  0: gpu-monitor   — nvidia-smi (canlı)"
if [[ "$MODE" == "full" ]]; then
echo "  1: mask-gen      — COCO maske üretimi"
echo "  2: synth-gen     — sentetik dataset (mask-gen sonrası otomatik başlar)"
fi
echo "  3: exp21-gpu0    — segmentasyon freeze=5"
echo "  4: exp22-gpu1    — segmentasyon freeze=3"
echo "  5: exp23-gpu2    — segmentasyon freeze=0  ← ana deney"
echo "  6: exp10-gpu3    — sınıflandırma"
echo "  7: logs          — exp23 log izleme"
echo ""
echo "Pencere geçişi: Ctrl+B → pencere numarası"
echo "Tüm pencereleri görmek: Ctrl+B → w"
