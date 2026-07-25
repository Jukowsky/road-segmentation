# Road Segmentation — Winter Road Surface Condition

Snow and segmentation: semi-supervised instance segmentation with limited annotation.

This project detects **road surface condition** (dry / wet / snow / ice / mud, on
asphalt / concrete / gravel) from traffic-camera imagery. Pixel-accurate,
per-class labeled data for this task barely exists, so the core of the project
is a **synthetic data pipeline** that manufactures large amounts of realistically
textured, perfectly labeled training images, plus a set of experiments that
train and evaluate several model families on that data.

## The core idea

1. **Borrow shapes, not semantics, from COCO.** `instances_train2017.json`
   annotations are used purely for their geometry — each COCO instance mask
   becomes an arbitrary binary region (foreground/background is assigned by
   `category_id % 2`, ignoring what the object actually is). This turns the
   ~118K COCO training images into a huge, diverse library of mask shapes.
2. **Texture those masks with real surface-condition crops.** Each binary mask
   is "painted" using real material crops from RSCD (Road Surface Condition
   Dataset), so every pixel gets an exact, known class (one of 27
   moisture × material combinations — see below). This yields a synthetic
   segmentation dataset with pixel-perfect ground truth, generated at scale
   (up to ~1M images: 118K masks × `SAMPLES_PER_MASK` texture variations).
3. **Train several model families** on the synthetic data (YOLO instance
   segmentation, YOLO semantic segmentation, YOLO classification, a
   sliding-window classifier baseline, RT-DETR detection) and compare them.
4. **Sanity-check against real, hand-labeled images.** A small
   `solid_label_testset/` of genuinely annotated frames is used to check
   whether a model generalizes to real surface conditions, or is just
   exploiting synthetic-texture/background shortcuts (see
   `scripts/utils/eval_solid_testset.py`).
5. **Qualitatively validate on live traffic cameras.** Real Swedish STA
   camera feeds (Grillom, Gålsjö, Husum, Långvattnet, Spjute-Bjästa) and RCM
   road-camera samples are used for visual inspection of predictions.

## Class taxonomy (27 classes)

Every class is a `{moisture state} × {material} [× severity]` combination:

| Moisture state | Materials |
|---|---|
| `dry` | asphalt (severe/slight/smooth), concrete (severe/slight/smooth), gravel, mud |
| `wet` | asphalt (severe/slight/smooth), concrete (severe/slight/smooth), gravel, mud |
| `water`-covered | asphalt (severe/slight/smooth), concrete (severe/slight/smooth), gravel, mud |
| winter | `fresh_snow`, `ice`, `melted_snow` |

Full list lives in the dataset configs, e.g. `config/masked_dataset_coco_full.yaml`.

## Repository layout

```
config/                 Ultralytics-style dataset YAMLs (data paths + class names)
datasets/                Raw inputs: coco/, rscd/ (RSCD material crops), COCO-derived
                         binary masks, real camera folders, rcm-samples, solid_label_testset/
scripts/
  train/                 One script per training experiment (see below)
  utils/                 Data generation, inference, visualization, comparison utilities
                         main.py holds the shared masking/compositing/export helpers
masked_rscd_dataset*/    Generated synthetic datasets (images + YOLO seg labels)
runs/, logs/             Ultralytics run artifacts and training logs (generated)
weights/                 Pretrained/base checkpoints used as training starting points
RT-DETR/                 Vendored RT-DETR repo (PaddlePaddle + PyTorch implementations)
notebooks/               Exploratory notebooks (RT-DETR, YOLO)
*_predictions/, *_qual*/,
compare_seg_cls/, wincls_*_results/
                         Per-experiment qualitative output folders (generated, not source)
```

Directories with `*` suffixes (e.g. `rcm_sample_predictions_lowered_threshold_trial7`,
`vvis_predictions_trial4_maskcoverage100_confthres_27`, `synthetic_predictions_1..8`)
are outputs from one-off inference/visualization runs, named after the
parameters that were varied in that run. They are safe to delete/regenerate.

## Data generation pipeline

Run from the repo root, in order:

1. `scripts/utils/generate_coco_binary_masks_full.py` — extracts binary masks
   from all COCO train2017 annotations → `datasets/coco_binary_masks_full/`.
2. `scripts/utils/generate_synthetic_dataset_full.py` — composites RSCD
   material textures onto those masks → `masked_rscd_dataset_coco_full/{images,labels}`
   (YOLO segmentation format). Resume-safe; set `SAMPLES_PER_MASK` to control
   variations per mask (default 9).
3. `scripts/utils/convert_yolo_to_sem_masks.py` — converts the YOLO polygon
   labels into dense semantic segmentation masks for the `sem_exp2x` (semantic
   head) experiments.

Both generation scripts skip files that already exist, so they can be
interrupted and safely re-run.

## Training experiments

All training scripts live in `scripts/train/` and wrap `ultralytics.YOLO`.
Naming follows `expNN`:

| Experiment(s) | Task | Notes |
|---|---|---|
| `exp10` | Classification | Retrain on the full ~1M RSCD dataset |
| `exp15`, `exp16` | Segmentation | Partial backbone unfreeze ablations on COCO-masked data |
| `exp18` | Segmentation | COCO-mask, freeze=5 baseline (reference point for later runs) |
| `exp21` / `exp22` / `exp23` | Segmentation | Full synthetic dataset, freeze=5 / freeze=3 / freeze=0 — `exp23` (freeze=0) is the main experiment |
| `exp24` / `exp25` / `exp26` | Semantic segmentation | Same freeze=5/3/0 ablation, mirrors exp21–23 with a semantic head |
| `train_yolo26m-cls-*` | Classification | Pretraining/freeze/backbone variants (COCO, RSCD, cropped, synthetic, masked) |
| `train_rtdetr*` | Detection | RT-DETR baselines (`-l`, `-s`, v2) |

Freeze-depth ablations (how many backbone layers stay frozen while
fine-tuning on synthetic data) are the recurring experimental axis across
segmentation, semantic segmentation, and classification.

## Running the full pipeline

`run_full_pipeline.sh` runs data generation + training end-to-end, sequentially:

```bash
bash run_full_pipeline.sh                    # steps 1-5
STEP_START=3 STEP_END=3 bash run_full_pipeline.sh   # run only step 3
```

Steps: (1) verify the RSCD dataset is present, (2) generate COCO binary masks,
(3) generate the synthetic dataset, (4) train classification (exp10-full),
(5) train the three segmentation freeze ablations (exp21/22/23) in parallel,
one per GPU.

`tmux_pipeline.sh` does the same but launches everything as tmux windows
across 4 GPUs (plus a live `nvidia-smi` monitor window), so long-running
generation and training jobs can be attached to and observed independently:

```bash
bash tmux_pipeline.sh          # full pipeline: mask-gen, synth-gen, all training
bash tmux_pipeline.sh train    # training only (assumes data is already generated)
tmux attach -t rscd
```

## Evaluation & comparison utilities

- `scripts/utils/eval_solid_testset.py` — validates a segmentation checkpoint
  against the real, hand-labeled `solid_label_testset/` to detect
  texture/background overfitting.
- `scripts/utils/compare_seg_cls.py` — side-by-side visual comparison of a
  segmentation model vs. the sliding-window classifier (`wincls.py`) on the
  same images.
- `scripts/utils/infer_exp21_*.py`, `infer_sem_exp26*.py` — run inference with
  specific trained checkpoints over real camera samples.
- `scripts/utils/visualize_*.py`, `visyolo*.py` — prediction/overlay
  visualization helpers.
- `results_exp10_classification.xlsx`, `results_exp21_exp22.xlsx` — exported
  metric comparisons across experiments.

## Environment

Two conda environments are used side by side:

- `canenv` — main environment (Ultralytics/YOLO stack), used for all
  `scripts/train/*.py` and `scripts/utils/*.py` scripts.
- `rtdetr` — separate environment for the vendored `RT-DETR/` repo
  (PaddlePaddle and PyTorch implementations have different dependency needs
  than Ultralytics).

There is no committed `requirements.txt`/`environment.yml` yet — set up each
environment to match the imports used in the scripts you intend to run
(`ultralytics`, `opencv-python`, `pycocotools`, `torch`, etc.).
