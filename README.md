# BMI Prediction from 3D Brain MRI

CNN models to predict **BMI** (and related cardiometabolic disease outcomes) from 3D structural MRI images.

## Overview

This repository contains two sets of scripts:

| Directory | Task |
|-----------|------|
| `.` (root) | Predict **BMI** as a continuous value from a 3D brain MRI scan using soft-label classification (KL-divergence loss over binned BMI values). |
| `e2e/` | Predict **9 cardiometabolic diseases** end-to-end from 3D brain MRI using binary cross-entropy loss. |

Both pipelines share the same SFCN/`BrainToBMINet` backbone architecture, adapted from:
> Peng et al., *Accurate brain age prediction with lightweight deep neural networks*, Medical Image Analysis, 2021. https://doi.org/10.1016/j.media.2021.102013

---

## Requirements

```
torch
torchvision
numpy
pandas
scipy
nibabel
scikit-learn
torcheval   # e2e/ only
```

---

## Data format

Each dataset requires:

1. **A metadata CSV** with at least the following columns:

   | Column | Description |
   |--------|-------------|
   | `subject_id` | Unique subject identifier |
   | `image_id` | Unique image identifier (used to build the filename) |
   | `bmi` | BMI value |
   | `r_correlation` | Image quality metric (scans with `r_correlation <= 0.4` are excluded) |
   | `days_since_baseline` | Days from baseline visit (used to compute age at scan) |
   | `age_at_baseline` | Age at baseline |

   For `e2e/`, the CSV also needs a column per disease label (see `list_disease` in `e2e/train.py`).

2. **Preprocessed NIfTI images** (`.nii.gz`) registered to MNI152 space, named:
   ```
   <img_dir>/<dataset>/temp_preprocessing_<image_id>.nii.gz
   ```

3. **A brain mask** NIfTI file (e.g. `MNI152_T1_1mm_brain_mask_dil.nii.gz`).

---

## BMI prediction (root)

### Training

```bash
python train.py \
    --data_dir /path/to/metadata/ \
    --mask_path /path/to/MNI152_T1_1mm_brain_mask_dil.nii.gz \
    --img_dir /path/to/images/ \
    --datasets ukbb,hcp,hcp-aging \
    --start_epoch 1 \
    --num_epoch 100 \
    --batch_size 8 \
    --ckpt_dir ./weights/
```

Resume training from a checkpoint:
```bash
python train.py ... --start_epoch 31 --num_epoch 70
```

### Inference

```bash
python pred.py \
    --data_dir /path/to/metadata/ \
    --mask_path /path/to/MNI152_T1_1mm_brain_mask_dil.nii.gz \
    --img_dir /path/to/images/ \
    --num_epoch 50 \
    --ckpt_dir ./weights/ \
    --out_dir ./pred/
```

### Feature map extraction

```bash
python extract_feature_map.py \
    --data_dir /path/to/metadata/ \
    --mask_path /path/to/MNI152_T1_1mm_brain_mask_dil.nii.gz \
    --img_dir /path/to/images/ \
    --ckpt_dir ./weights/ \
    --epoch 50 \
    --out_dir ./tbl/
```

---

## Disease prediction (e2e/)

Predicts 9 cardiometabolic disease labels using `BCEWithLogitsLoss`. Pre-split train/test CSVs are expected (e.g. from a k-fold cross-validation setup).

### Training

```bash
python e2e/train.py \
    --data_dir /path/to/metadata/ \
    --mask_path /path/to/MNI152_T1_1mm_brain_mask_dil.nii.gz \
    --img_dir /path/to/images/ukbb/ \
    --fold 0 \
    --start_epoch 1 \
    --num_epoch 30 \
    --batch_size 8
```

### Inference

```bash
python e2e/pred.py \
    --data_dir /path/to/metadata/ \
    --mask_path /path/to/MNI152_T1_1mm_brain_mask_dil.nii.gz \
    --img_dir /path/to/images/ukbb/ \
    --fold 0 \
    --epoch 30
```

---

## Model architecture

**SFCN / BrainToBMINet** — a lightweight 3D fully convolutional network with 6 convolutional blocks, average pooling, optional dropout, and a 1×1×1 convolutional output layer.

**BrainToDiseaseNet** — same convolutional backbone, but replaces the fixed average pool with an adaptive average pool and a fully-connected prediction head.

---

## Output files

| Script | Output |
|--------|--------|
| `train.py` | `log.txt`, `weights/checkpoint_epoch<NNN>.pth.tar` |
| `pred.py` | `pred/gt_<dataset>.npy`, `pred/pred_<dataset>.npy`, `pred/tbl_pred_epoch<N>.csv` |
| `extract_feature_map.py` | `tbl/fm_64.csv`, `tbl/tbl_fm_64.csv` |
| `e2e/train.py` | `log_fold<F>.txt`, `ckpt_fold<F>/checkpoint_epoch<NNN>.pth.tar` |
| `e2e/pred.py` | `log_pred_fold<F>.txt` |
