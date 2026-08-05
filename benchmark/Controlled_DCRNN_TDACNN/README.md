# MV-AFA Controlled Baselines: DCRNN + TDA-CNN

This folder contains two protocol-compatible baseline reimplementations for the
MV-AFA journal extension. Both models consume the same precomputed EEG window
cache as MV-AFA, then use the same subject-level split, class balancing,
threshold selection, and metric computation. The goal is to make the controlled
baseline comparison reflect model behavior rather than differences in
preprocessing or data partitioning.

| File | Baseline | Source |
|------|----------|--------|
| `dcrnn.py`   | Dynamic-graph GNN (DCGRU, diffusion conv + GRU, dynamic correlation graph) | Tang et al., ICLR 2022 (arXiv:2104.08336) |
| `tda_cnn.py` | Vietoris-Rips persistence -> persistence image -> 2-D CNN | Wang et al., Frontiers in Physiology 2023 (PMC10773586) |
| `protocol.py` | Shared split / balancing / threshold / metrics (the unified protocol) | MV-AFA paper §4 |
| `run_baseline.py` | Train + evaluate driver | MV-AFA controlled protocol |
| `export_cache_to_npz.py` | One-time helper to export the MV-AFA window cache | MV-AFA data layer |

## Install

```
pip install torch scikit-learn numpy ripser
```

`ripser` is required for a faithful H0+H1 TDA baseline. A NumPy H0-only fallback
keeps the file importable, but the reported baseline should be produced with
`ripser` installed.

## Input: precomputed window cache

The baselines consume the **same windowed cache MV-AFA uses** — a `.npz` with:

| key | shape / type | meaning |
|-----|--------------|---------|
| `X` | float32 `[N, 18, 512]` | 2 s windows, 18 bipolar channels, 256 Hz (CHB-MIT) |
| `y` | int `[N]` | 0 non-seizure / 1 seizure |
| `subject` | object `[N]` | subject id (subject-disjoint split) |
| `rec_id` | object `[N]` | recording id (event grouping) |
| `t_start` | float32 `[N]` | window start time (s) in its recording |
| `win_sec`, `step_sec` | scalar | 2.0, 4.0 |

> Export this from whatever MV-AFA already uses so preprocessing stays
> byte-identical across all baselines. Do **not** re-extract windows here.

If the MV-AFA data layer is stored outside this benchmark repository, point the
export helper to it explicitly:

```bash
python export_cache_to_npz.py \
    --dataset chb_mit \
    --raw-root /path/to/CHB-MIT-scalp-eeg-database-1.0.0 \
    --cache-dir /path/to/mvafa/cache \
    --mvafa-code-root /path/to/mvafa-eeg-seizure-code \
    --out chbmit_windows.npz
```

## Run
```bash
# Patient-independent split -> the controlled-comparison row on CHB-MIT.
python run_baseline.py --model dcrnn --cache chbmit_windows.npz --protocol pi
python run_baseline.py --model tda_cnn --cache chbmit_windows.npz --protocol pi

# LOSO -> mean ± std across all folds (the LOSO table)
python run_baseline.py --model dcrnn   --cache chbmit_windows.npz --protocol loso
```

Outputs the full MV-AFA metric set: **Acc, Sens, Spec, F1, AUROC, AUPRC,
event detection rate, mean/median delay, false-alarms/hour**, written to JSON.

## Results status

The table under `../../results/controlled_baseline_chbmit_estimated_placeholder.*`
contains draft estimated placeholder values used for manuscript planning. Those
values are included only so the paper table and the repository stay in sync
during development. They must be replaced by completed JSON/CSV outputs from
this runner before the numbers are used as final experimental evidence.

## Matching MV-AFA exactly
- Training mirrors the paper: AdamW, lr 1e-4, wd 1e-4, batch 16, dropout 0.2,
  50 epochs, class-weighted CE, seed 42.
- `--neg_pos_ratio 1.0` = balanced test (your **main-table** setting);
  `--neg_pos_ratio 5.0` reproduces the `neg-pos-ratio 5.0` LOSO optimization runs.
- `--threshold {f1,youden,sens_floor_spec}` with `--min_sens 0.70` reproduces
  the `sens_floor_spec` / `min-sensitivity 0.70` strategy. Threshold is chosen on
  **validation only** — keep this fixed across all baselines and MV-AFA.

## Three integration hooks (`# >>> SWAP <<<` in `protocol.py`)
If your real harness (BioCAS2026-Benchmark) already defines them, import those
instead so every baseline shares byte-identical logic:
`subject_disjoint_split`, `select_threshold`, `compute_metrics`.

## Important: report event-level metrics at natural prevalence
At CHB-MIT's natural window prevalence (~0.31 %), window-level F1 is inherently
near-zero for any detector — it is **not** a missed-seizure signal. For LOSO /
external validation, lead with **AUROC, event detection rate, delay, FA/h**
(SzCORE-style); keep F1 only with a prevalence note.

## Cost note (TDA)
VR persistence is the expensive offline step. `windows_to_images_parallel()`
fans it across CPU cores and you should **cache** the images — ripser gets no
benefit from GPU/MPS.
