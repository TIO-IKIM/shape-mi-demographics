# shape-demographics

**What does anatomical *shape* know about you?**

Predicting demographic and clinical attributes from the 3D **shape** of
anatomical structures alone (segmentations -> surface point clouds -> descriptors),
with image intensity removed. This repository accompanies the ShapeMI workshop
paper and contains the full, reproducible pipeline.

> **Reproducibility contract.** No number in the paper is hand-typed. Every
> figure and statistic is generated from files in `experiments/` by the
> pipeline. Re-run a stage, and the results regenerate.

If you find this work useful, please cite:

> G. Luijten, B. Hinrichs-Puladi, M. Engelke, C. Wachinger, and J. Egger,
> "What Does Anatomical Shape Know About You? A Multi-Organ, Shape-Only Study
> of Demographic Prediction, Attribution, and Privacy,"
> *under review at ShapeMI 2026*.

---

## Key Findings

- Sex from shape alone: **AUC 0.868** (95% CI 0.846-0.886); age: **MAE 9.3 yr** (R^2 0.33)
- **Shape > intensity** for sex (0.87 vs 0.73) -- the sex signal is *geometric*
- Most of the sex signal is **organ size**; scale-free shape alone still gives AUC 0.78
- Strongest sex cue: **hip bones (AUC 0.94)** -- recovers forensic pelvic dimorphism automatically
- Generalises under **leave-one-institution-out** (sex AUC 0.83 on the large held-out site)
  and **across modality** -- sex replicates on MRI (AUC 0.76)
- **Auto-labeling:** calibrated (ECE 0.04); 50% coverage -> 92.8% accuracy
- **Privacy:** shape descriptors make a single organ near-unique within the cohort

---

## Quick Start (no data download, ~2 minutes)

```bash
bash setup.sh                      # creates ./.venv and installs the package (pip)
source .venv/bin/activate
python example/run_example.py      # offline synthetic demo
```

The example synthesises organ masks with sex/age-dependent shape and runs the
**real** pipeline end-to-end (see `example/README.md`).

### Alternative: conda

```bash
conda env create -f environment.yml && conda activate shapedem && pip install -e .
```

Both `setup.sh` (venv) and `conda env create` have been tested on **macOS
(Apple Silicon M5)** and **Linux (NVIDIA GPU cluster)**. `setup.sh`
auto-detects a working Python version (3.10–3.13); override with
`PYTHON=python3.x bash setup.sh` if needed.

### Exact reproducibility (Linux / NVIDIA cluster only)

```bash
pip install -r requirements.lock.txt && pip install -e . --no-deps
```

The lock file pins every dependency to the exact version used on the compute
cluster (Linux x86_64, CUDA 12, Python 3.11). It includes GPU-specific
packages (`nvidia-nccl-cu12`, `torch` with CUDA support) that have no macOS
wheels — use the conda or venv install above on non-Linux machines.

---

## Data

| Dataset | Subjects | License | Source |
|---|---|---|---|
| TotalSegmentator v2 | 1,228 CT | CC BY 4.0 | Zenodo 8367088 |
| TotalSegmentator-MRI | 240 MRI | CC BY 4.0 | Zenodo 11367005 |

The CT dataset is used for all main analyses. The MRI dataset is used only for
the cross-modality experiment (step 4) and is downloaded automatically by
`python -m shapedem.cli crossmodal`.

Download the CT data (scripted and resumable):

```bash
bash scripts/run_full_extract.sh   # download + extract all shapes (~23.6 GB download)
```

### Paths

No paths are hard-coded. Large artifacts go under `$SHAPEDEM_ROOT`
(default `./_workspace`). The repo ships with the directory structure
pre-created and one sample patient per dataset:

```
_workspace/
  data/              # downloaded zips (run_full_extract.sh populates this)
  features/          # extracted point-cloud npz files (one dir per subject)
    totalseg/s0004/  # sample CT patient (all 20 organs)
    totalseg_mr/s0001/  # sample MRI patient (13 organs)
  cache/             # runtime cache
  example_output/    # output from run_example.py
```

`bash setup.sh` creates the virtual environment. The pipeline commands
(`extract`, `analyze`, etc.) populate `_workspace/` with data and results.
To use a larger disk:

```bash
export SHAPEDEM_ROOT=/path/to/large/disk    # <-- SET THIS to a disk with ~50 GB free
```

---

## Reproducing All Results

```bash
# 0. Sanity gate (no full download; pulls a few subjects via HTTP range requests)
python -m shapedem.cli smoke

# 1. Full extraction (requires the TotalSegmentator download above)
#    This is done by run_full_extract.sh

# 2. Analysis stages (run in order)
python -m shapedem.cli meta          # cache labels
python -m shapedem.cli analyze       # attribution, size-vs-shape, uniqueness, fusion
python -m shapedem.cli intensity     # per-organ HU features from raw CT
python -m shapedem.cli modality      # shape vs intensity vs both
python -m shapedem.cli stats         # bootstrap CIs, leave-one-institution-out
python -m shapedem.cli labeling      # selective prediction, calibration, learning curves

# 3. Deep models (requires: pip install torch)
#    GPU recommended; runs on CPU but much slower
python -m shapedem.cli deep                    # PointNet
python -m shapedem.cli deep --encoder dgcnn    # DGCNN

# 4. Cross-modality (downloads TotalSegmentator-MRI, ~2 GB)
python -m shapedem.cli crossmodal

# 5. Comparison table + figures + result macros
python -m shapedem.cli compare
python -m shapedem.cli figures       # -> experiments/figures/
python -m shapedem.cli tables        # -> experiments/tables/ + experiments/results_macros.tex
```

<details>
<summary><b>Stage details (click to expand)</b></summary>

### Why this order?

Steps 2, 3, and 4 are independent of each other — they all read from the
extracted features and write separate result files. Step 5 (`compare`,
`figures`, `tables`) needs all prior stages to have run because it reads
their outputs to build summary tables and plots.

### Per-stage breakdown

| Stage | What it does | Output | Time |
|---|---|---|---|
| `smoke` | Quick sanity check — downloads a few subjects via HTTP range requests, extracts shapes, runs a tiny CV. No full download needed. | `experiments/smoke_results.json` | ~2 min |
| `meta` | Parses the TotalSegmentator metadata CSV (age, sex, pathology, institute) and caches it as a label table. | `experiments/totalseg_labels.csv` | seconds |
| `analyze` | Trains and evaluates **classical models** (XGBoost, 5-fold stratified CV) for sex classification and age regression. Runs per-organ attribution, size-vs-shape decomposition, cross-domain transfer, uniqueness analysis, and multi-organ fusion. | `attribution.csv`, `size_vs_shape.csv`, `cross_domain.csv`, `uniqueness_curve.csv`, `fusion.json`, `organ_presence.csv` | ~5 min |
| `intensity` | Extracts Hounsfield unit statistics (mean, std, percentiles) from the raw CT volumes for each organ, using 48 parallel workers. | `experiments/intensity_features.csv` | ~10 min |
| `modality` | Compares shape-only vs intensity-only vs combined features for sex and age prediction. | `experiments/modality_comparison.csv` | ~1 min |
| `stats` | Computes bootstrap 95% CIs on headline metrics (out-of-fold predictions), leave-one-institution-out generalization, institute demographics, uniqueness sensitivity to quantization, and truncation bias. | `headline_stats.json`, `loio.csv`, `institute_demographics.csv`, `uniqueness_sensitivity.csv`, `truncation_bias.csv` | ~5 min |
| `labeling` | Auto-labeling viability: selective prediction (accuracy vs coverage trade-off), calibration (ECE), and data-efficiency learning curves. | `selective_sex.csv`, `calibration_sex.csv`, `learning_curve_sex.csv`, `learning_curve_age.csv`, `labeling_results.json` | ~3 min |
| `deep` | Trains a multi-organ multi-task **PointNet** (or DGCNN with `--encoder dgcnn`) using the dataset's official train/val/test split. Validation-based early stopping, averaged over 3 seeds. Requires `pip install torch`. | `deep_results.json` or `deep_dgcnn_results.json` | ~30 min (GPU) |
| `crossmodal` | Downloads TotalSegmentator-MRI (~2 GB), extracts MRI shapes, runs within-MRI CV and CT↔MRI cross-modality transfer. | `crossmodal_results.json`, `totalseg_mr_labels.csv` | ~15 min |
| `compare` | Reads results from all prior stages and assembles a single comparison table (classical vs deep vs SSM vs volume-only baselines). | `experiments/comparison.csv` | ~5 min |
| `figures` | Generates all plots (attribution bar charts, size-vs-shape, uniqueness curve, labeling, SSM modes) from the experiment CSVs. | `experiments/figures/*.png` and `.pdf` | seconds |
| `tables` | Generates LaTeX tables and `\newcommand` macros from experiment files, for use in the manuscript. | `experiments/tables/*.tex`, `experiments/results_macros.tex` | seconds |

</details>

---

## Repository Layout

```
shapedem/                  Python package
  cli.py                   command-line entry point
  config.py                YAML config loader (portable ${REPO}/${ROOT} tokens)
  data/totalseg.py         dataset access (HTTP range-requests + local zip)
  shapes/extract.py        mask -> marching cubes (RAS mm) -> point cloud + FOV-truncation QC
  shapes/descriptors.py    correspondence-free descriptors (size vs scale-free shape)
  shapes/ssm.py            ICP-correspondence SSM (PCA modes)
  intensity.py             per-organ Hounsfield-unit features
  pipeline.py              streaming, resumable extraction driver
  features.py              per-subject descriptor matrix (truncation-aware)
  models/multitask.py      multi-organ multi-task PointNet / DGCNN encoders
  train/baselines.py       GBM / RF / linear CV, bootstrap CIs, cross-domain transfer
  train/train_deep.py      deep training (early stopping, multi-seed)
  analysis/core.py         attribution, size-vs-shape, LOIO, uniqueness, model comparison
  analysis/figures.py      all figures
  analysis/tables.py       LaTeX tables + \newcommand macros from result files
  analysis/labeling.py     selective prediction, calibration, learning curves
  crossmodal.py            TotalSegmentator-MRI download + CT<->MRI transfer
configs/default.yaml       organs, paths, label definitions, thresholds
experiments/               result CSVs/JSONs, figures, tables — git-tracked
tests/                     unit tests (pytest)
example/                   offline synthetic demo
```

---

## Tests

```bash
pip install -e ".[dev]" && pytest -q
```

8 unit tests covering: shape extraction, truncation detection, sphere descriptors,
Umeyama alignment, cross-validation, config loading, deep model forward passes,
and intensity feature computation.

---

## Ethics & Responsible Use

We predict **sex and age only**. We deliberately do **not** predict race/ethnicity.
The privacy results are intended as a **caution** about sharing segmentations,
not a tool for re-identification.

## Use of Large Language Models

Large language models were used to assist with literature search, code
scaffolding, and drafting. All study design, code, experiments, results
and final text were reviewed by the first author. See `DISCLAIMER.md`.

## License

Code: MIT (see `LICENSE`). The TotalSegmentator dataset retains its own license
(CC BY 4.0); derived shapes are redistributed only where the dataset license permits.
