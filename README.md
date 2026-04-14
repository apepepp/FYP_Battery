# 🔋 Battery SoH Trajectory Prediction

A comprehensive pipeline for predicting **State of Health (SoH) trajectories** of lithium-ion battery cells from early-cycle features, using machine learning, deep learning, and physics-informed models.

---

## 📋 Overview

This repository contains two main scripts:

| Script | Description |
|---|---|
| `data_extraction_v4.py` | Extracts and unifies raw cycling data from **ISU-ILCC** and **UofM** datasets into a single CSV |
| `soh_trajectory_prediction_v4.py` | Trains and evaluates **8 models** to predict full SoH trajectories from early-cycle observations |

**Datasets used:**
- **ISU-ILCC** — 251 NMC/graphite polymer cells (250 mAh), 63 cycling conditions, varying charge rate, discharge rate, and depth of discharge (DoD)
- **UofM** — 21 NMC111/graphite pouch cells (5.0 Ah), with expansion, EIS, and C-rate characterisation data

---

## 📥 Downloading the Datasets

### ISU-ILCC Battery Aging Dataset
> 251 lithium-ion polymer cells (NMC, 250 mAh) cycled under 63 unique conditions at Iowa State University and Iowa Lakes Community College.

- **Download:** [OpenEnergy Hub — ISU-ILCC Battery Aging Dataset](https://openenergyhub.ornl.gov/explore/dataset/isu-ilcc-battery-aging-dataset/)
- Format: JSON files organised into `RPT_json/` and `Cycling_json/` sub-folders, split into `Release 1.0` and `Release 2.0`
- Metadata: provided as a CSV (`ISU.csv`) with columns: `Cell_ID`, `Group_Num`, `Charging_Crate`, `Discharging_Crate`, `Mean_DoD_pct`

After downloading, place files in the following structure:
```
ISU/
├── ISU.csv
├── valid_cells.csv
├── RPT_json/
│   ├── Release 1.0/
│   └── Release 2.0/
└── Cycling_json/
    ├── Release 1.0/
    └── Release 2.0/
```

### UofM Battery Expansion Dataset
> 21 NMC111/graphite pouch cells (5.0 Ah) with expansion, OCV, resistance, and C-rate characterisation data from the University of Michigan Battery Lab.

- **Download (Kaggle mirror):** [Michigan Expansion Battery Dataset — Kaggle](https://www.kaggle.com/datasets/sidharthdk/michigan-expansion-battery-dataset)
- **Download (Battery Archive):** [batteryarchive.org](https://batteryarchive.org)
- Related publication: *"Reversible and Irreversible Expansion of Lithium-Ion Batteries Under a Wide Range of Stress Factors"*, University of Michigan Battery Control Group
- Metadata: provided as a CSV (`UofM.csv`) with columns: `Cell_ID`, `Temperature_Type`, `DoD_Min_pct`, `DoD_Max_pct`, `Charge_Crate`, `Discharge_Crate`, `Discharge_Profile`

After downloading, place files in the following structure:
```
UofM/
├── UofM.csv
└── data/
    ├── 01/
    │   ├── cycling_wExpansion.csv
    │   ├── OCV_wExpansion.csv
    │   ├── Resistance.csv
    │   └── Crate_wExpansion.csv
    ├── 02/
    └── ... (up to 21/)
```

---

## 🗂️ Repository Structure

```
.
├── data_extraction_v4.py           # Step 1: Extract & unify raw data → CSV
├── soh_trajectory_prediction_v4.py # Step 2: Train & evaluate all models
├── ISU/                            # ISU-ILCC dataset (see download above)
├── UofM/                           # UofM dataset (see download above)
├── dataset_v2.csv                  # Output of Step 1 (generated)
└── results_v2/                     # Output of Step 2 (generated)
    ├── csv/
    │   ├── all_experiment_results.csv
    │   ├── per_cell_metrics.csv
    │   └── best_model_summary.csv
    ├── plots/
    └── models/
```

---

## ⚙️ Installation

```bash
pip install torch lightgbm xgboost scikit-learn pandas numpy matplotlib scipy shap tqdm
```

> **Note:** PyTorch, LightGBM, XGBoost, and SHAP are optional — the pipeline will skip models if their dependencies are not installed.

Python ≥ 3.8 recommended.

---

## 🚀 Usage

### Step 1 — Data Extraction

Extracts raw cycling data from both datasets and outputs a unified feature CSV.

```bash
python data_extraction_v4.py
```

**Output:** `dataset_v2.csv` — one row per EFC (Equivalent Full Cycle) per cell, containing SoH, incremental capacity features, physics-model parameters, expansion/temperature signals, and metadata.

**Key design decisions in the extraction:**
- **EFC normalisation** — ISU-ILCC data is indexed by Equivalent Full Cycles (`EFC = cumulative_Ah / Q_nominal`) to make DoD-diverse protocols comparable
- **SoH definition** — ISU: RPT C/5 capacity / initial C/5 capacity; UofM: C/20 characterisation capacity / nominal 5.0 Ah
- **IC features** — dQ/dV curves computed from RPT data, split into low / mid / high voltage bands (3.3–3.6 V, 3.6–3.9 V, 3.9–4.1 V)
- **Physics baseline** — fits `Q(n) = Q₀ − a√n − b·n` per cell; residuals are also stored as features

---

### Step 2 — SoH Trajectory Prediction

Trains all 8 models across all experiment configurations.

```bash
python soh_trajectory_prediction_v4.py --data_path dataset_v2.csv
```

**Optional flags:**

| Flag | Description | Example |
|---|---|---|
| `--n_early` | Subset of early-window sizes to evaluate | `--n_early 3 5 10` |
| `--models` | Subset of models to run | `--models lgb xgb gb` |
| `--skip_dl` | Skip DL models (use if no GPU) | `--skip_dl` |
| `--skip_pinn` | Skip PINN and HYBRID models | `--skip_pinn` |
| `--output_dir` | Directory for results | `--output_dir results_v2` |
| `--eval_mode` | Evaluation scope: `combined` or `full` | `--eval_mode combined` |

**Example (ML only, quick run):**
```bash
python soh_trajectory_prediction_v4.py \
    --data_path dataset_v2.csv \
    --n_early 5 10 \
    --models lgb xgb gb \
    --skip_dl --skip_pinn \
    --output_dir results_quick
```

---

## 🤖 Models

| Category | Model | Description |
|---|---|---|
| ML | **LightGBM** | Leaf-wise gradient boosting; handles NaN natively |
| ML | **XGBoost** | Histogram-based boosting with strong regularisation |
| ML | **GradientBoosting** | Sklearn Huber-loss boosting baseline |
| DL | **Transformer** | Self-attention encoder over early cycles + MLP decoder |
| DL | **LSTM+Attention** | Bidirectional LSTM with attention-weighted context |
| DL | **TCN** | Dilated causal convolutions for multi-scale temporal patterns |
| Physics | **PINN** | Physics-Informed NN enforcing `Q = Q₀ − a√n − b·n` + monotonicity |
| Physics | **HYBRID** | Dual-stream LSTM/GRU decomposing low-freq trend and high-freq residual |

All ML models use **RandomizedSearchCV** (30 iterations, 3-fold group-aware CV) for hyperparameter tuning. All DL models use **AdamW + ReduceLROnPlateau** with early stopping (patience = 20).

---

## 🧪 Experiment Design

### Early-Cycle Window Sweep
Models are evaluated on observations from the **first n cycles only** before predicting the full remaining trajectory:

```
n_early ∈ {3, 5, 10, 15, 30, 50}
```

### Feature Ablation
```
Feature sets: top-3, top-5, top-10, top-20, all features
```
Feature importance is determined by the best-performing ML model (LightGBM) using SHAP values.

### Data Partitions (5 splits)

| Split | Train | Test |
|---|---|---|
| Combined 80/20 | 80% of ISU + UofM cells | 20% of ISU + UofM cells |
| ISU-only | 80% ISU cells | 20% ISU cells |
| UofM-only | 80% UofM cells | 20% UofM cells |
| Cross-dataset A | All ISU cells | All UofM cells |
| Cross-dataset B | All UofM cells | All ISU cells |

All splits are **group-aware** — every cycle from a given cell is entirely in either train or test (no data leakage).

---

## 📊 Evaluation Metrics

| Metric | Description |
|---|---|
| Global R² | R² across all concatenated test predictions |
| Global MAE | Mean absolute error across all test cycles |
| Global RMSE | Root mean squared error across all test cycles |
| MAPE (%) | Mean absolute percentage error |
| Mean Cell R² | Average per-cell R² (measures trajectory quality) |

---

## 📁 Output Files

After running Step 2, the `results_v2/` directory contains:

```
results_v2/
├── csv/
│   ├── all_experiment_results.csv  # Full results for every model × split × n_early × feature_set
│   ├── per_cell_metrics.csv        # Per-cell R², MAE, RMSE for every experiment
│   └── best_model_summary.csv      # Best configuration summary per model
├── plots/
│   ├── per_split/                  # SoH trajectory plots per split
│   └── ...                         # Feature importance, SHAP, error distribution plots
└── models/
    ├── combined_80_20/             # Saved model checkpoints (.pt, .pkl)
    ├── ISU_only/
    └── ...
```

---

## 🔑 Key Features in `dataset_v2.csv`

| Column | Description |
|---|---|
| `cell_id` | Unique cell identifier |
| `dataset_source` | `ISU-ILCC` or `UofM` |
| `efc` | Equivalent Full Cycles (primary time axis) |
| `SoH` | State of Health (target variable, 0–1) |
| `Q_discharge` | Mean discharge capacity per EFC bin |
| `coulombic_eff` | Coulombic efficiency (Q_discharge / Q_charge) |
| `mean_dQdV_*` | Mean incremental capacity in low/mid/high voltage bands |
| `ic_peak_volt` | Voltage of the dQ/dV peak in the mid band |
| `Q_physics` | Physics model prediction `Q₀ − a√n − b·n` |
| `Q_residual` | Deviation from physics model |
| `physics_a`, `physics_b` | Fitted degradation law coefficients |
| `exp_range`, `exp_irrev` | Reversible and irreversible expansion (UofM only) |
| `R0_EIS`, `R_ct` | EIS resistance features (UofM only) |
| `charge_rate`, `discharge_rate`, `mean_dod` | Operating condition metadata |

---

## 📚 References

- **ISU-ILCC dataset:** Deng, Z. et al. *"Predicting Battery Lifetime Under Varying Usage Conditions from Early Aging Data."* [OpenEnergy Hub](https://openenergyhub.ornl.gov/explore/dataset/isu-ilcc-battery-aging-dataset/)
- **UofM dataset:** Mohtat, P. et al. *"Reversible and Irreversible Expansion of Lithium-Ion Batteries Under a Wide Range of Stress Factors."* [Battery Archive](https://batteryarchive.org) | [Kaggle](https://www.kaggle.com/datasets/sidharthdk/michigan-expansion-battery-dataset)
- **PINN reference:** Wang, Y. et al. *"Physics-informed neural network for lithium-ion battery degradation stable modeling and prognosis."* Nature Communications 15, 4332 (2024).
- **HYBRID reference:** Knowledge-Aware Model with Frequency-Adaptive Learning, arXiv:2510.02839 (2025).

---

## 📄 License

This project is released for research and educational use. The underlying datasets are subject to their respective licenses — please refer to the original dataset pages for terms of use.
