#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║  BATTERY SoH TRAJECTORY PREDICTION — COMPREHENSIVE PIPELINE        ║
║  Dataset: ISU-ILCC (251 cells) + UofM (21 cells), NMC chemistry    ║
║  Task: Predict remaining SoH vs cycle from early-cycle features    ║
╠══════════════════════════════════════════════════════════════════════╣
║  8 Models:                                                          ║
║    ML  — LightGBM, XGBoost, Gradient Boosting                      ║
║    DL  — Transformer Encoder, LSTM+Attention, TCN                   ║
║    Phys— PINN (Wang et al. 2024), HYBRID (dual-stream fusion)       ║
║  Experiments:                                                        ║
║    • n_early sweep: 3, 5, 10, 15, 30, 50                           ║
║    • Feature ablation: top-3, top-5, top-10, top-20, all           ║
║    • 5 Data Partitions:                                              ║
║        a. Combined ISU+UofM 80/20 (stratified)                      ║
║        b. ISU-ILCC only 80/20                                        ║
║        c. UofM only 80/20                                            ║
║        d. ISU train → UofM test (cross-dataset transfer)            ║
║        e. UofM train → ISU test (cross-dataset transfer)            ║
╚══════════════════════════════════════════════════════════════════════╝

USAGE:
    pip install torch lightgbm xgboost scikit-learn pandas numpy matplotlib shap tqdm
    python soh_trajectory_prediction.py --data_path dataset_v2.csv

    Optional flags:
      --n_early  3 5 10      (subset of early windows)
      --models   lgb xgb gb  (subset of models)
      --skip_dl              (skip DL if no GPU)
      --skip_pinn            (skip PINN/HYBRID)
      --output_dir results_v2
"""

# ═══════════════════════════════════════════════════════════════════
# SECTION 0: IMPORTS & CONFIGURATION
# ═══════════════════════════════════════════════════════════════════
import os, sys, time, json, pickle, warnings, argparse, random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                             r2_score, make_scorer)
from sklearn.ensemble import GradientBoostingRegressor
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")

# ── Reproducibility ──────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── Try imports that may not be available ─────────────────────────
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("⚠ LightGBM not installed — will skip LGB model")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("⚠ XGBoost not installed — will skip XGB model")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from torch.nn.utils.rnn import pad_sequence
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    DEVICE = None
    print("⚠ PyTorch not installed — will skip DL/PINN/HYBRID models")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("⚠ SHAP not installed — will use permutation importance instead")


# ═══════════════════════════════════════════════════════════════════
# SECTION 1: CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

# ── Column names ──────────────────────────────────────────────────
CELL_COL   = "cell_id"
CYCLE_COL  = "cycle"
EFC_COL    = "efc"
SOH_COL    = "SoH"
SOURCE_COL = "dataset_source"

# ── Features ──────────────────────────────────────────────────────
# SHARED features: available in >90% of BOTH ISU-ILCC and UofM
# These are the ONLY features used for combined modeling.
# Why: ISU has 0% EIS/expansion/Crate data, UofM has 0% RPT rate data.
# Using dataset-specific features would create structural NaN blocks
# that bias tree models toward one dataset.

CANDIDATE_FEATURES = [
    # ── Capacity ──
    "Q_discharge", "Q_charge", "coulombic_eff",
    # ── Incremental capacity (dQ/dV) ──
    "mean_dQdV_low", "var_dQdV_low",
    "mean_dQdV_mid", "var_dQdV_mid",
    "mean_dQdV_high", "var_dQdV_high",
    "ic_peak_volt", "ic_peak_height",
    # ── Physics-model parameters ──
    "Q_physics", "Q_residual",
    "physics_Q0", "physics_a", "physics_b",
    # ── Operating conditions (metadata, constant per cell) ──
    "charge_rate", "discharge_rate", "mean_dod",
    # ── Position (normalized cycle) ──
    "efc",
]

# We add engineered features from early cycles (slope, drop, etc.)
# at runtime — see build_features()

# ── Experiment grid ────────────────────────────────────────────────
N_EARLY_LIST = [3, 5, 10, 15, 30, 50]
FEATURE_TOP_K = [3, 5, 10, 20, None]  # None = all features
FEATURE_TOP_K_LABELS = ["top3", "top5", "top10", "top20", "all"]

# ── Model hyperparameter search spaces ─────────────────────────────
N_SEARCH_ITER = 30       # RandomizedSearchCV iterations
CV_FOLDS      = 3        # inner CV folds for hyperparam tuning
TEST_SIZE     = 0.20     # 80/20 group-aware split

# ── DL training ───────────────────────────────────────────────────
DL_EPOCHS     = 150
DL_BATCH      = 32
DL_PATIENCE   = 20
DL_LR         = 1e-3
DL_HIDDEN     = 64
DL_LAYERS     = 2
DL_DROPOUT    = 0.2

# ── PINN ────────────────────────────────────────────────────────────
PINN_EPOCHS   = 200
PINN_LR       = 1e-3
PINN_ALPHA    = 0.10     # physics loss weight
PINN_BETA     = 0.01     # monotonicity penalty weight
PINN_HIDDEN   = 128

# ── HYBRID ───────────────────────────────────────────────────────────
HYBRID_EPOCHS    = 200
HYBRID_LR        = 1e-3
HYBRID_HIDDEN    = 64
HYBRID_WINDOW_SG = 11     # Savitzky-Golay window (odd, >= 3)

# ── Palette ──────────────────────────────────────────────────────
COLORS = {
    "LightGBM":     "#2196F3",
    "XGBoost":      "#4CAF50",
    "GradBoost":    "#FF9800",
    "Transformer":  "#9C27B0",
    "LSTM_Attn":    "#E91E63",
    "TCN":          "#00BCD4",
    "PINN":         "#795548",
    "HYBRID":        "#607D8B",
    "Actual":       "#212121",
}

def mape(y_true, y_pred):
    """Mean Absolute Percentage Error (%)."""
    mask = np.abs(y_true) > 0.01
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def trajectory_r2(y_true_dict, y_pred_dict):
    """Average per-cell R² across all cells."""
    r2s = []
    for cid in y_true_dict:
        yt = np.array(y_true_dict[cid])
        yp = np.array(y_pred_dict[cid])
        if len(yt) > 1 and np.std(yt) > 1e-6:
            r2s.append(r2_score(yt, yp))
    return np.mean(r2s) if r2s else np.nan


# ═══════════════════════════════════════════════════════════════════
# SECTION 2: DATA LOADING & FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════

def load_and_clean(path: str) -> pd.DataFrame:
    """Load CSV, basic cleaning."""
    df = pd.read_csv(path, low_memory=False)
    # Drop rows with missing SoH or cycle
    df = df.dropna(subset=[SOH_COL, CYCLE_COL])
    # Filter extreme SoH
    df = df[(df[SOH_COL] >= 0.05) & (df[SOH_COL] <= 1.10)].copy()
    # Sort
    df = df.sort_values([CELL_COL, CYCLE_COL]).reset_index(drop=True)
    print(f"Loaded {path}: {df.shape[0]} rows, "
          f"{df[CELL_COL].nunique()} cells, "
          f"SoH [{df[SOH_COL].min():.3f}, {df[SOH_COL].max():.3f}]")
    return df


def build_early_features(df: pd.DataFrame, n_early: int) -> pd.DataFrame:
    """
    Extract features from the first n_early cycles of each cell.
    Returns one row per cell with aggregated early-cycle statistics
    plus per-cell metadata.

    DESIGN DECISION: We engineer slope/drop/trend features because
    the raw early-cycle values are nearly identical across cells
    (SoH ≈ 1.0 ± 0.02). The *rate* of early degradation is much
    more predictive than the *level*.
    """
    feats_list = []
    for cid, grp in df.groupby(CELL_COL):
        grp = grp.sort_values(CYCLE_COL)
        early = grp.head(n_early)
        if len(early) < max(3, n_early):
            continue

        feat = {CELL_COL: cid}
        feat[SOURCE_COL] = grp[SOURCE_COL].iloc[0]

        # ── Per-feature statistics from early window ──────────────
        for col in CANDIDATE_FEATURES:
            vals = early[col].dropna()
            if len(vals) == 0:
                feat[f"{col}_mean"] = np.nan
                feat[f"{col}_std"]  = np.nan
                continue
            feat[f"{col}_mean"] = vals.mean()
            feat[f"{col}_std"]  = vals.std() if len(vals) > 1 else 0.0

        # ── Engineered trend features (most predictive) ───────────
        cyc_arr = early[CYCLE_COL].values.astype(float)
        q_arr   = early["Q_discharge"].dropna().values
        soh_arr = early[SOH_COL].values

        # Capacity slope (linear fit of Q_discharge vs cycle)
        if len(q_arr) >= 2 and len(cyc_arr) >= 2:
            try:
                slope, intercept = np.polyfit(cyc_arr[:len(q_arr)], q_arr, 1)
                feat["Q_discharge_slope"] = slope
                feat["Q_discharge_intercept"] = intercept
            except:
                feat["Q_discharge_slope"] = 0.0
                feat["Q_discharge_intercept"] = q_arr[0] if len(q_arr) > 0 else np.nan
        else:
            feat["Q_discharge_slope"] = 0.0
            feat["Q_discharge_intercept"] = np.nan

        # SoH slope
        if len(soh_arr) >= 2:
            try:
                feat["SoH_slope"] = np.polyfit(cyc_arr[:len(soh_arr)], soh_arr, 1)[0]
            except:
                feat["SoH_slope"] = 0.0
        else:
            feat["SoH_slope"] = 0.0

        # Early drops
        feat["Q_early_drop"] = q_arr[0] - q_arr[-1] if len(q_arr) >= 2 else 0.0
        feat["SoH_early_drop"] = soh_arr[0] - soh_arr[-1] if len(soh_arr) >= 2 else 0.0

        # Coulombic efficiency trend
        ce = early["coulombic_eff"].dropna().values
        if len(ce) >= 2:
            feat["ce_trend"] = ce[-1] - ce[0]
        else:
            feat["ce_trend"] = 0.0

        # IC peak shift
        ic = early["ic_peak_volt"].dropna().values
        if len(ic) >= 2:
            feat["ic_peak_shift"] = ic[-1] - ic[0]
        else:
            feat["ic_peak_shift"] = 0.0

        # Initial SoH (for normalization context)
        feat["initial_SoH"] = soh_arr[0]

        # Max cycle for this cell (useful for context)
        feat["max_cycle"] = grp[CYCLE_COL].max()
        feat["total_cycles"] = len(grp)

        feats_list.append(feat)

    return pd.DataFrame(feats_list)


def build_sequence_data(df: pd.DataFrame, n_early: int,
                        feature_cols: List[str]) -> Dict:
    """
    Build sequence data for DL models.
    Input:  first n_early cycles as sequence → (n_early, n_features)
    Target: full SoH trajectory of the cell → (n_total_cycles,)

    Returns dict of {cell_id: {"X_early": np.array, "y_full": np.array,
                                "cycles_full": np.array, "source": str}}
    """
    data = {}
    for cid, grp in df.groupby(CELL_COL):
        grp = grp.sort_values(CYCLE_COL)
        if len(grp) < n_early:
            continue

        early = grp.head(n_early)
        X_early = early[feature_cols].fillna(0).values  # (n_early, n_feat)
        y_full  = grp[SOH_COL].values                   # (n_total,)
        cycles  = grp[CYCLE_COL].values                  # (n_total,)
        source  = grp[SOURCE_COL].iloc[0]

        data[cid] = {
            "X_early":    X_early,
            "y_full":     y_full,
            "cycles_full": cycles,
            "source":     source,
        }
    return data


# ═══════════════════════════════════════════════════════════════════
# SECTION 3: DATA SPLITTING
# ═══════════════════════════════════════════════════════════════════

def group_train_test_split(cell_ids: np.ndarray, sources: np.ndarray,
                           test_size=TEST_SIZE):
    """
    Group-aware stratified split by cell_id, preserving dataset ratio.

    DESIGN DECISION: We split by cell_id (not by row) so that all
    cycles of a cell are either entirely in train or entirely in test.
    This prevents data leakage from seeing future cycles of a test cell
    during training.

    We also stratify by dataset_source to ensure both ISU and UofM
    cells appear in both train and test sets proportionally.
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size,
                                 random_state=SEED)
    train_idx, test_idx = next(sss.split(cell_ids, sources))
    return cell_ids[train_idx], cell_ids[test_idx]


def loco_cv_splits(cell_ids: np.ndarray):
    """Leave-One-Cell-Out CV generator."""
    for cid in cell_ids:
        train_ids = cell_ids[cell_ids != cid]
        test_ids  = np.array([cid])
        yield train_ids, test_ids


# ═══════════════════════════════════════════════════════════════════
# SECTION 4: ML MODELS (LightGBM, XGBoost, Gradient Boosting)
# ═══════════════════════════════════════════════════════════════════

def get_ml_models() -> Dict:
    """
    Return dict of {name: (model_class, param_distributions)}.

    DESIGN DECISIONS:
    - LightGBM: Fastest tree booster, handles NaN natively, leaf-wise
      growth is ideal for our sparse battery features.
    - XGBoost: Strong regularization (alpha/lambda), histogram-based.
      Often generalizes better than LightGBM on small datasets.
    - GradientBoosting (sklearn): Reliable baseline, deterministic.
      Slower but well-studied. Uses Huber loss for robustness.

    WHY THESE 3:
    From our previous experiments, tree-based models consistently
    outperformed linear models (ElasticNet R²=0.58) and basic NNs
    (MLP R²=0.37). GradientBoosting was our best (R²=0.82).
    LightGBM and XGBoost are the SOTA tree boosters that can
    potentially beat it with proper tuning.
    """
    models = {}

    if HAS_LGB:
        models["LightGBM"] = {
            "class": lgb.LGBMRegressor,
            "fixed": {"random_state": SEED, "verbosity": -1,
                      "n_jobs": -1, "force_col_wise": True},
            "search": {
                "n_estimators":     [100, 200, 300, 500],
                "learning_rate":    [0.01, 0.03, 0.05, 0.1],
                "max_depth":        [3, 4, 5, 6, 8],
                "num_leaves":       [15, 31, 63, 127],
                "min_child_samples":[5, 10, 20, 30],
                "subsample":        [0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
                "reg_alpha":        [0.0, 0.01, 0.1, 1.0],
                "reg_lambda":       [0.0, 0.01, 0.1, 1.0],
            }
        }

    if HAS_XGB:
        models["XGBoost"] = {
            "class": xgb.XGBRegressor,
            "fixed": {"random_state": SEED, "verbosity": 0,
                      "n_jobs": -1, "tree_method": "hist"},
            "search": {
                "n_estimators":     [100, 200, 300, 500],
                "learning_rate":    [0.01, 0.03, 0.05, 0.1],
                "max_depth":        [3, 4, 5, 6, 8],
                "min_child_weight": [1, 3, 5, 10],
                "subsample":        [0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
                "reg_alpha":        [0.0, 0.01, 0.1, 1.0],
                "reg_lambda":       [0.0, 0.01, 0.1, 1.0],
                "gamma":            [0.0, 0.01, 0.1, 0.5],
            }
        }

    models["GradBoost"] = {
        "class": GradientBoostingRegressor,
        "fixed": {"random_state": SEED, "loss": "huber"},
        "search": {
            "n_estimators":     [100, 200, 300, 500],
            "learning_rate":    [0.01, 0.03, 0.05, 0.1],
            "max_depth":        [3, 4, 5, 6, 8],
            "min_samples_leaf": [2, 5, 10, 20],
            "min_samples_split":[2, 5, 10],
            "subsample":        [0.7, 0.8, 0.9, 1.0],
            "max_features":     [0.6, 0.7, 0.8, 0.9, 1.0],
        }
    }

    return models


def train_ml_model(model_name: str, model_cfg: Dict,
                   X_train: np.ndarray, y_train: np.ndarray,
                   groups_train: np.ndarray,
                   feature_names: List[str]) -> Tuple:
    """
    Train one ML model with RandomizedSearchCV.
    Returns (fitted_model, best_params, cv_score).
    """
    base_model = model_cfg["class"](**model_cfg["fixed"])

    # Group-aware CV for inner loop
    gss = GroupShuffleSplit(n_splits=CV_FOLDS, test_size=0.2,
                           random_state=SEED)

    search = RandomizedSearchCV(
        base_model,
        param_distributions=model_cfg["search"],
        n_iter=N_SEARCH_ITER,
        scoring="neg_mean_absolute_error",
        cv=gss.split(X_train, y_train, groups_train),
        random_state=SEED,
        n_jobs=-1,
        verbose=0,
    )
    t0 = time.time()
    search.fit(X_train, y_train)
    train_time = time.time() - t0
    return search.best_estimator_, search.best_params_, search.best_score_, train_time


def predict_trajectory_ml(model, df_cell: pd.DataFrame,
                          feature_names: List[str],
                          early_features: Dict,
                          n_early: int) -> np.ndarray:
    """
    Predict full SoH trajectory for a single cell using ML model.

    DESIGN DECISION: Each cycle gets the SAME early-feature vector
    (aggregated from first n_early cycles) plus a 'cycle_position' value
    that tells the model WHERE in the lifecycle we are predicting.
    This is the "early features + cycle position" paradigm.
    """
    all_cycles = df_cell[CYCLE_COL].values
    n_total = len(all_cycles)

    # Repeat the early-feature vector for each cycle
    feat_vec = np.array([[early_features.get(f, 0.0) for f in feature_names]])
    X_pred = np.tile(feat_vec, (n_total, 1))

    # Override the cycle_position column with actual cycle position
    if "cycle_position" in feature_names:
        efc_idx = feature_names.index("cycle_position")
        X_pred[:, efc_idx] = all_cycles

    return model.predict(X_pred)


# ═══════════════════════════════════════════════════════════════════
# SECTION 5: DL MODELS (Transformer, LSTM+Attention, TCN)
# ═══════════════════════════════════════════════════════════════════

if HAS_TORCH:

    class BatterySeqDataset(Dataset):
        """
        Each sample: (X_early, y_full, cycle_full, mask, cell_id)
        X_early: (n_early, n_feat)   — input sequence
        y_full:  (max_len,)          — padded target SoH trajectory
        """
        def __init__(self, cell_data: Dict, max_len: int):
            self.samples = []
            for cid, d in cell_data.items():
                x = torch.FloatTensor(d["X_early"])
                y = torch.FloatTensor(d["y_full"])
                c = torch.FloatTensor(d["cycles_full"])
                self.samples.append((x, y, c, cid))
            self.max_len = max_len

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, idx):
            x, y, c, cid = self.samples[idx]
            # Pad y and c to max_len
            padded_y = torch.full((self.max_len,), float("nan"))
            padded_c = torch.full((self.max_len,), 0.0)
            L = min(len(y), self.max_len)
            padded_y[:L] = y[:L]
            padded_c[:L] = c[:L]
            mask = ~torch.isnan(padded_y)
            padded_y[~mask] = 0.0  # replace NaN with 0 for loss computation
            return x, padded_y, padded_c, mask, cid


    # ── Positional Encoding ────────────────────────────────────────
    class PositionalEncoding(nn.Module):
        def __init__(self, d_model, max_len=5000):
            super().__init__()
            pe = torch.zeros(max_len, d_model)
            pos = torch.arange(0, max_len).unsqueeze(1).float()
            div = torch.exp(torch.arange(0, d_model, 2).float()
                            * (-np.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x):
            return x + self.pe[:, :x.size(1)]


    # ── Transformer Encoder Model ──────────────────────────────────
    class TransformerSoH(nn.Module):
        """
        Encodes early cycles with a Transformer encoder, then
        decodes full trajectory via MLP with cycle-position input.

        DESIGN DECISION: Transformer excels at capturing feature
        interactions across the early-cycle window. The self-attention
        mechanism learns which early cycles are most informative
        (e.g., first-cycle capacity vs cycle-5 CE drop).
        """
        def __init__(self, n_feat, d_model=DL_HIDDEN, nhead=4,
                     n_layers=DL_LAYERS, dropout=DL_DROPOUT, max_out=3000):
            super().__init__()
            self.input_proj = nn.Linear(n_feat, d_model)
            self.pos_enc = PositionalEncoding(d_model)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
                dropout=dropout, batch_first=True
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
            # Decoder: takes encoded context + cycle position → SoH
            self.decoder = nn.Sequential(
                nn.Linear(d_model + 1, d_model),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, 1),
            )
            self.max_out = max_out

        def forward(self, x_early, cycles_full):
            """
            x_early: (B, n_early, n_feat)
            cycles_full: (B, max_len) — normalized cycle positions
            Returns: (B, max_len) predicted SoH
            """
            h = self.input_proj(x_early)          # (B, n_early, d_model)
            h = self.pos_enc(h)
            h = self.encoder(h)                   # (B, n_early, d_model)
            context = h.mean(dim=1)               # (B, d_model) — global pool

            # Expand context to each output cycle
            B, T = cycles_full.shape
            ctx = context.unsqueeze(1).expand(B, T, -1)  # (B, T, d_model)
            cyc = cycles_full.unsqueeze(2)                # (B, T, 1)
            dec_in = torch.cat([ctx, cyc], dim=2)         # (B, T, d_model+1)
            out = self.decoder(dec_in).squeeze(-1)        # (B, T)
            return out


    # ── LSTM + Attention Model ─────────────────────────────────────
    class LSTMAttentionSoH(nn.Module):
        """
        Bidirectional LSTM encodes early cycles, attention-weighted
        context feeds the trajectory decoder.

        DESIGN DECISION: LSTM captures temporal ordering in early
        degradation (cycle 1→2→3...). Attention highlights which
        early cycles carry the most predictive signal. Bidirectional
        processing lets late early-cycles inform interpretation of
        earlier ones.
        """
        def __init__(self, n_feat, hidden=DL_HIDDEN, n_layers=DL_LAYERS,
                     dropout=DL_DROPOUT):
            super().__init__()
            self.lstm = nn.LSTM(n_feat, hidden, num_layers=n_layers,
                                batch_first=True, bidirectional=True,
                                dropout=dropout if n_layers > 1 else 0)
            self.attn_w = nn.Linear(hidden * 2, 1)
            self.decoder = nn.Sequential(
                nn.Linear(hidden * 2 + 1, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Linear(hidden // 2, 1),
            )

        def forward(self, x_early, cycles_full):
            h, _ = self.lstm(x_early)         # (B, n_early, hidden*2)
            # Attention
            alpha = torch.softmax(self.attn_w(h), dim=1)  # (B, n_early, 1)
            context = (alpha * h).sum(dim=1)               # (B, hidden*2)

            B, T = cycles_full.shape
            ctx = context.unsqueeze(1).expand(B, T, -1)
            cyc = cycles_full.unsqueeze(2)
            dec_in = torch.cat([ctx, cyc], dim=2)
            return self.decoder(dec_in).squeeze(-1)


    # ── Temporal Convolutional Network ─────────────────────────────
    class TCNBlock(nn.Module):
        def __init__(self, in_ch, out_ch, kernel=3, dilation=1, dropout=0.2):
            super().__init__()
            pad = (kernel - 1) * dilation
            self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=pad,
                                  dilation=dilation)
            self.bn   = nn.BatchNorm1d(out_ch)
            self.drop = nn.Dropout(dropout)
            self.relu = nn.ReLU()
            self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        def forward(self, x):
            out = self.conv(x)[:, :, :x.size(2)]  # causal trim
            out = self.relu(self.bn(out))
            out = self.drop(out)
            return out + self.residual(x)


    class TCNSoH(nn.Module):
        """
        Temporal CNN encodes early cycles via dilated causal convolutions.

        DESIGN DECISION: TCN has larger receptive field than LSTM
        for the same depth, and trains faster. Dilated convolutions
        capture multi-scale temporal patterns (short-term noise vs
        medium-term degradation trends in early cycles).
        """
        def __init__(self, n_feat, hidden=DL_HIDDEN, n_blocks=3,
                     dropout=DL_DROPOUT):
            super().__init__()
            layers = [TCNBlock(n_feat, hidden, dilation=1, dropout=dropout)]
            for i in range(1, n_blocks):
                layers.append(TCNBlock(hidden, hidden, dilation=2**i,
                                       dropout=dropout))
            self.tcn = nn.Sequential(*layers)
            self.decoder = nn.Sequential(
                nn.Linear(hidden + 1, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Linear(hidden // 2, 1),
            )

        def forward(self, x_early, cycles_full):
            # x_early: (B, n_early, n_feat) → conv needs (B, n_feat, n_early)
            h = self.tcn(x_early.permute(0, 2, 1))  # (B, hidden, n_early)
            context = h.mean(dim=2)                   # (B, hidden)

            B, T = cycles_full.shape
            ctx = context.unsqueeze(1).expand(B, T, -1)
            cyc = cycles_full.unsqueeze(2)
            dec_in = torch.cat([ctx, cyc], dim=2)
            return self.decoder(dec_in).squeeze(-1)


    # ── PINN Model ──────────────────────────────────────────────────
    class PINNSoH(nn.Module):
        """
        Physics-Informed Neural Network.
        Based on: Wang et al. "Physics-informed neural network for
        lithium-ion battery degradation stable modeling and prognosis."
        Nature Communications 15, 4332 (2024).

        DESIGN DECISION: The physics loss enforces the empirical
        degradation law Q(n) = Q0 - a*sqrt(n) - b*n, penalizing
        predictions that violate known electrochemistry (SEI growth
        ∝ sqrt(n), lithium plating ∝ n). The monotonicity penalty
        ensures SoH never increases during aging — a hard physical
        constraint that data-driven models often violate.
        """
        def __init__(self, n_feat, hidden=PINN_HIDDEN, dropout=0.2):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(n_feat, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(hidden + 1, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Linear(hidden // 2, 1),
            )
            # Learnable physics parameters (per-batch, shared)
            self.physics_head = nn.Sequential(
                nn.Linear(hidden, 3),  # outputs: Q0, a, b
            )

        def forward(self, x_early, cycles_full):
            # x_early: (B, n_early, n_feat)
            h = self.encoder(x_early.mean(dim=1))  # (B, hidden)
            physics_params = self.physics_head(h)    # (B, 3)

            B, T = cycles_full.shape
            ctx = h.unsqueeze(1).expand(B, T, -1)
            cyc = cycles_full.unsqueeze(2)
            dec_in = torch.cat([ctx, cyc], dim=2)
            y_pred = self.decoder(dec_in).squeeze(-1)

            return y_pred, physics_params

        def physics_loss(self, y_pred, cycles_full, physics_params, mask):
            """
            Enforce Q(n) = Q0 - a*sqrt(n) - b*n.
            """
            Q0 = physics_params[:, 0:1]  # (B, 1)
            a  = torch.abs(physics_params[:, 1:2])
            b  = torch.abs(physics_params[:, 2:3])
            n  = cycles_full.clamp(min=0)
            Q_physics = Q0 - a * torch.sqrt(n + 1e-6) - b * n  # (B, T)
            physics_err = ((y_pred - Q_physics) ** 2 * mask.float()).sum() / (mask.sum() + 1e-8)
            return physics_err

        def monotonicity_loss(self, y_pred, mask):
            """Penalize SoH increases between consecutive cycles."""
            diff = y_pred[:, 1:] - y_pred[:, :-1]  # should be ≤ 0
            violations = torch.relu(diff)  # positive means increase
            m = mask[:, 1:] & mask[:, :-1]
            return (violations ** 2 * m.float()).sum() / (m.sum() + 1e-8)


    # ── HYBRID Model ────────────────────────────────────────────────
    class HYBRIDDualStream(nn.Module):
        """
        Knowledge-Aware Model with Frequency-Adaptive Learning.
        Inspired by: arXiv:2510.02839 (2025).

        DESIGN DECISION: Battery degradation has two frequency
        components: (1) smooth long-term capacity fade (low-freq)
        driven by SEI growth, and (2) noisy short-term fluctuations
        (high-freq) from temperature, rate variations, rest periods.
        
        Dual streams:
          - LSTM for low-freq trend (smooth degradation)
          - GRU for high-freq residual (cycle-to-cycle noise)
        Attention fusion combines both with learned weights.
        """
        def __init__(self, n_feat, hidden=HYBRID_HIDDEN, dropout=0.2):
            super().__init__()
            # Low-freq stream (LSTM for long-term dependencies)
            self.lstm_low = nn.LSTM(n_feat, hidden, num_layers=2,
                                    batch_first=True, dropout=dropout)
            # High-freq stream (GRU for fast dynamics)
            self.gru_high = nn.GRU(n_feat, hidden, num_layers=2,
                                    batch_first=True, dropout=dropout)
            # Attention fusion
            self.attn_fc = nn.Sequential(
                nn.Linear(hidden * 2, hidden),
                nn.Tanh(),
                nn.Linear(hidden, 2),  # weights for low/high
            )
            # Decoder
            self.decoder = nn.Sequential(
                nn.Linear(hidden + 1, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Linear(hidden // 2, 1),
            )

        def _decompose(self, x_early):
            """
            Decompose input into low-freq + high-freq via moving average.
            x_early: (B, n_early, n_feat)
            """
            # Simple moving average as low-freq component
            kernel = min(3, x_early.size(1))
            if kernel < 2:
                return x_early, torch.zeros_like(x_early)
            # Causal moving average along time axis
            low = x_early.clone()
            for t in range(1, x_early.size(1)):
                start = max(0, t - kernel + 1)
                low[:, t] = x_early[:, start:t+1].mean(dim=1)
            high = x_early - low
            return low, high

        def forward(self, x_early, cycles_full):
            x_low, x_high = self._decompose(x_early)

            h_low, _  = self.lstm_low(x_low)    # (B, n_early, hidden)
            h_high, _ = self.gru_high(x_high)   # (B, n_early, hidden)

            # Last hidden states
            ctx_low  = h_low[:, -1]    # (B, hidden)
            ctx_high = h_high[:, -1]   # (B, hidden)

            # Attention fusion
            combined = torch.cat([ctx_low, ctx_high], dim=1)  # (B, hidden*2)
            weights = torch.softmax(self.attn_fc(combined), dim=1)  # (B, 2)
            context = weights[:, 0:1] * ctx_low + weights[:, 1:2] * ctx_high

            B, T = cycles_full.shape
            ctx = context.unsqueeze(1).expand(B, T, -1)
            cyc = cycles_full.unsqueeze(2)
            dec_in = torch.cat([ctx, cyc], dim=2)
            return self.decoder(dec_in).squeeze(-1)


    # ── DL Training Loop ──────────────────────────────────────────
    def train_dl_model(model, train_dataset, val_dataset,
                       model_name: str, epochs=DL_EPOCHS,
                       lr=DL_LR, patience=DL_PATIENCE,
                       is_pinn=False, alpha=PINN_ALPHA, beta=PINN_BETA):
        """Train a DL model with early stopping on validation loss."""
        model = model.to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=patience // 3,
            min_lr=1e-6
        )

        train_loader = DataLoader(train_dataset, batch_size=DL_BATCH,
                                  shuffle=True)
        val_loader   = DataLoader(val_dataset, batch_size=DL_BATCH,
                                  shuffle=False)

        best_val_loss = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(epochs):
            # ── Train ──
            model.train()
            train_loss_sum = 0.0
            n_train = 0
            for X, y, cyc, mask, _ in train_loader:
                X    = X.to(DEVICE)
                y    = y.to(DEVICE)
                cyc  = cyc.to(DEVICE)
                mask = mask.to(DEVICE)

                optimizer.zero_grad()

                if is_pinn:
                    y_pred, phys_params = model(X, cyc)
                    data_loss = ((y_pred - y) ** 2 * mask.float()).sum() / (mask.sum() + 1e-8)
                    phys_loss = model.physics_loss(y_pred, cyc, phys_params, mask)
                    mono_loss = model.monotonicity_loss(y_pred, mask)
                    loss = data_loss + alpha * phys_loss + beta * mono_loss
                else:
                    y_pred = model(X, cyc)
                    loss = ((y_pred - y) ** 2 * mask.float()).sum() / (mask.sum() + 1e-8)

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss_sum += loss.item() * mask.sum().item()
                n_train += mask.sum().item()

            # ── Validate ──
            model.eval()
            val_loss_sum = 0.0
            n_val = 0
            with torch.no_grad():
                for X, y, cyc, mask, _ in val_loader:
                    X    = X.to(DEVICE)
                    y    = y.to(DEVICE)
                    cyc  = cyc.to(DEVICE)
                    mask = mask.to(DEVICE)

                    if is_pinn:
                        y_pred, _ = model(X, cyc)
                    else:
                        y_pred = model(X, cyc)

                    val_loss = ((y_pred - y) ** 2 * mask.float()).sum()
                    val_loss_sum += val_loss.item()
                    n_val += mask.sum().item()

            avg_val = val_loss_sum / (n_val + 1e-8)
            scheduler.step(avg_val)

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if (epoch + 1) % 20 == 0 or epoch == 0:
                avg_train = train_loss_sum / (n_train + 1e-8)
                print(f"    [{model_name}] Epoch {epoch+1}/{epochs}  "
                      f"train_mse={avg_train:.6f}  val_mse={avg_val:.6f}")

            if no_improve >= patience:
                print(f"    [{model_name}] Early stopping at epoch {epoch+1}")
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        model = model.cpu()
        return model


    def predict_dl(model, dataset, is_pinn=False):
        """Get predictions from a DL model. Returns {cell_id: y_pred}."""
        model = model.to(DEVICE).eval()
        loader = DataLoader(dataset, batch_size=DL_BATCH, shuffle=False)
        preds = {}
        with torch.no_grad():
            for X, y, cyc, mask, cids in loader:
                X   = X.to(DEVICE)
                cyc = cyc.to(DEVICE)

                if is_pinn:
                    y_pred, _ = model(X, cyc)
                else:
                    y_pred = model(X, cyc)

                y_pred = y_pred.cpu().numpy()
                mask_np = mask.numpy()

                for i, cid in enumerate(cids):
                    m = mask_np[i]
                    preds[cid] = y_pred[i][m]
        model = model.cpu()
        return preds


# ═══════════════════════════════════════════════════════════════════
# SECTION 6: FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════

def compute_feature_importance(model, X_train, y_train, feature_names,
                               model_name: str) -> pd.DataFrame:
    """
    Compute feature importance via:
    1. Native tree importance (for tree models)
    2. SHAP values (if available)
    3. Permutation importance (fallback)
    """
    results = []

    # Native importance (tree models)
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
        for i, fn in enumerate(feature_names):
            results.append({"feature": fn, "importance": imp[i],
                            "method": "native_tree"})

    # SHAP
    if HAS_SHAP and model_name in ["LightGBM", "XGBoost", "GradBoost"]:
        try:
            # Use a subsample for speed
            n_shap = min(500, len(X_train))
            idx = np.random.choice(len(X_train), n_shap, replace=False)
            X_shap = X_train[idx]

            if model_name in ["LightGBM", "XGBoost"]:
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_shap)
            shap_imp = np.abs(shap_values).mean(axis=0)
            for i, fn in enumerate(feature_names):
                results.append({"feature": fn, "importance": shap_imp[i],
                                "method": "shap"})
        except Exception as e:
            print(f"    SHAP failed for {model_name}: {e}")

    # Permutation importance (always computed as cross-check)
    from sklearn.inspection import permutation_importance
    try:
        perm = permutation_importance(model, X_train, y_train,
                                      n_repeats=10, random_state=SEED,
                                      n_jobs=-1, scoring="neg_mean_absolute_error")
        for i, fn in enumerate(feature_names):
            results.append({"feature": fn, "importance": perm.importances_mean[i],
                            "method": "permutation"})
    except Exception as e:
        print(f"    Permutation importance failed: {e}")

    return pd.DataFrame(results)


def rank_features(importance_df: pd.DataFrame,
                  method="shap") -> List[str]:
    """Rank features by importance, return ordered list."""
    sub = importance_df[importance_df["method"] == method]
    if len(sub) == 0:
        sub = importance_df[importance_df["method"] == "native_tree"]
    if len(sub) == 0:
        sub = importance_df[importance_df["method"] == "permutation"]
    ranked = sub.sort_values("importance", ascending=False)
    return ranked["feature"].tolist()


# ═══════════════════════════════════════════════════════════════════
# SECTION 7: EVALUATION & METRICS
# ═══════════════════════════════════════════════════════════════════

def evaluate_predictions(y_true_dict: Dict, y_pred_dict: Dict,
                         model_name: str, config_tag: str) -> Dict:
    """
    Compute per-cell and aggregate metrics.
    Returns dict with all metrics.
    """
    all_true, all_pred = [], []
    per_cell = []

    for cid in y_true_dict:
        if cid not in y_pred_dict:
            continue
        yt = np.array(y_true_dict[cid])
        yp = np.array(y_pred_dict[cid])
        n  = min(len(yt), len(yp))
        yt, yp = yt[:n], yp[:n]

        if len(yt) < 2:
            continue

        cell_r2 = r2_score(yt, yp) if np.std(yt) > 1e-6 else np.nan
        cell_mae = mean_absolute_error(yt, yp)
        cell_rmse = np.sqrt(mean_squared_error(yt, yp))
        cell_mape = mape(yt, yp)

        per_cell.append({
            "cell_id": cid, "r2": cell_r2, "mae": cell_mae,
            "rmse": cell_rmse, "mape": cell_mape, "n_cycles": n
        })

        all_true.extend(yt.tolist())
        all_pred.extend(yp.tolist())

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    result = {
        "model":       model_name,
        "config":      config_tag,
        "global_R2":   r2_score(all_true, all_pred) if len(all_true) > 1 else np.nan,
        "global_MAE":  mean_absolute_error(all_true, all_pred),
        "global_RMSE": np.sqrt(mean_squared_error(all_true, all_pred)),
        "global_MAPE": mape(all_true, all_pred),
        "mean_cell_R2":   np.nanmean([c["r2"]   for c in per_cell]),
        "mean_cell_MAE":  np.nanmean([c["mae"]  for c in per_cell]),
        "mean_cell_RMSE": np.nanmean([c["rmse"] for c in per_cell]),
        "mean_cell_MAPE": np.nanmean([c["mape"] for c in per_cell]),
        "n_cells":     len(per_cell),
        "per_cell":    per_cell,
    }
    return result


# ═══════════════════════════════════════════════════════════════════
# SECTION 8: PLOTTING
# ═══════════════════════════════════════════════════════════════════

def plot_trajectory_comparison(predictions: Dict, actuals: Dict,
                               cycles: Dict, example_cells: List[str],
                               title: str, save_path: str):
    """
    Plot actual vs predicted SoH trajectory for example cells.
    All models overlaid on the same plot for each cell.
    """
    n_cells = len(example_cells)
    fig, axes = plt.subplots(1, n_cells, figsize=(6 * n_cells, 5),
                             sharey=True)
    if n_cells == 1:
        axes = [axes]

    for ax, cid in zip(axes, example_cells):
        # Actual
        if cid in actuals:
            cyc = cycles.get(cid, np.arange(len(actuals[cid])))
            ax.plot(cyc, actuals[cid], color=COLORS["Actual"],
                    linewidth=2.5, label="Actual", zorder=10)

        # Each model
        for model_name, pred_dict in predictions.items():
            if cid in pred_dict:
                yp = pred_dict[cid]
                cyc_pred = cycles.get(cid, np.arange(len(yp)))[:len(yp)]
                ax.plot(cyc_pred, yp, color=COLORS.get(model_name, "#999"),
                        linewidth=1.5, alpha=0.8, label=model_name)

        ax.set_xlabel("Cycle (EFC)", fontsize=11)
        ax.set_ylabel("SoH", fontsize=11)
        ax.set_title(cid, fontsize=12, fontweight="bold")
        ax.axhline(0.8, color="red", linestyle="--", alpha=0.4,
                    linewidth=1, label="80% EoL")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.4, 1.10)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=min(len(labels), 5), fontsize=9,
               bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_feature_importance_bar(importance_df: pd.DataFrame,
                                model_name: str, save_path: str,
                                top_n=20):
    """Bar chart of feature importance."""
    for method in ["shap", "native_tree", "permutation"]:
        sub = importance_df[(importance_df["method"] == method)]
        if len(sub) == 0:
            continue

        sub = sub.sort_values("importance", ascending=True).tail(top_n)
        fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.3)))
        ax.barh(sub["feature"], sub["importance"],
                color=COLORS.get(model_name, "#2196F3"), alpha=0.8)
        ax.set_xlabel("Importance", fontsize=11)
        ax.set_title(f"{model_name} — Feature Importance ({method})",
                     fontsize=13, fontweight="bold")
        ax.grid(True, axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_path.replace(".png", f"_{method}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_n_early_heatmap(results_df: pd.DataFrame, metric: str,
                         save_path: str):
    """Heatmap: n_early × model, colored by metric."""
    pivot = results_df.pivot_table(index="n_early", columns="model",
                                   values=metric, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                   vmin=pivot.values.min(), vmax=pivot.values.max())
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"n={x}" for x in pivot.index])
    # Annotate values
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=8, color="black" if val > (pivot.values.min() + pivot.values.max())/2 else "white")
    ax.set_title(f"{metric} by n_early × Model", fontsize=13,
                 fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_feature_ablation_heatmap(results_df: pd.DataFrame, metric: str,
                                  save_path: str):
    """Heatmap: feature_set × model, colored by metric."""
    pivot = results_df.pivot_table(index="feature_set", columns="model",
                                   values=metric, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=9)
    ax.set_title(f"{metric} by Feature Set × Model", fontsize=13,
                 fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_fitting_diagnosis(results_list: List[Dict], save_path: str):
    """Scatter plot: actual vs predicted for each model (best config)."""
    n_models = len(results_list)
    cols = min(4, n_models)
    rows = (n_models + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    axes = np.array(axes).flatten() if n_models > 1 else [axes]

    for ax, res in zip(axes, results_list):
        all_y, all_p = [], []
        for cell in res["per_cell"]:
            # We need to access the raw predictions — stored separately
            pass
        ax.set_title(f"{res['model']}\nR²={res['global_R2']:.3f}  "
                     f"RMSE={res['global_RMSE']:.4f}",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Actual SoH")
        ax.set_ylabel("Predicted SoH")

    for ax in axes[len(results_list):]:
        ax.set_visible(False)

    fig.suptitle("Fitting Diagnosis — Actual vs Predicted", fontsize=14,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_actual_vs_predicted_scatter(all_preds: Dict, all_actuals: Dict,
                                     save_path: str):
    """
    One scatter subplot per model: all predicted vs actual SoH points.
    """
    models = list(all_preds.keys())
    n = len(models)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for ax, model_name in zip(axes_flat, models):
        yt_all, yp_all = [], []
        pred_dict = all_preds[model_name]
        for cid in pred_dict:
            if cid in all_actuals:
                yt = np.array(all_actuals[cid])
                yp = np.array(pred_dict[cid])
                n_min = min(len(yt), len(yp))
                yt_all.extend(yt[:n_min].tolist())
                yp_all.extend(yp[:n_min].tolist())

        yt_all = np.array(yt_all)
        yp_all = np.array(yp_all)

        ax.scatter(yt_all, yp_all, s=1, alpha=0.3,
                   color=COLORS.get(model_name, "#2196F3"))
        ax.plot([0.4, 1.1], [0.4, 1.1], "k--", linewidth=1, alpha=0.5)
        r2 = r2_score(yt_all, yp_all) if len(yt_all) > 1 else np.nan
        rmse = np.sqrt(mean_squared_error(yt_all, yp_all)) if len(yt_all) > 1 else np.nan
        ax.set_title(f"{model_name}\nR²={r2:.4f}  RMSE={rmse:.4f}",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Actual SoH")
        ax.set_ylabel("Predicted SoH")
        ax.set_xlim(0.4, 1.1)
        ax.set_ylim(0.4, 1.1)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle("Fitting Diagnosis — Actual vs Predicted SoH", fontsize=14,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ═══════════════════════════════════════════════════════════════════
# SECTION 9: MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="dataset_v2.csv")
    parser.add_argument("--output_dir", default="results_v5")
    parser.add_argument("--n_early", nargs="*", type=int, default=None,
                        help="Override n_early list (e.g. --n_early 10 30)")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Subset of models (e.g. --models lgb xgb)")
    parser.add_argument("--skip_dl", action="store_true",
                        help="Skip DL models")
    parser.add_argument("--skip_pinn", action="store_true",
                        help="Skip PINN/HYBRID models")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: n_early=[10,30], all features only")
    parser.add_argument(
        "--eval_mode", default="combined", choices=["standard", "combined"],
        help=(
            "standard  = evaluate on ALL cycles of test cells.\n"
            "combined  = evaluate ONLY on future cycles (> n_early) of "
            "test cells.  Cross-cell split + temporal extrapolation "
            "simultaneously.  This is the recommended mode for thesis "
            "because it prevents early-cycle SoH \u2248 1.0 from inflating "
            "R\u00b2.  Default: combined."
        ),
    )
    args = parser.parse_args()

    OUT = args.output_dir
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(f"{OUT}/models", exist_ok=True)
    os.makedirs(f"{OUT}/plots", exist_ok=True)
    os.makedirs(f"{OUT}/csv", exist_ok=True)
    os.makedirs(f"{OUT}/pickles", exist_ok=True)
    os.makedirs(f"{OUT}/pickles", exist_ok=True)

    n_early_list = args.n_early if args.n_early else N_EARLY_LIST
    if args.quick:
        n_early_list = [10, 30]

    print("═" * 65)
    print("  BATTERY SoH TRAJECTORY PREDICTION PIPELINE")
    print(f"  Output: {OUT}/")
    print(f"  n_early: {n_early_list}")
    print(f"  eval_mode: {args.eval_mode}")
    print(f"  Device: {DEVICE if HAS_TORCH else 'CPU (no PyTorch)'}")
    if args.eval_mode == "combined":
        print("  \u2192 COMBINED mode: test cells scored on FUTURE cycles only")
        print("    (cycles > n_early).  Early cycles excluded from metrics.")
    print("═" * 65)

    # ── Load data ──────────────────────────────────────────────────
    df = load_and_clean(args.data_path)

    # ── Select example cells for trajectory plots ─────────────────
    # Pick 2 ISU cells (one high-DoD with many cycles, one low-DoD)
    # and 2 UofM cells (one 100% DoD, one 50% DoD)
    isu_cells = df[df[SOURCE_COL] == "ISU-ILCC"].groupby(CELL_COL).agg(
        n_cycles=(CYCLE_COL, "count"),
        min_soh=(SOH_COL, "min"),
        dod=("mean_dod", "first"),
    ).reset_index()
    # Pick a deep-degradation ISU cell and a moderate one
    isu_deep = isu_cells.sort_values("min_soh").iloc[0][CELL_COL]
    isu_mid  = isu_cells[(isu_cells["n_cycles"] > 200) &
                         (isu_cells["min_soh"] < 0.85)].sort_values(
                             "n_cycles", ascending=False).iloc[0][CELL_COL]

    uofm_cells = df[df[SOURCE_COL] == "UofM"].groupby(CELL_COL).agg(
        n_cycles=(CYCLE_COL, "count"),
        min_soh=(SOH_COL, "min"),
        dod=("mean_dod", "first"),
    ).reset_index()
    uofm_full = uofm_cells[uofm_cells["dod"] > 0.9].sort_values(
        "n_cycles", ascending=False).iloc[0][CELL_COL]
    uofm_partial = uofm_cells[uofm_cells["dod"] < 0.9]
    if len(uofm_partial) > 0:
        uofm_half = uofm_partial.sort_values("n_cycles", ascending=False).iloc[0][CELL_COL]
    else:
        uofm_half = uofm_cells.sort_values("n_cycles", ascending=False).iloc[1][CELL_COL]

    EXAMPLE_CELLS = [isu_deep, isu_mid, uofm_full, uofm_half]
    print(f"\nExample cells for plots: {EXAMPLE_CELLS}")

    # Build actual trajectories dict for examples
    actual_trajectories = {}
    actual_cycles = {}
    for cid in df[CELL_COL].unique():
        grp = df[df[CELL_COL] == cid].sort_values(CYCLE_COL)
        actual_trajectories[cid] = grp[SOH_COL].values
        actual_cycles[cid] = grp[CYCLE_COL].values

    # ══════════════════════════════════════════════════════════════
    # PHASE 1: Feature importance (using n_early=30 as reference)
    # ══════════════════════════════════════════════════════════════
    print("\n" + "═" * 65)
    print("  PHASE 1: FEATURE IMPORTANCE ANALYSIS")
    print("═" * 65)

    ref_n_early = 30
    early_df = build_early_features(df, n_early=ref_n_early)
    feature_cols = [c for c in early_df.columns
                    if c not in [CELL_COL, SOURCE_COL, "max_cycle",
                                 "total_cycles", "initial_SoH"]]
    # Remove columns that are all NaN
    feature_cols = [c for c in feature_cols
                    if early_df[c].notna().sum() > 0.5 * len(early_df)]
    print(f"  {len(feature_cols)} valid features after NaN filtering")

    # Build ML training data (target = final SoH at each cycle)
    # For feature importance, we use a flat representation:
    # each row = one (cell, cycle) pair with early features + cycle position
    print("  Building training matrix for feature ranking...")

    # ── Create flat training data ──
    # For each cell: early features (constant) + cycle position → SoH at that cycle
    train_rows = []
    for _, row in early_df.iterrows():
        cid = row[CELL_COL]
        cell_df = df[df[CELL_COL] == cid].sort_values(CYCLE_COL)
        # Sample up to 100 cycles per cell to keep manageable
        if len(cell_df) > 100:
            idx = np.linspace(0, len(cell_df) - 1, 100, dtype=int)
            cell_df = cell_df.iloc[idx]
        for _, c_row in cell_df.iterrows():
            r = {f: row[f] for f in feature_cols}
            r["cycle_position"] = c_row[CYCLE_COL]  # override with actual cycle
            r[SOH_COL] = c_row[SOH_COL]
            r[CELL_COL] = cid
            r[SOURCE_COL] = row[SOURCE_COL]
            train_rows.append(r)

    train_flat = pd.DataFrame(train_rows)

    # Ensure cycle_position is in feature_cols
    if "cycle_position" not in feature_cols:
        feature_cols.append("cycle_position")

    X_all = train_flat[feature_cols].fillna(0).values
    y_all = train_flat[SOH_COL].values
    cells_all = train_flat[CELL_COL].values
    sources_all = train_flat[SOURCE_COL].values

    # Split
    unique_cells = early_df[CELL_COL].values
    unique_sources = early_df[SOURCE_COL].values
    train_cells, test_cells = group_train_test_split(unique_cells,
                                                      unique_sources)
    train_mask = np.isin(cells_all, train_cells)
    test_mask  = np.isin(cells_all, test_cells)

    X_train, y_train = X_all[train_mask], y_all[train_mask]
    X_test,  y_test  = X_all[test_mask],  y_all[test_mask]
    groups_train = cells_all[train_mask]

    print(f"  Train: {len(X_train)} rows ({len(train_cells)} cells)")
    print(f"  Test:  {len(X_test)} rows ({len(test_cells)} cells)")

    # ── Save Phase-1 reference train/test split ───────────────────────────
    _p1_pkl = f"{OUT}/pickles/phase1_reference_data.pkl"
    with open(_p1_pkl, "wb") as _f:
        pickle.dump({
            "X_train":       X_train,
            "y_train":       y_train,
            "X_test":        X_test,
            "y_test":        y_test,
            "groups_train":  groups_train,
            "train_cells":   train_cells,
            "test_cells":    test_cells,
            "feature_cols":  feature_cols,
            "n_early":       ref_n_early,
            "description":   "Phase-1 reference split (n_early=30, all features)",
        }, _f)
    print(f"  Saved: {_p1_pkl}")

    # ── Train reference model for feature ranking ──
    ml_models = get_ml_models()
    ref_model_name = "LightGBM" if HAS_LGB else ("XGBoost" if HAS_XGB else "GradBoost")
    print(f"\n  Training {ref_model_name} for feature ranking...")

    ref_cfg = ml_models[ref_model_name]
    ref_model, ref_params, ref_cv, ref_time = train_ml_model(
        ref_model_name, ref_cfg, X_train, y_train, groups_train, feature_cols
    )
    print(f"  Best CV MAE: {-ref_cv:.4f}  (trained in {ref_time:.1f}s)")
    print(f"  Best params: {ref_params}")

    # Compute importance
    importance_df = compute_feature_importance(
        ref_model, X_train, y_train, feature_cols, ref_model_name
    )
    importance_df.to_csv(f"{OUT}/csv/feature_importance_{ref_model_name}.csv",
                         index=False)
    print(f"  Saved feature importance CSV")

    # Plot feature importance
    plot_feature_importance_bar(importance_df, ref_model_name,
                                f"{OUT}/plots/feature_importance.png")

    # Rank features for ablation
    feature_ranking = rank_features(importance_df)
    print(f"\n  Feature ranking (top 10):")
    for i, fn in enumerate(feature_ranking[:10], 1):
        print(f"    {i:2d}. {fn}")

    # ══════════════════════════════════════════════════════════════
    # PHASE 2: DEFINE 5 DATA PARTITION STRATEGIES
    # ══════════════════════════════════════════════════════════════
    #
    # Why 5 splits?
    # a. Combined 80/20 → baseline: can the model generalise across
    #    both datasets when trained on a mix?
    # b. ISU-only 80/20 → within-dataset: how well does a model
    #    trained & tested on ISU's 63 conditions generalise?
    # c. UofM-only 80/20 → within-dataset: same but for UofM (only
    #    21 cells, so expect higher variance).
    # d. ISU→UofM (cross-dataset transfer): train on ISU, test on
    #    ALL UofM → does knowledge transfer across lab protocols?
    # e. UofM→ISU (cross-dataset transfer): train on UofM, test on
    #    ALL ISU → harder direction (21 train cells → 251 test).
    #
    # Splits d & e are the most important for FYP because they test
    # cross-laboratory generalisation — the holy grail for BMS.

    isu_cell_ids  = df[df[SOURCE_COL]=="ISU-ILCC"][CELL_COL].unique()
    uofm_cell_ids = df[df[SOURCE_COL]=="UofM"][CELL_COL].unique()

    def _split_80_20(cell_ids):
        """Random 80/20 split of cell IDs."""
        n = len(cell_ids)
        idx = np.random.RandomState(SEED).permutation(n)
        k = max(1, int(n * TEST_SIZE))
        return cell_ids[idx[k:]], cell_ids[idx[:k]]

    SPLIT_STRATEGIES = {
        "combined_80_20": {
            "desc": "Combined ISU+UofM, stratified 80/20 by cell",
            "train": train_cells,  # from Phase 1 stratified split
            "test":  test_cells,
        },
        "ISU_only": {
            "desc": "ISU-ILCC only, 80/20 split",
            **dict(zip(["train","test"], _split_80_20(isu_cell_ids))),
        },
        "UofM_only": {
            "desc": "UofM only, 80/20 split",
            **dict(zip(["train","test"], _split_80_20(uofm_cell_ids))),
        },
        "ISU_train_UofM_test": {
            "desc": "Train on ALL ISU, test on ALL UofM (cross-dataset)",
            "train": isu_cell_ids,
            "test":  uofm_cell_ids,
        },
        "UofM_train_ISU_test": {
            "desc": "Train on ALL UofM, test on ALL ISU (cross-dataset)",
            "train": uofm_cell_ids,
            "test":  isu_cell_ids,}
    }

    print("\n" + "═" * 65)
    print("  DATA PARTITION STRATEGIES")
    print("═" * 65)
    for sname, scfg in SPLIT_STRATEGIES.items():
        print(f"  {sname:28s}  train={len(scfg['train']):4d} cells  "
              f"test={len(scfg['test']):4d} cells  | {scfg['desc']}")

    # ══════════════════════════════════════════════════════════════
    # PHASE 3: TRAIN ALL MODELS × ALL SPLITS
    # ══════════════════════════════════════════════════════════════
    print("\n" + "═" * 65)
    print("  PHASE 3: MODEL TRAINING ACROSS ALL SPLITS")
    print("═" * 65)

    all_results = []            # every single experiment row
    best_config_per_model = {}  # global best: model_name → result
    # Per-split best predictions for plotting
    split_predictions = {}      # {split_name: {model_name: {cid: y_pred}}}
    split_actuals     = {}      # {split_name: {cid: y_true}}

    for split_name, split_cfg in SPLIT_STRATEGIES.items():
        sp_train = split_cfg["train"]
        sp_test  = split_cfg["test"]

        print(f"\n{'█' * 65}")
        print(f"  SPLIT: {split_name}")
        print(f"  {split_cfg['desc']}")
        print(f"  Train: {len(sp_train)} cells | Test: {len(sp_test)} cells")
        print(f"{'█' * 65}")

        os.makedirs(f"{OUT}/models/{split_name}", exist_ok=True)

        split_best = {}  # model → best metrics for THIS split
        split_preds_cur = {}   # model → {cid: y_pred}
        split_actuals_cur = {} # cid → y_true

        # ─────────────────────────────────────────────────────────
        # ML MODELS: n_early × feature_set sweep
        # ─────────────────────────────────────────────────────────
        for n_early in n_early_list:
            early_df_n = build_early_features(df, n_early)
            # Only keep cells present in both early_df and split
            eligible = set(early_df_n[CELL_COL].values)
            sp_tr = np.array([c for c in sp_train if c in eligible])
            sp_te = np.array([c for c in sp_test  if c in eligible])
            if len(sp_tr) < 3 or len(sp_te) < 1:
                print(f"  [n_early={n_early}] Too few cells after filtering — skip")
                continue

            feat_cols_n = [c for c in early_df_n.columns
                           if c not in [CELL_COL, SOURCE_COL, "max_cycle",
                                        "total_cycles", "initial_SoH"]]
            feat_cols_n = [c for c in feat_cols_n
                           if early_df_n[c].notna().sum() > 0.5 * len(early_df_n)]
            if "cycle_position" not in feat_cols_n:
                feat_cols_n.append("cycle_position")

            # Build flat training matrix
            rows_n = []
            for _, row in early_df_n.iterrows():
                cid = row[CELL_COL]
                cell_df = df[df[CELL_COL]==cid].sort_values(CYCLE_COL)
                if len(cell_df) > 100:
                    idx = np.linspace(0, len(cell_df)-1, 100, dtype=int)
                    cell_df = cell_df.iloc[idx]
                for _, c_row in cell_df.iterrows():
                    r = {f: row.get(f, np.nan) for f in feat_cols_n}
                    r["cycle_position"] = c_row[CYCLE_COL]
                    r[SOH_COL] = c_row[SOH_COL]
                    r[CELL_COL] = cid
                    r[SOURCE_COL] = row[SOURCE_COL]
                    rows_n.append(r)
            flat_n = pd.DataFrame(rows_n)
            cells_n = flat_n[CELL_COL].values

            tr_mask = np.isin(cells_n, sp_tr)
            te_mask = np.isin(cells_n, sp_te)

            # ── Save ML flat train/test data for this split × n_early ──────
            _ml_pkl = f"{OUT}/pickles/ml_data_{split_name}_n{n_early}.pkl"
            if not os.path.exists(_ml_pkl):
                with open(_ml_pkl, "wb") as _f:
                    pickle.dump({
                        "X_train":        flat_n.loc[tr_mask, feat_cols_n].fillna(0).values,
                        "y_train":        flat_n.loc[tr_mask, SOH_COL].values,
                        "X_test":         flat_n.loc[te_mask, feat_cols_n].fillna(0).values,
                        "y_test":         flat_n.loc[te_mask, SOH_COL].values,
                        "train_cells":    sp_tr,
                        "test_cells":     sp_te,
                        "feature_cols":   feat_cols_n,
                        "cell_ids_train": cells_n[tr_mask],
                        "cell_ids_test":  cells_n[te_mask],
                        "split":          split_name,
                        "n_early":        n_early,
                    }, _f)
                print(f"  Saved: {_ml_pkl}")

            # Feature ablation
            valid_ranking = [f for f in feature_ranking if f in feat_cols_n]
            if "cycle_position" not in valid_ranking:
                valid_ranking.append("cycle_position")

            feat_set_configs = []
            for k, label in zip(FEATURE_TOP_K, FEATURE_TOP_K_LABELS):
                if args.quick and label != "all":
                    continue
                if k is None:
                    feats = feat_cols_n
                else:
                    feats = valid_ranking[:k]
                    if "cycle_position" not in feats:
                        feats.append("cycle_position")
                feat_set_configs.append((label, feats))

            for feat_label, feats in feat_set_configs:
                X_tr = flat_n.loc[tr_mask, feats].fillna(0).values
                y_tr = flat_n.loc[tr_mask, SOH_COL].values
                g_tr = cells_n[tr_mask]

                if len(X_tr) < 10:
                    continue

                for model_name, model_cfg in ml_models.items():
                    if args.models and model_name.lower() not in \
                       [m.lower() for m in args.models]:
                        continue

                    tag = f"{split_name}_n{n_early}_{feat_label}"
                    print(f"\n  [{split_name}] {model_name} | n={n_early} | "
                          f"{feat_label} ({len(feats)}f)...", end="")

                    try:
                        fitted, best_p, cv_score, train_sec = train_ml_model(
                            model_name, model_cfg, X_tr, y_tr, g_tr, feats)

                        t_pred = time.time()
                        y_true_d, y_pred_d = {}, {}     # for metrics (may be future-only)
                        y_pred_full, y_true_full = {}, {} # for plotting (always all cycles)
                        for cid in sp_te:
                            # NEW:
                            cd = flat_n[flat_n[CELL_COL]==cid]
                            if len(cd)==0: continue

                            # Use full cell dataframe (all cycles, not subsampled)
                            # for plotting so trajectory is not truncated at EFC ~100
                            cell_full = df[df[CELL_COL]==cid].sort_values(CYCLE_COL)
                            early_row = early_df_n[early_df_n[CELL_COL]==cid].iloc[0].to_dict()
                            yp_full = predict_trajectory_ml(fitted, cell_full, feats, early_row, n_early)
                            yt_full = cell_full[SOH_COL].values

                            # Always store full for plots
                            y_pred_full[cid] = yp_full
                            y_true_full[cid] = yt_full

                            # For metrics: use subsampled flat_n predictions (consistent with training)
                            Xc = cd[feats].fillna(0).values
                            yp_all = fitted.predict(Xc)
                            yt_all = cd[SOH_COL].values

                            if args.eval_mode == "combined":
                                # COMBINED: only score future cycles (> n_early)
                                cyc_vals = cd["cycle_position"].values if "cycle_position" in cd.columns \
                                           else cd[CYCLE_COL].values if CYCLE_COL in cd.columns \
                                           else np.arange(len(cd))
                                future = cyc_vals > n_early
                                if future.sum() < 2:
                                    continue
                                y_pred_d[cid] = yp_all[future]
                                y_true_d[cid] = yt_all[future]
                            else:
                                y_pred_d[cid] = yp_all
                                y_true_d[cid] = yt_all
                        pred_sec = time.time() - t_pred

                        metrics = evaluate_predictions(y_true_d, y_pred_d,
                                                       model_name, tag)
                        metrics.update({
                            "split": split_name,
                            "n_early": n_early,
                            "feature_set": feat_label,
                            "n_features": len(feats),
                            "eval_mode": args.eval_mode,
                            "best_params": json.dumps(best_p),
                            "train_time_sec": round(train_sec, 2),
                            "predict_time_sec": round(pred_sec, 4),
                            "n_train_cells": len(sp_tr),
                            "n_test_cells": len(sp_te),
                        })
                        all_results.append(metrics)
                        mode_tag = "[future]" if args.eval_mode=="combined" else ""
                        print(f"  R²={metrics['global_R2']:.4f}  "
                              f"MAE={metrics['global_MAE']:.4f}  "
                              f"t={train_sec:.0f}s {mode_tag}")

                        # Save model
                        mp = f"{OUT}/models/{split_name}/{model_name}_{tag}.pkl"
                        with open(mp, "wb") as f:
                            pickle.dump({"model": fitted, "features": feats,
                                         "params": best_p, "n_early": n_early,
                                         "split": split_name}, f)

                        # Track best per split (store FULL predictions for plots)
                        if model_name not in split_best or \
                           metrics["global_R2"] > split_best[model_name]["global_R2"]:
                            split_best[model_name] = metrics
                            split_preds_cur[model_name] = y_pred_full
                            for cid in y_true_full:
                                split_actuals_cur[cid] = y_true_full[cid]

                        # Track global best
                        gk = model_name
                        if gk not in best_config_per_model or \
                           metrics["global_R2"] > best_config_per_model[gk]["global_R2"]:
                            best_config_per_model[gk] = metrics

                    except Exception as e:
                        print(f"  ERR: {e}")

        # ─────────────────────────────────────────────────────────
        # DL MODELS (Transformer, LSTM+Attn, TCN) + PINN + HYBRID
        # ─────────────────────────────────────────────────────────
        if HAS_TORCH and not args.skip_dl:
            SEQ_FEATURES = [f for f in CANDIDATE_FEATURES
                            if f in df.columns and df[f].notna().mean() > 0.5]
            # Pick best n_early from ML for this split, else 30
            sp_ml_res = [r for r in all_results
                         if r["split"]==split_name and r["model"] in ml_models]
            if sp_ml_res:
                sp_best_ne = max(sp_ml_res, key=lambda x: x["global_R2"])["n_early"]
            else:
                sp_best_ne = 30

            dl_ne_list = sorted(set([ne for ne in n_early_list if ne >= 5] + [sp_best_ne]))

            for n_early in dl_ne_list:
                seq_data = build_sequence_data(df, n_early, SEQ_FEATURES)
                if not seq_data:
                    continue

                # Scale
                all_X = np.concatenate([d["X_early"] for d in seq_data.values()])
                scaler = StandardScaler().fit(all_X)
                gmc = df[CYCLE_COL].max()
                for cid in seq_data:
                    seq_data[cid]["X_early"] = scaler.transform(seq_data[cid]["X_early"])
                    seq_data[cid]["cycles_full"] = seq_data[cid]["cycles_full"] / gmc

                train_sq = {c: seq_data[c] for c in seq_data if c in sp_train}
                test_sq  = {c: seq_data[c] for c in seq_data if c in sp_test}

                # ── Save DL sequence data for this split × n_early ────────────
                _seq_pkl = f"{OUT}/pickles/seq_data_{split_name}_n{n_early}.pkl"
                with open(_seq_pkl, "wb") as _f:
                    pickle.dump({
                        "train_seq":      train_sq,
                        "test_seq":       test_sq,
                        "features":       SEQ_FEATURES,
                        "n_early":        n_early,
                        "split":          split_name,
                        "scaler":         scaler,
                        "max_cycle_norm": gmc,
                    }, _f)
                print(f"  Saved: {_seq_pkl}")
                if len(train_sq) < 3 or len(test_sq) < 1:
                    continue

                ml_ = max(len(d["y_full"]) for d in seq_data.values())
                train_ds = BatterySeqDataset(train_sq, ml_)
                test_ds  = BatterySeqDataset(test_sq, ml_)
                nf = len(SEQ_FEATURES)

                # All DL + PINN + HYBRID
                model_list = [
                    ("Transformer", lambda: TransformerSoH(nf, d_model=DL_HIDDEN,
                        nhead=4, n_layers=DL_LAYERS, dropout=DL_DROPOUT, max_out=ml_), False),
                    ("LSTM_Attn", lambda: LSTMAttentionSoH(nf, hidden=DL_HIDDEN,
                        n_layers=DL_LAYERS, dropout=DL_DROPOUT), False),
                    ("TCN", lambda: TCNSoH(nf, hidden=DL_HIDDEN, n_blocks=3,
                        dropout=DL_DROPOUT), False),
                ]
                if not args.skip_pinn:
                    model_list += [
                        ("PINN", lambda: PINNSoH(nf, hidden=PINN_HIDDEN), True),
                        ("HYBRID", lambda: HYBRIDDualStream(nf, hidden=HYBRID_HIDDEN), False),
                    ]

                for mname, model_fn, is_pinn in model_list:
                    if args.models and mname.lower() not in \
                       [m.lower() for m in args.models]:
                        continue

                    tag = f"{split_name}_n{n_early}_all"
                    ep = PINN_EPOCHS if mname=="PINN" else (
                         HYBRID_EPOCHS if mname=="HYBRID" else DL_EPOCHS)
                    lr = PINN_LR if mname=="PINN" else (
                         HYBRID_LR if mname=="HYBRID" else DL_LR)

                    print(f"\n  [{split_name}] {mname} | n={n_early}...", end="")
                    try:
                        mdl = model_fn()
                        t0 = time.time()
                        fitted = train_dl_model(
                            mdl, train_ds, test_ds, mname,
                            epochs=ep, lr=lr, patience=DL_PATIENCE,
                            is_pinn=is_pinn, alpha=PINN_ALPHA, beta=PINN_BETA)
                        tr_s = time.time() - t0

                        t0 = time.time()
                        y_pred_d_raw = predict_dl(fitted, test_ds, is_pinn=is_pinn)
                        pr_s = time.time() - t0
                        y_true_d_raw = {c: seq_data[c]["y_full"] for c in test_sq}

                        # Apply combined eval mask if needed
                        if args.eval_mode == "combined":
                            y_pred_d, y_true_d = {}, {}
                            for c in y_true_d_raw:
                                # Get original (unnormalized) cycle array
                                orig_cyc = seq_data[c]["cycles_full"] * gmc
                                yt = np.array(y_true_d_raw[c])
                                yp_raw = y_pred_d_raw.get(c)
                                if yp_raw is None:
                                    continue
                                yp = np.array(yp_raw)
                                n_min = min(len(yt), len(yp), len(orig_cyc))
                                future = orig_cyc[:n_min] > n_early
                                if future.sum() < 2:
                                    continue
                                y_true_d[c] = yt[:n_min][future]
                                y_pred_d[c] = yp[:n_min][future]
                        else:
                            y_pred_d = y_pred_d_raw
                            y_true_d = y_true_d_raw

                        metrics = evaluate_predictions(y_true_d, y_pred_d,
                                                       mname, tag)
                        metrics.update({
                            "split": split_name,
                            "n_early": n_early,
                            "feature_set": "all",
                            "n_features": nf,
                            "eval_mode": args.eval_mode,
                            "best_params": json.dumps({"hidden": DL_HIDDEN,
                                "lr": lr, "epochs": ep}),
                            "train_time_sec": round(tr_s, 2),
                            "predict_time_sec": round(pr_s, 4),
                            "n_train_cells": len(sp_train),
                            "n_test_cells": len(sp_test),
                        })
                        all_results.append(metrics)
                        mode_tag = "[future]" if args.eval_mode=="combined" else ""
                        print(f"  R²={metrics['global_R2']:.4f}  "
                              f"t={tr_s:.0f}s {mode_tag}")

                        # Save
                        pt_path = f"{OUT}/models/{split_name}/{mname}_{tag}.pt"
                        torch.save({"model_state": fitted.state_dict(),
                                    "features": SEQ_FEATURES,
                                    "n_early": n_early}, pt_path)
                        pkl_path = f"{OUT}/models/{split_name}/{mname}_{tag}.pkl"
                        with open(pkl_path, "wb") as f:
                            pickle.dump({"model_class": mname,
                                         "features": SEQ_FEATURES,
                                         "n_early": n_early,
                                         "split": split_name}, f)

                        # Store FULL predictions for plotting
                        if mname not in split_best or \
                           metrics["global_R2"] > split_best[mname]["global_R2"]:
                            split_best[mname] = metrics
                            split_preds_cur[mname] = y_pred_d_raw
                            for cid in y_true_d_raw:
                                split_actuals_cur[cid] = y_true_d_raw[cid]

                        gk = mname
                        if gk not in best_config_per_model or \
                           metrics["global_R2"] > best_config_per_model[gk]["global_R2"]:
                            best_config_per_model[gk] = metrics

                    except Exception as e:
                        print(f"  ERR: {e}")

        # Store per-split predictions for plotting
        split_predictions[split_name] = split_preds_cur
        split_actuals[split_name] = split_actuals_cur

    # ══════════════════════════════════════════════════════════════
    # PHASE 5: SAVE RESULTS & GENERATE PLOTS
    # ══════════════════════════════════════════════════════════════
    print("\n" + "═" * 65)
    print("  PHASE 5: RESULTS & PLOTS")
    print("═" * 65)

    os.makedirs(f"{OUT}/plots/per_split", exist_ok=True)

    # Short labels for display
    split_labels_short = {
        "combined_80_20":       "Combined 80/20",
        "ISU_only":             "ISU-only",
        "UofM_only":            "UofM-only",
        "ISU_train_UofM_test":  "ISU \u2192 UofM",
        "UofM_train_ISU_test":  "UofM \u2192 ISU",
    }

    # ── Save comprehensive results CSV ──
    results_df = pd.DataFrame([{k: v for k, v in r.items() if k != "per_cell"}
                                for r in all_results])
    results_df.to_csv(f"{OUT}/csv/all_experiment_results.csv", index=False)
    print(f"  Saved: {OUT}/csv/all_experiment_results.csv")

    # ── Pickle: full results list ─────────────────────────────────────
    with open(f"{OUT}/pickles/all_results.pkl", "wb") as _f:
        pickle.dump(all_results, _f)
    print(f"  Saved: {OUT}/pickles/all_results.pkl")

    # ── Pickle: per-split predictions & actuals (numpy arrays) ────────
    with open(f"{OUT}/pickles/split_predictions.pkl", "wb") as _f:
        pickle.dump(split_predictions, _f)
    print(f"  Saved: {OUT}/pickles/split_predictions.pkl")

    with open(f"{OUT}/pickles/split_actuals.pkl", "wb") as _f:
        pickle.dump(split_actuals, _f)
    print(f"  Saved: {OUT}/pickles/split_actuals.pkl")

    # ── Pickle: best config per model ─────────────────────────────────
    with open(f"{OUT}/pickles/best_config_per_model.pkl", "wb") as _f:
        pickle.dump(best_config_per_model, _f)
    print(f"  Saved: {OUT}/pickles/best_config_per_model.pkl")

    # ── Per-cell results ──
    per_cell_rows = []
    for r in all_results:
        for pc in r.get("per_cell", []):
            pc2 = pc.copy()
            pc2["model"]       = r["model"]
            pc2["split"]       = r.get("split", "combined_80_20")
            pc2["n_early"]     = r["n_early"]
            pc2["feature_set"] = r["feature_set"]
            pc2["eval_mode"]   = r.get("eval_mode", args.eval_mode)
            per_cell_rows.append(pc2)
    pd.DataFrame(per_cell_rows).to_csv(f"{OUT}/csv/per_cell_metrics.csv",
                                        index=False)
    print(f"  Saved: {OUT}/csv/per_cell_metrics.csv")

    # ── Summary table (best per model × split) ──
    print("\n" + "─" * 65)
    print("  BEST CONFIGURATION PER MODEL (across all splits)")
    print("─" * 65)
    summary_rows = []
    for mn, metrics in sorted(best_config_per_model.items()):
        row = {
            "Model": mn,
            "Eval Mode": metrics.get("eval_mode", args.eval_mode),
            "Best Split": metrics.get("split", "combined_80_20"),
            "n_early": metrics["n_early"],
            "Features": metrics.get("feature_set", "all"),
            "R²": f"{metrics['global_R2']:.4f}",
            "MAE": f"{metrics['global_MAE']:.4f}",
            "RMSE": f"{metrics['global_RMSE']:.4f}",
            "MAPE (%)": f"{metrics['global_MAPE']:.2f}",
            "Cell R²": f"{metrics['mean_cell_R2']:.4f}",
            "Train (s)": metrics.get("train_time_sec", ""),
            "Predict (s)": metrics.get("predict_time_sec", ""),
        }
        summary_rows.append(row)
        print(f"  {mn:15s}  split={metrics.get('split','?'):25s}  "
              f"n={metrics['n_early']:2d}  R²={metrics['global_R2']:.4f}  "
              f"MAE={metrics['global_MAE']:.4f}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{OUT}/csv/best_model_summary.csv", index=False)

    # ── Per-split summary table ──
    split_summary_rows = []
    for sname in SPLIT_STRATEGIES:
        sp_res = [r for r in all_results if r.get("split") == sname]
        if not sp_res:
            continue
        for r in sp_res:
            split_summary_rows.append({
                "Split": sname,
                "Eval Mode": r.get("eval_mode", args.eval_mode),
                "Model": r["model"],
                "n_early": r["n_early"],
                "Features": r.get("feature_set", "all"),
                "R²": r["global_R2"],
                "MAE": r["global_MAE"],
                "RMSE": r["global_RMSE"],
                "MAPE": r["global_MAPE"],
                "Cell R²": r["mean_cell_R2"],
                "Train (s)": r.get("train_time_sec", ""),
                "Predict (s)": r.get("predict_time_sec", ""),
                "n_train": r.get("n_train_cells", ""),
                "n_test": r.get("n_test_cells", ""),
            })
    pd.DataFrame(split_summary_rows).to_csv(
        f"{OUT}/csv/per_split_all_results.csv", index=False)
    print(f"  Saved: {OUT}/csv/per_split_all_results.csv")

    # ================================================================
    # PLOT 1: PER-SPLIT TRAJECTORY COMPARISON (actual vs predicted)
    # ================================================================
    print("\n  Generating per-split trajectory comparison plots...")
    for sname, preds_dict in split_predictions.items():
        if not preds_dict:
            continue
        actuals_dict = split_actuals.get(sname, {})
        if not actuals_dict:
            continue

        # Pick up to 4 example test cells for this split
        test_cids = list(actuals_dict.keys())
        isu_ex = [c for c in test_cids
                  if df[df[CELL_COL]==c][SOURCE_COL].iloc[0]=="ISU-ILCC"]
        uofm_ex = [c for c in test_cids
                   if df[df[CELL_COL]==c][SOURCE_COL].iloc[0]=="UofM"]
        examples = (isu_ex[:2] + uofm_ex[:2])[:4]
        if not examples:
            examples = test_cids[:4]

        n_ex = len(examples)
        fig, axes = plt.subplots(1, n_ex, figsize=(6*n_ex, 5), sharey=True)
        if n_ex == 1:
            axes = [axes]

        for ax, cid in zip(axes, examples):
            # Actual trajectory
            if cid in actual_trajectories:
                ax.plot(actual_cycles.get(cid, np.arange(len(actual_trajectories[cid]))),
                        actual_trajectories[cid],
                        color=COLORS["Actual"], linewidth=2.5,
                        label="Actual", zorder=10)

            # Each model's prediction
            for mname, pred_d in preds_dict.items():
                if cid in pred_d:
                    yp = pred_d[cid]
                    cyc_p = actual_cycles.get(cid,
                                np.arange(len(yp)))[:len(yp)]
                    ax.plot(cyc_p, yp,
                            color=COLORS.get(mname, "#999"),
                            linewidth=1.5, alpha=0.8, label=mname)

            ax.set_xlabel("Cycle (EFC)", fontsize=11)
            ax.set_ylabel("SoH", fontsize=11)
            src_label = df[df[CELL_COL]==cid][SOURCE_COL].iloc[0]
            ax.set_title(f"{cid} ({src_label})",
                         fontsize=11, fontweight="bold")
            ax.axhline(0.8, color="red", ls="--", alpha=0.4, lw=1,
                        label="80% EoL")
            if args.eval_mode == "combined":
                # Show n_early boundary: predictions to the RIGHT
                # are the "true future" being evaluated
                best_ne_here = max(
                    (r["n_early"] for r in all_results
                     if r.get("split")==sname), default=30)
                ax.axvline(best_ne_here, color="green", ls=":",
                           alpha=0.6, lw=1.5, label=f"n_early={best_ne_here}")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0.4, 1.10)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center",
                   ncol=min(len(labels), 5), fontsize=9,
                   bbox_to_anchor=(0.5, -0.08))
        split_label = SPLIT_STRATEGIES[sname]["desc"]
        fig.suptitle(f"SoH Trajectory — {split_label}",
                     fontsize=13, fontweight="bold", y=1.02)
        fig.tight_layout()
        fig.savefig(f"{OUT}/plots/per_split/trajectory_{sname}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: trajectory_{sname}.png")

    # ================================================================
    # PLOT 2: CROSS-SPLIT R² HEATMAP (split × model)
    # ================================================================
    print("  Generating cross-split R² heatmap...")
    try:
        # Build pivot: best R² per (split, model) — pick best n_early/feat
        heat_rows = []
        for r in all_results:
            heat_rows.append({
                "split": r.get("split", "combined_80_20"),
                "model": r["model"],
                "R2": r["global_R2"],
            })
        heat_df = pd.DataFrame(heat_rows)
        pivot = heat_df.groupby(["split", "model"])["R2"].max().unstack()
        # Order splits logically
        split_order = [s for s in ["combined_80_20", "ISU_only" 
                                    "UofM_only","ISU_train_UofM_test", "UofM_train_ISU_test"]
                       if s in pivot.index]
        pivot = pivot.reindex(split_order)

        fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns)*1.2),
                                        max(4, len(pivot.index)*0.8)))
        im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto",
                       vmin=min(0, np.nanmin(pivot.values)),
                       vmax=1.0)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=10)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([split_labels_short.get(s, s)
                            for s in pivot.index], fontsize=10)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if np.isfinite(val):
                    txt_color = "white" if val < 0.3 else "black"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            fontsize=9, fontweight="bold", color=txt_color)
        ax.set_title("Best R² — Data Partition × Model",
                     fontsize=14, fontweight="bold", pad=12)
        fig.colorbar(im, ax=ax, shrink=0.8, label="R²")
        fig.tight_layout()
        fig.savefig(f"{OUT}/plots/cross_split_R2_heatmap.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: cross_split_R2_heatmap.png")
    except Exception as e:
        print(f"    Cross-split R² heatmap failed: {e}")

    # ================================================================
    # PLOT 3: PER-SPLIT ACTUAL VS PREDICTED SCATTER
    # ================================================================
    print("  Generating per-split actual vs predicted scatter...")
    for sname, preds_dict in split_predictions.items():
        if not preds_dict:
            continue
        actuals_d = split_actuals.get(sname, {})
        if not actuals_d:
            continue

        models = list(preds_dict.keys())
        n = len(models)
        cols = min(4, n)
        rows = max(1, (n + cols - 1) // cols)
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
        axes_flat = np.array(axes).flatten() if n > 1 else [axes]

        for ax, mname in zip(axes_flat, models):
            yt_all, yp_all = [], []
            for cid in preds_dict[mname]:
                if cid in actuals_d:
                    yt = np.array(actuals_d[cid])
                    yp = np.array(preds_dict[mname][cid])
                    nmin = min(len(yt), len(yp))
                    yt_all.extend(yt[:nmin].tolist())
                    yp_all.extend(yp[:nmin].tolist())
            yt_all, yp_all = np.array(yt_all), np.array(yp_all)
            if len(yt_all) < 2:
                ax.set_visible(False)
                continue

            ax.scatter(yt_all, yp_all, s=1, alpha=0.3,
                       color=COLORS.get(mname, "#2196F3"))
            ax.plot([0.4, 1.1], [0.4, 1.1], "k--", lw=1, alpha=0.5)
            r2v = r2_score(yt_all, yp_all)
            rmse_v = np.sqrt(mean_squared_error(yt_all, yp_all))
            ax.set_title(f"{mname}\nR²={r2v:.4f}  RMSE={rmse_v:.4f}",
                         fontsize=10, fontweight="bold")
            ax.set_xlabel("Actual SoH")
            ax.set_ylabel("Predicted SoH")
            ax.set_xlim(0.4, 1.1); ax.set_ylim(0.4, 1.1)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)

        for ax in axes_flat[n:]:
            ax.set_visible(False)

        split_label = SPLIT_STRATEGIES[sname]["desc"]
        fig.suptitle(f"Actual vs Predicted — {split_label}",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(f"{OUT}/plots/per_split/scatter_{sname}.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: scatter_{sname}.png")

    # ================================================================
    # PLOT 4: ERROR DISTRIBUTION BOXPLOTS PER SPLIT
    # ================================================================
    print("  Generating error distribution boxplots...")
    try:
        pc_df = pd.DataFrame(per_cell_rows)
        if len(pc_df) > 0:
            for metric, ylabel in [("r2", "Per-cell R²"),
                                    ("mae", "Per-cell MAE"),
                                    ("rmse", "Per-cell RMSE")]:
                fig, ax = plt.subplots(figsize=(14, 6))
                # Group by split × model
                splits_in_data = [s for s in SPLIT_STRATEGIES
                                  if s in pc_df["split"].unique()]
                positions = []
                labels = []
                all_data = []
                pos = 0
                for sname in splits_in_data:
                    sp_data = pc_df[pc_df["split"]==sname]
                    models_here = sorted(sp_data["model"].unique())
                    for mname in models_here:
                        vals = sp_data[sp_data["model"]==mname][metric].dropna()
                        # Use best config only per (split, model, cell)
                        best_vals = sp_data[sp_data["model"]==mname].groupby(
                            "cell_id")[metric].max().values
                        all_data.append(best_vals)
                        positions.append(pos)
                        short_split = split_labels_short.get(sname, sname)[:10]
                        labels.append(f"{mname}\n{short_split}")
                        pos += 1
                    pos += 0.5  # gap between splits

                bp = ax.boxplot(all_data, positions=positions,
                                widths=0.6, patch_artist=True,
                                showfliers=True, flierprops=dict(ms=2))
                # Color by model
                for i, (patch, lbl) in enumerate(zip(bp["boxes"], labels)):
                    mname = lbl.split("\n")[0]
                    patch.set_facecolor(COLORS.get(mname, "#999"))
                    patch.set_alpha(0.7)

                ax.set_xticks(positions)
                ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
                ax.set_ylabel(ylabel, fontsize=11)
                ax.set_title(f"{ylabel} Distribution by Split × Model",
                             fontsize=13, fontweight="bold")
                ax.grid(True, axis="y", alpha=0.3)
                fig.tight_layout()
                fig.savefig(f"{OUT}/plots/error_boxplot_{metric}.png",
                            dpi=150, bbox_inches="tight")
                plt.close(fig)
            print(f"    Saved: error_boxplot_r2/mae/rmse.png")
    except Exception as e:
        print(f"    Error distribution boxplots failed: {e}")

    # ================================================================
    # PLOT 5: RADAR CHART — MODEL COMPARISON (best split per model)
    # ================================================================
    print("  Generating radar chart...")
    try:
        radar_models = sorted(best_config_per_model.keys())
        categories = ["R²", "1-MAE", "1-RMSE", "1-MAPE/100", "Cell R²"]
        N_cat = len(categories)
        angles = np.linspace(0, 2 * np.pi, N_cat, endpoint=False).tolist()
        angles += angles[:1]  # close the loop

        fig, ax = plt.subplots(figsize=(8, 8),
                               subplot_kw=dict(polar=True))
        for mname in radar_models:
            m = best_config_per_model[mname]
            vals = [
                max(0, m["global_R2"]),
                max(0, 1 - m["global_MAE"]),
                max(0, 1 - m["global_RMSE"]),
                max(0, 1 - m["global_MAPE"] / 100),
                max(0, m["mean_cell_R2"]),
            ]
            vals += vals[:1]
            ax.plot(angles, vals, linewidth=2, alpha=0.8,
                    label=mname, color=COLORS.get(mname, "#999"))
            ax.fill(angles, vals, alpha=0.1,
                    color=COLORS.get(mname, "#999"))

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_title("Model Performance Radar (best config)",
                     fontsize=14, fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1),
                  fontsize=9)
        fig.tight_layout()
        fig.savefig(f"{OUT}/plots/radar_model_comparison.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: radar_model_comparison.png")
    except Exception as e:
        print(f"    Radar chart failed: {e}")

    # ================================================================
    # PLOT 6: TRANSFER LEARNING DEGRADATION ANALYSIS
    # ================================================================
    print("  Generating transfer learning analysis...")
    try:
        # Compare within-dataset vs cross-dataset R² for each model
        tl_rows = []
        for mname in set(r["model"] for r in all_results):
            for sname in SPLIT_STRATEGIES:
                sp_res = [r for r in all_results
                          if r["model"]==mname and r.get("split")==sname]
                if sp_res:
                    best_r2 = max(r["global_R2"] for r in sp_res)
                    tl_rows.append({"model": mname, "split": sname,
                                    "R2": best_r2})

        if tl_rows:
            tl_df = pd.DataFrame(tl_rows)
            models_tl = sorted(tl_df["model"].unique())
            n_m = len(models_tl)

            # Grouped bar chart: split on x-axis, models as groups
            fig, ax = plt.subplots(figsize=(14, 6))
            x = np.arange(len(split_order))
            width = 0.8 / max(1, n_m)

            for i, mname in enumerate(models_tl):
                vals = []
                for sn in split_order:
                    sub = tl_df[(tl_df["model"]==mname) &
                                (tl_df["split"]==sn)]
                    vals.append(sub["R2"].max() if len(sub) > 0 else 0)
                offset = (i - n_m/2 + 0.5) * width
                bars = ax.bar(x + offset, vals, width, alpha=0.8,
                              color=COLORS.get(mname, "#999"),
                              label=mname)
                for bar, val in zip(bars, vals):
                    if val != 0:
                        ax.text(bar.get_x() + bar.get_width()/2,
                                bar.get_height() + 0.01,
                                f"{val:.2f}", ha="center", va="bottom",
                                fontsize=7, rotation=90)

            ax.set_xticks(x)
            ax.set_xticklabels([split_labels_short.get(s, s)
                                for s in split_order],
                               fontsize=10, rotation=15, ha="right")
            ax.set_ylabel("Best R²", fontsize=12)
            ax.set_title("Transfer Learning Analysis — R² by Data Partition",
                         fontsize=14, fontweight="bold")
            ax.legend(fontsize=8, ncol=min(4, n_m),
                      loc="upper right")
            ax.axhline(0, color="black", lw=0.5)
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(f"{OUT}/plots/transfer_learning_analysis.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"    Saved: transfer_learning_analysis.png")
    except Exception as e:
        print(f"    Transfer learning analysis failed: {e}")

    # ================================================================
    # PLOT 7: n_early HEATMAP (per split, combined)
    # ================================================================
    if len(results_df) > 0 and "n_early" in results_df.columns:
        print("  Generating n_early heatmaps...")
        for metric in ["global_R2", "global_MAE", "global_RMSE"]:
            try:
                # Combined across all splits
                plot_n_early_heatmap(results_df, metric,
                                     f"{OUT}/plots/heatmap_{metric}_by_n_early.png")
                # Per split
                for sname in SPLIT_STRATEGIES:
                    sub = results_df[results_df["split"]==sname]
                    if len(sub) > 0:
                        plot_n_early_heatmap(
                            sub, metric,
                            f"{OUT}/plots/per_split/heatmap_{metric}_{sname}.png")
            except Exception as e:
                print(f"    Heatmap {metric} failed: {e}")

    # ================================================================
    # PLOT 8: FEATURE ABLATION HEATMAP
    # ================================================================
    if len(results_df) > 0 and "feature_set" in results_df.columns:
        ablation_df = results_df[results_df["feature_set"].notna()]
        if len(ablation_df) > 0:
            print("  Generating feature ablation heatmaps...")
            for metric in ["global_R2", "global_MAE"]:
                try:
                    plot_feature_ablation_heatmap(
                        ablation_df, metric,
                        f"{OUT}/plots/heatmap_{metric}_by_feature_set.png")
                except Exception as e:
                    print(f"    Feature ablation heatmap failed: {e}")

    # ================================================================
    # PLOT 9: COMPUTATION TIME COMPARISON
    # ================================================================
    if len(results_df) > 0 and "train_time_sec" in results_df.columns:
        print("  Generating computation time comparison...")
        time_rows = []
        for mn, metrics in best_config_per_model.items():
            time_rows.append({
                "Model": mn,
                "Train (s)": metrics.get("train_time_sec", 0),
                "Predict (s)": metrics.get("predict_time_sec", 0),
            })
        time_df = pd.DataFrame(time_rows).sort_values("Train (s)",
                                                       ascending=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        ax = axes[0]
        bars = ax.barh(time_df["Model"], time_df["Train (s)"],
                       color=[COLORS.get(m, "#999")
                              for m in time_df["Model"]],
                       alpha=0.85)
        for bar, val in zip(bars, time_df["Train (s)"]):
            ax.text(bar.get_width() + 0.5,
                    bar.get_y() + bar.get_height()/2,
                    f"{val:.1f}s", va="center", fontsize=9)
        ax.set_xlabel("Training Time (seconds)", fontsize=11)
        ax.set_title("Training Time", fontsize=12, fontweight="bold")
        ax.grid(True, axis="x", alpha=0.3)

        ax = axes[1]
        bars = ax.barh(time_df["Model"], time_df["Predict (s)"],
                       color=[COLORS.get(m, "#999")
                              for m in time_df["Model"]],
                       alpha=0.85)
        for bar, val in zip(bars, time_df["Predict (s)"]):
            ax.text(bar.get_width() + 0.001,
                    bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}s", va="center", fontsize=9)
        ax.set_xlabel("Prediction Time (seconds)", fontsize=11)
        ax.set_title("Prediction Time", fontsize=12, fontweight="bold")
        ax.grid(True, axis="x", alpha=0.3)

        fig.suptitle("Computation Time Comparison (best config)",
                     fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(f"{OUT}/plots/computation_time_comparison.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: computation_time_comparison.png")

        time_df.to_csv(f"{OUT}/csv/computation_times.csv", index=False)

    # ================================================================
    # PLOT 10: SoH DISTRIBUTION — PER SPLIT
    # ================================================================
    print("  Generating SoH distribution per split...")
    try:
        n_splits = len(SPLIT_STRATEGIES)
        fig, axes = plt.subplots(1, n_splits, figsize=(5*n_splits, 4),
                                  sharey=True)
        if n_splits == 1:
            axes = [axes]
        for ax, (sname, scfg) in zip(axes, SPLIT_STRATEGIES.items()):
            tr_soh = df[df[CELL_COL].isin(scfg["train"])][SOH_COL].values
            te_soh = df[df[CELL_COL].isin(scfg["test"])][SOH_COL].values
            ax.hist(tr_soh, bins=40, alpha=0.6, label="Train",
                    color="#2196F3")
            ax.hist(te_soh, bins=40, alpha=0.6, label="Test",
                    color="#FF9800")
            ax.set_xlabel("SoH")
            ax.set_title(split_labels_short.get(sname, sname),
                         fontsize=10, fontweight="bold")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        axes[0].set_ylabel("Count")
        fig.suptitle("SoH Distribution: Train vs Test per Split",
                     fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig.savefig(f"{OUT}/plots/soh_distribution_per_split.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: soh_distribution_per_split.png")
    except Exception as e:
        print(f"    SoH distribution plot failed: {e}")

    # ================================================================
    # PLOT 11: DEGRADATION CURVE GALLERY (test cells, best model)
    # ================================================================
    print("  Generating degradation curve gallery...")
    try:
        # For "combined_80_20" split, plot a grid of test cells
        ref_split = "combined_80_20"
        if ref_split in split_predictions and split_predictions[ref_split]:
            preds_ref = split_predictions[ref_split]
            acts_ref  = split_actuals.get(ref_split, {})
            # Find best model for this split
            sp_res = [r for r in all_results
                      if r.get("split")==ref_split]
            if sp_res:
                best_mn = max(sp_res, key=lambda x: x["global_R2"])["model"]
                pred_best = preds_ref.get(best_mn, {})

                gallery_cells = list(acts_ref.keys())[:12]
                n_g = len(gallery_cells)
                cols_g = min(4, n_g)
                rows_g = max(1, (n_g + cols_g - 1) // cols_g)
                fig, axes = plt.subplots(rows_g, cols_g,
                                          figsize=(5*cols_g, 4*rows_g),
                                          sharey=True)
                axes_flat = np.array(axes).flatten()

                for ax, cid in zip(axes_flat, gallery_cells):
                    cyc = actual_cycles.get(cid,
                              np.arange(len(acts_ref[cid])))
                    ax.plot(cyc, acts_ref[cid],
                            color=COLORS["Actual"], lw=2, label="Actual")
                    if cid in pred_best:
                        yp = pred_best[cid]
                        ax.plot(cyc[:len(yp)], yp,
                                color=COLORS.get(best_mn, "#E91E63"),
                                lw=1.5, ls="--", label=best_mn)
                    ax.set_title(cid, fontsize=9)
                    ax.axhline(0.8, color="red", ls=":", alpha=0.3)
                    ax.grid(True, alpha=0.2)
                    ax.set_ylim(0.5, 1.08)

                for ax in axes_flat[n_g:]:
                    ax.set_visible(False)

                handles, labels = axes_flat[0].get_legend_handles_labels()
                fig.legend(handles, labels, loc="lower center",
                           ncol=2, fontsize=9,
                           bbox_to_anchor=(0.5, -0.03))
                fig.suptitle(f"Degradation Gallery — Combined Split "
                             f"({best_mn})",
                             fontsize=13, fontweight="bold")
                fig.tight_layout()
                fig.savefig(f"{OUT}/plots/degradation_gallery.png",
                            dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"    Saved: degradation_gallery.png")
    except Exception as e:
        print(f"    Degradation gallery failed: {e}")

    # ================================================================
    # PLOT 12: PER-DATASET PERFORMANCE COMPARISON (bar chart)
    # ================================================================
    try:
        if len(per_cell_rows) > 0:
            print("  Generating per-dataset performance comparison...")
            pc_df = pd.DataFrame(per_cell_rows)
            cell_source = df.groupby(CELL_COL)[SOURCE_COL].first().to_dict()
            pc_df["source"] = pc_df["cell_id"].map(cell_source)

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            for ax, metric, label in zip(axes, ["r2", "mae"],
                                          ["R²", "MAE"]):
                best_cfgs = pc_df.groupby(
                    ["model", "cell_id"])[metric].max().reset_index()
                best_cfgs["source"] = best_cfgs["cell_id"].map(cell_source)
                for src in ["ISU-ILCC", "UofM"]:
                    sub = best_cfgs[best_cfgs["source"]==src]
                    grouped = sub.groupby("model")[metric].mean()
                    ax.bar([f"{m}\n{src[:3]}" for m in grouped.index],
                           grouped.values, alpha=0.7, label=src)
                ax.set_ylabel(label)
                ax.set_title(f"{label} by Model × Dataset")
                ax.legend()
                ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            fig.savefig(f"{OUT}/plots/performance_by_dataset.png",
                        dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"    Saved: performance_by_dataset.png")
    except Exception as e:
        print(f"    Per-dataset comparison failed: {e}")

    # ══════════════════════════════════════════════════════════════
    # DONE
    # ══════════════════════════════════════════════════════════════
    print("\n" + "═" * 65)
    print("  PIPELINE COMPLETE")
    print(f"  All outputs saved to: {OUT}/")
    print(f"  Models:  {OUT}/models/<split_name>/")
    print(f"  Plots:   {OUT}/plots/  (+ per_split/)")
    print(f"  CSVs:    {OUT}/csv/")
    print("═" * 65)

    # Print final leaderboard
    print("\n  FINAL LEADERBOARD (best config across all splits):")
    if len(summary_rows) > 0:
        lb = pd.DataFrame(summary_rows).sort_values("R²", ascending=False)
        print(lb.to_string(index=False))

    # Per-split leaderboard
    print("\n  PER-SPLIT LEADERBOARD:")
    for sname in SPLIT_STRATEGIES:
        sp_res = [r for r in all_results if r.get("split")==sname]
        if sp_res:
            best = max(sp_res, key=lambda x: x["global_R2"])
            print(f"    {sname:28s}  best={best['model']:15s}  "
                  f"R²={best['global_R2']:.4f}  MAE={best['global_MAE']:.4f}")

    return results_df


if __name__ == "__main__":
    main()
