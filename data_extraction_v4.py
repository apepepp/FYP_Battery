"""
data_extraction_v4.py
=====================
Final extraction pipeline: ISU-ILCC + UofM → unified CSV.

Key design (v4 — EFC-based):
─────────────────────────────
ISU-ILCC:
  • X-axis = Equivalent Full Cycles (EFC):
        EFC(n) = cumulative_Ah_discharged(1..n) / Q_nominal
    This normalises across DoD protocols: 25 tiny 4%-DoD cycles = 1 EFC,
    1 full 100%-DoD cycle = 1 EFC.  All 63 conditions become comparable.
  • Downsampled to 1 row per integer EFC (or per 0.5 EFC if cell has
    few EFCs) to keep CSV manageable.
  • SoH = RPT C/5 capacity / initial RPT C/5 capacity, linearly
    interpolated onto the EFC axis.
  • Per-EFC features: mean cycling Q_discharge, coulombic efficiency,
    IC features from RPT (forward-filled), rate_fade (C/2 vs C/5).
  • Metadata from ISU.csv: charge_rate, discharge_rate, mean_dod.

UofM:
  • X-axis = raw cycle number (21 cells × 300-600 cycles = manageable).
    EFC is also computed for cross-dataset comparability.
  • Q_nom = 5.0 Ah (rated).
  • 100%-DoD cells: SoH = Q_discharge / 5.0 Ah.
  • 50%-DoD cells: SoH from periodic C/20 characterisation, interpolated.
  • Metadata from UofM.csv.

Output: all_cells_features_unified.csv

Usage:
    python data_extraction_v4.py
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import interpolate, signal
from scipy.linalg import lstsq as scipy_lstsq

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# ════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════
V_GRID            = np.linspace(3.0, 4.18, 1000)
SOH_THRESH        = 0.80
RANDOM_STATE      = 42
UOFM_NOMINAL_AH   = 5.0     # rated capacity for UofM pouch cells
ISU_NOMINAL_AH     = 0.250   # rated capacity for ISU polymer cells (250 mAh)
EFC_STEP           = 1.0     # downsample ISU to 1 row per this many EFCs

UOFM_BASE  = "UofM/data"
ISU_BASE   = "ISU"
ISU_META_CSV  = "ISU.csv"
UOFM_META_CSV = "UofM.csv"


# ════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ════════════════════════════════════════════════════════════

def fit_physics_model(x, y):
    """Fit Q(x) = Q0 - a*sqrt(x) - b*x.  Returns (Q_pred, Q0, a, b)."""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    mask = np.isfinite(y) & np.isfinite(x) & (x >= 0)
    if mask.sum() < 4:
        return np.full_like(x, np.nan), np.nan, np.nan, np.nan
    xm, ym = x[mask], y[mask]
    if np.std(xm) < 1e-9:
        return np.full_like(x, np.nan), np.nan, np.nan, np.nan
    A = np.column_stack([np.ones(mask.sum()), np.sqrt(xm), xm])
    try:
        c, _, _, _ = scipy_lstsq(A, ym, lapack_driver="gelsy")
        Q0, a, b = c
    except Exception:
        Q0, a, b = ym[0], 0.0, 0.0
    return Q0 - a * np.sqrt(x) - b * x, Q0, a, b


def compute_ic_features(Q_arr, V_arr, v_grid=V_GRID):
    """Incremental capacity features from a Q-V curve."""
    _nan = {k: np.nan for k in [
        "mean_dQdV_low","var_dQdV_low","mean_dQdV_mid","var_dQdV_mid",
        "mean_dQdV_high","var_dQdV_high","ic_peak_volt","ic_peak_height"]}
    try:
        Q_arr = np.asarray(Q_arr, dtype=float)
        V_arr = np.asarray(V_arr, dtype=float)
        ok = np.isfinite(Q_arr) & np.isfinite(V_arr)
        Q_arr, V_arr = Q_arr[ok], V_arr[ok]
        if len(Q_arr) < 20:
            return _nan
        idx = np.argsort(V_arr)
        V_s, Q_s = V_arr[idx], Q_arr[idx]
        _, ui = np.unique(V_s, return_index=True)
        V_s, Q_s = V_s[ui], Q_s[ui]
        if len(V_s) < 10:
            return _nan
        tck = interpolate.splrep(V_s, Q_s, s=1e-4, k=3)
        Q_interp = interpolate.splev(v_grid, tck)
        dQdV = np.gradient(Q_interp, v_grid)
        dQdV = signal.savgol_filter(dQdV, window_length=21, polyorder=3)
        lo  = (v_grid >= 3.3) & (v_grid < 3.6)
        mid = (v_grid >= 3.6) & (v_grid < 3.9)
        hi  = (v_grid >= 3.9) & (v_grid <= 4.1)
        mi = np.where(mid)[0]
        if len(mi) == 0:
            return _nan
        pk = np.argmax(dQdV[mid]) + mi[0]
        return {
            "mean_dQdV_low":  float(np.mean(dQdV[lo])),
            "var_dQdV_low":   float(np.var(dQdV[lo])),
            "mean_dQdV_mid":  float(np.mean(dQdV[mid])),
            "var_dQdV_mid":   float(np.var(dQdV[mid])),
            "mean_dQdV_high": float(np.mean(dQdV[hi])),
            "var_dQdV_high":  float(np.var(dQdV[hi])),
            "ic_peak_volt":   float(v_grid[pk]),
            "ic_peak_height": float(dQdV[pk]),
        }
    except Exception:
        return _nan


def _clean_cap_list(lst):
    out = []
    for v in lst:
        if isinstance(v, list):
            flat = [x for sub in v for x in (sub if isinstance(sub, list) else [sub])]
            out.append(float(flat[0]) if flat else np.nan)
        else:
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                out.append(np.nan)
    return np.array(out)


def _find_col(df, candidates):
    """Find a column by name.  Also matches 'Name [unit]' style headers."""
    for c in candidates:
        if c in df.columns:
            return c
    # Fallback: match base name ignoring bracketed units
    # e.g. candidate "Capacity" matches "Capacity [Ah]"
    for c in candidates:
        for col in df.columns:
            base = col.split("[")[0].strip()
            if base.lower() == c.lower():
                return col
    return None


# ════════════════════════════════════════════════════════════
# METADATA
# ════════════════════════════════════════════════════════════

def load_isu_metadata(csv_path=ISU_META_CSV):
    if not os.path.exists(csv_path):
        print(f"  [META] ISU.csv not found at {csv_path}")
        return {}
    df = pd.read_csv(csv_path)
    meta = {}
    for _, r in df.iterrows():
        meta[r["Cell_ID"]] = {
            "group_num":      int(r["Group_Num"]),
            "charge_rate":    float(r["Charging_Crate"]),
            "discharge_rate": float(r["Discharging_Crate"]),
            "mean_dod":       float(r["Mean_DoD_pct"]) / 100.0,
        }
    print(f"  [META] ISU: {len(meta)} cells loaded")
    return meta


def load_uofm_metadata(csv_path=UOFM_META_CSV):
    if not os.path.exists(csv_path):
        print(f"  [META] UofM.csv not found at {csv_path}")
        return {}
    df = pd.read_csv(csv_path)
    meta = {}
    for _, r in df.iterrows():
        dch = r.get("Discharge_Crate", np.nan)
        try:
            dch = float(dch)
        except (TypeError, ValueError):
            dch = np.nan
        meta[r["Cell_ID"]] = {
            "temperature_type":  str(r.get("Temperature_Type", "")),
            "dod_min":           float(r.get("DoD_Min_pct", 0)) / 100.0,
            "dod_max":           float(r.get("DoD_Max_pct", 100)) / 100.0,
            "charge_rate":       float(r.get("Charge_Crate", np.nan)),
            "discharge_rate":    dch,
            "discharge_profile": str(r.get("Discharge_Profile", "Constant")),
        }
    for cid in meta:
        meta[cid]["mean_dod"] = meta[cid]["dod_max"] - meta[cid]["dod_min"]
    print(f"  [META] UofM: {len(meta)} cells loaded")
    return meta


# ════════════════════════════════════════════════════════════
# DATASET A — UofM  (per-cycle, manageable size)
# ════════════════════════════════════════════════════════════

_CYC_COLS = ["Time","Current","Voltage","Expansion",
             "Temperature","Q","Capacity","cycle_number"]

def _read_cycling_csv(path):
    raw = pd.read_csv(path, header=0)
    if raw.shape[1] == len(_CYC_COLS):
        raw.columns = _CYC_COLS
    else:
        rename = {raw.columns[i]: _CYC_COLS[i]
                  for i in range(min(len(raw.columns), len(_CYC_COLS)))}
        raw = raw.rename(columns=rename)
    raw["cycle_number"] = pd.to_numeric(raw["cycle_number"],
                                        errors="coerce").fillna(0).astype(int)
    return raw

def _read_csv_flex(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path, header=0)
    df.columns = [c.strip() for c in df.columns]
    for c in ["cycle_number","Cycle_Number","Cycle","cycle",
              "Cycle number","Cycle Number","cycle number"]:
        if c in df.columns:
            df = df.rename(columns={c: "cycle_number"})
            break
    if "cycle_number" in df.columns:
        df["cycle_number"] = pd.to_numeric(df["cycle_number"],
                                           errors="coerce").fillna(0).astype(int)
    return df


def _extract_c20_soh_map(ocv_path, q_nom):
    """C/20 characterisation → {cycle_number: SoH}."""
    result = {}
    if not os.path.exists(ocv_path):
        return result
    try:
        ocv = _read_csv_flex(ocv_path)
        cap_col = _find_col(ocv, ["Capacity","capacity"])
        if cap_col is None or "cycle_number" not in ocv.columns:
            return result
        for cn, grp in ocv.groupby("cycle_number"):
            q = float(np.abs(grp[cap_col]).max())
            if q > 0 and q_nom > 0:
                result[int(cn)] = q / q_nom
    except Exception:
        pass
    return result


def _extract_ocv_ic(ocv_path):
    result = {}
    if not os.path.exists(ocv_path):
        return result
    try:
        ocv = _read_csv_flex(ocv_path)
        q_col = _find_col(ocv, ["Q","q"])
        v_col = _find_col(ocv, ["Voltage","voltage","V"])
        if q_col is None or v_col is None or "cycle_number" not in ocv.columns:
            return result
        for cn, grp in ocv.groupby("cycle_number"):
            ic = compute_ic_features(grp[q_col].values, grp[v_col].values)
            cap_col = _find_col(grp, ["Capacity","capacity"])
            ic["Q_C20"] = float(np.abs(grp[cap_col]).max()) if cap_col else np.nan
            result[int(cn)] = ic
    except Exception:
        pass
    return result


def _extract_resistance(res_path):
    """Extract R0_EIS, R_ct, Z_mag_mean from EIS and R_DC from HPPC pulses."""
    result = {}
    if not os.path.exists(res_path):
        return result
    try:
        df = _read_csv_flex(res_path)
        if "cycle_number" not in df.columns:
            return result
        re_col   = _find_col(df, ["Re","re"])
        im_col   = _find_col(df, ["-Im","Im","im"])
        freq_col = _find_col(df, ["Frequency","frequency"])
        mag_col  = _find_col(df, ["Magnitude","magnitude"])
        i_col    = _find_col(df, ["Current","current"])
        v_col    = _find_col(df, ["Voltage","voltage"])

        for cn, grp in df.groupby("cycle_number"):
            feats = {}

            # ── EIS features ──────────────────────────────────────
            if freq_col and re_col:
                eis = grp[grp[freq_col] > 0].dropna(subset=[re_col, freq_col])
                if len(eis) >= 3:
                    eis_s = eis.sort_values(freq_col, ascending=False)
                    re_arr = eis_s[re_col].values
                    # R0: Re at the point where |Im| is smallest (closest to real axis)
                    if im_col and im_col in eis_s.columns:
                        im_arr = eis_s[im_col].values
                        min_im = np.argmin(np.abs(im_arr))
                        feats["R0_EIS"] = float(re_arr[min_im])
                    else:
                        # Fallback: Re at highest frequency
                        feats["R0_EIS"] = float(re_arr[0])
                    # R_ct: approximate semicircle diameter = max(Re) - R0
                    r0 = feats["R0_EIS"]
                    re_max = float(np.max(re_arr))
                    feats["R_ct"] = re_max - r0 if re_max > r0 else np.nan

            if mag_col and mag_col in grp.columns:
                eis_mag = grp[grp[freq_col] > 0] if freq_col else grp
                if len(eis_mag) > 0:
                    feats["Z_mag_mean"] = float(eis_mag[mag_col].mean())

            # ── HPPC R_DC from discharge pulses ───────────────────
            if i_col and v_col and i_col in grp.columns:
                non_eis = grp[grp[freq_col] == 0] if freq_col else grp
                i_arr = non_eis[i_col].values
                v_arr = non_eis[v_col].values
                r_dc_list = []
                for j in range(1, len(i_arr)):
                    # Detect transition: rest → discharge pulse
                    if abs(i_arr[j-1]) < 10 and i_arr[j] < -500:
                        dv = abs(v_arr[j-1] - v_arr[j])
                        di = abs(i_arr[j] / 1000.0)  # mA → A
                        if di > 0:
                            r_dc_list.append(dv / di)  # Ohm
                if r_dc_list:
                    feats["R_DC_mean"] = float(np.mean(r_dc_list))
                    feats["R_DC_max"]  = float(np.max(r_dc_list))

            result[int(cn)] = feats
    except Exception:
        pass
    return result


def _extract_crate_features(crate_path, q_nom):
    """Extract rate-capability features from Crate_wExpansion.csv."""
    result = {}
    if not os.path.exists(crate_path):
        return result
    try:
        df = _read_csv_flex(crate_path)
        if "cycle_number" not in df.columns:
            return result
        i_col = _find_col(df, ["Current","current"])
        cap_col = _find_col(df, ["Capacity","capacity"])
        if not i_col or not cap_col:
            return result
        # Rate thresholds for 5Ah cell: C/10=500, C/5=1000, C/2=2500, 1C=5000 mA
        rate_defs = [("Q_Crate_C10", q_nom * 0.1 * 1000),
                     ("Q_Crate_C5",  q_nom * 0.2 * 1000),
                     ("Q_Crate_C2",  q_nom * 0.5 * 1000),
                     ("Q_Crate_1C",  q_nom * 1.0 * 1000)]
        for cn, grp in df.groupby("cycle_number"):
            feats = {}
            ch = grp[grp[i_col] > 10]
            for rname, rate_ma in rate_defs:
                sub = ch[(ch[i_col] > rate_ma * 0.85) & (ch[i_col] < rate_ma * 1.15)]
                if len(sub) > 0:
                    feats[rname] = float(sub[cap_col].max())
            # Rate capability ratio: Q_1C / Q_C10 — lower = worse power fade
            if "Q_Crate_1C" in feats and "Q_Crate_C10" in feats and feats["Q_Crate_C10"] > 0:
                feats["rate_capability"] = feats["Q_Crate_1C"] / feats["Q_Crate_C10"]
            result[int(cn)] = feats
    except Exception:
        pass
    return result


def extract_uofm_features(cell_dir, cell_id, meta_dict):
    cyc_path   = os.path.join(cell_dir, "cycling_wExpansion.csv")
    ocv_path   = os.path.join(cell_dir, "OCV_wExpansion.csv")
    res_path   = os.path.join(cell_dir, "Resistance.csv")
    crate_path = os.path.join(cell_dir, "Crate_wExpansion.csv")

    if not os.path.exists(cyc_path):
        print(f"  [UofM] SKIP {cell_id}: no cycling file")
        return pd.DataFrame()

    cyc = _read_cycling_csv(cyc_path)
    cell_meta = meta_dict.get(cell_id, {})
    dod_max = cell_meta.get("dod_max", 1.0)
    is_partial = (dod_max < 1.0)
    Q_nom = UOFM_NOMINAL_AH

    rows = []
    cum_ah = 0.0
    for cyc_n, grp in cyc.groupby("cycle_number"):
        ch  = grp[grp["Current"] > 0]
        dch = grp[grp["Current"] < 0]
        Q_d = dch["Capacity"].max() if len(dch) > 0 else np.nan
        Q_c = ch["Capacity"].max()  if len(ch)  > 0 else np.nan
        if np.isfinite(Q_d):
            cum_ah += Q_d
        row = {
            "cycle":         cyc_n,
            "efc":           cum_ah / Q_nom if Q_nom > 0 else np.nan,
            "Q_discharge":   Q_d,
            "Q_charge":      Q_c,
            "coulombic_eff": Q_d / Q_c if (Q_c and Q_c > 0) else np.nan,
            "exp_max":       grp["Expansion"].max(),
            "exp_min":       grp["Expansion"].min(),
            "exp_range":     grp["Expansion"].max() - grp["Expansion"].min(),
            "exp_mean":      grp["Expansion"].mean(),
            "exp_irrev":     (grp["Expansion"].iloc[-1] - grp["Expansion"].iloc[0]
                              if len(grp) > 1 else np.nan),
            "temp_mean":     grp["Temperature"].mean(),
            "temp_max":      grp["Temperature"].max(),
            "temp_range":    grp["Temperature"].max() - grp["Temperature"].min(),
        }
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("cycle").reset_index(drop=True)

    # ── SoH ──────────────────────────────────────────────────────
    # Use C/20 characterisation for ALL cells (most accurate ground truth)
    c20_soh = _extract_c20_soh_map(ocv_path, Q_nom)
    if c20_soh and len(c20_soh) >= 2:
        cc = sorted(c20_soh.keys())
        ss = [c20_soh[c] for c in cc]
        df["SoH"] = np.interp(df["cycle"].values,
                              np.array(cc, dtype=float),
                              np.array(ss, dtype=float))
    elif c20_soh and len(c20_soh) == 1:
        df["SoH"] = list(c20_soh.values())[0]
    elif not is_partial:
        # Only fall back to Q_discharge/Q_nom for 100%-DoD cells
        df["SoH"] = df["Q_discharge"] / Q_nom
    else:
        # Last resort for partial DoD: use first discharge as reference
        Q_init = df["Q_discharge"].iloc[:5].median()
        df["SoH"] = df["Q_discharge"] / Q_init if Q_init > 0 else np.nan
        print(f"  [UofM] WARN {cell_id}: no C/20 data, SoH from cycling Q")

    # Clamp SoH to [0, 1.05] — values >1.05 indicate extraction error
    df["SoH"] = df["SoH"].clip(upper=1.05)

    # ── Periodic features ────────────────────────────────────────
    ic_rows    = _extract_ocv_ic(ocv_path)
    res_rows   = _extract_resistance(res_path)
    crate_rows = _extract_crate_features(crate_path, Q_nom)

    # Build union of cycling cycles + characterisation cycles for merge
    all_c = sorted(set(df["cycle"].unique()) |
                   set(ic_rows.keys()) |
                   set(res_rows.keys()) |
                   set(crate_rows.keys()))

    def _periodic(rd):
        if not rd:
            return pd.DataFrame({"cycle": all_c})
        t = pd.DataFrame({c: rd.get(c, {}) for c in all_c}).T
        t.index.name = "cycle"
        return t.reset_index()

    for src_df in [_periodic(ic_rows), _periodic(res_rows), _periodic(crate_rows)]:
        if len(src_df.columns) > 1:
            df = df.merge(src_df, on="cycle", how="left")
            for col in src_df.columns[1:]:
                if col in df.columns:
                    df[col] = df[col].ffill()

    if "Q_C20" in df.columns:
        df = df.rename(columns={"Q_C20": "Q_OCV"})
    else:
        df["Q_OCV"] = np.nan

    # ── Physics baseline ─────────────────────────────────────────
    if is_partial and df["SoH"].notna().sum() >= 4:
        Q_phys, Q0, a, b = fit_physics_model(df["efc"], df["SoH"])
    else:
        Q_phys, Q0, a, b = fit_physics_model(df["efc"], df["SoH"])
    df["Q_physics"]  = Q_phys
    df["Q_residual"] = df["SoH"] - Q_phys
    df["physics_Q0"] = Q0; df["physics_a"] = a; df["physics_b"] = b

    # ── Metadata ─────────────────────────────────────────────────
    df["cell_id"]           = cell_id
    df["dataset_source"]    = "UofM"
    df["has_expansion"]     = True
    df["has_EIS"]           = ("R0_EIS" in df.columns and df["R0_EIS"].notna().any())
    df["time_weeks"]        = np.nan
    df["charge_rate"]       = cell_meta.get("charge_rate", np.nan)
    df["discharge_rate"]    = cell_meta.get("discharge_rate", np.nan)
    df["mean_dod"]          = cell_meta.get("mean_dod", np.nan)
    df["temperature_type"]  = cell_meta.get("temperature_type", "")
    df["discharge_profile"] = cell_meta.get("discharge_profile", "Constant")
    df["group_num"]         = np.nan

    for col in ["Q_C5_rpt","Q_C2_rpt","rate_fade","Q_low_rate","Q_high_rate",
                "R_DC_mean","R_DC_max","R_ct","R0_EIS","Z_mag_mean",
                "Q_Crate_C10","Q_Crate_C5","Q_Crate_C2","Q_Crate_1C",
                "rate_capability"]:
        if col not in df.columns:
            df[col] = np.nan

    s = df["SoH"]
    partial_tag = " (C/20 SoH)" if (c20_soh and len(c20_soh) >= 2) else ""
    eis_tag = f" | EIS={df['R0_EIS'].notna().any()}" if 'R0_EIS' in df.columns else ""
    print(f"  [UofM] {cell_id}: {len(df)} cycles | {df['efc'].max():.0f} EFC"
          f" | SoH [{s.min():.3f}, {s.max():.3f}]{partial_tag}{eis_tag}")
    return df


# ════════════════════════════════════════════════════════════
# DATASET B — ISU-ILCC  (EFC-based, downsampled)
# ════════════════════════════════════════════════════════════

def load_isu_json(cell, subfolder):
    def _load(folder, cell, subfolder):
        path = os.path.join(ISU_BASE, folder, subfolder, f"{cell}.json")
        with open(path, "r") as f:
            return json.loads(json.load(f))
    return _load("RPT_json", cell, subfolder), _load("Cycling_json", cell, subfolder)


def _parse_wk(label):
    try:
        return float(str(label).strip().replace("*","").replace("Week","").strip())
    except Exception:
        return np.nan


def _build_cycle_week(cycling_dict, N):
    """Map each raw cycle → week number."""
    week_arr = np.full(N, np.nan)
    for key in ["time_series_discharge","time_series_charge"]:
        ts = cycling_dict.get(key, None)
        if ts is None or not isinstance(ts, dict):
            continue
        wc = {}
        for wl, entries in ts.items():
            if isinstance(entries, (list, np.ndarray)):
                wc[wl] = len(entries)
            elif isinstance(entries, dict):
                wc[wl] = len(entries.get("start", []))
        sw = sorted(wc.items(), key=lambda kv: _parse_wk(kv[0]))
        cursor = 0
        for wl, cnt in sw:
            wf = _parse_wk(wl)
            end = min(cursor + cnt, N)
            week_arr[cursor:end] = wf
            cursor = end
            if cursor >= N:
                break
        if cursor < N and cursor > 0:
            week_arr[cursor:] = week_arr[cursor - 1]
        break
    return week_arr


def extract_isu_features(cell, subfolder, meta_dict):
    """
    ISU extraction with EFC-based downsampling.

    1. Load all raw cycles → compute cumulative Ah → EFC
    2. Load RPT C/5 capacity → compute SoH at each RPT checkpoint
    3. Map RPT checkpoints to the EFC axis
    4. Downsample to 1 row per EFC_STEP (default 1 EFC)
    5. Interpolate SoH + RPT features onto downsampled EFC grid
    """
    try:
        RPT_dict, cycling_dict = load_isu_json(cell, subfolder)
    except FileNotFoundError:
        print(f"  [ISU] SKIP {cell}: JSON not found")
        return pd.DataFrame()
    except Exception as e:
        print(f"  [ISU] SKIP {cell}: {e}")
        return pd.DataFrame()

    # ── Raw cycling data ─────────────────────────────────────────
    cap_dch_raw = _clean_cap_list(cycling_dict.get("capacity_discharge", []))
    cap_chg_raw = _clean_cap_list(cycling_dict.get("capacity_charge", []))
    N_raw = len(cap_dch_raw)
    if N_raw == 0:
        print(f"  [ISU] SKIP {cell}: no cycling data")
        return pd.DataFrame()

    cell_meta = meta_dict.get(cell, {})

    # ── Q_nom from initial RPT C/5 ──────────────────────────────
    cap_c5_rpt = _clean_cap_list(RPT_dict.get("capacity_discharge_C_5", []))
    cap_c2_rpt = _clean_cap_list(RPT_dict.get("capacity_discharge_C_2", []))
    Q_nom = cap_c5_rpt[0] if len(cap_c5_rpt) > 0 and np.isfinite(cap_c5_rpt[0]) else np.nan
    if np.isnan(Q_nom) or Q_nom <= 0:
        valid = cap_c5_rpt[np.isfinite(cap_c5_rpt)]
        Q_nom = float(np.nanquantile(valid, 0.95)) if len(valid) > 0 else ISU_NOMINAL_AH

    # ── Compute EFC for every raw cycle ──────────────────────────
    # EFC(n) = cumulative Ah discharged / Q_nominal
    cum_ah = np.nancumsum(np.where(np.isfinite(cap_dch_raw), cap_dch_raw, 0))
    efc_raw = cum_ah / Q_nom

    # ── Week mapping for raw cycles ──────────────────────────────
    week_raw = _build_cycle_week(cycling_dict, N_raw)

    # ── RPT SoH at each checkpoint ──────────────────────────────
    n_rpt = len(cap_c5_rpt)
    rpt_soh = cap_c5_rpt / Q_nom  # SoH at each RPT

    # Map RPT checkpoints to EFC values.
    # RPT at week W → find the last raw cycle of that week → its EFC.
    rpt_weeks = np.array([float(i) for i in range(n_rpt)])  # fallback: 0,1,2,...
    rpt_efc = []
    for rpt_i in range(n_rpt):
        wk = rpt_weeks[rpt_i]
        # Find the last raw cycle at or before this RPT week
        matches = np.where(week_raw <= wk + 0.01)[0]
        if len(matches) > 0:
            rpt_efc.append(efc_raw[matches[-1]])
        elif rpt_i == 0:
            rpt_efc.append(0.0)  # Week 0 RPT → EFC=0
        else:
            # Estimate: linear interpolation from week to EFC
            if efc_raw[-1] > 0 and rpt_weeks[-1] > 0:
                rpt_efc.append(wk / rpt_weeks[-1] * efc_raw[-1])
            else:
                rpt_efc.append(np.nan)
    rpt_efc = np.array(rpt_efc, dtype=float)

    # ── RPT IC features ──────────────────────────────────────────
    Q_C5_qv = RPT_dict.get("QV_discharge_C_5", {}).get("Q", [])
    V_C5_qv = RPT_dict.get("QV_discharge_C_5", {}).get("V", [])
    rpt_ic_list = []
    for i in range(n_rpt):
        if (i < len(Q_C5_qv) and isinstance(Q_C5_qv[i], (list, np.ndarray))
                and len(Q_C5_qv[i]) > 0 and i < len(V_C5_qv)):
            rpt_ic_list.append(compute_ic_features(
                np.array(Q_C5_qv[i], dtype=float),
                np.array(V_C5_qv[i], dtype=float)))
        else:
            rpt_ic_list.append({k: np.nan for k in [
                "mean_dQdV_low","var_dQdV_low","mean_dQdV_mid","var_dQdV_mid",
                "mean_dQdV_high","var_dQdV_high","ic_peak_volt","ic_peak_height"]})

    # ── Downsample to EFC grid ───────────────────────────────────
    efc_max = efc_raw[-1]
    if efc_max < 2:
        # Very few EFCs — use finer grid
        efc_grid = np.arange(0, efc_max + 0.5, 0.5)
    else:
        efc_grid = np.arange(0, efc_max + EFC_STEP, EFC_STEP)

    if len(efc_grid) == 0:
        efc_grid = np.array([0.0])

    # ── Interpolate SoH onto EFC grid ────────────────────────────
    # Use only valid RPT points
    valid = np.isfinite(rpt_efc) & np.isfinite(rpt_soh)
    if valid.sum() >= 2:
        soh_on_grid = np.interp(efc_grid, rpt_efc[valid], rpt_soh[valid])
    elif valid.sum() == 1:
        soh_on_grid = np.full(len(efc_grid), rpt_soh[valid][0])
    else:
        soh_on_grid = np.full(len(efc_grid), np.nan)

    # ── Interpolate RPT C/5, C/2, rate_fade onto EFC grid ────────
    def _interp_rpt_feat(rpt_vals):
        v = np.isfinite(rpt_efc) & np.isfinite(rpt_vals)
        if v.sum() >= 2:
            return np.interp(efc_grid, rpt_efc[v], rpt_vals[v])
        elif v.sum() == 1:
            return np.full(len(efc_grid), rpt_vals[v][0])
        return np.full(len(efc_grid), np.nan)

    c5_on_grid = _interp_rpt_feat(cap_c5_rpt)
    c2_on_grid = _interp_rpt_feat(cap_c2_rpt)
    rf_on_grid = np.where((c5_on_grid > 0) & np.isfinite(c5_on_grid),
                          c2_on_grid / c5_on_grid, np.nan)

    # ── Forward-fill IC features onto EFC grid ───────────────────
    # For each EFC grid point, find the latest RPT checkpoint ≤ that EFC.
    def _get_rpt_idx(efc_val):
        applicable = np.where((rpt_efc <= efc_val + 0.01) & np.isfinite(rpt_efc))[0]
        return int(applicable[-1]) if len(applicable) > 0 else 0

    # ── Aggregate cycling features per EFC bin ───────────────────
    # For each EFC grid interval, compute mean Q_discharge, CE, etc.
    efc_bin_idx = np.digitize(efc_raw, efc_grid) - 1  # which bin each raw cycle falls in
    efc_bin_idx = np.clip(efc_bin_idx, 0, len(efc_grid) - 1)

    bin_Q_dch_mean = np.full(len(efc_grid), np.nan)
    bin_Q_chg_mean = np.full(len(efc_grid), np.nan)
    bin_CE_mean    = np.full(len(efc_grid), np.nan)
    bin_raw_cycles = np.zeros(len(efc_grid), dtype=int)

    for bi in range(len(efc_grid)):
        mask = efc_bin_idx == bi
        if mask.sum() == 0:
            continue
        bin_raw_cycles[bi] = mask.sum()
        d = cap_dch_raw[mask]
        c = cap_chg_raw[mask] if len(cap_chg_raw) == N_raw else np.full(mask.sum(), np.nan)
        bin_Q_dch_mean[bi] = np.nanmean(d)
        bin_Q_chg_mean[bi] = np.nanmean(c)
        valid_ce = np.isfinite(d) & np.isfinite(c) & (c > 0)
        if valid_ce.any():
            bin_CE_mean[bi] = np.nanmean(d[valid_ce] / c[valid_ce])

    # ── Assemble rows ────────────────────────────────────────────
    rows = []
    for gi in range(len(efc_grid)):
        rpt_idx = _get_rpt_idx(efc_grid[gi])
        ic = rpt_ic_list[rpt_idx] if rpt_idx < len(rpt_ic_list) else {
            k: np.nan for k in rpt_ic_list[0]} if rpt_ic_list else {}

        row = {
            "cycle":          int(efc_grid[gi]),  # EFC as the "cycle" axis
            "efc":            float(efc_grid[gi]),
            "time_weeks":     np.nan,  # could be computed but EFC is primary
            "raw_cycles_in_bin": int(bin_raw_cycles[gi]),
            "Q_discharge":    float(bin_Q_dch_mean[gi]),   # mean cycling Q per EFC bin
            "Q_charge":       float(bin_Q_chg_mean[gi]),
            "coulombic_eff":  float(bin_CE_mean[gi]),
            "SoH":            float(soh_on_grid[gi]),
            # No expansion/temperature for ISU
            "exp_max": np.nan, "exp_min": np.nan, "exp_range": np.nan,
            "exp_mean": np.nan, "exp_irrev": np.nan,
            "temp_mean": np.nan, "temp_max": np.nan, "temp_range": np.nan,
            # RPT features (interpolated)
            "Q_C5_rpt":    float(c5_on_grid[gi]),
            "Q_C2_rpt":    float(c2_on_grid[gi]),
            "rate_fade":   float(rf_on_grid[gi]),
            "Q_low_rate":  float(c5_on_grid[gi]),
            "Q_high_rate": float(c2_on_grid[gi]),
            "Q_OCV":       np.nan,
            # Resistance / EIS / Crate (ISU has none)
            "R_DC_mean": np.nan, "R_DC_max": np.nan,
            "R0_EIS": np.nan, "R_ct": np.nan, "Z_mag_mean": np.nan,
            "Q_Crate_C10": np.nan, "Q_Crate_C5": np.nan,
            "Q_Crate_C2": np.nan, "Q_Crate_1C": np.nan,
            "rate_capability": np.nan,
            # Metadata
            "charge_rate":       cell_meta.get("charge_rate", np.nan),
            "discharge_rate":    cell_meta.get("discharge_rate", np.nan),
            "mean_dod":          cell_meta.get("mean_dod", np.nan),
            "group_num":         cell_meta.get("group_num", np.nan),
            "temperature_type":  "",
            "discharge_profile": "Constant",
        }
        row.update(ic)
        rows.append(row)

    df = pd.DataFrame(rows)

    # ── Physics baseline on SoH vs EFC ───────────────────────────
    Q_phys, Q0, a, b = fit_physics_model(df["efc"], df["SoH"])
    df["Q_physics"]  = Q_phys
    df["Q_residual"] = df["SoH"] - Q_phys
    df["physics_Q0"] = Q0; df["physics_a"] = a; df["physics_b"] = b

    # ── Tags ─────────────────────────────────────────────────────
    df["cell_id"]        = cell
    df["dataset_source"] = "ISU-ILCC"
    df["has_expansion"]  = False
    df["has_EIS"]        = False

    s = df["SoH"]
    print(f"  [ISU] {cell}: {N_raw} raw cycles → {len(df)} EFC rows"
          f" | max EFC={efc_max:.0f} | Q_nom={Q_nom:.4f} Ah"
          f" | SoH [{s.min():.3f}, {s.max():.3f}]"
          f" | {n_rpt} RPTs | DoD={cell_meta.get('mean_dod','?')}")
    return df


# ════════════════════════════════════════════════════════════
# UNIFIED SCHEMA
# ════════════════════════════════════════════════════════════
UNIFIED_COLS = [
    "cell_id", "dataset_source", "cycle", "efc", "time_weeks",
    "Q_discharge", "Q_charge", "coulombic_eff", "SoH",
    "exp_max","exp_min","exp_range","exp_mean","exp_irrev",
    "temp_mean","temp_max","temp_range",
    "mean_dQdV_low","var_dQdV_low","mean_dQdV_mid","var_dQdV_mid",
    "mean_dQdV_high","var_dQdV_high","ic_peak_volt","ic_peak_height",
    "Q_OCV","Q_C5_rpt","Q_C2_rpt",
    "Q_low_rate","Q_high_rate","rate_fade",
    "R_DC_mean","R_DC_max","R0_EIS","R_ct","Z_mag_mean",
    "Q_Crate_C10","Q_Crate_C5","Q_Crate_C2","Q_Crate_1C","rate_capability",
    "Q_physics","Q_residual","physics_Q0","physics_a","physics_b",
    "has_expansion","has_EIS",
    "charge_rate","discharge_rate","mean_dod",
    "group_num","temperature_type","discharge_profile",
]

def enforce_schema(df):
    for col in UNIFIED_COLS:
        if col not in df.columns:
            df[col] = np.nan
    return df[UNIFIED_COLS]


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

def build_unified_dataset(
    uofm_cells=None,
    isu_cells=None,
    isu_valid_csv="valid_cells.csv",
    output_csv="all_cells_features_unified.csv",
):
    all_dfs = []
    isu_meta  = load_isu_metadata()
    uofm_meta = load_uofm_metadata()

    if uofm_cells:
        print(f"\n── Extracting UofM cells ({len(uofm_cells)}) ──────────")
        for cid, folder in uofm_cells.items():
            df = extract_uofm_features(folder, cid, uofm_meta)
            if len(df) > 0:
                all_dfs.append(enforce_schema(df))

    BATCH2 = ["G57C1","G57C2","G57C3","G57C4","G58C1",
              "G26C3","G49C1","G49C2","G49C3","G49C4",
              "G50C1","G50C3","G50C4"]
    if isu_cells is None and os.path.exists(isu_valid_csv):
        isu_cells = pd.read_csv(isu_valid_csv).values.flatten().tolist()

    if isu_cells:
        print(f"\n── Extracting ISU-ILCC cells ({len(isu_cells)}) ────────")
        for cell in isu_cells:
            sf = "Release 2.0" if cell in BATCH2 else "Release 1.0"
            df = extract_isu_features(cell, sf, isu_meta)
            if len(df) > 0:
                all_dfs.append(enforce_schema(df))

    if not all_dfs:
        raise RuntimeError("No data extracted.")

    unified = (pd.concat(all_dfs, ignore_index=True)
                 .sort_values(["dataset_source","cell_id","efc"])
                 .reset_index(drop=True))
    unified.to_csv(output_csv, index=False)

    print(f"\n{'='*60}")
    print(f" Saved → {output_csv}")
    print(f" Total rows : {len(unified):,}")
    print(f" Total cells: {unified['cell_id'].nunique()}")
    print(f"\n Per dataset:")
    print(unified.groupby("dataset_source").agg(
        cells=("cell_id","nunique"), rows=("SoH","count"),
        SoH_mean=("SoH","mean"), SoH_min=("SoH","min"),
        SoH_max=("SoH","max"),
        EFC_mean=("efc","mean"), EFC_max=("efc","max"),
    ).to_string())
    print(f"\n Feature coverage (% non-NaN):")
    fc = [c for c in UNIFIED_COLS if c not in
          ["cell_id","dataset_source","cycle","efc","time_weeks",
           "has_expansion","has_EIS","temperature_type",
           "discharge_profile","group_num"]]
    print((unified[fc].notna().mean().sort_values() * 100).to_string())
    return unified


if __name__ == "__main__":
    print("Working directory:", os.getcwd())
    print("UofM base exists:", os.path.exists(UOFM_BASE))
    print("ISU base exists: ", os.path.exists(ISU_BASE))
    print("ISU.csv exists:  ", os.path.exists(ISU_META_CSV))
    print("UofM.csv exists: ", os.path.exists(UOFM_META_CSV))

    UOFM_CELLS = {
        f"Cell_{i:02d}": os.path.join(UOFM_BASE, f"{i:02d}")
        for i in range(1, 22)
    }
    ISU_CELLS = None

    build_unified_dataset(
        uofm_cells    = UOFM_CELLS,
        isu_cells     = ISU_CELLS,
        isu_valid_csv = os.path.join(ISU_BASE, "valid_cells.csv"),
        output_csv    = "dataset_v2.csv",
    )
