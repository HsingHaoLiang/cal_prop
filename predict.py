"""
- Support CSV with multiple rows: (smiles, T(K)) for *_real tasks (per-smiles, per-T).
- Support CSV with multiple rows: (smiles, Tr)  for *_reduced tasks (per-smiles, per-Tr).
    * If CSV contains only smiles column, you can still use --temps (for *_real) or --trs (for *_reduced).
other can see help in argument

Tasks:
  - pvap_real      : input (smiles, T[K]) -> predict lnP(Pa) (via Tc reduce -> pvap -> +lnPc)
  - both_real      : pvap_real + also output Tb/Tc/lnPc/w ensemble mean/std
  - pvap_reduced   : input (smiles, Tr)   -> predict lnPr
  - both_reduced   : pvap_reduced + Tb/Tc/lnPc/w mean/std
  - TbTcPcw        : only Tb/Tc/lnPc/w mean/std per smiles

Example: 
  python predict.py \
    --task both_real --rep FP \
    --input-csv smiles.csv --smiles-col smiles --temp-col "T(K)" \
    --i-list 1-10 --j-list 1-10 \
    --out-csv out.csv
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")

def parse_int_list(s: str, name: str):
    s = (s or "").strip()
    if not s:
        raise ValueError(f"--{name} is required, e.g. --{name} 0-4,7,9")
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a = int(a.strip()); b = int(b.strip())
            step = 1 if a <= b else -1
            out.extend(list(range(a, b + step, step)))
        else:
            out.append(int(part))
    if not out:
        raise ValueError(f"--{name} parsed empty list.")
    return sorted(set(out))


def parse_float_list(s: str, name: str):
    s = (s or "").strip()
    if not s:
        raise ValueError(f"--{name} is required.")
    vals = []
    for part in s.split(","):
        part = part.strip()
        if part == "":
            continue
        vals.append(float(part))
    if not vals:
        raise ValueError(f"--{name} parsed empty list.")
    return vals


def _norm_colname(c: str) -> str:
    return str(c).strip().lower()


def _find_col_case_insensitive(df: pd.DataFrame, want: str):
    want_n = _norm_colname(want)
    for c in df.columns:
        if _norm_colname(c) == want_n:
            return c
    return None

def load_stats(stats_json: str) -> dict:
    assert os.path.exists(stats_json), f"missing: {stats_json}"
    with open(stats_json, "r", encoding="utf-8") as f:
        return json.load(f)


def get_stats(STATS: dict, rep: str, i: int, stats_json: str) -> dict:
    if rep in STATS and str(i) in STATS[rep]:
        return STATS[rep][str(i)]
    k = f"{rep}_i{i}"
    if k in STATS:
        return STATS[k]
    raise KeyError(f"cannot find stats for rep={rep}, i={i} in {stats_json}")


def normalize_data(data, mean, std):
    data = np.asarray(data, dtype=np.float32)
    mean = float(mean); std = float(std)
    if std == 0.0:
        std = 1.0
    return (data - mean) / std


def renorm(y_norm, mean, std):
    y_norm = np.asarray(y_norm, dtype=np.float32)
    return y_norm * float(std) + float(mean)


def stack_mean_std(pred_list_1d):
    S = np.stack([np.asarray(p).reshape(-1) for p in pred_list_1d], axis=1)
    return S.mean(axis=1), S.std(axis=1, ddof=1)


def shift_by_id_lastpoint(y_pred_norm: np.ndarray, pvap_norm: np.ndarray, id_vector: np.ndarray) -> np.ndarray:
    ids = id_vector.reshape(-1)
    y = np.asarray(y_pred_norm).copy().reshape(-1, 1)
    pv = np.asarray(pvap_norm).reshape(-1, 1)

    last_index = {}
    for idx, uid in enumerate(ids):
        last_index[uid] = idx

    for uid, li in last_index.items():
        shift = float(pv[li] - y[li])
        mask = (ids == uid)
        y[mask] += shift
    return y

def split_columns(arr: np.ndarray) -> dict:
    ncol = arr.shape[1]
    return {
        "id":       arr[:, 0:1],
        "x":        arr[:, 1:ncol-34],
        "T_PRSAC":  arr[:, ncol-34:ncol-23],  # 11
        "P_PRSAC":  arr[:, ncol-23:ncol-12],  # 11
        "T_predi":  arr[:, ncol-12:ncol-11],  # 1  
        "P_predi":  arr[:, ncol-11:ncol-10],  # 1  
        "Tb_PRSAC": arr[:, ncol-10:ncol-9],   # 1
        "Tc_PRSAC": arr[:, ncol-9:ncol-8],    # 1
        "Pc_PRSAC": arr[:, ncol-8:ncol-7],    # 1  (lnPc)
        "pvap":     arr[:, ncol-7:ncol-6],    # 1  
        "Tb":       arr[:, ncol-6:ncol-5],    # 1
        "Tc":       arr[:, ncol-5:ncol-4],    # 1
        "Pc":       arr[:, ncol-4:ncol-3],    # 1
        "optional": arr[:, ncol-3:ncol-2],    # 1
        "w_exp":    arr[:, ncol-2:ncol-1],    # 1
        "w_PRSAC":  arr[:, ncol-1:ncol],      # 1
    }


def prep_inputs_pvap(test, stats):
    x_norm  = normalize_data(test["x"],        stats["ave_molecule"], stats["std_molecule"])
    Tn      = normalize_data(test["T_PRSAC"],  stats["ave_T"],     stats["std_T"])
    Tp      = normalize_data(test["T_predi"],  stats["ave_T"],     stats["std_T"])
    Pn      = normalize_data(test["P_PRSAC"],  stats["ave_P"],     stats["std_P"])
    Pp      = normalize_data(test["P_predi"],  stats["ave_P"],     stats["std_P"])
    Tb_n    = normalize_data(test["Tb_PRSAC"], stats["ave_Tb"],    stats["std_Tb"])
    Tc_n    = normalize_data(test["Tc_PRSAC"], stats["ave_Tc"],    stats["std_Tc"])
    Pc_n    = normalize_data(test["Pc_PRSAC"], stats["ave_Pc"],    stats["std_Pc"])
    optional_n = normalize_data(test["optional"],    stats["ave_optional"], stats["std_optional"])
    pvap_n  = normalize_data(test["pvap"],     stats["ave_P"],     stats["std_P"])

    return {
        "x_FP": np.hstack([x_norm, optional_n]).astype(np.float32, copy=False),
        "x_T":  np.hstack([Tn, Tp]).astype(np.float32, copy=False),
        "x_P":  np.hstack([Pn, Pp]).astype(np.float32, copy=False),
        "x_Tb": np.hstack([Tb_n, Tc_n]).astype(np.float32, copy=False),
        "Pc":   Pc_n.astype(np.float32, copy=False),
        "pvap_norm": pvap_n.astype(np.float32, copy=False),
    }


def prep_inputs_TbTcPcw(test, stats):
    x_norm   = normalize_data(test["x"],        stats["ave_molecule"], stats["std_molecule"])
    optional_n  = normalize_data(test["optional"],    stats["ave_optional"], stats["std_optional"])
    Tb_n     = normalize_data(test["Tb_PRSAC"], stats["ave_Tb"],    stats["std_Tb"])
    Tc_n     = normalize_data(test["Tc_PRSAC"], stats["ave_Tc"],    stats["std_Tc"])
    Pc_n     = normalize_data(test["Pc_PRSAC"], stats["ave_Pc"],    stats["std_Pc"])
    wP_n     = normalize_data(test["w_PRSAC"],  stats["ave_w"],     stats["std_w"])

    return {
        "x_FP": np.hstack([x_norm, optional_n]).astype(np.float32, copy=False),
        "TbTc": np.hstack([Tb_n, Tc_n]).astype(np.float32, copy=False),
        "Pc":   Pc_n.astype(np.float32, copy=False),
        "w":    wP_n.astype(np.float32, copy=False),
    }


def _unpack_tb_tcpw_pred(pred):
    if isinstance(pred, dict):
        keys = list(pred.keys())

        def pick(*names):
            for n in names:
                if n in pred:
                    return pred[n]
            return None

        yTb = pick("Tb", "tb", "Tb_out", "output_Tb")
        yTc = pick("Tc", "tc", "Tc_out", "output_Tc")
        yPc = pick("Pc", "pc", "lnPc", "lnPc_out", "output_Pc")
        yW  = pick("w", "W", "omega", "acentric", "w_out", "output_w")

        if any(v is None for v in [yTb, yTc, yPc, yW]):
            raise KeyError(f"predict() returned dict keys={keys}, cannot map to Tb/Tc/Pc/w")
        return yTb, yTc, yPc, yW

    if isinstance(pred, (list, tuple)) and len(pred) == 4:
        return pred[0], pred[1], pred[2], pred[3]

    raise TypeError(f"Unexpected TbTcPcw predict() output type={type(pred)}")


def wagner_lnPr(Tr, c1, c2, c3, c4):
    Tr = float(Tr)
    if Tr >= 1.0:
        return 0.0
    if Tr <= 0.0:
        return np.nan
    delta = 1.0 - Tr
    return (c1*delta + c2*delta**1.5 + c3*delta**2.5 + c4*delta**5) / Tr


def make_Tr_grid(rng, n):
    start = float(np.random.uniform(rng[0], rng[1]))
    return np.linspace(start, 1.0, n)


def load_merged_sheets(xlsx_path: str):
    fp  = pd.read_excel(xlsx_path, sheet_name="FP")
    sig = pd.read_excel(xlsx_path, sheet_name="sigma_profile")
    wag = pd.read_excel(xlsx_path, sheet_name="PRSAC Wagner")
    opt = pd.read_excel(xlsx_path, sheet_name="optional features")

    for df in (fp, sig, wag, opt):
        if "id" not in df.columns:
            raise ValueError("All sheets must contain 'id' column.")
        df["id"] = df["id"].astype(str)

    if "smiles" not in fp.columns:
        raise ValueError("Sheet 'FP' must contain 'smiles' for smiles->id lookup.")
    fp["smiles"] = fp["smiles"].astype(str)

    smiles2id = (
        fp[["smiles", "id"]]
        .dropna()
        .drop_duplicates("smiles", keep="first")
        .set_index("smiles")["id"]
        .to_dict()
    )

    fp_feat_cols = [c for c in fp.columns if str(c).lower().startswith("fp")]
    sig_cols     = [c for c in sig.columns if str(c).lower().startswith("sigmaprofile")]

    if len(fp_feat_cols) == 0:
        raise ValueError("Sheet 'FP' has no fp* columns.")
    if len(sig_cols) == 0:
        raise ValueError("Sheet 'sigma_profile' has no sigmaprofile* columns.")

    for c in ["wagner0", "wagner1", "wagner2", "wagner3", "lnPc(Pa)"]:
        if c not in wag.columns:
            raise ValueError(f"Sheet 'PRSAC Wagner' missing column: {c}")
    for c in ["wagner0","wagner1","wagner2","wagner3","lnPc(Pa)"]:
        wag[c] = pd.to_numeric(wag[c], errors="coerce")

    rename_map = {}
    for c in opt.columns:
        cl = str(c).strip().lower()
        if cl in ["tb(k)", "tb"]:
            rename_map[c] = "Tb(K)"
        elif cl in ["tc(k)", "tc"]:
            rename_map[c] = "Tc(K)"
        elif cl in ["lnpc(pa)", "lnpc"]:
            rename_map[c] = "lnPc(Pa)"
        elif cl == "v":
            rename_map[c] = "V"
        elif cl in ["acentric", "omega", "w"]:
            rename_map[c] = "w"
    if rename_map:
        opt = opt.rename(columns=rename_map)

    if "lnPc(Pa)" not in opt.columns:
        opt["lnPc(Pa)"] = np.nan
    if "V" not in opt.columns:
        raise ValueError("Sheet 'optional features' missing column 'V'.")

    for c in ["Tb(K)", "Tc(K)", "lnPc(Pa)", "V", "w"]:
        if c in opt.columns:
            opt[c] = pd.to_numeric(opt[c], errors="coerce")

    fp_map  = fp.set_index("id")[fp_feat_cols]
    sig_map = sig.set_index("id")[sig_cols]
    wag_map = wag.set_index("id")[["wagner0","wagner1","wagner2","wagner3","lnPc(Pa)"]]
    opt_map = opt.set_index("id")[["Tb(K)","Tc(K)","lnPc(Pa)","V","w"]]

    has_opt_lnpc = opt["lnPc(Pa)"].notna().any()

    return {
        "smiles2id": smiles2id,
        "fp_cols": fp_feat_cols,
        "sigma_cols": sig_cols,
        "fp_map": fp_map,
        "sigma_map": sig_map,
        "wag_map": wag_map,
        "opt_map": opt_map,
        "has_opt_lnpc": has_opt_lnpc,
    }


def build_base_features_for_smiles(SHEETS, smiles: str, rep_builder: str, strict_V: bool):
    smiles = str(smiles)
    if smiles not in SHEETS["smiles2id"]:
        raise KeyError(f"smiles not found in FP sheet: {smiles}")

    id_str = SHEETS["smiles2id"][smiles]
    id_num = pd.to_numeric(id_str, errors="coerce")
    if not np.isfinite(id_num):
        raise ValueError(f"id is not numeric for smiles={smiles}: id='{id_str}'")
    id_num = float(id_num)

    rep_builder = rep_builder.lower()
    if rep_builder not in ["fp", "sigma"]:
        raise ValueError("rep_builder must be fp or sigma")

    if rep_builder == "fp":
        vec_cols = SHEETS["fp_cols"]
        vec = pd.to_numeric(SHEETS["fp_map"].loc[id_str, vec_cols], errors="coerce").to_numpy(dtype=float)
    else:
        vec_cols = SHEETS["sigma_cols"]
        vec = pd.to_numeric(SHEETS["sigma_map"].loc[id_str, vec_cols], errors="coerce").to_numpy(dtype=float)

    sum_vec = float(np.nansum(vec))

    wrow = SHEETS["wag_map"].loc[id_str]
    c1, c2, c3, c4 = [float(wrow[k]) for k in ["wagner0","wagner1","wagner2","wagner3"]]
    lnPc_wag = float(wrow["lnPc(Pa)"])

    orow = SHEETS["opt_map"].loc[id_str]
    Tb0 = float(orow["Tb(K)"]) if np.isfinite(orow["Tb(K)"]) else np.nan
    Tc0 = float(orow["Tc(K)"]) if np.isfinite(orow["Tc(K)"]) else np.nan
    w0  = float(orow["w"]) if np.isfinite(orow["w"]) else np.nan
    V   = float(orow["V"]) if np.isfinite(orow["V"]) else np.nan

    if strict_V and (not np.isfinite(V) or V == 0.0):
        raise ValueError(f"id={id_str} has invalid V (V={V}). Cannot compute sum/V.")

    if SHEETS["has_opt_lnpc"]:
        lnPc_opt = float(orow["lnPc(Pa)"]) if np.isfinite(orow["lnPc(Pa)"]) else np.nan
        lnPc0 = lnPc_opt if np.isfinite(lnPc_opt) else lnPc_wag
    else:
        lnPc0 = lnPc_wag

    sum_over_V = (sum_vec / V) if (np.isfinite(V) and V != 0.0) else np.nan

    return {
        "smiles": smiles,
        "id_num": id_num,
        "vec": vec.astype(np.float32),
        "c": (c1, c2, c3, c4),
        "Tb0": Tb0,
        "Tc0": Tc0,
        "lnPc0": lnPc0,
        "w0": w0,
        "sum_over_V": float(sum_over_V) if np.isfinite(sum_over_V) else np.nan,
    }


def build_arr_for_TbTcPcw(base_list):
    rows = []
    for b in base_list:
        row = []
        row.append(b["id_num"])
        row.extend(b["vec"].tolist())

        row.extend([0.0] * 11)
        row.extend([0.0] * 11)

        row.append(0.0)
        row.append(0.0)

        row.append(float(b["Tb0"]) if np.isfinite(b["Tb0"]) else 0.0)
        row.append(float(b["Tc0"]) if np.isfinite(b["Tc0"]) else 0.0)
        row.append(float(b["lnPc0"]) if np.isfinite(b["lnPc0"]) else 0.0)

        row.extend([0.0, 0.0, 0.0, 0.0])

        row.append(float(b["sum_over_V"]) if np.isfinite(b["sum_over_V"]) else 0.0)
        row.append(0.0)
        row.append(float(b["w0"]) if np.isfinite(b["w0"]) else 0.0)

        rows.append(row)

    return np.asarray(rows, dtype=np.float32)


def build_arr_for_pvap_member_real(
    base_list, temps_map, Tc_pred, lnPc_pred, Tb_pred, w_pred,
    tr_start_range, n_grid, include_terminal=True
):
    rows = []
    rows_per_smiles_list = []
    smiles_for_rows = []
    T_for_rows = []

    for idx, b in enumerate(base_list):
        c1, c2, c3, c4 = b["c"]
        Tc_i = float(Tc_pred[idx])
        lnPc_i = float(lnPc_pred[idx])
        Tb_i = float(Tb_pred[idx])
        w_i = float(w_pred[idx])

        T_use = temps_map.get(b["smiles"], [])
        n_use = 0

        for T in T_use:
            T = float(T)
            Tr = T / Tc_i if Tc_i != 0 else np.nan

            Tr_grid = make_Tr_grid(tr_start_range, n_grid)
            lnPr_grid = np.array([wagner_lnPr(t, c1, c2, c3, c4) for t in Tr_grid], dtype=float)
            lnPr_cal = float(wagner_lnPr(Tr, c1, c2, c3, c4)) if np.isfinite(Tr) else np.nan

            row = []
            row.append(b["id_num"])
            row.extend(b["vec"].tolist())
            row.extend(Tr_grid.tolist())
            row.extend(lnPr_grid.tolist())
            row.append(Tr)          
            row.append(lnPr_cal)    
            row.append(Tb_i)
            row.append(Tc_i)
            row.append(lnPc_i)      
            row.extend([0.0, 0.0, 0.0, 0.0])
            row.append(float(b["sum_over_V"]) if np.isfinite(b["sum_over_V"]) else 0.0)
            row.append(0.0)
            row.append(w_i)
            rows.append(row)

            smiles_for_rows.append(b["smiles"])
            T_for_rows.append(T)
            n_use += 1

        if include_terminal:
            Tr_grid = make_Tr_grid(tr_start_range, n_grid)
            lnPr_grid = np.array([wagner_lnPr(t, c1, c2, c3, c4) for t in Tr_grid], dtype=float)

            row = []
            row.append(b["id_num"])
            row.extend(b["vec"].tolist())
            row.extend(Tr_grid.tolist())
            row.extend(lnPr_grid.tolist())
            row.append(1.0)
            row.append(0.0)
            row.append(Tb_i)
            row.append(Tc_i)
            row.append(lnPc_i)
            row.extend([0.0, 0.0, 0.0, 0.0])
            row.append(float(b["sum_over_V"]) if np.isfinite(b["sum_over_V"]) else 0.0)
            row.append(0.0)
            row.append(w_i)
            rows.append(row)

            smiles_for_rows.append(b["smiles"])
            T_for_rows.append(np.nan)
            n_use += 1

        rows_per_smiles_list.append(n_use)

    return np.asarray(rows, dtype=np.float32), rows_per_smiles_list, smiles_for_rows, T_for_rows


def build_arr_for_pvap_reduced_per_smiles(base_list, trs_map, tr_start_range, n_grid):
    rows = []
    smiles_for_rows = []
    Tr_for_rows = []
    rows_per_smiles_list = []

    for idx, b in enumerate(base_list):
        c1, c2, c3, c4 = b["c"]

        Tr_use = list(trs_map.get(b["smiles"], []))
        if not any(abs(t - 1.0) < 1e-12 for t in Tr_use):
            Tr_use.append(1.0)

        Tr_use = sorted(set([float(t) for t in Tr_use]))

        for Tr in Tr_use:
            Tr = float(Tr)
            Tr_grid = make_Tr_grid(tr_start_range, n_grid)
            lnPr_grid = np.array([wagner_lnPr(t, c1, c2, c3, c4) for t in Tr_grid], dtype=float)
            lnPr_cal = 0.0 if abs(Tr - 1.0) < 1e-12 else float(wagner_lnPr(Tr, c1, c2, c3, c4))

            row = []
            row.append(b["id_num"])
            row.extend(b["vec"].tolist())
            row.extend(Tr_grid.tolist())
            row.extend(lnPr_grid.tolist())
            row.append(Tr)
            row.append(lnPr_cal)
            row.append(float(b["Tb0"]) if np.isfinite(b["Tb0"]) else 0.0)
            row.append(float(b["Tc0"]) if np.isfinite(b["Tc0"]) else 0.0)
            row.append(float(b["lnPc0"]) if np.isfinite(b["lnPc0"]) else 0.0)
            row.extend([0.0, 0.0, 0.0, 0.0])
            row.append(float(b["sum_over_V"]) if np.isfinite(b["sum_over_V"]) else 0.0)
            row.append(0.0)
            row.append(float(b["w0"]) if np.isfinite(b["w0"]) else 0.0)
            rows.append(row)

            smiles_for_rows.append(b["smiles"])
            Tr_for_rows.append(Tr)

        rows_per_smiles_list.append(len(Tr_use))

    return np.asarray(rows, dtype=np.float32), smiles_for_rows, np.asarray(Tr_for_rows, dtype=np.float32), rows_per_smiles_list


def dump_space_txt(arr: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in arr:
            f.write(" ".join(str(x) for x in r.tolist()) + "\n")

def _progress_init(enabled: bool, label: str, n_rows: int, n_models: int):
    if not enabled:
        return None
    total = int(n_rows) * int(n_models)
    total = max(total, 1)
    print(f"[INFO] {label}: rows={int(n_rows)} models={int(n_models)} total_work={total}")
    return {"label": label, "n_rows": int(n_rows), "n_models": int(n_models), "total": total, "done": 0, "last_pct": -1}

def _progress_step(st, done_rows: int, *, i=None, j=None):
    if st is None:
        return
    st["done"] += int(done_rows)
    pct = int(st["done"] * 100 / st["total"])

    if pct >= st["last_pct"] + 5 or pct == 100:
        tail = ""
        if i is not None and j is not None:
            tail = f" [i={i}, j={j}]"
        print(f"[PROGRESS] {st['label']}: {st['done']}/{st['total']} ({pct}%)")
        st["last_pct"] = pct


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--xlsx-path", default="input_features.xlsx")

    p.add_argument("--smiles", nargs="*", default=[], help='One or more SMILES. Example: --smiles "CCO" "CCN"')
    p.add_argument("--input-csv", "--smiles-csv",  dest="input_csv", default="", help="CSV file containing smiles (and optionally per-row temps/Tr).")
    p.add_argument("--smiles-col", default="smiles", help="smiles column in input-csv.")
    p.add_argument("--temp-col", default="T(K)", help="temperature column in input-csv.")
    p.add_argument("--tr-col", default="Tr", help="Tr column in input-csv.")
    p.add_argument("--skip-missing", action="store_true")

    p.add_argument("--temps", default="", help="Comma-separated temperatures in K, e.g. 298.15,350,400")
    p.add_argument("--trs", default="", help="Comma-separated Tr for reduced tasks")

    p.add_argument("--tr-start-range", default="0.30,0.31")
    p.add_argument("--n-grid", type=int, default=11)
    p.add_argument("--strict-V", action="store_true")
    p.add_argument("--seed", type=int, default=None)

    p.add_argument("--model-dir", default="./model_save")
    p.add_argument("--stats-json", default="normalization_stats.json")
    p.add_argument("--rep", default="FP", choices=["FP", "sigma"])
    p.add_argument("--task", default="both_real",
                   choices=["pvap_reduced", "pvap_real", "TbTcPcw", "both_reduced", "both_real"])
    p.add_argument("--i-list", required=True)
    p.add_argument("--j-list", required=True)

    p.add_argument("--out-csv", default="")
    p.add_argument("--debug-txt", default="",)

    p.add_argument("--progress", action="store_true", help="Show progress as done/total for ensemble inference.")

    return p.parse_args()

def main():
    args = parse_args()

    if args.seed is not None:
        np.random.seed(int(args.seed))

    is_real = args.task in ["pvap_real", "both_real"]
    is_reduced = args.task in ["pvap_reduced", "both_reduced"]

    smiles_list_in = []
    temps_map_in = {}  
    trs_map_in = {}    
    source_df = None

    if args.smiles and len(args.smiles) > 0:
        smiles_list_in = [str(s).strip() for s in args.smiles if str(s).strip()]

    elif args.input_csv.strip():
        df = pd.read_csv(args.input_csv)
        source_df = df

        sc = _find_col_case_insensitive(df, args.smiles_col)
        if sc is None:
            raise ValueError(f"--smiles-col '{args.smiles_col}' not found in {args.input_csv}")

        smiles_list_in = df[sc].astype(str).map(str.strip).tolist()

        if is_real:
            tc = _find_col_case_insensitive(df, args.temp_col)
            if tc is not None:
                for smi, t in zip(df[sc].astype(str).map(str.strip), df[tc]):
                    if smi == "" or pd.isna(t):
                        continue
                    temps_map_in.setdefault(smi, []).append(float(t))
        if is_reduced:
            trc = _find_col_case_insensitive(df, args.tr_col)
            if trc is not None:
                for smi, tr in zip(df[sc].astype(str).map(str.strip), df[trc]):
                    if smi == "" or pd.isna(tr):
                        continue
                    trs_map_in.setdefault(smi, []).append(float(tr))
    else:
        raise ValueError("Provide either --smiles or --smiles-csv or --input-csv")

    seen = set()
    smiles_order_list = []
    for smi in smiles_list_in:
        if smi not in seen:
            seen.add(smi)
            smiles_order_list.append(smi)

    T_fallback = parse_float_list(args.temps, "temps") if args.temps.strip() else []
    Tr_fallback = parse_float_list(args.trs, "trs") if args.trs.strip() else []

    if is_real:
        if not temps_map_in:
            if not T_fallback:
                raise ValueError(f"task={args.task} requires either CSV column '{args.temp_col}' or --temps.")
            temps_map_in = {smi: list(T_fallback) for smi in smiles_order_list}
    if is_reduced:
        if not trs_map_in:
            if not Tr_fallback:
                raise ValueError(f"task={args.task} requires either CSV column '{args.tr_col}' or --trs.")
            trs_map_in = {smi: list(Tr_fallback) for smi in smiles_order_list}

    if temps_map_in:
        for smi in list(temps_map_in.keys()):
            temps_map_in[smi] = sorted(set([float(x) for x in temps_map_in[smi]]))
    if trs_map_in:
        for smi in list(trs_map_in.keys()):
            trs_map_in[smi] = sorted(set([float(x) for x in trs_map_in[smi]]))

    rng = parse_float_list(args.tr_start_range, "tr-start-range")
    if len(rng) != 2:
        raise ValueError("--tr-start-range must be like '0.30,0.31'")
    tr_start_range = (float(rng[0]), float(rng[1]))

    assert os.path.exists(args.xlsx_path), f"missing: {args.xlsx_path}"
    SHEETS = load_merged_sheets(args.xlsx_path)

    assert os.path.exists(args.stats_json), f"missing: {args.stats_json}"
    STATS = load_stats(args.stats_json)
    I_LIST = parse_int_list(args.i_list, "i-list")
    J_LIST = parse_int_list(args.j_list, "j-list")

    rep_builder = "fp" if args.rep.lower() == "fp" else "sigma"

    base_list = []
    skipped = []
    for smi in smiles_order_list:
        try:
            b = build_base_features_for_smiles(SHEETS, smi, rep_builder, strict_V=bool(args.strict_V))
            base_list.append(b)
        except Exception as e:
            if args.skip_missing:
                skipped.append((smi, str(e)))
                continue
            raise

    if not base_list:
        raise RuntimeError("No valid smiles after filtering/skip.")

    smiles_order = {b["smiles"]: k for k, b in enumerate(base_list)}

    valid_smiles = set(smiles_order.keys())
    if temps_map_in:
        temps_map = {s: temps_map_in.get(s, []) for s in smiles_order.keys()}
    else:
        temps_map = {}
    if trs_map_in:
        trs_map = {s: trs_map_in.get(s, []) for s in smiles_order.keys()}
    else:
        trs_map = {}

    member_Tb = []
    member_Tc = []
    member_lnPc = []
    member_w = []

    arr_ttpcw = build_arr_for_TbTcPcw(base_list)
    test_ttpcw = split_columns(arr_ttpcw)

    _st_tb = _progress_init(args.progress, 'TbTcPcw', n_rows=len(arr_ttpcw), n_models=len(I_LIST) * len(J_LIST))

    for i in I_LIST:
        stats = get_stats(STATS, args.rep, i, args.stats_json)
        inp_ttpcw = prep_inputs_TbTcPcw(test_ttpcw, stats)

        for j in J_LIST:
            model_path = os.path.join(args.model_dir, f"TbTcPcw_{args.rep}_{i}_{j}.h5")
            assert os.path.exists(model_path), f"missing model: {model_path}"
            m = load_model(model_path, compile=False)

            pred = m.predict([inp_ttpcw["x_FP"], inp_ttpcw["TbTc"], inp_ttpcw["Pc"], inp_ttpcw["w"]],
                             verbose=0)
            yTb, yTc, yPc, yW = _unpack_tb_tcpw_pred(pred)

            Tb = renorm(yTb, stats["ave_Tb"], stats["std_Tb"]).reshape(-1)
            Tc = renorm(yTc, stats["ave_Tc"], stats["std_Tc"]).reshape(-1)
            lnPc = renorm(yPc, stats["ave_Pc"], stats["std_Pc"]).reshape(-1)
            w_pred = renorm(yW, stats["ave_w"], stats["std_w"]).reshape(-1)

            member_Tb.append(Tb)
            member_Tc.append(Tc)
            member_lnPc.append(lnPc)
            member_w.append(w_pred)

            _progress_step(_st_tb, done_rows=len(arr_ttpcw), i=i, j=j)

    Tb_mean, Tb_std = stack_mean_std(member_Tb)
    Tc_mean, Tc_std = stack_mean_std(member_Tc)
    lnPc_mean, lnPc_std = stack_mean_std(member_lnPc)
    w_mean, w_std = stack_mean_std(member_w)

    if args.task == "TbTcPcw":
        out = pd.DataFrame({
            "smiles": [b["smiles"] for b in base_list],
            "Tb_mean(K)": Tb_mean,
            "Tb_std": Tb_std,
            "Tc_mean(K)": Tc_mean,
            "Tc_std": Tc_std,
            "lnPc_mean(Pa)": lnPc_mean,
            "lnPc_std": lnPc_std,
            "w_mean": w_mean,
            "w_std": w_std,
        })
        out["__order"] = out["smiles"].map(smiles_order).astype(int)
        out = out.sort_values(["__order"]).reset_index(drop=True).drop(columns="__order")

        out_csv = args.out_csv.strip() if args.out_csv.strip() else f"ensemble_{args.task}_{args.rep}.csv"
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        out.to_csv(out_csv, index=False)
        print("[INFO] wrote:", out_csv)

        if skipped:
            print("[WARN] skipped smiles:")
            for smi, msg in skipped[:20]:
                print(" ", smi, "->", msg)
        return

    if is_reduced:
        arr_pvap, smiles_for_rows, Tr_out, rows_per_smiles_list = build_arr_for_pvap_reduced_per_smiles(
            base_list=base_list,
            trs_map=trs_map,
            tr_start_range=tr_start_range,
            n_grid=args.n_grid
        )

        if args.debug_txt.strip():
            dump_space_txt(arr_pvap, args.debug_txt.strip())
            print("[INFO] wrote debug txt:", args.debug_txt.strip())

        test = split_columns(arr_pvap)

        _st_pvap = _progress_init(args.progress, 'Pvap(reduced)', n_rows=len(arr_pvap), n_models=len(I_LIST) * len(J_LIST))

        all_lnPr = []

        for i in I_LIST:
            stats = get_stats(STATS, args.rep, i, args.stats_json)
            inp_pvap = prep_inputs_pvap(test, stats)

            for j in J_LIST:
                model_path = os.path.join(args.model_dir, f"Pvap_{args.rep}_{i}_{j}.h5")
                assert os.path.exists(model_path), f"missing model: {model_path}"
                m = load_model(model_path, compile=False)

                y = m.predict([inp_pvap["x_FP"], inp_pvap["x_T"], inp_pvap["x_P"], inp_pvap["x_Tb"], inp_pvap["Pc"]],
                              verbose=0)
                if isinstance(y, (list, tuple)):
                    y = y[0]
                elif isinstance(y, dict):
                    y = y.get("pvap", list(y.values())[0])

                y_adj = shift_by_id_lastpoint(y, inp_pvap["pvap_norm"], test["id"])
                y_lnPr = renorm(y_adj, stats["ave_P"], stats["std_P"]).reshape(-1)
                all_lnPr.append(y_lnPr)

                _progress_step(_st_pvap, done_rows=len(arr_pvap), i=i, j=j)

        lnPr_mean, lnPr_std = stack_mean_std(all_lnPr)

        out = pd.DataFrame({
            "smiles": smiles_for_rows,
            "Tr": Tr_out,
            "lnPr_mean": lnPr_mean,
            "lnPr_std": lnPr_std,
        })

        if args.task == "both_reduced":
            Tb_mean_r, Tb_std_r = [], []
            Tc_mean_r, Tc_std_r = [], []
            lnPc_mean_r, lnPc_std_r = [], []
            w_mean_r, w_std_r = [], []

            for sidx, rN in enumerate(rows_per_smiles_list):
                Tb_mean_r.extend([Tb_mean[sidx]] * rN); Tb_std_r.extend([Tb_std[sidx]] * rN)
                Tc_mean_r.extend([Tc_mean[sidx]] * rN); Tc_std_r.extend([Tc_std[sidx]] * rN)
                lnPc_mean_r.extend([lnPc_mean[sidx]] * rN); lnPc_std_r.extend([lnPc_std[sidx]] * rN)
                w_mean_r.extend([w_mean[sidx]] * rN); w_std_r.extend([w_std[sidx]] * rN)

            out["Tb_mean(K)"] = np.asarray(Tb_mean_r, dtype=np.float32)
            out["Tb_std"] = np.asarray(Tb_std_r, dtype=np.float32)
            out["Tc_mean(K)"] = np.asarray(Tc_mean_r, dtype=np.float32)
            out["Tc_std"] = np.asarray(Tc_std_r, dtype=np.float32)
            out["lnPc_mean(Pa)"] = np.asarray(lnPc_mean_r, dtype=np.float32)
            out["lnPc_std"] = np.asarray(lnPc_std_r, dtype=np.float32)
            out["w_mean"] = np.asarray(w_mean_r, dtype=np.float32)
            out["w_std"] = np.asarray(w_std_r, dtype=np.float32)

        out["__order"] = out["smiles"].map(smiles_order).astype(int)
        out = out.sort_values(["__order", "Tr"]).reset_index(drop=True).drop(columns="__order")

        out_csv = args.out_csv.strip() if args.out_csv.strip() else f"ensemble_{args.task}_{args.rep}.csv"
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        out.to_csv(out_csv, index=False)
        print("[INFO] wrote:", out_csv)

        if skipped:
            print("[WARN] skipped smiles:")
            for smi, msg in skipped[:20]:
                print(" ", smi, "->", msg)
        return

    all_lnP_members = []
    all_Tr_members = []

    debug_txt_written = False

    _st_pvap_real = None

    smiles_for_rows = None
    T_for_rows = None
    rows_per_smiles_list = None

    for idx_m in range(len(member_Tc)):
        Tc_m = np.asarray(member_Tc[idx_m], dtype=np.float32)
        lnPc_m = np.asarray(member_lnPc[idx_m], dtype=np.float32)
        Tb_m = np.asarray(member_Tb[idx_m], dtype=np.float32)
        w_m = np.asarray(member_w[idx_m], dtype=np.float32)

        arr_pvap_m, rows_per_smiles_list_m, smiles_for_rows_m, T_for_rows_m = build_arr_for_pvap_member_real(
            base_list=base_list,
            temps_map=temps_map,
            Tc_pred=Tc_m,
            lnPc_pred=lnPc_m,
            Tb_pred=Tb_m,
            w_pred=w_m,
            tr_start_range=tr_start_range,
            n_grid=args.n_grid,
            include_terminal=True
        )

        if smiles_for_rows is None:
            smiles_for_rows = smiles_for_rows_m
            T_for_rows = T_for_rows_m
            rows_per_smiles_list = rows_per_smiles_list_m

            _st_pvap_real = _progress_init(args.progress, 'Pvap(real)', n_rows=arr_pvap_m.shape[0], n_models=len(member_Tc))

        if args.debug_txt.strip() and (not debug_txt_written):
            dump_space_txt(arr_pvap_m, args.debug_txt.strip())
            print("[INFO] wrote debug txt (member-0):", args.debug_txt.strip())
            debug_txt_written = True

        test = split_columns(arr_pvap_m)

        Tr_m = test["T_predi"].reshape(-1)
        all_Tr_members.append(Tr_m)

        k = idx_m
        i_pos = k // len(J_LIST)
        j_pos = k % len(J_LIST)
        i = I_LIST[i_pos]
        j = J_LIST[j_pos]

        stats = get_stats(STATS, args.rep, i, args.stats_json)
        inp_pvap = prep_inputs_pvap(test, stats)

        model_path = os.path.join(args.model_dir, f"Pvap_{args.rep}_{i}_{j}.h5")
        assert os.path.exists(model_path), f"missing model: {model_path}"
        m = load_model(model_path, compile=False)

        y = m.predict([inp_pvap["x_FP"], inp_pvap["x_T"], inp_pvap["x_P"], inp_pvap["x_Tb"], inp_pvap["Pc"]],
                      verbose=0)
        if isinstance(y, (list, tuple)):
            y = y[0]
        elif isinstance(y, dict):
            y = y.get("pvap", list(y.values())[0])

        y_adj = shift_by_id_lastpoint(y, inp_pvap["pvap_norm"], test["id"])
        lnPr_pred = renorm(y_adj, stats["ave_P"], stats["std_P"]).reshape(-1)

        lnPc_rep = []
        for sidx, rN in enumerate(rows_per_smiles_list_m):
            lnPc_rep.extend([float(lnPc_m[sidx])] * rN)
        lnPc_rep = np.asarray(lnPc_rep, dtype=np.float32)

        lnP_pred = lnPr_pred + lnPc_rep
        all_lnP_members.append(lnP_pred)

        _progress_step(_st_pvap_real, done_rows=arr_pvap_m.shape[0], i=i, j=j)

    lnP_mean, lnP_std = stack_mean_std(all_lnP_members)
    Tr_mean, Tr_std = stack_mean_std(all_Tr_members)

    Tb_mean_r, Tb_std_r = [], []
    Tc_mean_r, Tc_std_r = [], []
    lnPc_mean_r, lnPc_std_r = [], []
    w_mean_r, w_std_r = [], []

    for sidx, rN in enumerate(rows_per_smiles_list):
        Tb_mean_r.extend([Tb_mean[sidx]] * rN); Tb_std_r.extend([Tb_std[sidx]] * rN)
        Tc_mean_r.extend([Tc_mean[sidx]] * rN); Tc_std_r.extend([Tc_std[sidx]] * rN)
        lnPc_mean_r.extend([lnPc_mean[sidx]] * rN); lnPc_std_r.extend([lnPc_std[sidx]] * rN)
        w_mean_r.extend([w_mean[sidx]] * rN); w_std_r.extend([w_std[sidx]] * rN)

    out = pd.DataFrame({
        "smiles": smiles_for_rows,
        "T(K)": np.asarray(T_for_rows, dtype=np.float32),
        "Tr_mean": Tr_mean,
        "Tr_std": Tr_std,
        "lnP_mean(Pa)": lnP_mean,
        "lnP_std": lnP_std,
        "Tb_mean(K)": np.asarray(Tb_mean_r, dtype=np.float32),
        "Tb_std": np.asarray(Tb_std_r, dtype=np.float32),
        "Tc_mean(K)": np.asarray(Tc_mean_r, dtype=np.float32),
        "Tc_std": np.asarray(Tc_std_r, dtype=np.float32),
        "lnPc_mean(Pa)": np.asarray(lnPc_mean_r, dtype=np.float32),
        "lnPc_std": np.asarray(lnPc_std_r, dtype=np.float32),
        "w_mean": np.asarray(w_mean_r, dtype=np.float32),
        "w_std": np.asarray(w_std_r, dtype=np.float32),
    })

    out["__order"] = out["smiles"].map(smiles_order).astype(int)
    out = out.sort_values(["__order", "T(K)"]).reset_index(drop=True).drop(columns="__order")

    out_csv = args.out_csv.strip() if args.out_csv.strip() else f"ensemble_{args.task}_{args.rep}.csv"
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    out.to_csv(out_csv, index=False)
    print("[INFO] wrote:", out_csv)

    if skipped:
        print("[WARN] skipped smiles:")
        for smi, msg in skipped[:20]:
            print(" ", smi, "->", msg)


if __name__ == "__main__":
    main()
