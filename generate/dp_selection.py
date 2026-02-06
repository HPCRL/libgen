# dp_selection.py
import pandas as pd
import numpy as np
import itertools
import json
from typing import List, Tuple, Dict, Any

SHAPE_COLS = ["M", "N", "K", "L"]
KERNEL_COLS = ["cta_m", "cta_n", "cta_k", "stages", "atom_m", "atom_n", "atom_k"]
FILTER_EQ = {
    "a_major": "m",
    "b_major": "n",
    "c_major": "n",
    "cta_n": 128,
    "atom_m": 2,
    "atom_n": 2,
    "atom_k": 1,
}
TIME_COL = "avg_us"

def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    for col, val in filters.items():
        out = out[out[col] == val]
    return out.reset_index(drop=True)

def identify_shapes_and_kernels(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    shapes = df[SHAPE_COLS].drop_duplicates().reset_index(drop=True).copy()
    shapes["shape_id"] = np.arange(len(shapes), dtype=int)
    kernels = df[KERNEL_COLS].drop_duplicates().reset_index(drop=True).copy()
    kernels["kernel_id"] = np.arange(len(kernels), dtype=int)
    return shapes, kernels

def attach_ids(df: pd.DataFrame, shapes: pd.DataFrame, kernels: pd.DataFrame) -> pd.DataFrame:
    return df.merge(shapes, on=SHAPE_COLS, how="left").merge(kernels, on=KERNEL_COLS, how="left")

def build_time_matrix(df_with_ids: pd.DataFrame, num_kernels: int, num_shapes: int) -> np.ndarray:
    T = np.full((num_kernels, num_shapes), np.inf, dtype=float)
    for _, r in df_with_ids.iterrows():
        d = int(r["kernel_id"]); s = int(r["shape_id"])
        t = float(r[TIME_COL])
        if t < T[d, s]:
            T[d, s] = t
    return T

def times_to_lop(T: np.ndarray, eps: float = 0.0) -> np.ndarray:
    T_proc = T.copy()
    if eps > 0:
        T_proc = T_proc + eps * np.nanmean(T_proc[np.isfinite(T_proc)])
    best = np.nanmin(T_proc, axis=0)
    best[~np.isfinite(best)] = np.nan
    best[best == 0] = 1e-12
    LOP = T_proc / best - 1.0
    LOP[:, ~np.isfinite(best)] = np.nan
    return LOP

def dominance_prune(LOP: np.ndarray, apply: bool = False) -> Tuple[np.ndarray, List[int]]:
    D = LOP.shape[0]
    keep = np.ones(D, dtype=bool)
    if apply:
        for i in range(D):
            if not keep[i]:
                continue
            for j in range(D):
                if i == j or not keep[j]:
                    continue
                a = LOP[i]; b = LOP[j]
                finite = np.isfinite(a) & np.isfinite(b)
                if not np.any(finite):
                    continue
                if np.all(a[finite] <= b[finite]) and np.any(a[finite] < b[finite]):
                    keep[j] = False
    idxs = [i for i, k in enumerate(keep) if k]
    return LOP[keep], idxs

def score_envelope(envelope: np.ndarray, weights: np.ndarray = None) -> float:
    if weights is None:
        weights = np.ones_like(envelope, dtype=float)
    env = envelope.copy()
    env[~np.isfinite(env)] = 10.0
    return float(np.sum(weights * env))

def enumerate_envelopes(LOP: np.ndarray, V_values: List[int], weights: np.ndarray = None):
    D, S = LOP.shape
    if weights is None:
        weights = np.ones(S, dtype=float)

    summary_rows = []
    envelopes = {}
    best_assignments = {}

    for V in V_values:
        if V > D or V <= 0:
            continue
        best_score = np.inf
        best_subset = None
        best_env = None
        best_assign = None

        for subset in itertools.combinations(range(D), V):
            env = np.nanmin(LOP[list(subset), :], axis=0)
            score = score_envelope(env, weights=weights)
            summary_rows.append({
                "V": V, "subset": subset, "sum_loss": score,
                "max_loss": float(np.nanmax(env)),
            })
            envelopes[(V, subset)] = env

            if score < best_score:
                best_score = score
                best_subset = subset
                best_env = env
                assign = np.zeros(S, dtype=int)
                for s in range(S):
                    vals = [(d, LOP[d, s]) for d in subset]
                    vals = [(d, (v if np.isfinite(v) else 10.0)) for d, v in vals]
                    d_star = min(vals, key=lambda x: x[1])[0]
                    assign[s] = d_star
                best_assign = assign

        if best_subset is not None:
            best_assignments[V] = {
                "subset": best_subset,
                "sum_loss": best_score,
                "envelope": best_env,
                "assignments": best_assign,
            }

    return summary_rows, envelopes, best_assignments

def save_results(out_prefix: str,
                 summary_rows,
                 envelopes,
                 best_assignments,
                 shapes: pd.DataFrame,
                 kernels: pd.DataFrame,
                 kept_kernel_indices):
    kernel_map = {new_idx: kept_kernel_indices[new_idx] for new_idx in range(len(kept_kernel_indices))}
    kernel_desc = []
    for new_idx in range(len(kept_kernel_indices)):
        row = kernels.iloc[new_idx][KERNEL_COLS].to_dict()
        row["kernel_row_index"] = new_idx
        row["original_kernel_index"] = kernel_map[new_idx]
        kernel_desc.append(row)

    summary_df = pd.DataFrame(summary_rows)
    def subset_to_str(subset):
        return ";".join(str(int(x)) for x in subset)
    if not summary_df.empty:
        summary_df["subset_str"] = summary_df["subset"].apply(subset_to_str)
    summary_path = f"{out_prefix}_envelopes_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    envelopes_json = {}
    for (V, subset), env in envelopes.items():
        key = f"V={V}|subset={subset_to_str(subset)}"
        envelopes_json[key] = list(map(lambda x: None if not np.isfinite(x) else float(x), env))
    envelopes_path = f"{out_prefix}_envelopes.json"
    with open(envelopes_path, "w") as f:
        json.dump(envelopes_json, f, indent=2)

    for V, info in best_assignments.items():
        subset = info["subset"]
        assign = info["assignments"]
        env = info["envelope"]
        rows = []
        for s in range(len(shapes)):
            k_id = int(assign[s])
            k_row = kernels.iloc[k_id][KERNEL_COLS].to_dict()
            shape_row = shapes.iloc[s][SHAPE_COLS].to_dict()
            rows.append({
                **{f"shape_{k}": v for k, v in shape_row.items()},
                **{f"kernel_{k}": v for k, v in k_row.items()},
                "shape_id": s,
                "kernel_row_index": k_id,
                "subset_indices": ";".join(map(str, subset)),
                "shape_loss": float(env[s]) if np.isfinite(env[s]) else None,
            })
        out_df = pd.DataFrame(rows)
        out_df.to_csv(f"{out_prefix}_best_assignments_V{V}.csv", index=False)

    kernels_desc_df = pd.DataFrame(kernel_desc)
    kernels_desc_df.to_csv(f"{out_prefix}_kernel_descriptors.csv", index=False)

    return {
        "summary_csv": summary_path,
        "envelopes_json": envelopes_path,
        "best_assignments": [f"{out_prefix}_best_assignments_V{V}.csv" for V in sorted(best_assignments.keys())],
        "kernel_descriptors_csv": f"{out_prefix}_kernel_descriptors.csv",
    }

def run_dp_selection(df: pd.DataFrame, out_prefix: str, prune: bool = False, V_values: List[int] = [2,3,4,5]):
    df_filtered = apply_filters(df, FILTER_EQ)
    shapes, kernels = identify_shapes_and_kernels(df_filtered)
    df_ids = attach_ids(df_filtered, shapes, kernels)
    T = build_time_matrix(df_ids, num_kernels=len(kernels), num_shapes=len(shapes))
    LOP = times_to_lop(T, eps=0.0)
    LOP_p, kept_idxs = dominance_prune(LOP, apply=prune)
    S = LOP_p.shape[1] if LOP_p.ndim == 2 else 0
    weights = np.ones(S, dtype=float)
    summary_rows, envelopes, best_assignments = enumerate_envelopes(LOP_p, V_values=V_values, weights=weights)
    return save_results(out_prefix, summary_rows, envelopes, best_assignments, shapes, kernels, kept_idxs)
