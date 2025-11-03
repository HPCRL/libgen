#!/usr/bin/env python3
import argparse, csv, json
from pathlib import Path

try:
    import yaml
except Exception:
    yaml = None

import sys
sys.path.insert(0, str(Path(__file__).parent))
import tensorop_gemm_tunable as gemm_mod  # type: ignore


def gflops(M, N, K, us):
    if us <= 0:
        return float("nan")
    return (2.0 * M * N * K) / (us * 1e-6) / 1e9


def _as_list(v, default):
    if v is None:
        return list(default)
    if isinstance(v, (list, tuple)):
        return list(v)
    return [v]


def load_config(path: Path):
    txt = path.read_text()
    if yaml is not None and path.suffix.lower() in (".yml", ".yaml"):
        cfg = yaml.safe_load(txt)
    else:
        cfg = json.loads(txt)

    # Existing knobs (keep full lists)
    ctas = cfg.get("cta_list", [])
    stages = cfg.get("stages", [])
    atoms = cfg.get("atom_layouts", [])
    layouts = cfg.get("layouts", [])

    # New knobs
    lda_list = _as_list(cfg.get("lda_align_elems"), [1, 8, 16])
    ldb_list = _as_list(cfg.get("ldb_align_elems"), [1, 8, 16])
    ldc_list = _as_list(cfg.get("ldc_align_elems"), [1, 8, 16])
    ep_out_list = _as_list(cfg.get("epilogue_elems_per_access"), [4, 8])
    beta0_list = _as_list(cfg.get("beta_zero_special"), [True])

    # Baselines for "skip old configs" logic
    baselines = cfg.get("new_knob_baselines", {
        "lda_align_elems": 1,
        "ldb_align_elems": 1,
        "ldc_align_elems": 1,
        "epilogue_elems_per_access": 4,
        "beta_zero_special": True,
    })
    require_any_change = bool(cfg.get("require_any_new_knob_change", True))

    return {
        "cta_list": ctas,
        "stages": stages,
        "atom_layouts": atoms,
        "layouts": layouts,
        "lda_align_elems": lda_list,
        "ldb_align_elems": ldb_list,
        "ldc_align_elems": ldc_list,
        "epilogue_elems_per_access": ep_out_list,
        "beta_zero_special": beta0_list,
        "baselines": baselines,
        "require_any_change": require_any_change,
        "iters": int(cfg.get("iters", 50)),
        "warmup": int(cfg.get("warmup", 5)),
        "use_cold_l2": bool(cfg.get("use_cold_l2", False)),
    }


def parse_triplet(s: str):
    s = s.strip().lower().replace(" ", "")
    if "x" in s:
        a,b,c = s.split("x")
    else:
        a,b,c = s.split(",")
    return (int(a), int(b), int(c))


def parse_atom(s: str):
    return parse_triplet(s)


def parse_layout_triplet(s: str):
    t = s.strip().lower()
    assert len(t) == 3, f"layout triplet must be 3 letters (got {t})"
    a,b,c = t[0], t[1], t[2]
    assert a in ("m","k") and b in ("n","k") and c in ("n","m"), f"invalid layout '{t}'"
    return (a,b,c)


def load_problems(csv_path: Path):
    probs = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            probs.append((int(row["m"]), int(row["n"]), int(row["k"])))
    return probs


def main():
    p = argparse.ArgumentParser(description="Sweep existing knobs × NEW vectorization knobs, skipping old-default-only cases")
    p.add_argument("--problems_csv", type=str, default=str(Path(__file__).with_name("problems.csv")))
    p.add_argument("--config", type=str, default=str(Path(__file__).with_name("tune_config.yaml")))
    p.add_argument("--out", type=str, default=str(Path(__file__).with_name("sweep_results_new_align.csv")))
    p.add_argument("--skip_ref_check", action="store_true")
    args = p.parse_args()

    cfg = load_config(Path(args.config))
    problems = load_problems(Path(args.problems_csv))

    # Normalize string forms
    ctas = [parse_triplet(x) if isinstance(x, str) else tuple(x) for x in cfg["cta_list"]]
    atoms = [parse_atom(x) if isinstance(x, str) else tuple(x) for x in cfg["atom_layouts"]]
    layouts = [parse_layout_triplet(x) if isinstance(x, str) else tuple(x) for x in cfg["layouts"]]

    rows, tried, ok = [], 0, 0
    for (M, N, K) in problems:
        for (a_major, b_major, c_major) in layouts:
            for (cta_m, cta_n, cta_k) in ctas:
                for stages in cfg["stages"]:
                    for (atom_m, atom_n, atom_k) in atoms:
                        for lda_align in cfg["lda_align_elems"]:
                            for ldb_align in cfg["ldb_align_elems"]:
                                for ldc_align in cfg["ldc_align_elems"]:
                                    for ep_elems in cfg["epilogue_elems_per_access"]:
                                        for beta0 in cfg["beta_zero_special"]:
                                            # Skip if all new knobs equal their baselines and require change
                                            if cfg["require_any_change"]:
                                                base = cfg["baselines"]
                                                if (lda_align == base["lda_align_elems"] and
                                                    ldb_align == base["ldb_align_elems"] and
                                                    ldc_align == base["ldc_align_elems"] and
                                                    ep_elems == base["epilogue_elems_per_access"] and
                                                    bool(beta0) == bool(base["beta_zero_special"])):
                                                    # old-default-only combo -> skip
                                                    continue

                                            tried += 1
                                            meta = dict(
                                                M=M, N=N, K=K, L=1,
                                                a_major=a_major, b_major=b_major, c_major=c_major,
                                                cta_m=cta_m, cta_n=cta_n, cta_k=cta_k,
                                                stages=stages,
                                                atom_m=atom_m, atom_n=atom_n, atom_k=atom_k,
                                                lda_align_elems=lda_align,
                                                ldb_align_elems=ldb_align,
                                                ldc_align_elems=ldc_align,
                                                epilogue_elems_per_access=ep_elems,
                                                beta_zero_special=bool(beta0),
                                            )
                                            try:
                                                elapsed = gemm_mod.run(
                                                    a_major=a_major, b_major=b_major, c_major=c_major,
                                                    ab_dtype=gemm_mod.cutlass.Float16,
                                                    c_dtype=gemm_mod.cutlass.Float16,
                                                    acc_dtype=gemm_mod.cutlass.Float32,
                                                    mnkl=(M, N, K, 1),
                                                    atom_layout_mnk=(atom_m, atom_n, atom_k),
                                                    warmup_iterations=cfg["warmup"],
                                                    iterations=cfg["iters"],
                                                    skip_ref_check=args.skip_ref_check,
                                                    use_cold_l2=cfg["use_cold_l2"],
                                                    cta_tiler=(cta_m, cta_n, cta_k),
                                                    num_stages=stages,
                                                    # NEW knobs
                                                    lda_align_elems=lda_align,
                                                    ldb_align_elems=ldb_align,
                                                    ldc_align_elems=ldc_align,
                                                    epilogue_elems_per_access=ep_elems,
                                                    beta_zero_special=bool(beta0),
                                                )
                                                ok += 1
                                                rows.append({**meta, "avg_us": elapsed, "gflops": gflops(M, N, K, elapsed)})
                                                print(f"[OK] {meta} -> {elapsed:.2f} us, {gflops(M,N,K,elapsed):.2f} GFLOPs")
                                            except AssertionError as e:
                                                print(f"[skip] {meta} -> invalid combo: {e}")
                                            except Exception as e:
                                                print(f"[fail] {meta} -> {type(e).__name__}: {e}")

    # write only successful rows
    fieldnames = [
        "M","N","K","L",
        "a_major","b_major","c_major",
        "cta_m","cta_n","cta_k","stages",
        "atom_m","atom_n","atom_k",
        "lda_align_elems","ldb_align_elems","ldc_align_elems",
        "epilogue_elems_per_access","beta_zero_special",
        "avg_us","gflops",
    ]
    out_csv = Path(args.out)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"Completed {ok}/{len(rows) + (tried-ok)} valid runs. Results -> {out_csv}")


if __name__ == "__main__":
    main()
