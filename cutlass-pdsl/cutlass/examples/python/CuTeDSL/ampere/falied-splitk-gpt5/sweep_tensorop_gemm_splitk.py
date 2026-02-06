"""
sweep_tensorop_gemm_splitk.py

Runs the split-K CuTe DSL tensorcore GEMM example across a sweep of tunable params
(CTA tiles, stages, atom layouts, memory layouts, split_k_list) for a LIST of problem sizes.
Only records rows that pass correctness checking.

Defaults:
  --problems_csv ./problems.csv
  --config       ./tune_config_splitk.yaml
  --out          ./sweep_results_splitk.csv
"""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

THIS_DIR = Path(__file__).parent
RUN_ONE = THIS_DIR / "run_one_config_splitk.py"
if not RUN_ONE.exists():
    raise FileNotFoundError(f"Missing helper script: {RUN_ONE}")

FIELDNAMES = [
    "M","N","K","L",
    "a_major","b_major","c_major",
    "cta_m","cta_n","cta_k","stages",
    "atom_m","atom_n","atom_k",
    "split_k",
    "avg_us","gflops",
]


def ensure_csv_with_header(path: Path, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()


def parse_cta_list(entries):
    out = []
    for item in entries:
        a = str(item).strip().lower().replace(" ", "")
        ms = [int(z) for z in (a.split('x') if 'x' in a else a.split(':'))]
        if len(ms) != 3:
            raise ValueError(f"Bad CTA shape '{item}' (want MxNxK)")
        out.append(tuple(ms))
    return out


def parse_atom_layouts(entries):
    out = []
    for item in entries:
        a = str(item).strip().lower().replace(" ", "")
        ms = [int(z) for z in (a.split('x') if 'x' in a else a.split(':'))]
        if len(ms) != 3:
            raise ValueError(f"Bad atom layout '{item}' (want MxNxK atoms)")
        out.append(tuple(ms))
    return out


def parse_layout_triplets(entries):
    valid_a = set(["m", "k"]) ; valid_b = set(["n", "k"]) ; valid_c = set(["n", "m"])
    out = []
    for trip in entries:
        t = str(trip).strip().lower()
        if len(t) != 3:
            raise ValueError("Layout triplet must be 3 letters (e.g., mnn)")
        a, b, c = t[0], t[1], t[2]
        if a not in valid_a or b not in valid_b or c not in valid_c:
            raise ValueError(f"Invalid layout triplet '{t}'")
        out.append((a, b, c))
    return out


def load_config(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    txt = path.read_text()
    if yaml is not None and (path.suffix in [".yml", ".yaml"]):
        cfg = yaml.safe_load(txt)
    else:
        cfg = json.loads(txt)
    return {
        "cta_list": parse_cta_list(cfg["cta_list"]),
        "stages": [int(s) for s in cfg["stages"]],
        "atom_layouts": parse_atom_layouts(cfg["atom_layouts"]),
        "layouts": parse_layout_triplets(cfg["layouts"]),
        "iters": int(cfg.get("iters", 50)),
        "warmup": int(cfg.get("warmup", 5)),
        "use_cold_l2": bool(cfg.get("use_cold_l2", False)),
        "timeout_sec": int(cfg.get("timeout_sec", 300)),
        "split_k_list": [int(v) for v in cfg.get("split_k_list", [1])],
    }


def load_problems(csv_path: Path):
    probs = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            probs.append((int(row["m"]), int(row["n"]), int(row["k"])))
    return probs


def gflops(M, N, K, us):
    if us <= 0:
        return float("nan")
    return (2.0 * M * N * K) / (us * 1e-6) / 1e9


def run_one_subprocess(cfg_row, iters, warmup, use_cold_l2, timeout_sec, skip_ref_check=False):
    args = [
        sys.executable, str(RUN_ONE),
        "--M", str(cfg_row["M"]),
        "--N", str(cfg_row["N"]),
        "--K", str(cfg_row["K"]),
        "--L", str(cfg_row["L"]),
        "--a_major", cfg_row["a_major"],
        "--b_major", cfg_row["b_major"],
        "--c_major", cfg_row["c_major"],
        "--cta_m", str(cfg_row["cta_m"]),
        "--cta_n", str(cfg_row["cta_n"]),
        "--cta_k", str(cfg_row["cta_k"]),
        "--stages", str(cfg_row["stages"]),
        "--atom_m", str(cfg_row["atom_m"]),
        "--atom_n", str(cfg_row["atom_n"]),
        "--atom_k", str(cfg_row["atom_k"]),
        "--split_k", str(cfg_row["split_k"]),
        "--iters", str(iters),
        "--warmup", str(warmup),
    ]
    if use_cold_l2: args.append("--use_cold_l2")
    if skip_ref_check: args.append("--skip_ref_check")

    try:
        proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout_sec)
        stdout = (proc.stdout or '').strip(); stderr = (proc.stderr or '').strip()
    except subprocess.TimeoutExpired:
        return {"ok": False, "kind": "fail", "error": f"timeout after {timeout_sec}s"}

    if stdout:
        try:
            return json.loads(stdout.splitlines()[-1])
        except json.JSONDecodeError:
            return {"ok": False, "kind": "fail", "error": f"bad-json: {stdout[-200:]} stderr: {stderr[-200:]}"}
    else:
        return {"ok": False, "kind": "fail", "error": f"empty-stdout rc={proc.returncode} stderr: {stderr[-200:]}"}


def main():
    p = argparse.ArgumentParser(description="Sweep split-K tuning params for CuTe tensorop GEMM over many problems")
    p.add_argument("--problems_csv", type=str, default=str(Path(__file__).with_name("problems.csv")))
    p.add_argument("--config", type=str, default=str(Path(__file__).with_name("tune_config_splitk.yaml")))
    p.add_argument("--out", type=str, default=str(Path(__file__).with_name("sweep_results_splitk.csv")))
    p.add_argument("--skip_ref_check", action="store_true")
    p.add_argument("--timeout_sec", type=int, default=None)
    args = p.parse_args()

    problems_csv = Path(args.problems_csv)
    cfg_path = Path(args.config)
    out_csv = Path(args.out)

    cfg = load_config(cfg_path)
    problems = load_problems(problems_csv)

    ensure_csv_with_header(out_csv, FIELDNAMES)

    tried = 0; succeeded = 0
    for (M, N, K) in problems:
        for (a_major, b_major, c_major) in cfg["layouts"]:
            for cta in cfg["cta_list"]:
                for stages in cfg["stages"]:
                    for atoms in cfg["atom_layouts"]:
                        for split_k in cfg["split_k_list"]:
                            tried += 1
                            cfg_row = {
                                "M": M, "N": N, "K": K, "L": 1,
                                "a_major": a_major, "b_major": b_major, "c_major": c_major,
                                "cta_m": cta[0], "cta_n": cta[1], "cta_k": cta[2],
                                "stages": stages,
                                "atom_m": atoms[0], "atom_n": atoms[1], "atom_k": atoms[2],
                                "split_k": split_k,
                            }
                            res = run_one_subprocess(
                                cfg_row,
                                iters=cfg["iters"],
                                warmup=cfg["warmup"],
                                use_cold_l2=cfg["use_cold_l2"],
                                timeout_sec=(args.timeout_sec if args.timeout_sec is not None else cfg["timeout_sec"]),
                                skip_ref_check=args.skip_ref_check,
                            )
                            if res.get("ok"):
                                elapsed = float(res["elapsed_us"]) ; succeeded += 1
                                row_out = { **cfg_row, "avg_us": elapsed, "gflops": gflops(M,N,K,elapsed) }
                                with out_csv.open("a", newline="") as f:
                                    w = csv.DictWriter(f, fieldnames=FIELDNAMES)
                                    w.writerow({k: row_out.get(k, "") for k in FIELDNAMES}); f.flush()
                                print(f"[OK] {cfg_row} -> {elapsed:.2f} us, {gflops(M,N,K,elapsed):.2f} GFLOPs")
                            else:
                                kind = res.get("kind", "fail"); err = res.get("error", "")
                                tag = "[skip]" if kind == "skip" else "[fail]"
                                print(f"{tag} {cfg_row} -> {err}")
    print(f"Completed {succeeded}/{tried} valid+checked configs. Results -> {out_csv}")


if __name__ == "__main__":
    main()
