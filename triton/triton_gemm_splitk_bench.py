import csv
import itertools
import torch
import triton
import triton.language as tl
from triton.testing import do_bench

# ============================================================
# Autotune sweep space (as requested)
#   - CTA = BM x BN x BK from cta_list
#   - num_stages in [3,4,5,6]
#   - SPLIT_K swept (atomic add accumulation)
#   - num_warps swept
# ============================================================

CTA_LIST_STR = [
    # 16x*
    "16x8x16","16x8x32","16x8x64","16x8x128","16x8x256",
    "16x16x16","16x16x32","16x16x64","16x16x128","16x16x256",
    "16x32x16","16x32x32","16x32x64","16x32x128","16x32x256",
    "16x64x16","16x64x32","16x64x64","16x64x128","16x64x256",
    "16x128x16","16x128x32","16x128x64","16x128x128","16x128x256",
    "16x256x16","16x256x32","16x256x64","16x256x128","16x256x256",
    # 32x*
    "32x8x16","32x8x32","32x8x64","32x8x128","32x8x256",
    "32x16x16","32x16x32","32x16x64","32x16x128","32x16x256",
    "32x32x16","32x32x32","32x32x64","32x32x128","32x32x256",
    "32x64x16","32x64x32","32x64x64","32x64x128","32x64x256",
    "32x128x16","32x128x32","32x128x64","32x128x128","32x128x256",
    "32x256x16","32x256x32","32x256x64","32x256x128","32x256x256",
    # 64x*
    "64x8x16","64x8x32","64x8x64","64x8x128","64x8x256",
    "64x16x16","64x16x32","64x16x64","64x16x128","64x16x256",
    "64x32x16","64x32x32","64x32x64","64x32x128","64x32x256",
    "64x64x16","64x64x32","64x64x64","64x64x128","64x64x256",
    "64x128x16","64x128x32","64x128x64","64x128x128","64x128x256",
    "64x256x16","64x256x32","64x256x64","64x256x128","64x256x256",
    # 128x*
    "128x8x16","128x8x32","128x8x64","128x8x128","128x8x256",
    "128x16x16","128x16x32","128x16x64","128x16x128","128x16x256",
    "128x32x16","128x32x32","128x32x64","128x32x128","128x32x256",
    "128x64x16","128x64x32","128x64x64","128x64x128","128x64x256",
    "128x128x16","128x128x32","128x128x64","128x128x128","128x128x256",
    "128x256x16","128x256x32","128x256x64","128x256x128","128x256x256",
    # 256x*
    "256x8x16","256x8x32","256x8x64","256x8x128","256x8x256",
    "256x16x16","256x16x32","256x16x64","256x16x128","256x16x256",
    "256x32x16","256x32x32","256x32x64","256x32x128","256x32x256",
    "256x64x16","256x64x32","256x64x64","256x64x128","256x64x256",
    "256x128x16","256x128x32","256x128x64","256x128x128","256x128x256",
    "256x256x16","256x256x32","256x256x64","256x256x128","256x256x256",
]

STAGES_LIST = [3, 4, 5, 6]
SPLITK_LIST = [1, 2, 4, 8, 16]          # edit if you want to go bigger
WARPS_LIST = [1, 2, 4, 8]               # common Triton choices


def parse_cta(s: str):
    bm, bn, bk = s.split("x")
    return int(bm), int(bn), int(bk)


def build_autotune_configs():
    cfgs = []
    for cta_str, stages, split_k, warps in itertools.product(
        CTA_LIST_STR, STAGES_LIST, SPLITK_LIST, WARPS_LIST
    ):
        BM, BN, BK = parse_cta(cta_str)

        # Optional pruning: SPLIT_K shouldn't exceed K in practice, but K is runtime here.
        # We'll keep everything; autotune will select what's best for each (M,N,K) key.

        cfgs.append(
            triton.Config(
                {"BM": BM, "BN": BN, "BK": BK, "SPLIT_K": split_k},
                num_warps=warps,
                num_stages=stages,
            )
        )
    return cfgs


MATMUL_CONFIGS = build_autotune_configs()

# ============================================================
# Kernel (masked, shape-generic)
# ============================================================

@triton.autotune(
    configs=MATMUL_CONFIGS,
    key=["M", "N", "K"],
    reset_to_zero=["C_ptr"],  # critical for atomic_add accumulation across configs
)
@triton.jit
def gemm_splitk_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_mn = tl.program_id(0)
    pid_k  = tl.program_id(1)

    grid_n = tl.cdiv(N, BN)
    pid_m = pid_mn // grid_n
    pid_n = pid_mn %  grid_n

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)

    # Split-K partition of K
    k_per = tl.cdiv(K, SPLIT_K)
    k0 = pid_k * k_per
    k1 = tl.minimum(K, (pid_k + 1) * k_per)

    acc = tl.zeros((BM, BN), dtype=tl.float32)

    k = k0
    while k < k1:
        offs_k = k + tl.arange(0, BK)
        k_mask = offs_k < k1  # CRITICAL: within this split

        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)

        acc += tl.dot(a, b)
        k += BK

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.atomic_add(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def triton_gemm_splitk(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    assert out.shape == (M, N)

    def grid(meta):
        return (triton.cdiv(M, meta["BM"]) * triton.cdiv(N, meta["BN"]),
                meta["SPLIT_K"])

    gemm_splitk_kernel[grid](
        a, b, out,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
    )


# ============================================================
# Benchmark + CSV
# ============================================================

def tflops(M, N, K, ms):
    return (2.0 * M * N * K) / (ms * 1e-3) / 1e12


@torch.no_grad()
def benchmark_one(M, N, K, dtype=torch.float16, device="cuda",
                  warmup_ms=25, rep_ms=200, reset_out_inside_bench=False):
    a = torch.randn((M, K), device=device, dtype=dtype)
    b = torch.randn((K, N), device=device, dtype=dtype)

    c_triton = torch.empty((M, N), device=device, dtype=torch.float32)
    c_cublas = torch.empty((M, N), device=device, dtype=dtype)

    # Warmup triggers autotune/compilation outside timing
    c_triton.zero_()
    triton_gemm_splitk(a, b, c_triton)
    torch.cuda.synchronize()

    def triton_fn():
        if reset_out_inside_bench:
            c_triton.zero_()
        triton_gemm_splitk(a, b, c_triton)

    ms_triton = do_bench(triton_fn, warmup=warmup_ms, rep=rep_ms, return_mode="median")
    torch.cuda.synchronize()

    def cublas_fn():
        torch.matmul(a, b, out=c_cublas)

    ms_cublas = do_bench(cublas_fn, warmup=warmup_ms, rep=rep_ms, return_mode="median")
    torch.cuda.synchronize()

    # Correctness metric vs fp32 ref (outside timing)
    ref = a.float() @ b.float()
    max_abs = (c_triton - ref).abs().max().item()

    return {
        "M": M, "N": N, "K": K,
        "dtype": str(dtype).replace("torch.", ""),
        "num_autotune_configs": len(MATMUL_CONFIGS),
        "reset_out_inside_bench": int(reset_out_inside_bench),
        "triton_ms_median": float(ms_triton),
        "cublas_ms_median": float(ms_cublas),
        "triton_tflops": float(tflops(M, N, K, ms_triton)),
        "cublas_tflops": float(tflops(M, N, K, ms_cublas)),
        "speedup_cublas_over_triton": float(ms_cublas / ms_triton),
        "triton_max_abs_err_vs_fp32ref": float(max_abs),
    }


def write_csv(rows, path):
    if not rows:
        raise ValueError("No rows to write")
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    torch.manual_seed(0)

    # WARNING: This space is enormous:
    # 150 CTAs * 4 stages * 5 split-k * 4 warps = 12,000 configs per key!
    print(f"Autotune configs: {len(MATMUL_CONFIGS)}")

    shapes = [
        (256, 256, 4096),
        (1024, 256, 2048),
        (128, 128, 8192),
        (777, 513, 1337),
    ]

    reset_out_inside_bench = False

    rows = []
    for (M, N, K) in shapes:
        rows.append(
            benchmark_one(
                M, N, K,
                dtype=torch.float16,
                warmup_ms=25,
                rep_ms=200,
                reset_out_inside_bench=reset_out_inside_bench,
            )
        )

    out_csv = "new_bench_triton_cta_stages_splitk_warps_vs_cublas.csv"
    write_csv(rows, out_csv)
    print(f"Wrote {len(rows)} rows to {out_csv}")
