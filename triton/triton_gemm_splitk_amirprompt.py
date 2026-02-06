import math
import torch
import triton
import triton.language as tl
from triton.testing import do_bench  # :contentReference[oaicite:0]{index=0}

# ============================================================
# 1) Triton Split-K GEMM (shape-generic: M/N/K are runtime)
#    + Autotune with reset_to_zero for atomic_add outputs
# ============================================================

MATMUL_CONFIGS = [
    triton.Config({"BM": 128, "BN": 128, "BK": 32, "SPLIT_K": 1}, num_warps=8, num_stages=3),
    triton.Config({"BM": 128, "BN": 128, "BK": 32, "SPLIT_K": 2}, num_warps=8, num_stages=3),
    triton.Config({"BM": 128, "BN":  64, "BK": 32, "SPLIT_K": 4}, num_warps=4, num_stages=4),
    triton.Config({"BM":  64, "BN": 128, "BK": 32, "SPLIT_K": 4}, num_warps=4, num_stages=4),
    triton.Config({"BM":  64, "BN":  64, "BK": 32, "SPLIT_K": 8}, num_warps=4, num_stages=5),
]

@triton.autotune(
    configs=MATMUL_CONFIGS,
    key=["M", "N", "K"],
    reset_to_zero=["C_ptr"],  # required: kernel accumulates into C via atomic_add :contentReference[oaicite:1]{index=1}
)
@triton.jit
def gemm_splitk_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,  # runtime sizes (no tl.constexpr)
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
        k_mask = offs_k < k1  # CRITICAL: stay within this split's K-range

        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)

        acc += tl.dot(a, b)
        k += BK

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.atomic_add(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def triton_gemm_splitk(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor):
    """
    a: [M,K] fp16/bf16, b: [K,N] fp16/bf16
    out: [M,N] fp32 (recommended for atomic_add robustness)
    """
    assert a.is_cuda and b.is_cuda and out.is_cuda
    assert a.ndim == 2 and b.ndim == 2 and out.ndim == 2
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    assert out.shape == (M, N)
    assert out.dtype == torch.float32

    def grid(meta):
        return (triton.cdiv(M, meta["BM"]) * triton.cdiv(N, meta["BN"]), meta["SPLIT_K"])

    gemm_splitk_kernel[grid](
        a, b, out,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
    )


# ============================================================
# 2) Benchmarking helpers (Triton do_bench) + cuBLAS (torch)
# ============================================================

def tflops(M, N, K, ms):
    # GEMM FLOPs ~ 2*M*N*K
    return (2.0 * M * N * K) / (ms * 1e-3) / 1e12


@torch.no_grad()
def benchmark_one(M, N, K, dtype=torch.float16, device="cuda",
                  warmup_ms=25, rep_ms=100, reset_out_inside_bench=False):
    # Inputs
    a = torch.randn((M, K), device=device, dtype=dtype)
    b = torch.randn((K, N), device=device, dtype=dtype)

    # Outputs
    c_triton = torch.empty((M, N), device=device, dtype=torch.float32)
    c_cublas = torch.empty((M, N), device=device, dtype=dtype)

    # Warmup to trigger Triton autotune/compile outside timing
    c_triton.zero_()
    triton_gemm_splitk(a, b, c_triton)
    torch.cuda.synchronize()

    # --- Triton timing ---
    # NOTE: If reset_out_inside_bench=True, we include c_triton.zero_() time in the measurement.
    # If False, out will accumulate across iterations (wrong values) but timing is usually what you want.
    def triton_fn():
        if reset_out_inside_bench:
            c_triton.zero_()
        triton_gemm_splitk(a, b, c_triton)

    ms_triton = do_bench(triton_fn, warmup=warmup_ms, rep=rep_ms, return_mode="median")  # :contentReference[oaicite:2]{index=2}
    torch.cuda.synchronize()

    # --- cuBLAS timing (torch.matmul / torch.mm) ---
    # Use out= to avoid measuring allocation.
    def cublas_fn():
        torch.matmul(a, b, out=c_cublas)

    ms_cublas = do_bench(cublas_fn, warmup=warmup_ms, rep=rep_ms, return_mode="median")  # :contentReference[oaicite:3]{index=3}
    torch.cuda.synchronize()

    # Correctness check (optional quick)
    # Compare Triton (fp32 out) to torch fp32 reference
    ref = (a.float() @ b.float())
    max_abs = (c_triton - ref).abs().max().item()

    return {
        "M": M, "N": N, "K": K, "dtype": str(dtype).replace("torch.", ""),
        "triton_ms": float(ms_triton),
        "cublas_ms": float(ms_cublas),
        "triton_tflops": float(tflops(M, N, K, ms_triton)),
        "cublas_tflops": float(tflops(M, N, K, ms_cublas)),
        "triton_max_abs_err_vs_fp32ref": max_abs,
        "includes_out_zero_time_in_triton_ms": bool(reset_out_inside_bench),
    }


def print_results(rows):
    headers = [
        "M", "N", "K", "dtype",
        "Triton ms", "cuBLAS ms",
        "Triton TFLOP/s", "cuBLAS TFLOP/s",
        "speedup (cuBLAS/Triton)",
        "max_abs_err"
    ]
    print("\t".join(headers))
    for r in rows:
        speedup = r["cublas_ms"] / r["triton_ms"]
        print(
            f'{r["M"]}\t{r["N"]}\t{r["K"]}\t{r["dtype"]}\t'
            f'{r["triton_ms"]:.4f}\t{r["cublas_ms"]:.4f}\t'
            f'{r["triton_tflops"]:.2f}\t{r["cublas_tflops"]:.2f}\t'
            f'{speedup:.2f}\t'
            f'{r["triton_max_abs_err_vs_fp32ref"]:.3e}'
        )


if __name__ == "__main__":
    torch.manual_seed(0)

    # Shapes to benchmark (edit as you like)
    shapes = [
        (256, 256, 4096),
        (1024, 256, 2048),
        (128, 128, 8192),
        (777, 513, 1337),
    ]

    # If True: measured time includes out.zero_() each iteration (slower but correct per-iter outputs).
    # If False: best for kernel-only timing (outputs accumulate; ignore correctness during timing).
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

    print_results(rows)
