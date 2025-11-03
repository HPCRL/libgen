#!/usr/bin/env python3
"""
tensorop_gemm_tunable.py

Adds NEW knobs (alignment-driven vectorization + epilogue vector width + beta0):
- lda_align_elems / ldb_align_elems / ldc_align_elems
- epilogue_elems_per_access
- beta_zero_special

Keeps your existing tunables: cta_tiler, num_stages, atom_layout_mnk, layouts.
Uses CUTLASS Python backend so this file is runnable as-is; if you prefer your
CuTe DSL builder, swap out 'build_gemm_op' and 'launch' accordingly, keeping the
same arguments and padded leading dimensions.
"""

from __future__ import annotations
import argparse, time
from typing import Tuple, Type

import numpy as np
import cutlass
from cutlass.backend import GemmOperationUniversal, TensorDescription, MathInstruction, TileDescription
from cutlass.backend.library import MathOperation
from cutlass.backend.utils.device import device_synchronize


def round_up(x: int, a: int) -> int:
    if a <= 0:
        return x
    return ((x + a - 1) // a) * a


def alloc_tensor(shape, dtype, init_zero=True):
    arr = np.zeros(shape, dtype=dtype) if init_zero else np.random.randn(*shape).astype(dtype)
    return cutlass.tensor.Tensor(arr, device="cuda")


class TensorOpGemm:
    def __init__(
        self,
        ab_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        atom_layout_mnk: Tuple[int, int, int],
        cta_tiler: tuple | None = None,
        num_stages: int | None = None,
        beta_zero_special: bool = True,
        lda_align_elems: int = 1,
        ldb_align_elems: int = 1,
        ldc_align_elems: int = 1,
        epilogue_elems_per_access: int = 4,
    ):
        self.ab_dtype = ab_dtype
        self.c_dtype = c_dtype
        self.acc_dtype = acc_dtype
        self.atom_layout_mnk = tuple(atom_layout_mnk)
        self.cta_tiler = cta_tiler if cta_tiler is not None else (128, 128, 32)
        self.num_stages = int(num_stages) if num_stages is not None else 3

        # NEW knobs
        self.beta_zero_special = bool(beta_zero_special)
        self.lda_align_elems = int(lda_align_elems)
        self.ldb_align_elems = int(ldb_align_elems)
        self.ldc_align_elems = int(ldc_align_elems)
        self.epilogue_elems_per_access = int(epilogue_elems_per_access)

        bM, bN, bK = self.cta_tiler
        assert bM % 16 == 0 and bN % 8 == 0 and bK % 16 == 0, \
            "CTA tile must be a multiple of mma.sync m16n8k16"
        am, an, ak = self.atom_layout_mnk
        assert am > 0 and an > 0 and ak > 0, "atom_layout_mnk must be positive"

    def build_gemm_op(self):
        # For Ampere FP16 TC
        A_desc = TensorDescription(self.ab_dtype, cutlass.layout.RowMajor)
        B_desc = TensorDescription(self.ab_dtype, cutlass.layout.ColumnMajor)
        C_desc = TensorDescription(self.c_dtype, cutlass.layout.RowMajor)

        instruction = MathInstruction(
            shape=[16, 8, 16],
            A=A_desc.element,
            B=B_desc.element,
            C=self.acc_dtype,
            opcode_class=MathOperation.multiply_add,
        )

        tile = TileDescription(
            threadblock_shape=list(self.cta_tiler),
            stages=self.num_stages,
            warp_count=[4, 2, 1],
            math_instruction=instruction
        )

        op = GemmOperationUniversal(
            arch=80,
            tile_description=tile,
            A=A_desc, B=B_desc, C=C_desc,
            element_epilogue=self.c_dtype,
            epilogue_functor=cutlass.epilogue.LinearCombination,
            epilogue_vector_length=self.epilogue_elems_per_access
        )
        return op

    def launch(self, *, A, B, C, alpha: float, beta: float,
               layouts, lda: int, ldb: int, ldc: int,
               mnk, warmup: int, iters: int,
               skip_ref_check: bool, use_cold_l2: bool) -> float:
        M, N, K = mnk
        a_major, b_major, c_major = layouts  # currently informational; tensors already padded

        op = self.build_gemm_op()
        problem_size = cutlass.gemm.GemmCoord(M, N, K)

        args = cutlass.gemm.GemmArguments(
            op, problem_size,
            A, B, C, C,
            alpha, beta,
            cutlass.gemm.GemmUniversalMode.Gemm
        )

        # Warmup
        for _ in range(max(0, warmup)):
            op.run(args)
        device_synchronize()

        # Time
        start = time.perf_counter_ns()
        for _ in range(max(1, iters)):
            op.run(args)
        device_synchronize()
        end = time.perf_counter_ns()

        return (end - start) / iters / 1e3


def run(
    a_major: str,
    b_major: str,
    c_major: str,
    ab_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    acc_dtype: Type[cutlass.Numeric],
    mnkl: Tuple[int, int, int, int],
    atom_layout_mnk: Tuple[int, int, int],
    warmup_iterations: int,
    iterations: int,
    skip_ref_check: bool,
    use_cold_l2: bool,
    cta_tiler=None,
    num_stages=None,
    beta_zero_special: bool = True,
    lda_align_elems: int = 1,
    ldb_align_elems: int = 1,
    ldc_align_elems: int = 1,
    epilogue_elems_per_access: int = 4,
) -> float:

    M, N, K, L = mnkl
    assert L == 1, "Batched L>1 not implemented in this sample"
    a_major = a_major.lower()
    b_major = b_major.lower()
    c_major = c_major.lower()
    assert a_major in ("m", "k")
    assert b_major in ("n", "k")
    assert c_major in ("n", "m")

    # Compute padded leading dims by major
    lda_unaligned = K if a_major == "m" else M
    ldb_unaligned = N if b_major == "n" else K
    ldc_unaligned = N if c_major == "n" else M

    lda = round_up(lda_unaligned, lda_align_elems)
    ldb = round_up(ldb_unaligned, ldb_align_elems)
    ldc = round_up(ldc_unaligned, ldc_align_elems)

    # Allocate tensors with padded LDs (keep logical M,N,K)
    # Row-major tensors: shape (rows, ld)
    # Col-major tensors: emulate via row-major with padded second dim; CUTLASS wrapper reads strides from array
    if a_major == 'm':  # A is MxK row-major with ld=lda
        A = alloc_tensor((M, lda), np.float16, init_zero=False)
    else:                # A is KxM col-major with ld=M -> emulate as (K, lda)
        A = alloc_tensor((K, lda), np.float16, init_zero=False)

    if b_major == 'n':  # B is KxN col-major with ld=ldb -> emulate as (K, ldb)
        B = alloc_tensor((K, ldb), np.float16, init_zero=False)
    else:                # B is NxK row-major with ld=ldb -> (N, ldb)
        B = alloc_tensor((N, ldb), np.float16, init_zero=False)

    if c_major == 'n':  # C is MxN row-major with ld=ldc
        C = alloc_tensor((M, ldc), np.float16, init_zero=True)
    else:                # C is NxM col-major -> emulate as (N, ldc)
        C = alloc_tensor((N, ldc), np.float16, init_zero=True)

    alpha = 1.0
    beta = 0.0 if beta_zero_special else 1.0

    gemm = TensorOpGemm(
        ab_dtype, c_dtype, acc_dtype, atom_layout_mnk,
        cta_tiler=cta_tiler, num_stages=num_stages,
        beta_zero_special=beta_zero_special,
        lda_align_elems=lda_align_elems,
        ldb_align_elems=ldb_align_elems,
        ldc_align_elems=ldc_align_elems,
        epilogue_elems_per_access=epilogue_elems_per_access,
    )

    avg_us = gemm.launch(
        A=A, B=B, C=C,
        alpha=alpha, beta=beta,
        layouts=(a_major, b_major, c_major),
        lda=lda, ldb=ldb, ldc=ldc,
        mnk=(M, N, K),
        warmup=warmup_iterations,
        iters=iterations,
        skip_ref_check=skip_ref_check,
        use_cold_l2=use_cold_l2,
    )
    return avg_us


# CLI passthrough
def parse_triplet(s: str):
    return tuple(int(x.strip()) for x in s.split(","))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--a_major", default="m")
    parser.add_argument("--b_major", default="n")
    parser.add_argument("--c_major", default="n")
    parser.add_argument("--mnkl", type=parse_triplet, default=(4096,4096,4096,1))
    parser.add_argument("--atom_layout_mnk", type=parse_triplet, default=(2,2,1))
    parser.add_argument("--cta_tiler", type=parse_triplet, default=(128,128,32))
    parser.add_argument("--num_stages", type=int, default=3)

    parser.add_argument("--beta_zero_special", action="store_true")
    parser.add_argument("--lda_align_elems", type=int, default=1)
    parser.add_argument("--ldb_align_elems", type=int, default=1)
    parser.add_argument("--ldc_align_elems", type=int, default=1)
    parser.add_argument("--epilogue_elems_per_access", type=int, default=4)

    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup_iterations", type=int, default=5)
    parser.add_argument("--skip_ref_check", action="store_true")
    parser.add_argument("--use_cold_l2", action="store_true")
    args = parser.parse_args()

    elapsed = run(
        a_major=args.a_major, b_major=args.b_major, c_major=args.c_major,
        ab_dtype=cutlass.Float16, c_dtype=cutlass.Float16, acc_dtype=cutlass.Float32,
        mnkl=args.mnkl, atom_layout_mnk=args.atom_layout_mnk,
        warmup_iterations=args.warmup_iterations, iterations=args.iterations,
        skip_ref_check=args.skip_ref_check, use_cold_l2=args.use_cold_l2,
        cta_tiler=args.cta_tiler, num_stages=args.num_stages,
        beta_zero_special=args.beta_zero_special,
        lda_align_elems=args.lda_align_elems,
        ldb_align_elems=args.ldb_align_elems,
        ldc_align_elems=args.ldc_align_elems,
        epilogue_elems_per_access=args.epilogue_elems_per_access,
    )
    print(f"Elapsed (avg): {elapsed:.2f} us")
