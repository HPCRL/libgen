# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import argparse
import math
import time
from typing import Tuple, Type

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
from cutlass.cute.runtime import from_dlpack

"""
A split-K dense GEMM (C = A * B) for the NVIDIA Ampere architecture using CUTE DSL.

Split-K parallelism: Instead of having each thread block compute an entire output tile,
we divide the K dimension across multiple thread blocks. Each block computes a partial
sum for its assigned K-slice, and the results are then reduced.

This is particularly beneficial for GEMM problems with small M and/or N dimensions but
large K dimension, where the standard approach would have insufficient parallelism.

The kernel works in two phases:
1. Partial GEMM: Each thread block computes a partial result for a subset of K
2. Reduction: Partial results are accumulated into the final output

To run this example:

.. code-block:: bash

    python tensorop_gemm_splitk.py                                           \\
      --mnkl 64,64,8192,1 --atom_layout_mnk 2,2,1                           \\
      --ab_dtype Float16                                                    \\
      --c_dtype Float16 --acc_dtype Float32                                 \\
      --a_major m --b_major n --c_major n                                   \\
      --split_k 16

"""


class TensorOpGemmSplitK:
    def __init__(
        self,
        ab_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        atom_layout_mnk: Tuple[int, int, int],
        split_k: int = 1,
        cta_tiler: tuple | None = None,
        num_stages: int | None = None,
    ):
        self.ab_dtype = ab_dtype
        self.c_dtype = c_dtype
        self.acc_dtype = acc_dtype
        # Tunable kernel shape and pipeline stages
        self.cta_tiler = cta_tiler if cta_tiler is not None else (128, 128, 32)
        self.num_stages = int(num_stages) if num_stages is not None else 3
        self.atom_layout_mnk = atom_layout_mnk
        self.split_k = int(split_k)
        atom_lay_M, atom_lay_N, atom_lay_K = self.atom_layout_mnk
        self.num_threads = atom_lay_M * atom_lay_N * atom_lay_K * 32

        self.bM, self.bN, self.bK = self.cta_tiler
        self.mma_inst_shape = (16, 8, 16)
        mmaM, mmaN, mmaK = self.mma_inst_shape

        assert (
            self.bM % (atom_lay_M * mmaM) == 0
        ), "bM must be divisible by MMA instruction"
        assert (
            self.bN % (atom_lay_N * mmaN) == 0
        ), "bN must be divisible by MMA instruction"
        assert atom_lay_K == 1, "this example does not support atom layout K > 1"
        assert self.bK % mmaK == 0, "bK must be divisible by MMA instruction"
        assert self.num_stages >= 3, "num_stages must be greater than or equal to 3"
        assert self.split_k >= 1, "split_k must be at least 1"

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        mWorkspace: cute.Tensor,  # (M, N, L, split_k) in accumulator precision
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        # For split-K, we need a workspace to store partial results
        # The workspace is passed in as a pre-allocated tensor
        
        self.a_major_mode = utils.LayoutEnum.from_tensor(mA)
        self.b_major_mode = utils.LayoutEnum.from_tensor(mB)
        self.c_major_mode = utils.LayoutEnum.from_tensor(mC)

        M, N, L = mC.shape

        # ///////////////////////////////////////////////////////////////////////////////
        # Shared memory layout (same as regular version)
        # ///////////////////////////////////////////////////////////////////////////////

        ab_copy_bits = 128
        sA_layout = self._make_smem_layout_AB(
            mA.element_type,
            self.a_major_mode,
            ab_copy_bits,
            (self.cta_tiler[0], self.cta_tiler[2], self.num_stages),
        )
        sB_layout = self._make_smem_layout_AB(
            mB.element_type,
            self.b_major_mode,
            ab_copy_bits,
            (self.cta_tiler[1], self.cta_tiler[2], self.num_stages),
        )

        smem_size = cute.size_in_bytes(mA.element_type, sA_layout) + cute.size_in_bytes(mB.element_type, sB_layout)

        # ///////////////////////////////////////////////////////////////////////////////
        # Tiled copy (same as regular version)
        # ///////////////////////////////////////////////////////////////////////////////

        atom_async_copy = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
            ),
            mA.element_type,
            num_bits_per_copy=ab_copy_bits,
        )

        tiled_copy_A = self._make_gmem_tiled_copy_AB(
            atom_async_copy, mA.element_type, self.a_major_mode, ab_copy_bits
        )
        tiled_copy_B = self._make_gmem_tiled_copy_AB(
            atom_async_copy, mB.element_type, self.b_major_mode, ab_copy_bits
        )

        # ///////////////////////////////////////////////////////////////////////////////
        # Tiled MMA
        # ///////////////////////////////////////////////////////////////////////////////

        op = cute.nvgpu.warp.MmaF16BF16Op(
            self.ab_dtype, self.acc_dtype, self.mma_inst_shape
        )

        permutation_mnk = (
            self.atom_layout_mnk[0] * self.mma_inst_shape[0],
            self.atom_layout_mnk[1] * self.mma_inst_shape[1] * 2,
            self.atom_layout_mnk[2] * self.mma_inst_shape[2],
        )

        tC = cute.make_layout(self.atom_layout_mnk)
        tiled_mma = cute.make_tiled_mma(
            op,
            tC,
            permutation_mnk=permutation_mnk,
        )

        # Grid dimensions: M tiles x N tiles x (L batches * split_k slices)
        grid_dim_m = (M + self.bM - 1) // self.bM
        grid_dim_n = (N + self.bN - 1) // self.bN
        grid_dim_split = L * self.split_k
        
        # Rasterization for better data reuse
        raster_factor = 1
        if grid_dim_n > 5:
            raster_factor = 8
        elif grid_dim_n > 2:
            raster_factor = 4
        elif grid_dim_n > 1:
            raster_factor = 2
            
        rasterization_remap_grid_dim = (
            grid_dim_m * raster_factor,
            (grid_dim_n + raster_factor - 1) // raster_factor,
            grid_dim_split,
        )

        # Launch the split-K kernel
        self.kernel_splitk(
            mA,
            mB,
            mWorkspace,
            sA_layout,
            sB_layout,
            tiled_copy_A,
            tiled_copy_B,
            tiled_mma,
            raster_factor,
        ).launch(
            grid=rasterization_remap_grid_dim,
            block=[self.num_threads, 1, 1],
            smem=smem_size,
        )
        
        # Launch reduction kernel
        self.kernel_reduce(
            mWorkspace,
            mC,
            epilogue_op,
        ).launch(
            grid=((M + 15) // 16, (N + 15) // 16, L),
            block=[256, 1, 1],
            smem=0,
        )

    @cute.kernel
    def kernel_splitk(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mWorkspace: cute.Tensor,  # (M, N, L, split_k)
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        rasterization_factor: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()
        
        # Decode the z dimension: batch index and split_k index
        batch_idx = bidz // self.split_k
        split_idx = bidz % self.split_k
        
        # Compute the M, N tile indices
        grid_dim_m = (mA.shape[0] + self.bM - 1) // self.bM
        grid_dim_n = (mB.shape[0] + self.bN - 1) // self.bN
        
        offset_tile_x, offset_tile_y = self.raster_tile(
            bidx, bidy, rasterization_factor
        )
        
        # Early exit if CTA is out of range
        if grid_dim_m <= offset_tile_x or grid_dim_n <= offset_tile_y:
            pass
        else:
            # Compute K range for this split
            K = mA.shape[1]
            k_per_split = (K + self.split_k - 1) // self.split_k
            k_start = split_idx * k_per_split
            k_end = min(k_start + k_per_split, K)
            k_tiles = (k_end - k_start + self.bK - 1) // self.bK
            
            if k_start >= K:
                pass  # This split is out of range
            else:
                tiler_coord = (offset_tile_x, offset_tile_y, None)
                
                # Get tiles for this thread block
                gA_full = cute.local_tile(
                    mA[None, None, batch_idx],
                    tiler=self.cta_tiler,
                    coord=tiler_coord,
                    proj=(1, None, 1),
                )
                gB_full = cute.local_tile(
                    mB[None, None, batch_idx],
                    tiler=self.cta_tiler,
                    coord=tiler_coord,
                    proj=(None, 1, 1),
                )
                
                # Adjust pointers for this K split
                k_tile_start = k_start // self.bK
                gA = cute.domain_offset((0, k_start, 0), gA_full)
                gB = cute.domain_offset((0, k_start, 0), gB_full)
                
                # Handle residual in first tile
                residual_k = k_start - k_tile_start * self.bK
                if residual_k > 0:
                    gA = cute.domain_offset((0, -residual_k, 0), gA)
                    gB = cute.domain_offset((0, -residual_k, 0), gB)
                
                gA = cute.make_tensor(gA.iterator.align(16), gA.layout)
                gB = cute.make_tensor(gB.iterator.align(16), gB.layout)
                
                # Identity tensors for predication
                mcA = cute.make_identity_tensor(mA.layout.shape)
                mcB = cute.make_identity_tensor(mB.layout.shape)
                cA_full = cute.local_tile(
                    mcA[None, None, batch_idx],
                    tiler=self.cta_tiler,
                    coord=tiler_coord,
                    proj=(1, None, 1),
                )
                cB_full = cute.local_tile(
                    mcB[None, None, batch_idx],
                    tiler=self.cta_tiler,
                    coord=tiler_coord,
                    proj=(None, 1, 1),
                )
                
                cA = cute.domain_offset((0, k_start, 0), cA_full)
                cB = cute.domain_offset((0, k_start, 0), cB_full)
                if residual_k > 0:
                    cA = cute.domain_offset((0, -residual_k, 0), cA)
                    cB = cute.domain_offset((0, -residual_k, 0), cB)
                
                # Shared memory allocation
                smem = cutlass.utils.SmemAllocator()
                sA = smem.allocate_tensor(mA.element_type, sA_layout, 16)
                sB = smem.allocate_tensor(mB.element_type, sB_layout, 16)
                
                # Partition for copies
                thr_copy_A = tiled_copy_A.get_slice(tidx)
                thr_copy_B = tiled_copy_B.get_slice(tidx)
                tAgA = thr_copy_A.partition_S(gA)
                tAsA = thr_copy_A.partition_D(sA)
                tBgB = thr_copy_B.partition_S(gB)
                tBsB = thr_copy_B.partition_D(sB)
                
                tAcA = thr_copy_A.partition_S(cA)
                tBcB = thr_copy_B.partition_S(cB)
                
                # Predication tensors
                tApA = cute.make_fragment(
                    cute.make_layout(
                        (
                            tAgA.shape[0][1],
                            cute.size(tAgA, mode=[1]),
                            cute.size(tAgA, mode=[2]),
                        ),
                        stride=(cute.size(tAgA, mode=[1]), 1, 0),
                    ),
                    cutlass.Boolean,
                )
                tBpB = cute.make_fragment(
                    cute.make_layout(
                        (
                            tBsB.shape[0][1],
                            cute.size(tBsB, mode=[1]),
                            cute.size(tBsB, mode=[2]),
                        ),
                        stride=(cute.size(tBsB, mode=[1]), 1, 0),
                    ),
                    cutlass.Boolean,
                )
                
                # Set predicates
                for rest_v in range(tApA.shape[0]):
                    for m in range(tApA.shape[1]):
                        tApA[rest_v, m, 0] = cute.elem_less(
                            tAcA[(0, rest_v), m, 0, 0][0], mA.shape[0]
                        )
                for rest_v in range(tBpB.shape[0]):
                    for n in range(tBpB.shape[1]):
                        tBpB[rest_v, n, 0] = cute.elem_less(
                            tBcB[(0, rest_v), n, 0, 0][0], mB.shape[0]
                        )
                
                # Prefetch prologue
                tAsA.fill(0)
                tBsB.fill(0)
                cute.arch.sync_threads()
                
                num_smem_stages = cute.size(tAsA, mode=[3])
                k_tile_index = cutlass.Int32(0)
                
                # Load first k-tile
                for k in range(tApA.shape[2]):
                    if k_tile_index < k_tiles:
                        k_global = k_start + k_tile_index * self.bK + k * (self.bK // tApA.shape[2])
                        if k_global < k_end:
                            cute.copy(
                                tiled_copy_A,
                                tAgA[None, None, k, k_tile_index],
                                tAsA[None, None, k, 0],
                                pred=tApA[None, None, k],
                            )
                
                for k in range(tBpB.shape[2]):
                    if k_tile_index < k_tiles:
                        k_global = k_start + k_tile_index * self.bK + k * (self.bK // tBpB.shape[2])
                        if k_global < k_end:
                            cute.copy(
                                tiled_copy_B,
                                tBgB[None, None, k, k_tile_index],
                                tBsB[None, None, k, 0],
                                pred=tBpB[None, None, k],
                            )
                k_tile_index = k_tile_index + 1
                cute.arch.cp_async_commit_group()
                
                # Prefetch remaining stages
                for k_tile in range(1, min(num_smem_stages - 1, k_tiles)):
                    if k_tile_index >= k_tiles:
                        tApA.fill(0)
                        tBpB.fill(0)
                    cute.copy(
                        tiled_copy_A,
                        tAgA[None, None, None, k_tile_index],
                        tAsA[None, None, None, k_tile],
                        pred=tApA,
                    )
                    cute.copy(
                        tiled_copy_B,
                        tBgB[None, None, None, k_tile_index],
                        tBsB[None, None, None, k_tile],
                        pred=tBpB,
                    )
                    k_tile_index = k_tile_index + 1
                    cute.arch.cp_async_commit_group()
                
                # MMA setup
                thr_mma = tiled_mma.get_slice(tidx)
                tCsA = thr_mma.partition_A(sA)
                tCsB = thr_mma.partition_B(sB)
                tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
                tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
                
                # Get output tile in workspace
                gWork = cute.local_tile(
                    mWorkspace[None, None, batch_idx, split_idx],
                    tiler=(self.bM, self.bN, 1),
                    coord=(offset_tile_x, offset_tile_y, None),
                    proj=(1, 1, None),
                )
                tCgWork = thr_mma.partition_C(gWork)
                tCrC = tiled_mma.make_fragment_C(tCgWork)
                tCrC.fill(0.0)
                
                # Copy atoms for shared memory
                atom_copy_s2r_A = cute.make_copy_atom(
                    cute.nvgpu.warp.LdMatrix8x8x16bOp(
                        self.a_major_mode != utils.LayoutEnum.ROW_MAJOR, 4
                    ),
                    mA.element_type,
                )
                atom_copy_s2r_B = cute.make_copy_atom(
                    cute.nvgpu.warp.LdMatrix8x8x16bOp(
                        self.b_major_mode != utils.LayoutEnum.ROW_MAJOR, 4
                    ),
                    mB.element_type,
                )
                
                tiled_copy_s2r_A = cute.make_tiled_copy_A(atom_copy_s2r_A, tiled_mma)
                tiled_copy_s2r_B = cute.make_tiled_copy_B(atom_copy_s2r_B, tiled_mma)
                
                thr_copy_ldmatrix_A = tiled_copy_s2r_A.get_slice(tidx)
                thr_copy_ldmatrix_B = tiled_copy_s2r_B.get_slice(tidx)
                tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(sA)
                tCrA_copy_view = thr_copy_ldmatrix_A.retile(tCrA)
                tCsB_copy_view = thr_copy_ldmatrix_B.partition_S(sB)
                tCrB_copy_view = thr_copy_ldmatrix_B.retile(tCrB)
                
                smem_pipe_read = 0
                smem_pipe_write = num_smem_stages - 1
                
                tCsA_p = tCsA_copy_view[None, None, None, smem_pipe_read]
                tCsB_p = tCsB_copy_view[None, None, None, smem_pipe_read]
                
                # Prefetch register pipeline
                num_k_block = cute.size(tCrA, mode=[2])
                if num_k_block > 1:
                    cute.arch.cp_async_wait_group(num_smem_stages - 2)
                    cute.arch.sync_threads()
                    cute.copy(
                        tiled_copy_s2r_A,
                        tCsA_p[None, None, 0],
                        tCrA_copy_view[None, None, 0],
                    )
                    cute.copy(
                        tiled_copy_s2r_B,
                        tCsB_p[None, None, 0],
                        tCrB_copy_view[None, None, 0],
                    )
                
                # Main loop over K tiles
                for k_tile in range(k_tiles):
                    for k_block in cutlass.range(num_k_block, unroll_full=True):
                        if k_block == num_k_block - 1:
                            tCsA_p = tCsA_copy_view[None, None, None, smem_pipe_read]
                            tCsB_p = tCsB_copy_view[None, None, None, smem_pipe_read]
                            cute.arch.cp_async_wait_group(num_smem_stages - 2)
                            cute.arch.sync_threads()
                        
                        # Load next k_block
                        k_block_next = (k_block + 1) % num_k_block
                        cute.copy(
                            tiled_copy_s2r_A,
                            tCsA_p[None, None, k_block_next],
                            tCrA_copy_view[None, None, k_block_next],
                        )
                        cute.copy(
                            tiled_copy_s2r_B,
                            tCsB_p[None, None, k_block_next],
                            tCrB_copy_view[None, None, k_block_next],
                        )
                        
                        # Fetch next tile
                        if k_block == 0:
                            if k_tile + num_smem_stages - 1 < k_tiles:
                                cute.copy(
                                    tiled_copy_A,
                                    tAgA[None, None, None, k_tile_index],
                                    tAsA[None, None, None, smem_pipe_write],
                                    pred=tApA,
                                )
                        
                        # GEMM
                        cute.gemm(
                            tiled_mma,
                            tCrC,
                            tCrA[None, None, k_block],
                            tCrB[None, None, k_block],
                            tCrC,
                        )
                        
                        # Fetch B and update pipeline
                        if k_block == 0:
                            if k_tile + num_smem_stages - 1 < k_tiles:
                                cute.copy(
                                    tiled_copy_B,
                                    tBgB[None, None, None, k_tile_index],
                                    tBsB[None, None, None, smem_pipe_write],
                                    pred=tBpB,
                                )
                            k_tile_index = k_tile_index + 1
                            cute.arch.cp_async_commit_group()
                            smem_pipe_write = smem_pipe_read
                            smem_pipe_read = smem_pipe_read + 1
                            if smem_pipe_read == num_smem_stages:
                                smem_pipe_read = 0
                
                cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()
                
                # Write partial results to workspace (no epilogue yet)
                # Direct copy from registers to global memory
                cute.autovec_copy(tCrC, tCgWork)
        return

    @cute.kernel
    def kernel_reduce(
        self,
        mWorkspace: cute.Tensor,  # (M, N, L, split_k)
        mC: cute.Tensor,           # (M, N, L)
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """Reduce partial results and apply epilogue."""
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()
        
        # Each thread handles multiple elements
        M, N, L, _ = mWorkspace.shape
        threads_per_block = 256
        
        # Compute which output elements this thread handles
        tile_m = bidx * 16
        tile_n = bidy * 16
        batch = bidz
        
        # Each thread processes elements in a 16x16 tile
        for local_m in range(16):
            m = tile_m + local_m
            if cute.elem_less(m, M):
                for local_n in range(tidx, 16, threads_per_block):
                    n = tile_n + local_n
                    if cute.elem_less(n, N):
                        # Sum across split_k dimension
                        acc = self.acc_dtype(0.0)
                        for s in range(self.split_k):
                            acc = acc + mWorkspace[m, n, batch, s]
                        
                        # Apply epilogue and convert to output type
                        result = epilogue_op(acc).to(self.c_dtype)
                        mC[m, n, batch] = result
        return

    def _make_smem_layout_AB(self, dtype, major_mode, copy_bits, smem_tiler):
        major_mode_size = (
            smem_tiler[1] if major_mode == utils.LayoutEnum.ROW_MAJOR else smem_tiler[0]
        )
        major_mode_size = 64 if major_mode_size >= 64 else major_mode_size

        swizzle_bits = int(math.log2(major_mode_size * dtype.width // copy_bits))
        swizzle_bits = min(swizzle_bits, 3)

        layout_atom_outer = (
            cute.make_layout((8, major_mode_size), stride=(major_mode_size, 1))
            if major_mode == utils.LayoutEnum.ROW_MAJOR
            else cute.make_layout((major_mode_size, 8), stride=(1, major_mode_size))
        )
        layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 3),
            0,
            layout_atom_outer,
        )
        layout = cute.tile_to_shape(layout_atom, smem_tiler, (0, 1, 2))
        return layout

    def _make_gmem_tiled_copy_AB(self, atom_copy, dtype, major_mode, copy_bits):
        copy_elems = copy_bits // dtype.width
        shape_dim_1 = cute.size(self.bK) // copy_elems
        thread_layout = cute.make_layout(
            (self.num_threads // shape_dim_1, shape_dim_1), stride=(shape_dim_1, 1)
        )
        if major_mode != utils.LayoutEnum.ROW_MAJOR:
            shape_dim_0 = cute.size(self.bM) // copy_elems
            thread_layout = cute.make_layout(
                (shape_dim_0, self.num_threads // shape_dim_0), stride=(1, shape_dim_0)
            )
        value_layout = (
            cute.make_layout((1, copy_elems))
            if major_mode == utils.LayoutEnum.ROW_MAJOR
            else cute.make_layout((copy_elems, 1))
        )
        return cute.make_tiled_copy_tv(atom_copy, thread_layout, value_layout)

    def raster_tile(self, i, j, f):
        new_i = i // f
        new_j = (i % f) + (j * f)
        return (new_i, new_j)


def run(
    a_major: str,
    b_major: str,
    c_major: str,
    ab_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    acc_dtype: Type[cutlass.Numeric],
    mnkl: Tuple[int, int, int, int],
    atom_layout_mnk: Tuple[int, int, int],
    split_k: int = 1,
    warmup_iterations: int = 2,
    iterations: int = 100,
    skip_ref_check: bool = False,
    use_cold_l2: bool = False,
    **kwargs,
):
    print(f"Running Ampere tensor core GEMM with Split-K:")
    print(f"mnkl: {mnkl}")
    print(f"split_k: {split_k}")
    print(
        f"A dtype: {ab_dtype}, B dtype: {ab_dtype}, C dtype: {c_dtype}, Acc dtype: {acc_dtype}"
    )
    print(f"Matrix majors - A: {a_major}, B: {b_major}, C: {c_major}")
    print(f"Atoms layout: {atom_layout_mnk}")
    print(f"Warmup iterations: {warmup_iterations}")
    print(f"Iterations: {iterations}")
    print(f"Skip reference checking: {skip_ref_check}")
    print(f"Use cold L2: {use_cold_l2}")
    M, N, K, L = mnkl

    # Create and permute tensor A/B/C
    def create_and_permute_tensor(l, mode0, mode1, is_mode0_major, dtype):
        shape = (l, mode1, mode0) if is_mode0_major else (l, mode0, mode1)
        permute_order = (2, 1, 0) if is_mode0_major else (1, 2, 0)
        torch_tensor = (
            torch.empty(*shape, dtype=torch.int32)
            .random_(-2, 2)
            .to(dtype=cutlass_torch.dtype(dtype))
            .permute(permute_order)
            .cuda()
        )
        cute_tensor = (
            from_dlpack(torch_tensor, assumed_align=16)
            .mark_layout_dynamic(leading_dim=(1 if not is_mode0_major else 0))
            .mark_compact_shape_dynamic(
                mode=(1 if not is_mode0_major else 0),
                stride_order=(2, 0, 1) if not is_mode0_major else (2, 1, 0),
                divisibility=(128 // dtype.width),
            )
        )
        return cute_tensor, torch_tensor

    mA, a_torch = create_and_permute_tensor(L, M, K, a_major == "m", ab_dtype)
    mB, b_torch = create_and_permute_tensor(L, N, K, b_major == "n", ab_dtype)
    mC, c_torch = create_and_permute_tensor(L, M, N, c_major == "m", c_dtype)
    
    # Create workspace for split-K partial results (M, N, L, split_k)
    workspace_torch = torch.zeros(
        M, N, L, split_k,
        dtype=cutlass_torch.dtype(acc_dtype),
        device='cuda',
    )
    mWorkspace = from_dlpack(workspace_torch, assumed_align=16)

    tensor_op_gemm = TensorOpGemmSplitK(
        ab_dtype,
        c_dtype,
        acc_dtype,
        atom_layout_mnk,
        split_k=split_k,
        **{k: v for k, v in kwargs.items() if k in ('cta_tiler', 'num_stages')}
    )

    print("Compiling kernel with cute.compile ...")
    compiled_gemm = cute.compile(tensor_op_gemm, mA, mB, mC, mWorkspace)

    print("Executing GEMM kernel...")

    if not skip_ref_check:
        ref = torch.einsum(
            "mkl,nkl->mnl",
            a_torch.to(dtype=torch.float32),
            b_torch.to(dtype=torch.float32),
        ).to(cutlass_torch.dtype(c_dtype))
        compiled_gemm(mA, mB, mC, mWorkspace)
        print("Verifying results...")
        torch.testing.assert_close(c_torch.cpu(), ref.cpu(), atol=1e-02, rtol=1e-03)
        print("Results verified successfully!")

    def generate_tensors():
        a_workspace, _ = create_and_permute_tensor(L, M, K, a_major == "m", ab_dtype)
        b_workspace, _ = create_and_permute_tensor(L, N, K, b_major == "n", ab_dtype)
        c_workspace, _ = create_and_permute_tensor(L, M, N, c_major == "m", c_dtype)
        # Create workspace for this run
        work_torch = torch.zeros(
            M, N, L, split_k,
            dtype=cutlass_torch.dtype(acc_dtype),
            device='cuda',
        )
        work_workspace = from_dlpack(work_torch, assumed_align=16)
        return testing.JitArguments(a_workspace, b_workspace, c_workspace, work_workspace)

    workspace_count = 1
    if use_cold_l2:
        one_workspace_bytes = (
            a_torch.numel() * a_torch.element_size()
            + b_torch.numel() * b_torch.element_size()
            + c_torch.numel() * c_torch.element_size()
        )
        workspace_count = testing.get_workspace_count(
            one_workspace_bytes, warmup_iterations, iterations
        )

    avg_time_us = testing.benchmark(
        compiled_gemm,
        workspace_generator=generate_tensors,
        workspace_count=workspace_count,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
        use_cuda_graphs=False,
    )
    return avg_time_us


if __name__ == "__main__":

    def parse_comma_separated_ints(s: str) -> Tuple[int, ...]:
        try:
            return tuple(int(x.strip()) for x in s.split(","))
        except ValueError:
            raise argparse.ArgumentTypeError(
                "Invalid format. Expected comma-separated integers."
            )

    parser = argparse.ArgumentParser(
        description="Split-K GEMM example with CuTe on GPU"
    )
    parser.add_argument(
        "--mnkl", type=parse_comma_separated_ints, default=(64, 64, 8192, 1)
    )
    parser.add_argument(
        "--atom_layout_mnk", type=parse_comma_separated_ints, default=(2, 2, 1)
    )
    parser.add_argument(
        "--split_k", type=int, default=16,
        help="Number of K-dimension splits for parallelism."
    )
    parser.add_argument(
        "--cta_tiler", type=parse_comma_separated_ints, default=(128, 128, 32),
        help="CTA tile shape as M,N,K (e.g., 128,128,32)"
    )
    parser.add_argument(
        "--num_stages", type=int, default=3,
        help="Number of pipeline stages (>=3 recommended)"
    )
    parser.add_argument(
        "--ab_dtype",
        type=cutlass.dtype,
        choices=[cutlass.Float16],
        default=cutlass.Float16,
    )
    parser.add_argument(
        "--acc_dtype",
        type=cutlass.dtype,
        choices=[cutlass.Float32],
        default=cutlass.Float32,
    )
    parser.add_argument(
        "--c_dtype",
        type=cutlass.dtype,
        choices=[cutlass.Float16],
        default=cutlass.Float16,
    )
    parser.add_argument("--a_major", choices=["k", "m"], default="m")
    parser.add_argument("--b_major", choices=["k", "n"], default="n")
    parser.add_argument("--c_major", choices=["n", "m"], default="n")
    parser.add_argument("--warmup_iterations", default=2, type=int)
    parser.add_argument("--iterations", default=100, type=int)
    parser.add_argument("--skip_ref_check", action="store_true")
    parser.add_argument(
        "--use_cold_l2",
        action="store_true",
        default=False,
        help="Use circular buffer tensor sets to ensure L2 cold cache",
    )

    args = parser.parse_args()
    elapsed_time = run(
        args.a_major,
        args.b_major,
        args.c_major,
        args.ab_dtype,
        args.c_dtype,
        args.acc_dtype,
        args.mnkl,
        args.atom_layout_mnk,
        args.split_k,
        args.warmup_iterations,
        args.iterations,
        args.skip_ref_check,
        args.use_cold_l2,
        cta_tiler=args.cta_tiler,
        num_stages=args.num_stages,
    )
    print("PASS")
    print(f"Average execution time: {elapsed_time:.2f} us")
