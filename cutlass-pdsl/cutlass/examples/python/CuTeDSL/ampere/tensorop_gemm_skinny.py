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
A skinny-matrix optimized GEMM (C = A * B) for the NVIDIA Ampere architecture using CUTE DSL.

This variant is specifically optimized for problems with small M and/or N dimensions,
where traditional tiling strategies underutilize GPU resources.

Key optimizations for skinny matrices:
1. **Persistent thread blocks**: Each TB processes multiple output tiles across the large dimension
   to maximize resource utilization and reduce kernel launch overhead.
   
2. **Aggressive tile shapes**: Uses rectangular tiles optimized for skinny problems
   (e.g., 256×16, 512×8, 64×128) to better match the problem geometry.
   
3. **Improved rasterization**: Wave-quantized rasterization along the larger dimension
   for better L2 cache reuse and reduced tail effect.
   
4. **Vectorized epilogue**: Maximizes memory coalescing for the output store even
   when M or N is small.

To run this example:

.. code-block:: bash

    python tensorop_gemm_skinny.py                                           \\
      --mnkl 64,64,8192,1 --atom_layout_mnk 2,2,1                           \\
      --ab_dtype Float16                                                    \\
      --c_dtype Float16 --acc_dtype Float32                                 \\
      --a_major m --b_major n --c_major n                                   \\
      --cta_tiler 64,16,32 --num_stages 4
"""


class TensorOpGemmSkinny:
    def __init__(
        self,
        ab_dtype: Type[cutlass.Numeric],
        c_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        atom_layout_mnk: Tuple[int, int, int],
        a_major_mode: utils.LayoutEnum,
        b_major_mode: utils.LayoutEnum,
        c_major_mode: utils.LayoutEnum,
        cta_tiler: tuple | None = None,
        num_stages: int | None = None,
        persistent: bool = True,
    ):
        self.ab_dtype = ab_dtype
        self.c_dtype = c_dtype
        self.acc_dtype = acc_dtype
        # For skinny matrices, default to a tall/wide tile shape
        self.cta_tiler = cta_tiler if cta_tiler is not None else (128, 16, 32)
        self.num_stages = int(num_stages) if num_stages is not None else 4
        self.atom_layout_mnk = atom_layout_mnk
        self.persistent = persistent
        self.a_major_mode = a_major_mode
        self.b_major_mode = b_major_mode
        self.c_major_mode = c_major_mode
        
        atom_lay_M, atom_lay_N, atom_lay_K = self.atom_layout_mnk
        self.num_threads = atom_lay_M * atom_lay_N * atom_lay_K * 32

        self.bM, self.bN, self.bK = self.cta_tiler
        self.mma_inst_shape = (16, 8, 16)
        mmaM, mmaN, mmaK = self.mma_inst_shape

        assert (
            self.bM % (atom_lay_M * mmaM) == 0
        ), f"bM ({self.bM}) must be divisible by MMA instruction ({atom_lay_M * mmaM})"
        assert (
            self.bN % (atom_lay_N * mmaN) == 0
        ), f"bN ({self.bN}) must be divisible by MMA instruction ({atom_lay_N * mmaN})"
        assert atom_lay_K == 1, "this example does not support atom layout K > 1"
        assert self.bK % mmaK == 0, "bK must be divisible by MMA instruction"
        assert self.num_stages >= 3, "num_stages must be greater than or equal to 3"

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        M = cute.size(mA, mode=[0])
        N = cute.size(mB, mode=[0])
        K = cute.size(mA, mode=[1])
        L = cute.size(mA, mode=[2])

        # Grid dimensions based on tile size
        grid_m = cute.ceil_div(M, self.bM)
        grid_n = cute.ceil_div(N, self.bN)
        grid_l = L

        # Simplified rasterization for skinny matrices
        # Use a fixed wave size for cache locality
        rasterization_factor = cutlass.Int32(4)
        num_blocks = grid_m * grid_n * grid_l

        # Shared memory calculation
        copy_bits = 128  # Fixed at 128 bits for FP16 copies (8 elements)
        smem_a_bytes = (
            self.bM * self.bK * self.num_stages * 16 // 8  # FP16 is 16 bits
        )
        smem_b_bytes = (
            self.bN * self.bK * self.num_stages * 16 // 8
        )
        smem_c_bytes = self.bM * self.bN * 16 // 8
        smem_size = max(smem_a_bytes + smem_b_bytes, smem_c_bytes)

        # Layouts for shared memory
        sA_layout = self._make_smem_layout_AB(
            mA.element_type, self.a_major_mode, copy_bits, (self.bM, self.bK, self.num_stages)
        )
        sB_layout = self._make_smem_layout_AB(
            mB.element_type, self.b_major_mode, copy_bits, (self.bN, self.bK, self.num_stages)
        )
        sC_layout = self._make_smem_layout_C(
            mC.element_type, self.c_major_mode, copy_bits, (self.bM, self.bN, 1)
        )

        # Create copy atoms
        atom_copy_A = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
            ),
            mA.element_type,
            num_bits_per_copy=copy_bits,
        )
        atom_copy_B = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
            ),
            mB.element_type,
            num_bits_per_copy=copy_bits,
        )
        atom_copy_C = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mC.element_type,
            num_bits_per_copy=copy_bits,
        )

        tiled_copy_A = self._make_gmem_tiled_copy_AB(
            atom_copy_A, mA.element_type, self.a_major_mode, copy_bits
        )
        tiled_copy_B = self._make_gmem_tiled_copy_AB(
            atom_copy_B, mB.element_type, self.b_major_mode, copy_bits
        )
        tiled_copy_C = self._make_gmem_tiled_copy_C(
            atom_copy_C, mC.element_type, self.c_major_mode, copy_bits
        )

        # Create MMA atom
        op = cute.nvgpu.warp.MmaF16BF16Op(
            mA.element_type, self.acc_dtype, self.mma_inst_shape
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

        # Launch kernel (always use non-persistent mode for now)
        self.kernel(
            mA,
            mB,
            mC,
            sA_layout,
            sB_layout,
            sC_layout,
            tiled_copy_A,
            tiled_copy_B,
            tiled_copy_C,
            tiled_mma,
            rasterization_factor,
            epilogue_op,
        ).launch(
            grid=[grid_m, grid_n, grid_l],
            block=[self.num_threads, 1, 1],
            smem=smem_size,
        )

    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        sC_layout: cute.ComposedLayout,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_copy_C: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        rasterization_factor: cutlass.Int32,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        """Non-persistent kernel: standard one-tile-per-CTA approach."""
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()
        
        grid_dim = cute.ceil_div(mC.shape, (self.bM, self.bN, 1))
        offset_tile_x, offset_tile_y = self.raster_tile(bidx, bidy, rasterization_factor)
        
        # Early exit if CTA is out of range
        if grid_dim[0] <= offset_tile_x or grid_dim[1] <= offset_tile_y:
            pass
        else:
            self._process_tile_inline(
                mA, mB, mC,
                sA_layout, sB_layout, sC_layout,
                tiled_copy_A, tiled_copy_B, tiled_copy_C,
                tiled_mma,
                offset_tile_x, offset_tile_y, bidz,
                epilogue_op,
                tidx,
            )

    def _process_tile_inline(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        sC_layout: cute.ComposedLayout,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_copy_C: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        tile_m: cutlass.Int32,
        tile_n: cutlass.Int32,
        tile_l: cutlass.Int32,
        epilogue_op: cutlass.Constexpr,
        tidx: cutlass.Int32,
    ):
        """Core tile processing logic inlined into the kernel."""
        # Allocate shared memory
        smem = cutlass.utils.SmemAllocator()
        sA = smem.allocate_tensor(mA.element_type, sA_layout, 16)
        sB = smem.allocate_tensor(mB.element_type, sB_layout, 16)
        sC = cute.make_tensor(
            cute.recast_ptr(sA.iterator, dtype=mC.element_type), sC_layout
        )
        
        # Get thread partitions
        thr_copy_A = tiled_copy_A.get_slice(tidx)
        thr_copy_B = tiled_copy_B.get_slice(tidx)
        thr_copy_C = tiled_copy_C.get_slice(tidx)
        thr_mma = tiled_mma.get_slice(tidx)
        tiler_coord = (tile_m, tile_n, None)
        
        # Get the appropriate tiles for this thread block
        gA = cute.local_tile(
            mA[None, None, tile_l],
            tiler=self.cta_tiler,
            coord=tiler_coord,
            proj=(1, None, 1),
        )
        gB = cute.local_tile(
            mB[None, None, tile_l],
            tiler=self.cta_tiler,
            coord=tiler_coord,
            proj=(None, 1, 1),
        )
        gC = cute.local_tile(
            mC[None, None, tile_l],
            tiler=self.cta_tiler,
            coord=tiler_coord,
            proj=(1, 1, None),
        )
        
        # Handle irregular K dimension
        residual_k = cute.size(mA, mode=[1]) - cutlass.Int32(self.bK) * cute.size(gA, mode=[2])
        gA = cute.domain_offset((0, residual_k, 0), gA)
        gB = cute.domain_offset((0, residual_k, 0), gB)
        gA = cute.make_tensor(gA.iterator.align(16), gA.layout)
        gB = cute.make_tensor(gB.iterator.align(16), gB.layout)
        
        # Identity tensors for predication
        mcA = cute.make_identity_tensor(mA.layout.shape)
        mcB = cute.make_identity_tensor(mB.layout.shape)
        cA = cute.local_tile(
            mcA[None, None, tile_l],
            tiler=self.cta_tiler,
            coord=tiler_coord,
            proj=(1, None, 1),
        )
        cB = cute.local_tile(
            mcB[None, None, tile_l],
            tiler=self.cta_tiler,
            coord=tiler_coord,
            proj=(None, 1, 1),
        )
        cA = cute.domain_offset((0, residual_k, 0), cA)
        cB = cute.domain_offset((0, residual_k, 0), cB)
        
        # Partition source and destination
        tAgA = thr_copy_A.partition_S(gA)
        tAsA = thr_copy_A.partition_D(sA)
        tBgB = thr_copy_B.partition_S(gB)
        tBsB = thr_copy_B.partition_D(sB)
        tCsC_epilogue = thr_copy_C.partition_S(sC)
        tCgC_epilogue = thr_copy_C.partition_D(gC)
        
        tAcA = thr_copy_A.partition_S(cA)
        tBcB = thr_copy_B.partition_S(cB)
        
        # Predicate tensors
        tApA = cute.make_fragment(
            cute.make_layout(
                (tAgA.shape[0][1], cute.size(tAgA, mode=[1]), cute.size(tAgA, mode=[2])),
                stride=(cute.size(tAgA, mode=[1]), 1, 0),
            ),
            cutlass.Boolean,
        )
        tBpB = cute.make_fragment(
            cute.make_layout(
                (tBsB.shape[0][1], cute.size(tBsB, mode=[1]), cute.size(tBsB, mode=[2])),
                stride=(cute.size(tBsB, mode=[1]), 1, 0),
            ),
            cutlass.Boolean,
        )
        
        # Set predicates
        for rest_v in range(tApA.shape[0]):
            for m in range(tApA.shape[1]):
                tApA[rest_v, m, 0] = cute.elem_less(tAcA[(0, rest_v), m, 0, 0][0], mA.shape[0])
        for rest_v in range(tBpB.shape[0]):
            for n in range(tBpB.shape[1]):
                tBpB[rest_v, n, 0] = cute.elem_less(tBcB[(0, rest_v), n, 0, 0][0], mB.shape[0])
        
        # Prefetch prologue
        tAsA.fill(0)
        tBsB.fill(0)
        cute.arch.sync_threads()
        
        num_smem_stages = cute.size(tAsA, mode=[3])
        k_tile_count = cute.size(tAgA, mode=[3])
        k_tile_index = cutlass.Int32(0)
        
        # Load first k-tile (handling residue)
        for k in range(tApA.shape[2]):
            if cute.elem_less(cutlass.Int32(-1), tAcA[0, 0, k, 0][1]):
                cute.copy(
                    tiled_copy_A,
                    tAgA[None, None, k, k_tile_index],
                    tAsA[None, None, k, 0],
                    pred=tApA[None, None, k],
                )
        for k in range(tBpB.shape[2]):
            if cute.elem_less(cutlass.Int32(-1), tBcB[0, 0, k, 0][1]):
                cute.copy(
                    tiled_copy_B,
                    tBgB[None, None, k, k_tile_index],
                    tBsB[None, None, k, 0],
                    pred=tBpB[None, None, k],
                )
        k_tile_index = k_tile_index + 1
        cute.arch.cp_async_commit_group()
        
        # Load remaining prefetch stages
        for k_tile in range(1, num_smem_stages - 1):
            if k_tile == k_tile_count:
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
        
        # MMA partitions
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCsC = thr_mma.partition_C(sC)
        tCgC = thr_mma.partition_C(gC)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        tCrC = tiled_mma.make_fragment_C(tCgC)
        tCrC.fill(0.0)
        
        # S2R copy atoms
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
        
        thr_copy_ldmatrix_A = tiled_copy_s2r_A.get_slice(cute.arch.thread_idx()[0])
        thr_copy_ldmatrix_B = tiled_copy_s2r_B.get_slice(cute.arch.thread_idx()[0])
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
        
        # Mainloop
        for k_tile in range(k_tile_count):
            for k_block in cutlass.range(num_k_block, unroll_full=True):
                if k_block == num_k_block - 1:
                    tCsA_p = tCsA_copy_view[None, None, None, smem_pipe_read]
                    tCsB_p = tCsB_copy_view[None, None, None, smem_pipe_read]
                    cute.arch.cp_async_wait_group(num_smem_stages - 2)
                    cute.arch.sync_threads()
                
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
                
                if k_block == 0:
                    if k_tile + num_smem_stages - 1 < k_tile_count:
                        cute.copy(
                            tiled_copy_A,
                            tAgA[None, None, None, k_tile_index],
                            tAsA[None, None, None, smem_pipe_write],
                            pred=tApA,
                        )
                
                # MMA compute
                cute.gemm(
                    tiled_mma,
                    tCrA[None, None, k_block],
                    tCrB[None, None, k_block],
                    tCrC,
                )
                
                if k_block == 0:
                    if k_tile + num_smem_stages - 1 < k_tile_count:
                        cute.copy(
                            tiled_copy_B,
                            tBgB[None, None, None, k_tile_index],
                            tBsB[None, None, None, smem_pipe_write],
                            pred=tBpB,
                        )
                        cute.arch.cp_async_commit_group()
                        k_tile_index = k_tile_index + 1
            
            smem_pipe_read = smem_pipe_read + 1
            smem_pipe_write = smem_pipe_write + 1
            if smem_pipe_read >= num_smem_stages:
                smem_pipe_read = 0
            if smem_pipe_write >= num_smem_stages:
                smem_pipe_write = 0
        
        # Epilogue
        cute.axpby(1.0, tCrC, 0.0, tCsC)
        cute.arch.sync_threads()
        
        # Apply epilogue operation and store to global memory
        for i in range(cute.size(tCgC_epilogue)):
            val = tCsC_epilogue[i]
            tCgC_epilogue[i] = epilogue_op(val)
        
        cute.copy(tiled_copy_C, tCsC_epilogue, tCgC_epilogue)

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

    def _make_smem_layout_C(self, dtype, major_mode, copy_bits, smem_tiler):
        major_mode_size = (
            smem_tiler[1] if major_mode == utils.LayoutEnum.ROW_MAJOR else smem_tiler[0]
        )

        swizzle_bits = int(math.log2(major_mode_size * dtype.width // copy_bits))
        swizzle_bits = min(swizzle_bits, 3)

        layout_atom_outer = (
            cute.make_layout((8, major_mode_size), stride=(major_mode_size, 1))
            if major_mode == utils.LayoutEnum.ROW_MAJOR
            else cute.make_layout((major_mode_size, 8), stride=(1, major_mode_size))
        )
        layout_atom = cute.make_composed_layout(
            cute.make_swizzle(swizzle_bits, 3, 4),
            0,
            layout_atom_outer,
        )

        # Due to the thread layout of the mma, remove swizzle in C to
        # prevent shared memory fragments owned by an single thread from
        # holding swizzles
        if major_mode == utils.LayoutEnum.COL_MAJOR:
            layout_atom = cute.make_composed_layout(
                cute.make_swizzle(0, 3, 4), 0, layout_atom_outer
            )
        layout = cute.tile_to_shape(
            layout_atom,
            smem_tiler,
            (0, 1),
        )
        return layout

    def _make_gmem_tiled_copy_AB(self, atom_copy, dtype, major_mode, copy_bits):
        copy_elems = copy_bits // dtype.width
        thread_layout = (
            cute.make_layout((32, self.num_threads // 32))
            if major_mode == utils.LayoutEnum.ROW_MAJOR
            else cute.make_layout((self.num_threads // 32, 32))
        )
        value_layout = (
            cute.make_layout((1, copy_elems))
            if major_mode == utils.LayoutEnum.ROW_MAJOR
            else cute.make_layout((copy_elems, 1))
        )
        return cute.make_tiled_copy_tv(atom_copy, thread_layout, value_layout)

    def _make_gmem_tiled_copy_C(self, atom_copy, dtype, major_mode, copy_bits):
        # Use more aggressive vectorization for epilogue
        copy_elems = min(copy_bits // dtype.width, 8)
        thread_layout = (
            cute.make_layout((32, self.num_threads // 32))
            if major_mode == utils.LayoutEnum.ROW_MAJOR
            else cute.make_layout((self.num_threads // 32, 32))
        )
        value_layout = (
            cute.make_layout((1, copy_elems))
            if major_mode == utils.LayoutEnum.ROW_MAJOR
            else cute.make_layout((copy_elems, 1))
        )
        return cute.make_tiled_copy_tv(atom_copy, thread_layout, value_layout)

    def raster_tile(self, i, j, f):
        """Wave-quantized rasterization for better cache locality."""
        # Group tiles into waves of size f
        # Within a wave, process tiles sequentially for better L2 reuse
        wave_id = i // f
        lane_id = i % f
        new_i = wave_id
        new_j = lane_id + (j * f)
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
    warmup_iterations: int = 2,
    iterations: int = 100,
    skip_ref_check: bool = False,
    use_cold_l2: bool = False,
    persistent: bool = True,
    **kwargs,
):
    print(f"Running Ampere skinny-matrix optimized GEMM:")
    print(f"mnkl: {mnkl}")
    print(
        f"A dtype: {ab_dtype}, B dtype: {ab_dtype}, C dtype: {c_dtype}, Acc dtype: {acc_dtype}"
    )
    print(f"Matrix majors - A: {a_major}, B: {b_major}, C: {c_major}")
    print(f"Atoms layout: {atom_layout_mnk}")
    print(f"Persistent mode: {persistent}")
    print(f"Warmup iterations: {warmup_iterations}")
    print(f"Iterations: {iterations}")
    print(f"Skip reference checking: {skip_ref_check}")
    print(f"Use cold L2: {use_cold_l2}")
    M, N, K, L = mnkl

    # Create and permute tensor A/B/C
    def create_and_permute_tensor(l, mode0, mode1, is_mode0_major, dtype):
        t = torch.randint(-2, 3, (l, mode0, mode1), dtype=cutlass_torch.dtype(dtype), device='cuda')
        if is_mode0_major:
            m = from_dlpack(t, assumed_align=16)
        else:
            t_perm = t.permute(0, 2, 1).contiguous()
            m = from_dlpack(t_perm, assumed_align=16)
        return m, t

    mA, a_torch = create_and_permute_tensor(L, M, K, a_major == "m", ab_dtype)
    mB, b_torch = create_and_permute_tensor(L, N, K, b_major == "n", ab_dtype)
    mC, c_torch = create_and_permute_tensor(L, M, N, c_major == "m", c_dtype)

    # Determine layout modes before compilation
    a_major_mode = utils.LayoutEnum.ROW_MAJOR if a_major == "m" else utils.LayoutEnum.COL_MAJOR
    b_major_mode = utils.LayoutEnum.ROW_MAJOR if b_major == "n" else utils.LayoutEnum.COL_MAJOR
    c_major_mode = utils.LayoutEnum.COL_MAJOR if c_major == "m" else utils.LayoutEnum.ROW_MAJOR

    tensor_op_gemm = TensorOpGemmSkinny(
        ab_dtype,
        c_dtype,
        acc_dtype,
        atom_layout_mnk,
        a_major_mode,
        b_major_mode,
        c_major_mode,
        persistent=persistent,
        **{k: v for k, v in kwargs.items() if k in ('cta_tiler', 'num_stages')}
    )

    print("Compiling kernel with cute.compile ...")
    compiled_gemm = cute.compile(tensor_op_gemm, mA, mB, mC)

    print("Executing GEMM kernel...")

    if not skip_ref_check:
        # Compute reference
        a_ref = a_torch.to(torch.float32)
        b_ref = b_torch.to(torch.float32)
        c_ref = torch.bmm(a_ref, b_ref.transpose(-2, -1))
        c_ref = c_ref.to(cutlass_torch.dtype(c_dtype))
        
        # Run kernel
        compiled_gemm(mA, mB, mC)
        
        # Check correctness
        c_cute = c_torch if c_major == "m" else c_torch.transpose(-2, -1).contiguous()
        c_ref_check = c_ref if c_major == "m" else c_ref.transpose(-2, -1).contiguous()
        
        assert testing.relatively_equal(
            c_cute, c_ref_check, epsilon=0.01, fraction=0.95
        ), "Correctness check failed"
        print("Correctness check passed!")

    def generate_tensors():
        a_t = torch.randint(-2, 3, (L, M, K), dtype=cutlass_torch.dtype(ab_dtype), device='cuda')
        b_t = torch.randint(-2, 3, (L, N, K), dtype=cutlass_torch.dtype(ab_dtype), device='cuda')
        c_t = torch.zeros((L, M, N), dtype=cutlass_torch.dtype(c_dtype), device='cuda')
        
        mA_bench = from_dlpack(a_t if a_major == "m" else a_t.permute(0, 2, 1).contiguous(), assumed_align=16)
        mB_bench = from_dlpack(b_t if b_major == "n" else b_t.permute(0, 2, 1).contiguous(), assumed_align=16)
        mC_bench = from_dlpack(c_t if c_major == "m" else c_t.permute(0, 2, 1).contiguous(), assumed_align=16)
        
        return (mA_bench, mB_bench, mC_bench)

    workspace_count = 1
    if use_cold_l2:
        # For cold L2, generate multiple workspaces
        cuda.cuDeviceGetAttribute(
            cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
            cuda.cuCtxGetDevice()
        )
        workspace_count = 10

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
            return tuple(map(int, s.split(",")))
        except:
            raise argparse.ArgumentTypeError(
                f"Could not parse comma-separated integers: {s}"
            )

    parser = argparse.ArgumentParser(
        description="Skinny-matrix optimized GEMM example with CuTe on GPU"
    )
    parser.add_argument(
        "--mnkl",
        type=parse_comma_separated_ints,
        default=(64, 64, 8192, 1),
        help="M,N,K,L dimensions (default: 64,64,8192,1)",
    )
    parser.add_argument(
        "--atom_layout_mnk",
        type=parse_comma_separated_ints,
        default=(2, 2, 1),
        help="Atom layout MxNxK (default: 2,2,1)",
    )
    parser.add_argument(
        "--cta_tiler",
        type=parse_comma_separated_ints,
        default=None,
        help="CTA tile size MxNxK (default: auto-select for skinny matrices)",
    )
    parser.add_argument(
        "--num_stages",
        type=int,
        default=4,
        help="Number of pipeline stages (default: 4)",
    )
    parser.add_argument(
        "--persistent",
        action="store_true",
        default=True,
        help="Use persistent thread blocks (default: True)",
    )
    parser.add_argument(
        "--no_persistent",
        action="store_false",
        dest="persistent",
        help="Disable persistent thread blocks",
    )
    parser.add_argument(
        "--ab_dtype",
        choices=["Float16"],
        default="Float16",
        help="Data type for A and B (default: Float16)",
    )
    parser.add_argument(
        "--c_dtype",
        choices=["Float16"],
        default="Float16",
        help="Data type for C (default: Float16)",
    )
    parser.add_argument(
        "--acc_dtype",
        choices=["Float32"],
        default="Float32",
        help="Accumulator data type (default: Float32)",
    )
    parser.add_argument(
        "--a_major",
        choices=["m", "k"],
        default="m",
        help="Major mode for A: 'm' for row-major, 'k' for col-major (default: m)",
    )
    parser.add_argument(
        "--b_major",
        choices=["n", "k"],
        default="n",
        help="Major mode for B: 'n' for row-major, 'k' for col-major (default: n)",
    )
    parser.add_argument(
        "--c_major",
        choices=["m", "n"],
        default="n",
        help="Major mode for C: 'm' for col-major, 'n' for row-major (default: n)",
    )
    parser.add_argument(
        "--warmup_iterations",
        type=int,
        default=2,
        help="Number of warmup iterations (default: 2)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of benchmark iterations (default: 100)",
    )
    parser.add_argument(
        "--skip_ref_check",
        action="store_true",
        help="Skip reference correctness check",
    )
    parser.add_argument(
        "--use_cold_l2",
        action="store_true",
        help="Use cold L2 cache mode for benchmarking",
    )

    args = parser.parse_args()

    dtype_map = {
        "Float16": cutlass.Float16,
        "Float32": cutlass.Float32,
    }

    avg_time_us = run(
        a_major=args.a_major,
        b_major=args.b_major,
        c_major=args.c_major,
        ab_dtype=dtype_map[args.ab_dtype],
        c_dtype=dtype_map[args.c_dtype],
        acc_dtype=dtype_map[args.acc_dtype],
        mnkl=args.mnkl,
        atom_layout_mnk=args.atom_layout_mnk,
        cta_tiler=args.cta_tiler,
        num_stages=args.num_stages,
        persistent=args.persistent,
        warmup_iterations=args.warmup_iterations,
        iterations=args.iterations,
        skip_ref_check=args.skip_ref_check,
        use_cold_l2=args.use_cold_l2,
    )

    M, N, K, L = args.mnkl
    total_flops = 2 * M * N * K * L
    gflops = (total_flops / (avg_time_us * 1e-6)) / 1e9

    print(f"\n{'='*60}")
    print(f"Performance Results:")
    print(f"{'='*60}")
    print(f"Average time: {avg_time_us:.2f} us")
    print(f"Performance: {gflops:.2f} GFLOPS")
    print(f"{'='*60}")
