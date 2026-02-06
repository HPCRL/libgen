# Split-K variant of tensorop_gemm_tunable.py
# Kernel-level split-K using a workspace tensor (L dimension = split_k) and a post-reduction pass.
# Each CTA processes a distinct K-slice (z-dimension enumerates slices) and writes its partial tile
# to the corresponding L-plane. After the main kernel, we reduce across L to form the final result.

import argparse
import math
from typing import Tuple, Type

import torch
import cutlass
import cutlass.cute as cute
import cutlass.cute.testing as testing
import cutlass.torch as cutlass_torch
import cutlass.utils as utils
from cutlass.cute.runtime import from_dlpack

# Reuse kernel class but allow passing K-slice bounds via tensors sized to sliceK.
# We wrap original TensorOpGemm; kernel unchanged except we only see a sliced K.
from tensorop_gemm_tunable import TensorOpGemm as BaseTensorOpGemm  # type: ignore


def _ceil_div(a, b):
    return (a + b - 1) // b

class TensorOpGemmSplitK(BaseTensorOpGemm):
    """Extends base TensorOpGemm with in-kernel split-K using workspace L=split_k.

    - mA and mB are indexed at L=0 (we do not replicate batch along split_k).
    - mC (workspace) is shaped (M, N, split_k) so bidz selects the slice plane.
    - K is partitioned into contiguous slices; each CTA processes only its slice by
      offsetting base pointers and limiting the K mainloop tile-count.
    """

    def __init__(self, *args, split_k: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.split_k = int(max(1, split_k))

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        # Detect majors from tensors
        self.a_major_mode = utils.LayoutEnum.from_tensor(mA)
        self.b_major_mode = utils.LayoutEnum.from_tensor(mB)
        self.c_major_mode = utils.LayoutEnum.from_tensor(mC)

        # Shared memory layouts
        ab_copy_bits = 128
        sA_layout = self._make_smem_layout_AB(
            mA.element_type, self.a_major_mode, ab_copy_bits, (self.cta_tiler[0], self.cta_tiler[2], self.num_stages)
        )
        sB_layout = self._make_smem_layout_AB(
            mB.element_type, self.b_major_mode, ab_copy_bits, (self.cta_tiler[1], self.cta_tiler[2], self.num_stages)
        )
        sC_layout = self._make_smem_layout_C(
            mC.element_type, self.c_major_mode, ab_copy_bits, (self.cta_tiler[0], self.cta_tiler[1])
        )

        smem_size = max(
            cute.size_in_bytes(mC.element_type, sC_layout),
            cute.size_in_bytes(mA.element_type, sA_layout) + cute.size_in_bytes(mB.element_type, sB_layout),
        )

        # Tiled copies
        atom_async_copy = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(
                cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL
            ),
            mA.element_type,
            num_bits_per_copy=ab_copy_bits,
        )
        tiled_copy_A = self._make_gmem_tiled_copy_AB(atom_async_copy, mA.element_type, self.a_major_mode, ab_copy_bits)
        tiled_copy_B = self._make_gmem_tiled_copy_AB(atom_async_copy, mB.element_type, self.b_major_mode, ab_copy_bits)

        c_copy_bits = 128
        atom_sync_copy = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mC.element_type, num_bits_per_copy=c_copy_bits)
        tiled_copy_C = self._make_gmem_tiled_copy_C(atom_sync_copy, mC.element_type, self.c_major_mode, c_copy_bits)

        # Tiled MMA
        op = cute.nvgpu.warp.MmaF16BF16Op(self.ab_dtype, self.acc_dtype, self.mma_inst_shape)
        permutation_mnk = (
            self.atom_layout_mnk[0] * self.mma_inst_shape[0],
            self.atom_layout_mnk[1] * self.mma_inst_shape[1] * 2,
            self.atom_layout_mnk[2] * self.mma_inst_shape[2],
        )
        tC = cute.make_layout(self.atom_layout_mnk)
        tiled_mma = cute.make_tiled_mma(op, tC, permutation_mnk=permutation_mnk)

        # Grid over M,N and L=valid split_k slices (omit empty trailing slices)
        K_total = mA.shape[1]
        slice_len = (K_total + self.split_k - 1) // self.split_k
        valid_slices = (K_total + slice_len - 1) // slice_len
        grid_dim = cute.ceil_div((mC.shape[0], mC.shape[1], valid_slices), (self.bM, self.bN, 1))

        # Rasterization
        raster_factor = 1
        grid_dim_n = cute.size(grid_dim[1])
        if grid_dim_n > 5:
            raster_factor = 8
        elif grid_dim_n > 2:
            raster_factor = 4
        elif grid_dim_n > 1:
            raster_factor = 2
        rasterization_remap_grid_dim = (
            cute.size(grid_dim[0]) * raster_factor,
            (cute.size(grid_dim[1]) + raster_factor - 1) // raster_factor,
            cute.size(grid_dim[2]),
        )

        self.kernel(
            mA, mB, mC,
            sA_layout, sB_layout, sC_layout,
            tiled_copy_A, tiled_copy_B, tiled_copy_C,
            tiled_mma, raster_factor, epilogue_op,
        ).launch(
            grid=(rasterization_remap_grid_dim[0], rasterization_remap_grid_dim[1], cute.size(grid_dim[2])),
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
        # Thread/block coords
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()
        grid_dim = cute.ceil_div(mC.shape, (self.bM, self.bN, 1))
        offset_tile_x, offset_tile_y = self.raster_tile(bidx, bidy, rasterization_factor)
        if grid_dim[0] <= offset_tile_x or grid_dim[1] <= offset_tile_y:
            # Outside problem bounds: no work (keep DSL happy with no early return)
            pass
        else:
            tiler_coord = (offset_tile_x, offset_tile_y, None)

            # K slice bounds based on bidz
            K_total = mA.shape[1]
            slice_len = (K_total + self.split_k - 1) // self.split_k
            k_start = bidz * slice_len
            k_len = K_total - k_start
            if k_len > slice_len:
                k_len = slice_len
            if k_len <= 0:
                # No work for this z-slice
                pass
            else:
                mA_base = cute.domain_offset((0, cutlass.Int32(k_start), 0), mA)
                mB_base = cute.domain_offset((0, cutlass.Int32(k_start), 0), mB)
                gA = cute.local_tile(mA_base[None, None, 0], self.cta_tiler, tiler_coord, (1, None, 1))
                gB = cute.local_tile(mB_base[None, None, 0], self.cta_tiler, tiler_coord, (None, 1, 1))
                gC = cute.local_tile(mC[None, None, bidz], self.cta_tiler, tiler_coord, (1, 1, None))

                k_tiles_slice = (k_len + self.bK - 1) // self.bK
                residual_k = cutlass.Int32(k_len - self.bK * k_tiles_slice)
                gA = cute.domain_offset((0, residual_k, 0), gA)
                gB = cute.domain_offset((0, residual_k, 0), gB)
                gA = cute.make_tensor(gA.iterator.align(16), gA.layout)
                gB = cute.make_tensor(gB.iterator.align(16), gB.layout)

                mcA = cute.make_identity_tensor(mA.layout.shape)
                mcB = cute.make_identity_tensor(mB.layout.shape)
                cA = cute.local_tile(mcA[None, None, 0], self.cta_tiler, tiler_coord, (1, None, 1))
                cB = cute.local_tile(mcB[None, None, 0], self.cta_tiler, tiler_coord, (None, 1, 1))
                cA = cute.domain_offset((0, residual_k, 0), cA)
                cB = cute.domain_offset((0, residual_k, 0), cB)

                smem = cutlass.utils.SmemAllocator()
                sA = smem.allocate_tensor(mA.element_type, sA_layout, 16)
                sB = smem.allocate_tensor(mB.element_type, sB_layout, 16)
                sC = cute.make_tensor(cute.recast_ptr(sA.iterator, dtype=self.c_dtype), sC_layout)

                thr_copy_A = tiled_copy_A.get_slice(tidx)
                thr_copy_B = tiled_copy_B.get_slice(tidx)
                thr_copy_C = tiled_copy_C.get_slice(tidx)
                tAgA = thr_copy_A.partition_S(gA)
                tAsA = thr_copy_A.partition_D(sA)
                tBgB = thr_copy_B.partition_S(gB)
                tBsB = thr_copy_B.partition_D(sB)
                tCsC_epilogue = thr_copy_C.partition_S(sC)
                tCgC_epilogue = thr_copy_C.partition_D(gC)

                tAcA = thr_copy_A.partition_S(cA)
                tBcB = thr_copy_B.partition_S(cB)

                tApA = cute.make_fragment(
                    cute.make_layout((tAgA.shape[0][1], cute.size(tAgA, mode=[1]), cute.size(tAgA, mode=[2])),
                                     stride=(cute.size(tAgA, mode=[1]), 1, 0)),
                    cutlass.Boolean,
                )
                tBpB = cute.make_fragment(
                    cute.make_layout((tBsB.shape[0][1], cute.size(tBsB, mode=[1]), cute.size(tBsB, mode=[2])),
                                     stride=(cute.size(tBsB, mode=[1]), 1, 0)),
                    cutlass.Boolean,
                )
                for rest_v in range(tApA.shape[0]):
                    for m in range(tApA.shape[1]):
                        tApA[rest_v, m, 0] = cute.elem_less(tAcA[(0, rest_v), m, 0, 0][0], mA.shape[0])
                for rest_v in range(tBpB.shape[0]):
                    for n in range(tBpB.shape[1]):
                        tBpB[rest_v, n, 0] = cute.elem_less(tBcB[(0, rest_v), n, 0, 0][0], mB.shape[0])

                tAsA.fill(0)
                tBsB.fill(0)
                cute.arch.sync_threads()
                num_smem_stages = cute.size(tAsA, mode=[3])
                k_tile_count = cutlass.Int32(k_tiles_slice)
                k_tile_index = cutlass.Int32(0)
                for k in range(tApA.shape[2]):
                    if cute.elem_less(cutlass.Int32(-1), tAcA[0, 0, k, 0][1]):
                        cute.copy(tiled_copy_A, tAgA[None, None, k, k_tile_index], tAsA[None, None, k, 0], pred=tApA[None, None, k])
                for k in range(tBpB.shape[2]):
                    if cute.elem_less(cutlass.Int32(-1), tBcB[0, 0, k, 0][1]):
                        cute.copy(tiled_copy_B, tBgB[None, None, k, k_tile_index], tBsB[None, None, k, 0], pred=tBpB[None, None, k])
                k_tile_index = k_tile_index + 1
                cute.arch.cp_async_commit_group()

                # Prefetch remaining smem stages with simple predication (remove dynamic Python ifs)
                tApA_zero = cute.make_fragment_like(tApA)
                tApA_zero.fill(0)
                tBpB_zero = cute.make_fragment_like(tBpB)
                tBpB_zero.fill(0)
                for k_tile in range(1, num_smem_stages - 1):
                    # Predicate for whether this stage is within slice tile count
                    stage_pred = cute.elem_less(cutlass.Int32(k_tile), k_tile_count)
                    # Always issue copy; use zeroed fragment if out-of-range
                    cute.copy(tiled_copy_A, tAgA[None, None, None, k_tile_index], tAsA[None, None, None, k_tile], pred=tApA if stage_pred else tApA_zero)
                    cute.copy(tiled_copy_B, tBgB[None, None, None, k_tile_index], tBsB[None, None, None, k_tile], pred=tBpB if stage_pred else tBpB_zero)
                    k_tile_index = k_tile_index + 1
                    cute.arch.cp_async_commit_group()

                thr_mma = tiled_mma.get_slice(tidx)
                tCsA = thr_mma.partition_A(sA)
                tCsB = thr_mma.partition_B(sB)
                tCsC = thr_mma.partition_C(sC)
                tCgC = thr_mma.partition_C(gC)
                tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
                tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
                tCrC = tiled_mma.make_fragment_C(tCgC)
                tCrC.fill(0.0)

                atom_copy_s2r_A = cute.make_copy_atom(
                    cute.nvgpu.warp.LdMatrix8x8x16bOp(self.a_major_mode != utils.LayoutEnum.ROW_MAJOR, 4), mA.element_type)
                atom_copy_s2r_B = cute.make_copy_atom(
                    cute.nvgpu.warp.LdMatrix8x8x16bOp(self.b_major_mode != utils.LayoutEnum.ROW_MAJOR, 4), mB.element_type)
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

                num_k_block = cute.size(tCrA, mode=[2])
                # Unconditionally perform initial wait/copy for block 0 (predicated loads handle overrun)
                cute.arch.cp_async_wait_group(num_smem_stages - 2)
                cute.arch.sync_threads()
                cute.copy(tiled_copy_s2r_A, tCsA_p[None, None, 0], tCrA_copy_view[None, None, 0])
                cute.copy(tiled_copy_s2r_B, tCsB_p[None, None, 0], tCrB_copy_view[None, None, 0])

                for k_tile in range(k_tile_count):
                    for k_block in cutlass.range(num_k_block, unroll_full=True):
                        # Always refresh read pipe at start of each k_block iteration (acts like previous end-condition)
                        tCsA_p = tCsA_copy_view[None, None, None, smem_pipe_read]
                        tCsB_p = tCsB_copy_view[None, None, None, smem_pipe_read]
                        cute.arch.cp_async_wait_group(num_smem_stages - 2)
                        cute.arch.sync_threads()

                        k_block_next = (k_block + 1) % num_k_block
                        cute.copy(tiled_copy_s2r_A, tCsA_p[None, None, k_block_next], tCrA_copy_view[None, None, k_block_next])
                        cute.copy(tiled_copy_s2r_B, tCsB_p[None, None, k_block_next], tCrB_copy_view[None, None, k_block_next])

                        # Global->smem staging only once per k_tile (when k_block==0 implicitly via arithmetic mask)
                        do_stage = cute.elem_less(cutlass.Int32(k_block), cutlass.Int32(1))
                        next_stage_pred = cute.elem_less(cutlass.Int32(k_tile + num_smem_stages - 1), k_tile_count)
                        cute.copy(tiled_copy_A, tAgA[None, None, None, k_tile_index], tAsA[None, None, None, smem_pipe_write], pred=tApA if (do_stage and next_stage_pred) else tApA_zero)

                        cute.gemm(tiled_mma, tCrC, tCrA[None, None, k_block], tCrB[None, None, k_block], tCrC)

                        cute.copy(tiled_copy_B, tBgB[None, None, None, k_tile_index], tBsB[None, None, None, smem_pipe_write], pred=tBpB if (do_stage and next_stage_pred) else tBpB_zero)

                        # Pipe advance only once per k_tile (when do_stage true)
                        if do_stage:
                            k_tile_index = k_tile_index + 1
                            cute.arch.cp_async_commit_group()
                            smem_pipe_write = smem_pipe_read
                            smem_pipe_read = smem_pipe_read + 1
                            if smem_pipe_read == num_smem_stages:
                                smem_pipe_read = 0

                cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()

                tCrD = cute.make_fragment_like(tCrC, self.c_dtype)
                tCrD[None] = epilogue_op(tCrC.load()).to(self.c_dtype)
                cute.autovec_copy(tCrD, tCsC)

                ceilM, ceilN, _ = cute.ceil_div(mC.shape, (self.bM, self.bN, 1))
                mcC = cute.make_identity_tensor((cute.size(ceilM) * self.cta_tiler[0], cute.size(ceilN) * self.cta_tiler[1], 1))
                cC = cute.local_tile(mcC[None, None, bidz], self.cta_tiler, tiler_coord, (1, 1, None))
                tCcC = thr_copy_C.partition_S(cC)

                tCrC_epilogue = cute.make_fragment_like(tCsC_epilogue)
                cute.arch.sync_threads()
                cute.autovec_copy(tCsC_epilogue, tCrC_epilogue)

                tCpC = cute.make_fragment(
                    cute.make_layout((tCgC_epilogue.shape[0][1], cute.size(tCgC_epilogue, mode=[1]), cute.size(tCgC_epilogue, mode=[2])),
                                     stride=(cute.size(tCgC_epilogue, mode=[1]), 1, 0)),
                    cutlass.Boolean,
                )
                for rest_v in range(tCpC.shape[0]):
                    for m in range(tCpC.shape[1]):
                        tCpC[rest_v, m, 0] = cute.elem_less(tCcC[(0, rest_v), m, 0][0], mC.shape[0])

                for rest_v in range(tCpC.shape[0]):
                    for n in range(tCpC.shape[2]):
                        if cute.elem_less(tCcC[(0, rest_v), 0, n][1], mC.shape[1]):
                            cute.copy(tiled_copy_C, tCrC_epilogue[None, None, n], tCgC_epilogue[None, None, n], pred=tCpC[None, None, n])
                k_tile_index = cutlass.Int32(0)
                for k in range(tApA.shape[2]):
                    if cute.elem_less(cutlass.Int32(-1), tAcA[0, 0, k, 0][1]):
                        cute.copy(tiled_copy_A, tAgA[None, None, k, k_tile_index], tAsA[None, None, k, 0], pred=tApA[None, None, k])
                for k in range(tBpB.shape[2]):
                    if cute.elem_less(cutlass.Int32(-1), tBcB[0, 0, k, 0][1]):
                        cute.copy(tiled_copy_B, tBgB[None, None, k, k_tile_index], tBsB[None, None, k, 0], pred=tBpB[None, None, k])
                k_tile_index = k_tile_index + 1
                cute.arch.cp_async_commit_group()

                for k_tile in range(1, num_smem_stages - 1):
                    if k_tile == k_tile_count:
                        tApA.fill(0)
                        tBpB.fill(0)
                    cute.copy(tiled_copy_A, tAgA[None, None, None, k_tile_index], tAsA[None, None, None, k_tile], pred=tApA)
                    cute.copy(tiled_copy_B, tBgB[None, None, None, k_tile_index], tBsB[None, None, None, k_tile], pred=tBpB)
                    k_tile_index = k_tile_index + 1
                    cute.arch.cp_async_commit_group()

                # MMA partitions and accumulators
                thr_mma = tiled_mma.get_slice(tidx)
                tCsA = thr_mma.partition_A(sA)
                tCsB = thr_mma.partition_B(sB)
                tCsC = thr_mma.partition_C(sC)
                tCgC = thr_mma.partition_C(gC)
                tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
                tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
                tCrC = tiled_mma.make_fragment_C(tCgC)
                tCrC.fill(0.0)

                atom_copy_s2r_A = cute.make_copy_atom(
                    cute.nvgpu.warp.LdMatrix8x8x16bOp(self.a_major_mode != utils.LayoutEnum.ROW_MAJOR, 4), mA.element_type)
                atom_copy_s2r_B = cute.make_copy_atom(
                    cute.nvgpu.warp.LdMatrix8x8x16bOp(self.b_major_mode != utils.LayoutEnum.ROW_MAJOR, 4), mB.element_type)
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

                num_k_block = cute.size(tCrA, mode=[2])
                if num_k_block > 1:
                    cute.arch.cp_async_wait_group(num_smem_stages - 2)
                    cute.arch.sync_threads()
                    cute.copy(tiled_copy_s2r_A, tCsA_p[None, None, 0], tCrA_copy_view[None, None, 0])
                    cute.copy(tiled_copy_s2r_B, tCsB_p[None, None, 0], tCrB_copy_view[None, None, 0])

                for k_tile in range(k_tile_count):
                    for k_block in cutlass.range(num_k_block, unroll_full=True):
                        if k_block == num_k_block - 1:
                            tCsA_p = tCsA_copy_view[None, None, None, smem_pipe_read]
                            tCsB_p = tCsB_copy_view[None, None, None, smem_pipe_read]
                            cute.arch.cp_async_wait_group(num_smem_stages - 2)
                            cute.arch.sync_threads()

                        k_block_next = (k_block + 1) % num_k_block
                        cute.copy(tiled_copy_s2r_A, tCsA_p[None, None, k_block_next], tCrA_copy_view[None, None, k_block_next])
                        cute.copy(tiled_copy_s2r_B, tCsB_p[None, None, k_block_next], tCrB_copy_view[None, None, k_block_next])

                        if k_block == 0:
                            if k_tile + num_smem_stages - 1 < k_tile_count:
                                cute.copy(tiled_copy_A, tAgA[None, None, None, k_tile_index], tAsA[None, None, None, smem_pipe_write], pred=tApA)

                        cute.gemm(tiled_mma, tCrC, tCrA[None, None, k_block], tCrB[None, None, k_block], tCrC)

                        if k_block == 0:
                            if k_tile + num_smem_stages - 1 < k_tile_count:
                                cute.copy(tiled_copy_B, tBgB[None, None, None, k_tile_index], tBsB[None, None, None, smem_pipe_write], pred=tBpB)
                            k_tile_index = k_tile_index + 1
                            cute.arch.cp_async_commit_group()
                            smem_pipe_write = smem_pipe_read
                            smem_pipe_read = smem_pipe_read + 1
                            if smem_pipe_read == num_smem_stages:
                                smem_pipe_read = 0

                cute.arch.cp_async_wait_group(0)
                cute.arch.sync_threads()

                # Epilogue -> shared -> global
                tCrD = cute.make_fragment_like(tCrC, self.c_dtype)
                tCrD[None] = epilogue_op(tCrC.load()).to(self.c_dtype)
                cute.autovec_copy(tCrD, tCsC)

                ceilM, ceilN, _ = cute.ceil_div(mC.shape, (self.bM, self.bN, 1))
                mcC = cute.make_identity_tensor((cute.size(ceilM) * self.cta_tiler[0], cute.size(ceilN) * self.cta_tiler[1], 1))
                cC = cute.local_tile(mcC[None, None, bidz], self.cta_tiler, tiler_coord, (1, 1, None))
                tCcC = thr_copy_C.partition_S(cC)

                tCrC_epilogue = cute.make_fragment_like(tCsC_epilogue)
                cute.arch.sync_threads()
                cute.autovec_copy(tCsC_epilogue, tCrC_epilogue)

                tCpC = cute.make_fragment(
                    cute.make_layout((tCgC_epilogue.shape[0][1], cute.size(tCgC_epilogue, mode=[1]), cute.size(tCgC_epilogue, mode=[2])),
                                     stride=(cute.size(tCgC_epilogue, mode=[1]), 1, 0)),
                    cutlass.Boolean,
                )
                for rest_v in range(tCpC.shape[0]):
                    for m in range(tCpC.shape[1]):
                        tCpC[rest_v, m, 0] = cute.elem_less(tCcC[(0, rest_v), m, 0][0], mC.shape[0])

                for rest_v in range(tCpC.shape[0]):
                    for n in range(tCpC.shape[2]):
                        if cute.elem_less(tCcC[(0, rest_v), 0, n][1], mC.shape[1]):
                            cute.copy(tiled_copy_C, tCrC_epilogue[None, None, n], tCgC_epilogue[None, None, n], pred=tCpC[None, None, n])
            tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
            tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
            tCrC = tiled_mma.make_fragment_C(tCgC)
            tCrC.fill(0.0)

            atom_copy_s2r_A = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(self.a_major_mode != utils.LayoutEnum.ROW_MAJOR, 4), mA.element_type)
            atom_copy_s2r_B = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(self.b_major_mode != utils.LayoutEnum.ROW_MAJOR, 4), mB.element_type)
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

            num_k_block = cute.size(tCrA, mode=[2])
            if num_k_block > 1:
                cute.arch.cp_async_wait_group(num_smem_stages - 2)
                cute.arch.sync_threads()
                cute.copy(tiled_copy_s2r_A, tCsA_p[None, None, 0], tCrA_copy_view[None, None, 0])
                cute.copy(tiled_copy_s2r_B, tCsB_p[None, None, 0], tCrB_copy_view[None, None, 0])

            for k_tile in range(k_tile_count):
                for k_block in cutlass.range(num_k_block, unroll_full=True):
                    if k_block == num_k_block - 1:
                        tCsA_p = tCsA_copy_view[None, None, None, smem_pipe_read]
                        tCsB_p = tCsB_copy_view[None, None, None, smem_pipe_read]
                        cute.arch.cp_async_wait_group(num_smem_stages - 2)
                        cute.arch.sync_threads()

                    k_block_next = (k_block + 1) % num_k_block
                    cute.copy(tiled_copy_s2r_A, tCsA_p[None, None, k_block_next], tCrA_copy_view[None, None, k_block_next])
                    cute.copy(tiled_copy_s2r_B, tCsB_p[None, None, k_block_next], tCrB_copy_view[None, None, k_block_next])

                    if k_block == 0:
                        if k_tile + num_smem_stages - 1 < k_tile_count:
                            cute.copy(tiled_copy_A, tAgA[None, None, None, k_tile_index], tAsA[None, None, None, smem_pipe_write], pred=tApA)

                    cute.gemm(tiled_mma, tCrC, tCrA[None, None, k_block], tCrB[None, None, k_block], tCrC)

                    if k_block == 0:
                        if k_tile + num_smem_stages - 1 < k_tile_count:
                            cute.copy(tiled_copy_B, tBgB[None, None, None, k_tile_index], tBsB[None, None, None, smem_pipe_write], pred=tBpB)
                        k_tile_index = k_tile_index + 1
                        cute.arch.cp_async_commit_group()
                        smem_pipe_write = smem_pipe_read
                        smem_pipe_read = smem_pipe_read + 1
                        if smem_pipe_read == num_smem_stages:
                            smem_pipe_read = 0

            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()

            # Epilogue -> shared -> global
            tCrD = cute.make_fragment_like(tCrC, self.c_dtype)
            tCrD[None] = epilogue_op(tCrC.load()).to(self.c_dtype)
            cute.autovec_copy(tCrD, tCsC)

            ceilM, ceilN, _ = cute.ceil_div(mC.shape, (self.bM, self.bN, 1))
            mcC = cute.make_identity_tensor((cute.size(ceilM) * self.cta_tiler[0], cute.size(ceilN) * self.cta_tiler[1], 1))
            cC = cute.local_tile(mcC[None, None, bidz], self.cta_tiler, tiler_coord, (1, 1, None))
            tCcC = thr_copy_C.partition_S(cC)

            tCrC_epilogue = cute.make_fragment_like(tCsC_epilogue)
            cute.arch.sync_threads()
            cute.autovec_copy(tCsC_epilogue, tCrC_epilogue)

            tCpC = cute.make_fragment(
                cute.make_layout((tCgC_epilogue.shape[0][1], cute.size(tCgC_epilogue, mode=[1]), cute.size(tCgC_epilogue, mode=[2])),
                                 stride=(cute.size(tCgC_epilogue, mode=[1]), 1, 0)),
                cutlass.Boolean,
            )
            for rest_v in range(tCpC.shape[0]):
                for m in range(tCpC.shape[1]):
                    tCpC[rest_v, m, 0] = cute.elem_less(tCcC[(0, rest_v), m, 0][0], mC.shape[0])

            for rest_v in range(tCpC.shape[0]):
                for n in range(tCpC.shape[2]):
                    if cute.elem_less(tCcC[(0, rest_v), 0, n][1], mC.shape[1]):
                        cute.copy(tiled_copy_C, tCrC_epilogue[None, None, n], tCgC_epilogue[None, None, n], pred=tCpC[None, None, n])
        return


def run_splitk(
    a_major: str,
    b_major: str,
    c_major: str,
    ab_dtype: Type[cutlass.Numeric],
    c_dtype: Type[cutlass.Numeric],
    acc_dtype: Type[cutlass.Numeric],
    mnkl: Tuple[int, int, int, int],
    atom_layout_mnk: Tuple[int, int, int],
    split_k: int,
    warmup_iterations: int = 2,
    iterations: int = 100,
    skip_ref_check: bool = False,
    use_cold_l2: bool = False,
    cta_tiler: Tuple[int, int, int] = (128, 128, 32),
    num_stages: int = 3,
):
    M, N, K, L = mnkl
    assert L == 1, "Batch >1 not yet supported in split-K example"
    assert split_k >= 1

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

    # Inputs
    mA, a_torch = create_and_permute_tensor(L, M, K, a_major == 'm', ab_dtype)
    mB, b_torch = create_and_permute_tensor(L, N, K, b_major == 'n', ab_dtype)

    # FP32 workspace C with L=split_k (reuse the same layout helper to satisfy leading_dim requirements)
    mW, w_torch = create_and_permute_tensor(split_k, M, N, c_major == 'm', cutlass.Float32)

    # Kernel: store dtype = FP32 (workspace), accumulate dtype = FP32
    gemm_kernel = TensorOpGemmSplitK(
        ab_dtype, cutlass.Float32, acc_dtype, atom_layout_mnk,
        cta_tiler=cta_tiler, num_stages=num_stages, split_k=split_k
    )

    if not skip_ref_check:
        ref = torch.einsum('mkl,nkl->mnl', a_torch.to(dtype=torch.float32), b_torch.to(dtype=torch.float32)).to(torch.float32)

    compiled_gemm = cute.compile(gemm_kernel, mA, mB, mW)

    # Warmup: kernel + reduction
    for _ in range(max(0, warmup_iterations)):
        compiled_gemm(mA, mB, mW)
        torch.cuda.synchronize()
        _ = w_torch.sum(dim=2)

    import time as _time
    torch.cuda.synchronize()
    t0 = _time.time()
    c_final_fp32 = None
    for _ in range(iterations):
        compiled_gemm(mA, mB, mW)
        torch.cuda.synchronize()
        c_final_fp32 = w_torch.sum(dim=2)
    torch.cuda.synchronize()
    t1 = _time.time()
    avg_time_us = (t1 - t0) * 1e6 / max(1, iterations)

    # Cast to requested output dtype
    c_final = c_final_fp32.to(cutlass_torch.dtype(c_dtype))

    if not skip_ref_check:
        torch.testing.assert_close(c_final_fp32.cpu(), ref.cpu(), atol=1e-3, rtol=1e-5)
        print("Reference check PASS")

    return avg_time_us


if __name__ == '__main__':
    def parse_ints(s: str) -> Tuple[int, ...]:
        return tuple(int(x.strip()) for x in s.split(','))

    parser = argparse.ArgumentParser(description='Split-K tensor core GEMM (CuTe DSL)')
    parser.add_argument('--mnkl', type=parse_ints, default=(112,136,40,1))
    parser.add_argument('--atom_layout_mnk', type=parse_ints, default=(2,2,1))
    parser.add_argument('--cta_tiler', type=parse_ints, default=(128,128,32))
    parser.add_argument('--num_stages', type=int, default=3)
    parser.add_argument('--split_k', type=int, default=4, help='Number of reduction slices along K')
    parser.add_argument('--ab_dtype', type=cutlass.dtype, choices=[cutlass.Float16], default=cutlass.Float16)
    parser.add_argument('--acc_dtype', type=cutlass.dtype, choices=[cutlass.Float32], default=cutlass.Float32)
    parser.add_argument('--c_dtype', type=cutlass.dtype, choices=[cutlass.Float16], default=cutlass.Float16)
    parser.add_argument('--a_major', choices=['k','m'], default='m')
    parser.add_argument('--b_major', choices=['k','n'], default='n')
    parser.add_argument('--c_major', choices=['n','m'], default='n')
    parser.add_argument('--warmup_iterations', type=int, default=2)
    parser.add_argument('--iterations', type=int, default=50)
    parser.add_argument('--skip_ref_check', action='store_true')
    parser.add_argument('--use_cold_l2', action='store_true')
    args = parser.parse_args()

    avg_us = run_splitk(
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
    print('PASS')
    print(f'Average execution time (split-K): {avg_us:.2f} us')
