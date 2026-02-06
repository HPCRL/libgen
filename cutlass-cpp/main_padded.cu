// Export CUTLASS Repo: export CUTLASS=/path/to/cutlass
// nvcc -O3 -std=c++17 -arch=sm_86 -I"$CUTLASS"/include -o cutlass_dyn_gemm_padded main_padded.cu

// main_padded.cu
// CUTLASS Tensor Core GEMM with dynamic shapes using *padded leading dimensions*
// to preserve vectorized loads/stores (e.g., alignment=8 for FP16).
//
// A,B,C,D: FP16 row-major; Accumulator: FP32
// Pads K and N up to a multiple of the vector alignment and zero-fills the
// padded regions, so the kernel can keep vectorized accesses while supporting
// arbitrary (M, N, K).
//
// Run (examples):
//   ./cutlass_dyn_gemm_padded             # defaults M=1024 N=768 K=513 (odd K to show padding)
//   ./cutlass_dyn_gemm_padded 513 1027 257
//
// Notes:
// - We set AlignmentA=B=C/D=8 (elements), common for FP16 (8 half = 16B).
// - We form padded sizes: Kp = round_up(K, 8), Np = round_up(N, 8).
// - We allocate A[M x Kp], B[Kp x Np], C/D[M x Np] and zero-fill padded regions.
// - We launch GEMM for (M, Np, Kp). Afterward, we validate only the leading N columns.
//
// This keeps the original "style" while only changing how memory is allocated &
/*  strides are chosen to satisfy the alignment requirements. */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <cstring>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/arch/arch.h>
#include <cutlass/numeric_types.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>

#define CHECK_CUDA(call) do {                                     \
  cudaError_t status_ = (call);                                   \
  if (status_ != cudaSuccess) {                                   \
    std::cerr << "CUDA Error: " << cudaGetErrorString(status_)    \
              << " at " << __FILE__ << ":" << __LINE__ << "\n";   \
    std::exit(1);                                                 \
  }                                                               \
} while (0)

using Elem   = cutlass::half_t;
using Layout = cutlass::layout::RowMajor;

// Utility: convert float->half
inline cutlass::half_t f2h(float x) { return cutlass::half_t(x); }
// Utility: convert half->float
inline float h2f(cutlass::half_t h) { return static_cast<float>(h); }

// Round up x to next multiple of a (a>0)
static inline int round_up(int x, int a) {
  return ((x + a - 1) / a) * a;
}

// Fill host vector with random values in [-1,1]
template <typename T>
void fill_random(std::vector<T>& v, unsigned seed=1234) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& x : v) x = T(dist(rng));
}
// Specialization for half
template <>
void fill_random<cutlass::half_t>(std::vector<cutlass::half_t>& v, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& x : v) x = f2h(dist(rng));
}

// CPU reference GEMM: D = alpha * A * B + beta * C
// A(MxK) row-major, B(KxN) row-major, C/D(MxN) row-major
void cpu_gemm_ref(int M, int N, int K,
                  float alpha,
                  const std::vector<Elem>& A, int lda,
                  const std::vector<Elem>& B, int ldb,
                  float beta,
                  const std::vector<Elem>& C, int ldc,
                  std::vector<Elem>& D, int ldd) {
  std::vector<float> acc(M * N, 0.0f);
  // acc[m,n] = sum_k A[m,k]*B[k,n]
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float s = 0.f;
      for (int k = 0; k < K; ++k) {
        float a = h2f(A[m * lda + k]);
        float b = h2f(B[k * ldb + n]);
        s += a * b;
      }
      acc[m * N + n] = s;
    }
  }
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float c = h2f(C[m * ldc + n]);
      float d = alpha * acc[m * N + n] + beta * c;
      D[m * ldd + n] = f2h(d);
    }
  }
}

// Compare only the leading MxN region of D vs D_ref
bool allclose_half_region(const std::vector<Elem>& D,
                          const std::vector<Elem>& D_ref,
                          int M, int N, int ldd, int ldd_ref,
                          float atol = 5e-2f, float rtol = 1e-2f) {
  auto af = [&](Elem h){ return h2f(h); };
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float x = af(D    [m * ldd     + n]);
      float y = af(D_ref[m * ldd_ref + n]);
      float diff = std::abs(x - y);
      float tol  = atol + rtol * std::abs(y);
      if (diff > tol) {
        std::cerr << "Mismatch at (" << m << "," << n << "): D="
                  << x << " ref=" << y << " diff=" << diff
                  << " tol=" << tol << "\n";
        return false;
      }
    }
  }
  return true;
}

int main(int argc, char** argv) {
  // Problem sizes (runtime)
  int M = 1024, N = 768, K = 513; // choose odd K to demonstrate padding
  if (argc >= 4) {
    M = std::atoi(argv[1]);
    N = std::atoi(argv[2]);
    K = std::atoi(argv[3]);
  }

  std::cout << "Running CUTLASS Tensor Core GEMM with padded LDs:\n"
            << "  M=" << M << "  N=" << N << "  K=" << K << "\n";

  // Vector alignment in *elements* (FP16 -> 8 elems = 16 bytes)
  constexpr int AlignA = 8;
  constexpr int AlignB = 8;
  constexpr int AlignC = 8;

  // Padded sizes
  int Kp = round_up(K, AlignA);   // for A (row-major, contiguous: K)
  int Np = round_up(N, AlignB);   // for B/C/D (row-major, contiguous: N)

  // Leading dimensions (row-major tensors)
  int lda = Kp;   // A: M x Kp
  int ldb = Np;   // B: Kp x Np
  int ldc = Np;   // C: M x Np
  int ldd = Np;   // D: M x Np

  float alpha = 1.0f;
  float beta  = 1.0f;

  // Host buffers (padded)
  std::vector<Elem> hA(M * lda, f2h(0.0f));
  std::vector<Elem> hB(Kp * ldb, f2h(0.0f));
  std::vector<Elem> hC(M * ldc, f2h(0.0f));
  std::vector<Elem> hD(M * ldd, f2h(0.0f));
  std::vector<Elem> hD_ref(M * ldd, f2h(0.0f));

  // Fill the *valid* regions with random values; leave padded areas as zero.
  // A: fill columns [0..K-1] for all rows
  {
    std::vector<Elem> tmp(M * K);
    fill_random(tmp, 123);
    for (int m = 0; m < M; ++m) {
      std::memcpy(&hA[m * lda], &tmp[m * K], sizeof(Elem) * K);
    }
  }
  // B: valid region is K x N within Kp x Np
  {
    std::vector<Elem> tmp(K * N);
    fill_random(tmp, 456);
    for (int k = 0; k < K; ++k) {
      std::memcpy(&hB[k * ldb], &tmp[k * N], sizeof(Elem) * N);
    }
  }
  // C: valid region is M x N within M x Np
  {
    std::vector<Elem> tmp(M * N);
    fill_random(tmp, 789);
    for (int m = 0; m < M; ++m) {
      std::memcpy(&hC[m * ldc], &tmp[m * N], sizeof(Elem) * N);
    }
  }

  // Device allocations
  Elem *dA=nullptr, *dB=nullptr, *dC=nullptr, *dD=nullptr;
  CHECK_CUDA(cudaMalloc((void**)&dA, sizeof(Elem) * hA.size()));
  CHECK_CUDA(cudaMalloc((void**)&dB, sizeof(Elem) * hB.size()));
  CHECK_CUDA(cudaMalloc((void**)&dC, sizeof(Elem) * hC.size()));
  CHECK_CUDA(cudaMalloc((void**)&dD, sizeof(Elem) * hD.size()));

  CHECK_CUDA(cudaMemcpy(dA, hA.data(), sizeof(Elem) * hA.size(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB, hB.data(), sizeof(Elem) * hB.size(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dC, hC.data(), sizeof(Elem) * hC.size(), cudaMemcpyHostToDevice));

  // -----------------------------
  // Define a CUTLASS TensorOp GEMM kernel for Ampere+ with vector alignment
  // -----------------------------
  using ElementInput       = Elem;                 // A,B
  using ElementOutput      = Elem;                 // C,D
  using ElementAccumulator = float;                // Accumulator / compute

  // Epilogue: vectorized stores (ElementsPerAccess = AlignC)
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput,
      AlignC,                 // elements per vectorized store
      ElementAccumulator,
      ElementAccumulator      // alpha/beta in FP32
  >;

  using Gemm = cutlass::gemm::device::Gemm<
      // A
      ElementInput, cutlass::layout::RowMajor,
      // B
      ElementInput, cutlass::layout::RowMajor,
      // C/D
      ElementOutput, cutlass::layout::RowMajor,
      // Accumulator & math/arch
      ElementAccumulator,
      cutlass::arch::OpClassTensorOp,
      cutlass::arch::Sm80,
      // Tile shapes
      cutlass::gemm::GemmShape<128, 128, 64>,   // Threadblock
      cutlass::gemm::GemmShape<64, 64, 64>,     // Warp
      cutlass::gemm::GemmShape<16, 8, 8>,       // MMA (FP16 Tensor Core on Ampere)
      // Epilogue
      // EpilogueOp,
      // Swizzle & stages
      // cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
      // 2
  >;

  // Launch with *padded* problem size (M, Np, Kp)
  cutlass::gemm::GemmCoord problem_size(M, Np, Kp);

  EpilogueOp::Params epilogue_params(alpha, beta);

  typename Gemm::Arguments args{
      problem_size,
      {dA, lda},
      {dB, ldb},
      {dC, ldc},
      {dD, ldd},
      epilogue_params
  };

  Gemm gemm_op;

  // Quick preflight
  cutlass::Status status = gemm_op.can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Gemm::can_implement() says not supported: "
              << static_cast<int>(status) << "\n";
    return 1;
  }

  status = gemm_op.initialize(args);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Gemm::initialize() failed: " << static_cast<int>(status) << "\n";
    return 1;
  }

  status = gemm_op();
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Gemm::run() failed: " << static_cast<int>(status) << "\n";
    return 1;
  }

  // Copy back D
  CHECK_CUDA(cudaMemcpy(hD.data(), dD, sizeof(Elem) * hD.size(), cudaMemcpyDeviceToHost));

  // CPU reference (un-padded) to compare against the leading region
  {
    std::vector<Elem> D_ref(M * ldd, f2h(0.0f));
    cpu_gemm_ref(M, N, K, alpha,
                 hA, lda, hB, ldb, beta, hC, ldc, D_ref, ldd);
    // Compare only first N columns
    bool ok = allclose_half_region(hD, D_ref, M, N, ldd, ldd);
    if (ok) {
      std::cout << "[CHECK] PASS ✅ (within tolerance)\n";
    } else {
      std::cout << "[CHECK] FAIL ❌\n";
    }
  }

  // Cleanup
  cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dD);
  return 0;
}
