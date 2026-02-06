// main_padded_bench_3x.cu
// CUTLASS 3.x Tensor Core GEMM with dynamic shapes + padded leading dims
// Benchmarks min/max/avg time (CUDA events) and prints logical/executed GFLOPs.
// Usage:
//   ./cutlass3_dyn_gemm_padded_bench
//   ./cutlass3_dyn_gemm_padded_bench 513 1027 257
//   ./cutlass3_dyn_gemm_padded_bench 4096 4096 4096 100 10

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

// ---- CUTLASS 3.x includes (device + kernel + collectives) ----
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/layout/matrix.h>

#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/default_epilogue.hpp>
#include <cutlass/epilogue/thread/linear_combination.h>

#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>

#include <cutlass/util/host_tensor.h>       // only for size helpers if desired
#include <cutlass/util/packed_stride.hpp>    // make_cute_packed_stride
#include <cute/tensor.hpp>

#define CHECK_CUDA(call) do {                                     \
  cudaError_t status_ = (call);                                   \
  if (status_ != cudaSuccess) {                                   \
    std::cerr << "CUDA Error: " << cudaGetErrorString(status_)    \
              << " at " << __FILE__ << ":" << __LINE__ << "\n";   \
    std::exit(1);                                                 \
  }                                                               \
} while (0)

#define CHECK_CUTLASS(call) do {                                  \
  cutlass::Status st_ = (call);                                   \
  if (st_ != cutlass::Status::kSuccess) {                         \
    std::cerr << "CUTLASS Error: " << int(st_)                    \
              << " at " << __FILE__ << ":" << __LINE__ << "\n";   \
    std::exit(1);                                                 \
  }                                                               \
} while (0)

using Elem   = cutlass::half_t;
using Layout = cutlass::layout::RowMajor;

inline cutlass::half_t f2h(float x) { return cutlass::half_t(x); }
inline float h2f(cutlass::half_t h) { return static_cast<float>(h); }

static inline int round_up(int x, int a) { return ((x + a - 1) / a) * a; }

template <typename T>
void fill_random(std::vector<T>& v, unsigned seed=1234) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& x : v) x = T(dist(rng));
}
template <>
void fill_random<cutlass::half_t>(std::vector<cutlass::half_t>& v, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& x : v) x = f2h(dist(rng));
}

void cpu_gemm_ref(int M, int N, int K,
                  float alpha,
                  const std::vector<Elem>& A, int lda,
                  const std::vector<Elem>& B, int ldb,
                  float beta,
                  const std::vector<Elem>& C, int ldc,
                  std::vector<Elem>& D, int ldd) {
  std::vector<float> acc(M * N, 0.0f);
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
      float d = alpha * acc[m * N + n] + beta * h2f(C[m * ldc + n]);
      D[m * ldd + n] = f2h(d);
    }
  }
}

bool allclose_MxN(const Elem* D, int ldd,
                  const Elem* D_ref, int ldd_ref,
                  int M, int N,
                  float atol = 5e-2f, float rtol = 1e-2f) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      float x = h2f(D    [m * ldd     + n]);
      float y = h2f(D_ref[m * ldd_ref + n]);
      float diff = std::abs(x - y);
      float tol  = atol + rtol * std::abs(y);
      if (diff > tol) return false;
    }
  }
  return true;
}

int main(int argc, char** argv) {
  int M = 1024, N = 768, K = 512;
  int reps = 50, warmups = 10;
  if (argc >= 4) { M = std::atoi(argv[1]); N = std::atoi(argv[2]); K = std::atoi(argv[3]); }
  if (argc >= 5) reps    = std::atoi(argv[4]);
  if (argc >= 6) warmups = std::atoi(argv[5]);

  std::cout << "Running CUTLASS 3.x Tensor Core GEMM with dynamic sizes (RowMajor):\n"
            << "  M=" << M << "  N=" << N << "  K=" << K
            << " | reps=" << reps << " warmups=" << warmups << "\n";

  // 16B vectorization for fp16 => 8 elements
  constexpr int AlignA = 8;
  constexpr int AlignB = 8;
  constexpr int AlignC = 8;

  // pad K for A’s lda, pad N for B/C/D ldb/ldc/ldd (RowMajor)
  int Kp = round_up(K, AlignA);
  int Np = round_up(N, AlignB);

  int lda = Kp;
  int ldb = Np;
  int ldc = Np;
  int ldd = Np;

  float alpha = 1.0f;
  float beta  = 1.0f;

  // Host buffers with row-major padding
  std::vector<Elem> hA(M * lda, f2h(0.0f));
  std::vector<Elem> hB(Kp * ldb, f2h(0.0f));
  std::vector<Elem> hC(M * ldc, f2h(0.0f));
  std::vector<Elem> hD(M * ldd, f2h(0.0f));
  std::vector<Elem> hD_ref(M * ldd, f2h(0.0f));

  { std::vector<Elem> tmp(M * K); fill_random(tmp, 123);
    for (int m = 0; m < M; ++m) std::memcpy(&hA[m * lda], &tmp[m * K], sizeof(Elem) * K); }
  { std::vector<Elem> tmp(K * N); fill_random(tmp, 456);
    for (int k = 0; k < K; ++k) std::memcpy(&hB[k * ldb], &tmp[k * N], sizeof(Elem) * N); }
  { std::vector<Elem> tmp(M * N); fill_random(tmp, 789);
    for (int m = 0; m < M; ++m) std::memcpy(&hC[m * ldc], &tmp[m * N], sizeof(Elem) * N); }

  // Device buffers
  Elem *dA=nullptr, *dB=nullptr, *dC=nullptr, *dD=nullptr;
  CHECK_CUDA(cudaMalloc((void**)&dA, sizeof(Elem) * hA.size()));
  CHECK_CUDA(cudaMalloc((void**)&dB, sizeof(Elem) * hB.size()));
  CHECK_CUDA(cudaMalloc((void**)&dC, sizeof(Elem) * hC.size()));
  CHECK_CUDA(cudaMalloc((void**)&dD, sizeof(Elem) * hD.size()));

  CHECK_CUDA(cudaMemcpy(dA, hA.data(), sizeof(Elem) * hA.size(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB, hB.data(), sizeof(Elem) * hB.size(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dC, hC.data(), sizeof(Elem) * hC.size(), cudaMemcpyHostToDevice));

  // ----------------------------
  // CUTLASS 3.x GEMM definition
  // ----------------------------
  using ElementA = Elem;
  using ElementB = Elem;
  using ElementC = Elem;
  using ElementD = Elem; // epilogue output
  using ElementAccumulator = float;

  using ArchTag       = cutlass::arch::Sm80;                // RTX-3090
  using OperatorClass = cutlass::arch::OpClassTensorOp;

  using namespace cute;
  using TileShape    = cute::Shape<cute::Int<128>, cute::Int<128>, cute::Int<64>>; // TB tile
  using ClusterShape = cute::Shape<cute::Int<1>,   cute::Int<2>,   cute::Int<1>>;  // TB cluster

  // Build mainloop & epilogue collectives (3.x)
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass,
      ElementA, cutlass::layout::RowMajor, AlignA,
      ElementB, cutlass::layout::RowMajor, AlignB,
      ElementAccumulator,
      TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
    >::CollectiveOp;

  using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogue<
      cutlass::gemm::TagToStrideC_t<cutlass::layout::RowMajor>, // C stride tag
      cutlass::gemm::TagToStrideC_t<cutlass::layout::RowMajor>, // D stride tag
      cutlass::epilogue::thread::LinearCombination<ElementD, AlignC, ElementAccumulator, ElementAccumulator>>;

  // Kernel and device adapter (3.x API)
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int,int>,   // Problem shape: (M,N,K, batch)
      CollectiveMainloop,
      CollectiveEpilogue>;

  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  // Prepare cute-packed strides that include our padded leading dims.
  using StrideA = typename Gemm::GemmKernel::StrideA; // usually (lda, 1, batch_strideA)
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  // For RowMajor tensors, pass extents (rows, cols_padded, batch)
  StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, Kp, 1});
  StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, {Kp, Np, 1});
  StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, Np, 1});
  StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, Np, 1});

  // Problem shape (M,N,K,B) with B=1
  auto problem = cute::make_shape(M, N, K, 1);

  // Epilogue (alpha, beta)
  typename Gemm::EpilogueOutputOp::Params epilogue_params(ElementAccumulator(alpha), ElementAccumulator(beta));

  // Build 3.x Arguments: {Problem, MainloopArgs{A, strideA, B, strideB}, EpilogueArgs{params, C, strideC, D, strideD}}
  typename Gemm::Arguments arguments{
      problem,
      { dA, stride_A, dB, stride_B },
      { epilogue_params, dC, stride_C, dD, stride_D }
  };

  // can_implement + initialize
  {
    Gemm op;
    cutlass::Status st = op.can_implement(arguments);
    if (st != cutlass::Status::kSuccess) {
      std::cerr << "Gemm::can_implement() says not supported: " << int(st) << "\n";
      return 1;
    }
  }

  // ----------------------------
  // Reference check (CPU)
  // ----------------------------
  cpu_gemm_ref(M, N, K, alpha, hA, lda, hB, ldb, beta, hC, ldc, hD_ref, ldd);

  // ----------------------------
  // Warmups + Timed runs
  // ----------------------------
  Gemm gemm_op;
  CHECK_CUTLASS(gemm_op.initialize(arguments));

  for (int i = 0; i < warmups; ++i) {
    CHECK_CUTLASS(gemm_op.run());
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  float min_ms=1e30f, max_ms=0.f, sum_ms=0.f;
  cudaEvent_t startE, stopE; CHECK_CUDA(cudaEventCreate(&startE)); CHECK_CUDA(cudaEventCreate(&stopE));

  for (int i = 0; i < reps; ++i) {
    CHECK_CUDA(cudaEventRecord(startE, 0));
    CHECK_CUTLASS(gemm_op.run());
    CHECK_CUDA(cudaEventRecord(stopE, 0));
    CHECK_CUDA(cudaEventSynchronize(stopE));
    float ms=0.f; CHECK_CUDA(cudaEventElapsedTime(&ms, startE, stopE));
    min_ms = std::min(min_ms, ms);
    max_ms = std::max(max_ms, ms);
    sum_ms += ms;
  }
  CHECK_CUDA(cudaEventDestroy(startE)); CHECK_CUDA(cudaEventDestroy(stopE));
  CHECK_CUDA(cudaDeviceSynchronize());

  // Copy back D and verify on logical region (MxN)
  CHECK_CUDA(cudaMemcpy(hD.data(), dD, sizeof(Elem) * hD.size(), cudaMemcpyDeviceToHost));
  bool ok = allclose_MxN(hD.data(), ldd, hD_ref.data(), ldd, M, N);
  std::cout << "[CHECK] " << (ok ? "PASS ✅ (within tolerance)" : "FAIL ❌") << "\n";

  // Metrics: logical vs executed flops
  const double flops_logical  = 2.0 * double(M) * double(N)  * double(K);
  const double flops_exec     = 2.0 * double(M) * double(Np) * double(Kp);
  const double sec_min = min_ms / 1e3, sec_max = max_ms / 1e3, sec_avg = (sum_ms / reps) / 1e3;

  const double gflops_log_min = flops_logical / sec_max / 1e9;
  const double gflops_log_max = flops_logical / sec_min / 1e9;
  const double gflops_log_avg = flops_logical / sec_avg / 1e9;

  const double gflops_exe_min = flops_exec / sec_max / 1e9;
  const double gflops_exe_max = flops_exec / sec_min / 1e9;
  const double gflops_exe_avg = flops_exec / sec_avg / 1e9;

  std::cout << "Timing (CUDA events):\n"
            << "  min: " << min_ms << " ms,  max: " << max_ms << " ms,  avg: " << (sum_ms/reps) << " ms\n"
            << "GFLOPs (logical  MxNxK):\n"
            << "  min: " << gflops_log_min << ",  max: " << gflops_log_max <<  ",  avg: " << gflops_log_avg << "\n"
            << "GFLOPs (executed MxNpxKp):\n"
            << "  min: " << gflops_exe_min << ",  max: " << gflops_exe_max << ",  avg: " << gflops_exe_avg  << "\n";

  cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dD);
  return ok ? 0 : 1;
}
