// Export CUTLASS Repo: export CUTLASS=/path/to/cutlass
// nvcc -O3 -std=c++17 -arch=sm_80 -I"$CUTLASS"/include -o cutlass_dyn_gemm_padded_bench_3x main_padded_bench_3x.cu
//
// main_padded_bench_3x.cu
// CUTLASS 3.x Tensor Core GEMM with dynamic shapes using padded leading dimensions
// + simple benchmarking (CUDA events): runtime (min/avg ms) and GFLOPs.
// Reports both "logical" GFLOPs (2*M*N*K) and "executed" GFLOPs (2*M*Np*Kp).
//
// Run examples:
//   ./cutlass_dyn_gemm_padded_bench_3x
//   ./cutlass_dyn_gemm_padded_bench_3x 513 1027 257
//   ./cutlass_dyn_gemm_padded_bench_3x 4096 4096 4096 100 10   # reps=100, warmups=10
//
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cassert>
#include <cstring>
#include <algorithm>

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/default_epilogue.hpp>
#include <cutlass/epilogue/thread/linear_combination.h>

#include "cute/tensor.hpp"

#define CHECK_CUDA(call) do {                                     \
  cudaError_t status_ = (call);                                   \
  if (status_ != cudaSuccess) {                                   \
    std::cerr << "CUDA Error: " << cudaGetErrorString(status_)    \
              << " at " << __FILE__ << ":" << __LINE__ << "\n";   \
    std::exit(1);                                                 \
  }                                                               \
} while (0)

using Elem   = cutlass::half_t;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;

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
      float c = h2f(C[m * ldc + n]);
      float d = alpha * acc[m * N + n] + beta * c;
      D[m * ldd + n] = f2h(d);
    }
  }
}

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
      if (diff > tol) return false;
    }
  }
  return true;
}

int main(int argc, char** argv) {
  int M = 1024, N = 768, K = 513;
  int reps = 50, warmups = 10;
  if (argc >= 4) { M = std::atoi(argv[1]); N = std::atoi(argv[2]); K = std::atoi(argv[3]); }
  if (argc >= 5) { reps = std::max(1, std::atoi(argv[4])); }
  if (argc >= 6) { warmups = std::max(0, std::atoi(argv[5])); }

  std::cout << "Running CUTLASS 3.x Tensor Core GEMM with padded LDs:\n"
            << "  M=" << M << "  N=" << N << "  K=" << K
            << " | reps=" << reps << " warmups=" << warmups << "\n";

  constexpr int AlignA = 8;
  constexpr int AlignB = 8;
  constexpr int AlignC = 8;

  int Kp = round_up(K, AlignA);
  int Np = round_up(N, AlignB);

  int lda = Kp;
  int ldb = Np;
  int ldc = Np;
  int ldd = Np;

  float alpha = 1.0f;
  float beta  = 1.0f;

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

  Elem *dA=nullptr, *dB=nullptr, *dC=nullptr, *dD=nullptr;
  CHECK_CUDA(cudaMalloc((void**)&dA, sizeof(Elem) * hA.size()));
  CHECK_CUDA(cudaMalloc((void**)&dB, sizeof(Elem) * hB.size()));
  CHECK_CUDA(cudaMalloc((void**)&dC, sizeof(Elem) * hC.size()));
  CHECK_CUDA(cudaMalloc((void**)&dD, sizeof(Elem) * hD.size()));

  CHECK_CUDA(cudaMemcpy(dA, hA.data(), sizeof(Elem) * hA.size(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB, hB.data(), sizeof(Elem) * hB.size(), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dC, hC.data(), sizeof(Elem) * hC.size(), cudaMemcpyHostToDevice));

  // CUTLASS 3.x API setup
  using ElementA = Elem;
  using ElementB = Elem;
  using ElementC = Elem;
  using ElementAccumulator = float;
  using ElementCompute = float;

  using ArchTag = cutlass::arch::Sm80;
  using OpClass = cutlass::arch::OpClassTensorOp;

  // Tile shape for collective
  using TileShape = cute::Shape<cute::Int<128>, cute::Int<128>, cute::Int<64>>;
  
  // Cluster shape (no clustering for Sm80)
  using ClusterShape = cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>;

  // Step 1: Build the collective mainloop using CollectiveBuilder
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OpClass,
      ElementA, LayoutA, AlignA,
      ElementB, LayoutB, AlignB,
      ElementAccumulator,
      TileShape, ClusterShape,
      cutlass::gemm::collective::StageCountAuto,
      cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

  // Step 2: Define the epilogue fusion
  using EpilogueThreadOp = cutlass::epilogue::thread::LinearCombination<
      ElementC, 1,
      ElementAccumulator, ElementCompute
  >;

  // Step 3: Create the collective epilogue
  using CollectiveEpilogue = cutlass::epilogue::collective::DefaultEpilogue<
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      cutlass::gemm::TagToStrideC_t<LayoutC>,
      EpilogueThreadOp,
      cutlass::gemm::EpilogueDefault
  >;

  // Step 4: Compose the kernel
  using ProblemShape = cute::Shape<int, int, int>;  // M, N, K
  
  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
      ProblemShape,
      CollectiveMainloop,
      CollectiveEpilogue
  >;

  // Step 5: Create the device-level GEMM handle
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

  // Problem size with padded dimensions
  typename Gemm::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {M, Np, Kp},  // Problem shape (using padded N and K)
      {dA, cute::make_stride(lda, cute::Int<1>{})},           // A tensor with stride
      {dB, cute::make_stride(ldb, cute::Int<1>{})},           // B tensor with stride
      {{alpha, beta}, dC, cute::make_stride(ldc, cute::Int<1>{}), dD, cute::make_stride(ldd, cute::Int<1>{})}  // Epilogue args
  };

  Gemm gemm_op;

  // Check if implementation is supported
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::Status status = gemm_op.can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Gemm::can_implement() says not supported: "
              << static_cast<int>(status) << "\n";
    return 1;
  }

  // Allocate workspace if needed
  void* workspace = nullptr;
  if (workspace_size > 0) {
    CHECK_CUDA(cudaMalloc(&workspace, workspace_size));
  }

  // Initialize the GEMM object
  status = gemm_op.initialize(arguments, workspace);
  if (status != cutlass::Status::kSuccess) {
    std::cerr << "Gemm::initialize() failed: " << static_cast<int>(status) << "\n";
    return 1;
  }

  // Warmups
  for (int i = 0; i < warmups; ++i) {
    status = gemm_op.run();
    if (status != cutlass::Status::kSuccess) { 
      std::cerr << "Gemm::run() failed during warmup\n"; 
      return 1; 
    }
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  // Timed runs
  std::vector<float> times_ms;
  times_ms.reserve(reps);

  cudaEvent_t startE, stopE;
  CHECK_CUDA(cudaEventCreate(&startE));
  CHECK_CUDA(cudaEventCreate(&stopE));

  for (int i = 0; i < reps; ++i) {
    CHECK_CUDA(cudaEventRecord(startE));
    status = gemm_op.run();
    CHECK_CUDA(cudaEventRecord(stopE));
    if (status != cutlass::Status::kSuccess) { 
      std::cerr << "Gemm::run() failed\n"; 
      return 1; 
    }
    CHECK_CUDA(cudaEventSynchronize(stopE));
    float ms = 0.f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, startE, stopE));
    times_ms.push_back(ms);
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  CHECK_CUDA(cudaEventDestroy(startE));
  CHECK_CUDA(cudaEventDestroy(stopE));

  // Stats
  auto min_ms = *std::min_element(times_ms.begin(), times_ms.end());
  auto max_ms = *std::max_element(times_ms.begin(), times_ms.end());
  double avg_ms = 0.0;
  for (auto t : times_ms) avg_ms += t;
  avg_ms /= static_cast<double>(times_ms.size());

  // Copy back D for correctness
  CHECK_CUDA(cudaMemcpy(hD.data(), dD, sizeof(Elem) * hD.size(), cudaMemcpyDeviceToHost));

  // Reference & check only logical region MxN
  {
    std::vector<Elem> D_ref(M * ldd, f2h(0.0f));
    cpu_gemm_ref(M, N, K, alpha, hA, lda, hB, ldb, beta, hC, ldc, D_ref, ldd);
    bool ok = allclose_half_region(hD, D_ref, M, N, ldd, ldd);
    std::cout << "[CHECK] " << (ok ? "PASS ✅" : "FAIL ❌") << "\n";
  }

  // FLOP counts
  const double flops_logical  = 2.0 * static_cast<double>(M)  * static_cast<double>(N)  * static_cast<double>(K);
  const double flops_executed = 2.0 * static_cast<double>(M)  * static_cast<double>(Np) * static_cast<double>(Kp);

  const double sec_min = min_ms * 1e-3;
  const double sec_max = max_ms * 1e-3;
  const double sec_avg = avg_ms * 1e-3;

  const double gflops_logical_max  = flops_logical  / sec_min / 1e9;
  const double gflops_logical_min = flops_logical  / sec_max / 1e9;
  const double gflops_logical_avg  = flops_logical  / sec_avg / 1e9;
  const double gflops_exec_max     = flops_executed / sec_min / 1e9;
  const double gflops_exec_min     = flops_executed / sec_max / 1e9;
  const double gflops_exec_avg     = flops_executed / sec_avg / 1e9;

  std::cout << "Timing (CUDA events):\n"
            << "  min: " << min_ms << " ms,  max: " << max_ms << " ms,  avg: " << avg_ms << " ms\n"
            << "GFLOPs (logical  MxNxK):\n"
            << "  min: " << gflops_logical_min << ",  max: " << gflops_logical_max <<  ",  avg: " << gflops_logical_avg << "\n"
            << "GFLOPs (executed MxNpxKp):\n"
            << "  min: " << gflops_exec_min << ",  max: " << gflops_exec_max << ",  avg: " << gflops_exec_avg  << "\n";

  if (workspace) cudaFree(workspace);
  cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dD);
  return 0;
}