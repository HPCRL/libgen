/*
 * cuBLAS GEMM Runner for NCU Profiling
 * 
 * This program directly calls cuBLAS to run FP16 GEMM operations.
 * Replaces the PyTorch-based approach to ensure only cuBLAS kernels are executed.
 * 
 * Compilation:
 *   nvcc -o cublas_gemm_runner cublas_gemm_runner.cu -lcublas
 * 
 * Usage:
 *   ./cublas_gemm_runner --M <M> --N <N> --K <K> [--batch <L>] [--iterations <I>] [--warmup <W>]
 */

#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cstdlib>
#include <cstring>
#include <ctime>

// Error checking macros
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CUBLAS_CHECK(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << "Error code: " << status << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Function to initialize matrix with random FP16 values
void initializeRandomMatrix(half* h_matrix, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        // Generate random float between -1 and 1, convert to half
        float val = 2.0f * (float)rand() / RAND_MAX - 1.0f;
        h_matrix[i] = __float2half(val);
    }
}

// Print usage information
void printUsage(const char* program_name) {
    std::cout << "Usage: " << program_name << " --M <M> --N <N> --K <K> [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --M <value>          M dimension (required)" << std::endl;
    std::cout << "  --N <value>          N dimension (required)" << std::endl;
    std::cout << "  --K <value>          K dimension (required)" << std::endl;
    std::cout << "  --batch <value>      Batch size (default: 1)" << std::endl;
    std::cout << "  --iterations <value> Number of iterations (default: 1)" << std::endl;
    std::cout << "  --warmup <value>     Number of warmup iterations (default: 1)" << std::endl;
    std::cout << "  --help               Show this help message" << std::endl;
}

int main(int argc, char** argv) {
    // Parse command line arguments
    int M = 0, N = 0, K = 0;
    int batch_size = 1;
    int iterations = 1;
    int warmup = 1;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--M") == 0 && i + 1 < argc) {
            M = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--N") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--K") == 0 && i + 1 < argc) {
            K = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) {
            batch_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            warmup = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << argv[i] << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }
    
    // Validate required arguments
    if (M <= 0 || N <= 0 || K <= 0) {
        std::cerr << "Error: M, N, and K must be positive integers" << std::endl;
        printUsage(argv[0]);
        return 1;
    }
    
    // Calculate GFLOPS (2*M*N*K*batch_size operations)
    double gflops_per_iter = (2.0 * M * N * K * batch_size) / 1e9;
    
    std::cout << "Running cuBLAS GEMM: M=" << M << ", N=" << N << ", K=" << K 
              << ", L=" << batch_size << std::endl;
    
    // Initialize cuBLAS
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    
    // Set math mode to allow tensor core usage for FP16
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    
    // Seed random number generator
    srand(time(NULL));
    
    // Matrix dimensions for column-major storage
    // C (MxN) = A (MxK) * B (KxN)
    size_t size_A = M * K;
    size_t size_B = K * N;
    size_t size_C = M * N;
    
    // For batched operations, multiply by batch size
    if (batch_size > 1) {
        size_A *= batch_size;
        size_B *= batch_size;
        size_C *= batch_size;
    }
    
    // Allocate host memory
    half* h_A = (half*)malloc(size_A * sizeof(half));
    half* h_B = (half*)malloc(size_B * sizeof(half));
    half* h_C = (half*)malloc(size_C * sizeof(half));
    
    if (!h_A || !h_B || !h_C) {
        std::cerr << "Failed to allocate host memory" << std::endl;
        return 1;
    }
    
    // Initialize matrices with random values
    initializeRandomMatrix(h_A, size_A);
    initializeRandomMatrix(h_B, size_B);
    initializeRandomMatrix(h_C, size_C);
    
    // Allocate device memory
    half *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size_A * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B, size_B * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C, size_C * sizeof(half)));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, size_C * sizeof(half), cudaMemcpyHostToDevice));
    
    // Setup GEMM parameters
    // For column-major: C = alpha * A * B + beta * C
    const half h_alpha = __float2half(1.0f);
    const half h_beta = __float2half(0.0f);
    
    // Leading dimensions
    int lda = M;
    int ldb = K;
    int ldc = M;
    
    if (batch_size == 1) {
        // Regular GEMM
        std::cout << "Performing regular GEMM..." << std::endl;
        
        // Warmup iterations
        for (int i = 0; i < warmup; ++i) {
            CUBLAS_CHECK(cublasGemmEx(
                handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                M, N, K,
                &h_alpha,
                d_A, CUDA_R_16F, lda,
                d_B, CUDA_R_16F, ldb,
                &h_beta,
                d_C, CUDA_R_16F, ldc,
                CUBLAS_COMPUTE_16F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP
            ));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Actual iterations with timing
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iterations; ++i) {
            CUBLAS_CHECK(cublasGemmEx(
                handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                M, N, K,
                &h_alpha,
                d_A, CUDA_R_16F, lda,
                d_B, CUDA_R_16F, ldb,
                &h_beta,
                d_C, CUDA_R_16F, ldc,
                CUBLAS_COMPUTE_16F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP
            ));
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        float avg_time_ms = milliseconds / iterations;
        float avg_time_us = avg_time_ms * 1000.0f;
        double gflops = gflops_per_iter / (avg_time_ms / 1000.0);
        
        std::cout << "[PERF] avg_time_us: " << avg_time_us << std::endl;
        std::cout << "[PERF] gflops: " << gflops << std::endl;
        
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        
    } else {
        // Strided batched GEMM
        std::cout << "Performing strided batched GEMM..." << std::endl;
        
        long long int strideA = M * K;
        long long int strideB = K * N;
        long long int strideC = M * N;
        
        // Warmup iterations
        for (int i = 0; i < warmup; ++i) {
            CUBLAS_CHECK(cublasGemmStridedBatchedEx(
                handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                M, N, K,
                &h_alpha,
                d_A, CUDA_R_16F, lda, strideA,
                d_B, CUDA_R_16F, ldb, strideB,
                &h_beta,
                d_C, CUDA_R_16F, ldc, strideC,
                batch_size,
                CUBLAS_COMPUTE_16F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP
            ));
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Actual iterations with timing
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iterations; ++i) {
            CUBLAS_CHECK(cublasGemmStridedBatchedEx(
                handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                M, N, K,
                &h_alpha,
                d_A, CUDA_R_16F, lda, strideA,
                d_B, CUDA_R_16F, ldb, strideB,
                &h_beta,
                d_C, CUDA_R_16F, ldc, strideC,
                batch_size,
                CUBLAS_COMPUTE_16F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP
            ));
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        float avg_time_ms = milliseconds / iterations;
        float avg_time_us = avg_time_ms * 1000.0f;
        double gflops = gflops_per_iter / (avg_time_ms / 1000.0);
        
        std::cout << "[PERF] avg_time_us: " << avg_time_us << std::endl;
        std::cout << "[PERF] gflops: " << gflops << std::endl;
        
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }
    
    // Copy result back to host (optional, but good for validation)
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C * sizeof(half), cudaMemcpyDeviceToHost));
    
    std::cout << "Result shape: ";
    if (batch_size > 1) {
        std::cout << "(" << batch_size << ", " << M << ", " << N << ")" << std::endl;
    } else {
        std::cout << "(" << M << ", " << N << ")" << std::endl;
    }
    std::cout << "cuBLAS GEMM completed successfully" << std::endl;
    
    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUBLAS_CHECK(cublasDestroy(handle));
    
    return 0;
}
