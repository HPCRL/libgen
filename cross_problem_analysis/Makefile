# Makefile for cuBLAS GEMM Runner
# This program directly calls cuBLAS for profiling

CXX = nvcc
TARGET = cublas_gemm_runner
SRC = cublas_gemm_runner.cu

# Compiler flags
CXXFLAGS = -O3 -std=c++11
LDFLAGS = -lcublas

# CUDA architecture (adjust for your GPU)
# Common options:
#   -arch=sm_70 for V100
#   -arch=sm_75 for Turing (RTX 20xx)
#   -arch=sm_80 for A100
#   -arch=sm_86 for RTX 30xx
#   -arch=sm_89 for RTX 40xx
ARCH = -arch=sm_80

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(ARCH) $(SRC) $(LDFLAGS) -o $(TARGET)

clean:
	rm -f $(TARGET)

test: $(TARGET)
	@echo "Testing cuBLAS GEMM runner..."
	@echo "Test 1: Small GEMM (M=128, N=128, K=128)"
	./$(TARGET) --M 128 --N 128 --K 128
	@echo ""
	@echo "Test 2: Batched GEMM (M=256, N=256, K=256, batch=4)"
	./$(TARGET) --M 256 --N 256 --K 256 --batch 4
	@echo ""
	@echo "All tests passed!"

help:
	@echo "Makefile for cuBLAS GEMM Runner"
	@echo ""
	@echo "Targets:"
	@echo "  all (default) - Build the runner"
	@echo "  clean         - Remove built files"
	@echo "  test          - Run basic tests"
	@echo "  help          - Show this help"
	@echo ""
	@echo "Usage:"
	@echo "  make           # Build"
	@echo "  make test      # Build and test"
	@echo "  make clean     # Clean up"
	@echo ""
	@echo "To change GPU architecture, edit ARCH variable in Makefile"
	@echo "Current: $(ARCH)"
