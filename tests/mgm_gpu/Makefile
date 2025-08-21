# Makefile for MGM Stereo GPU Tests

# Compiler settings
CXX = g++
NVCC = nvcc 
CXXFLAGS = -std=c++11 -O2 -w -I ../../spgpu/correlator/
TEST_DEFINES = -DCOMPILE_FOR_TESTING

# CUDA architecture (adjust based on your GPU)
CUDA_ARCH = -arch=sm_70

# Package config for dependencies
OPENCV_FLAGS = -I/usr/local/include/opencv4 -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_calib3d -lopencv_cudaimgproc -lopencv_cudastereo -lopencv_imgcodecs

GDAL_FLAGS = -I/usr/include/gdal -lgdal

CUDA_FLAGS = -DHAVE_OPENCV_CUDA

# Combine all flags
NVCC_FLAGS = $(CUDA_ARCH) $(TEST_DEFINES) $(CUDA_FLAGS) $(OPENCV_FLAGS) $(GDAL_FLAGS) -I ../../spgpu/correlator/

CXX_FLAGS = $(CXXFLAGS) $(TEST_DEFINES) $(CUDA_FLAGS) $(OPENCV_FLAGS) $(GDAL_FLAGS)

# File names
BASE_NAME = mgm_gpu
ORIGINAL_SOURCE = ../../spgpu/correlator/$(BASE_NAME).cu
TEST_SOURCE = test_$(BASE_NAME).cu
TEST_EXECUTABLE = test_$(BASE_NAME)
TEST_LOG = test_results.log

# Default target
.PHONY: all
all: test

# Build the test executable
$(TEST_EXECUTABLE): $(TEST_SOURCE) $(ORIGINAL_SOURCE)
	@echo "Building test executable..."
	@echo "Compiler flags: $(NVCC_FLAGS)"
	$(NVCC) $(TEST_SOURCE) -o $(TEST_EXECUTABLE) $(NVCC_FLAGS)
	@echo "Build successful."

# Run tests
.PHONY: test
test: $(TEST_EXECUTABLE)
	@echo "Running regression tests..."
	@echo "Test started at: $$(date)" > $(TEST_LOG)
	@echo "======================================" >> $(TEST_LOG)
	./$(TEST_EXECUTABLE) all 2>&1 | tee -a $(TEST_LOG)
	@echo "======================================" >> $(TEST_LOG)
	@echo "Test completed at: $$(date)" >> $(TEST_LOG)
	@echo ""
	@echo "Test results saved to: $(TEST_LOG)"

# Run tests with verbose output
.PHONY: test-verbose
test-verbose: $(TEST_EXECUTABLE)
	@echo "Running regression tests with verbose output..."
	@echo "Test started at: $$(date)" > $(TEST_LOG)
	@echo "======================================" >> $(TEST_LOG)
	./$(TEST_EXECUTABLE) all 2>&1 | tee -a $(TEST_LOG)
	@echo "======================================" >> $(TEST_LOG)
	@echo "Test completed at: $$(date)" >> $(TEST_LOG)
	@echo ""
	@echo "Detailed test results:"
	@cat $(TEST_LOG)

# Run specific test
.PHONY: test-specific
test-specific: $(TEST_EXECUTABLE)
	@echo "Running specific test: $(TEST_NAME)"
	./$(TEST_EXECUTABLE) run $(TEST_NAME)

# Clean up build artifacts
.PHONY: clean
clean:
	@echo "Cleaning up..."
	@rm -f $(TEST_EXECUTABLE)
	@rm -f $(TEST_LOG)
	@rm -f test_*.tif
	@echo "Clean complete."

# Debug build
.PHONY: debug
debug: NVCC_FLAGS += -g -G
debug: $(TEST_EXECUTABLE)
	@echo "Debug build complete."