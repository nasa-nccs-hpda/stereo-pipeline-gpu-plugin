# Makefile for OpenCV BM GPU Regression Tests

# Compiler settings
CXX = g++
CXXFLAGS = -std=c++11 -w -O2 -I ../../spgpu/correlator/
TEST_DEFINES = -DCOMPILE_FOR_TESTING

# Package config for dependencies
OPENCV_FLAGS = -I/usr/local/include/opencv4 -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_calib3d -lopencv_cudaimgproc -lopencv_cudastereo -lopencv_imgcodecs

GDAL_FLAGS = -I/usr/include/gdal -lgdal

CUDA_FLAGS = -DHAVE_OPENCV_CUDA

# Combine all flags
ALL_FLAGS = $(CXXFLAGS) $(TEST_DEFINES) $(CUDA_FLAGS) $(OPENCV_FLAGS) $(GDAL_FLAGS)

# File names
BASE_NAME = opencv_csbp_gpu
ORIGINAL_SOURCE = ../../spgpu/correlator/$(BASE_NAME).cpp
TEST_SOURCE = test_$(BASE_NAME).cpp
TEST_EXECUTABLE = test_$(BASE_NAME)
TEST_LOG = test_results.log

# Default target
.PHONY: all
all: test

# Build the test executable
$(TEST_EXECUTABLE): $(TEST_SOURCE) $(ORIGINAL_SOURCE)
	@echo "Building test executable..."
	@echo "Compiler flags: $(ALL_FLAGS)"
	$(CXX) $(TEST_SOURCE) -o $(TEST_EXECUTABLE) $(ALL_FLAGS)
	@echo "Build successful."

# Run tests
.PHONY: test
test: $(TEST_EXECUTABLE)
	@echo "Running regression tests..."
	@echo "Test started at: $$(date)" > $(TEST_LOG)
	@echo "======================================" >> $(TEST_LOG)
	./$(TEST_EXECUTABLE) 2>&1 | tee -a $(TEST_LOG)
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
	./$(TEST_EXECUTABLE) 2>&1 | tee -a $(TEST_LOG)
	@echo "======================================" >> $(TEST_LOG)
	@echo "Test completed at: $$(date)" >> $(TEST_LOG)
	@echo ""
	@echo "Detailed test results:"
	@cat $(TEST_LOG)

# Quick test (without logging)
.PHONY: test-quick
test-quick: $(TEST_EXECUTABLE)
	@echo "Running quick regression tests..."
	./$(TEST_EXECUTABLE)

# Clean up build artifacts
.PHONY: clean
clean:
	@echo "Cleaning up..."
	@rm -f $(TEST_EXECUTABLE)
	@rm -f $(TEST_LOG)
	@rm -f test_*.tif
	@rm -f *.o
	@echo "Clean complete."

# Clean up all test artifacts including temporary files
.PHONY: clean-all
clean-all: clean
	@echo "Cleaning all test artifacts..."
	@rm -f *.tif
	@rm -f *.log
	@rm -f core.*
	@echo "Deep clean complete."

# Debug build
.PHONY: debug
debug: CXXFLAGS = -std=c++11 -Wall -Wextra -g -DDEBUG
debug: $(TEST_EXECUTABLE)
	@echo "Debug build complete."

# Continuous integration target
.PHONY: ci
ci: clean check-deps test
	@echo "CI pipeline completed successfully."
