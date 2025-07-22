CXX = g++

CXXFLAGS = -std=c++17 -w -O2 -I/usr/local/include/opencv4 -I/usr/include/gdal

LDFLAGS = -lpthread -lgdal -lopencv_core

# Source files
MAIN_SRC = ../../spgpu/correlator/opencv_bm_gpu.cpp
MAIN_TARGET = opencv_bm_gpu

# Object files for shared functions
SHARED_OBJS = opencv_bm_gpu.o

# Test files
TEST_SRC = stereo_tests.cpp performance_tests.cpp cmdline_tests.cpp
TEST_RUNNER = run_tests.cpp
TEST_TARGET = test_runner

# Individual test targets
INDIVIDUAL_TESTS = stereo_tests performance_tests cmdline_tests

# All targets
all: $(MAIN_TARGET) $(TEST_TARGET)

# Main program build
$(MAIN_TARGET): $(MAIN_SRC)
	$(CXX) $(CXXFLAGS) -DCOMPILE_FOR_TESTING -c $^ $(LDFLAGS)

# Shared object file with functions needed by tests
opencv_bm_gpu.o: $(MAIN_SRC)
	$(CXX) $(CXXFLAGS) -DCOMPILE_FOR_TESTING -c $<

# Test runner executable (links with shared functions)
$(TEST_TARGET): $(TEST_RUNNER) $(TEST_SRC) $(SHARED_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

# Individual test executables (for debugging)
stereo_tests: stereo_tests.cpp $(SHARED_OBJS)
	$(CXX) $(CXXFLAGS) -DRUN_STANDALONE -o $@ $^ $(LDFLAGS)

performance_tests: performance_tests.cpp $(SHARED_OBJS)
	$(CXX) $(CXXFLAGS) -DRUN_STANDALONE -o $@ $^ $(LDFLAGS)

cmdline_tests: cmdline_tests.cpp $(SHARED_OBJS)
	$(CXX) $(CXXFLAGS) -DRUN_STANDALONE -o $@ $^ $(LDFLAGS)

# Clean targets
clean:
	rm -f $(MAIN_TARGET) $(TEST_TARGET) $(INDIVIDUAL_TESTS) $(SHARED_OBJS) *.tif *.o

# Run all tests
test: $(TEST_TARGET) $(MAIN_TARGET)
	./$(TEST_TARGET)

# Run individual test suites
test-stereo: stereo_tests $(MAIN_TARGET)
	./stereo_tests

test-performance: performance_tests $(MAIN_TARGET)
	./performance_tests

test-cmdline: cmdline_tests $(MAIN_TARGET)
	./cmdline_tests

# Install targets (optional)
install: $(MAIN_TARGET)
	cp $(MAIN_TARGET) /usr/local/bin/

uninstall:
	rm -f /usr/local/bin/$(MAIN_TARGET)

.PHONY: all clean test test-stereo test-performance test-cmdline install uninstall