// run_tests.cpp
#include "test_utils.h"
#include <iostream>

// Declare all test functions
// Stereo tests
void testRobustNormalize();
void testDisparityComputation();
void testGeoTIFFIO();
void testTiledProcessing();

// Performance tests
void testPerformanceScaling();
void testMemoryScaling();
void testPerformanceRegression();

// Command line interface tests
void testBasicCommandLine();
void testParameterizedCommandLine();

int main() {
    std::cout << "======== Starting Regression Tests ========" << std::endl;
    TestRunner runner;
    
    // Stereo tests
    runner.runTest("RobustNormalize", testRobustNormalize);
    runner.runTest("DisparityComputation", testDisparityComputation);
    runner.runTest("GeoTIFFIO", testGeoTIFFIO);
    runner.runTest("TiledProcessing", testTiledProcessing);
    
    // Performance tests
    runner.runTest("PerformanceScaling", testPerformanceScaling);
    runner.runTest("MemoryScaling", testMemoryScaling);
    runner.runTest("PerformanceRegression", testPerformanceRegression);
    
    // Command line interface tests
    runner.runTest("BasicCommandLine", testBasicCommandLine);
    runner.runTest("ParameterizedCommandLine", testParameterizedCommandLine);
    
    // Print summary
    runner.printSummary();
    
    return runner.getFailureCount();
}