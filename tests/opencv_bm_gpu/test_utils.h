// test_utils.h
#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <chrono>
#include <cmath>
#include <sstream>
#include <exception>

class TestRunner {
private:
    struct TestResult {
        std::string test_name;
        bool passed;
        std::string error_message;
        double duration_ms;
    };
    
    std::vector<TestResult> results;
    
public:
    void runTest(const std::string& test_name, std::function<void()> test_func) {
        std::cout << "Running: " << test_name << "... ";
        
        auto start = std::chrono::high_resolution_clock::now();
        TestResult result;
        result.test_name = test_name;
        
        try {
            test_func();
            result.passed = true;
            std::cout << "PASSED";
        } catch (const std::exception& e) {
            result.passed = false;
            result.error_message = e.what();
            std::cout << "FAILED - " << e.what();
        } catch (...) {
            result.passed = false;
            result.error_message = "Unknown exception";
            std::cout << "FAILED - Unknown exception";
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        result.duration_ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << " (" << result.duration_ms << "ms)" << std::endl;
        
        results.push_back(result);
    }
    
    void printSummary() {
        int passed = 0;
        int failed = 0;
        double total_time = 0;
        
        std::cout << "\n========== TEST SUMMARY ==========\n";
        for (const auto& result : results) {
            if (result.passed) {
                passed++;
            } else {
                failed++;
                std::cout << "FAILED: " << result.test_name << " - " << result.error_message << std::endl;
            }
            total_time += result.duration_ms;
        }
        
        std::cout << "\nTests run: " << (passed + failed) << std::endl;
        std::cout << "Passed: " << passed << std::endl;
        std::cout << "Failed: " << failed << std::endl;
        std::cout << "Total time: " << total_time << "ms" << std::endl;
        
        if (failed == 0) {
            std::cout << "All tests passed!" << std::endl;
        }
    }
    
    int getFailureCount() const {
        int failed = 0;
        for (const auto& result : results) {
            if (!result.passed) failed++;
        }
        return failed;
    }
};

// Assertion functions
void assertTrue(bool condition, const std::string& message = "") {
    if (!condition) {
        throw std::runtime_error("Assertion failed: " + message);
    }
}

void assertFalse(bool condition, const std::string& message = "") {
    assertTrue(!condition, message);
}

void assertEqual(int expected, int actual, const std::string& message = "") {
    if (expected != actual) {
        std::ostringstream oss;
        oss << "Expected: " << expected << ", Actual: " << actual;
        if (!message.empty()) oss << " (" << message << ")";
        throw std::runtime_error(oss.str());
    }
}

void assertEqual(double expected, double actual, double tolerance = 1e-6, const std::string& message = "") {
    if (std::abs(expected - actual) > tolerance) {
        std::ostringstream oss;
        oss << "Expected: " << expected << ", Actual: " << actual << " (tolerance: " << tolerance << ")";
        if (!message.empty()) oss << " (" << message << ")";
        throw std::runtime_error(oss.str());
    }
}

void assertLess(double value, double threshold, const std::string& message = "") {
    if (value >= threshold) {
        std::ostringstream oss;
        oss << "Expected " << value << " < " << threshold;
        if (!message.empty()) oss << " (" << message << ")";
        throw std::runtime_error(oss.str());
    }
}

void assertGreater(double value, double threshold, const std::string& message = "") {
    if (value <= threshold) {
        std::ostringstream oss;
        oss << "Expected " << value << " > " << threshold;
        if (!message.empty()) oss << " (" << message << ")";
        throw std::runtime_error(oss.str());
    }
}

void assertNear(double expected, double actual, double tolerance, const std::string& message = "") {
    assertEqual(expected, actual, tolerance, message);
}

void assertNotNull(void* ptr, const std::string& message = "") {
    if (ptr == nullptr) {
        throw std::runtime_error("Pointer is null: " + message);
    }
}

#endif // TEST_UTILS_H