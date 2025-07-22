// performance_tests.cpp
#include "test_utils.h"
#include "opencv_bm_gpu_functions.h"
#include <opencv2/opencv.hpp>
#include <opencv2/cudastereo.hpp>
#include <chrono>
#include <cstdlib>

// Create realistic test data for performance testing
void createRealisticTestData(cv::Mat& left, cv::Mat& right, int width, int height) {
    left = cv::Mat(height, width, CV_8U);
    right = cv::Mat(height, width, CV_8U);
    
    // Fill with random data
    cv::randu(left, cv::Scalar(0), cv::Scalar(255));
    
    // Create right image with simulated disparities
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int shift = 5 + (y % 10);  // Variable disparity
            int src_x = x + shift;
            if (src_x < width) {
                right.at<uchar>(y, x) = left.at<uchar>(y, src_x);
            } else {
                right.at<uchar>(y, x) = 0;
            }
        }
    }
    
    // Apply some blur for realism
    cv::GaussianBlur(left, left, cv::Size(3, 3), 0);
    cv::GaussianBlur(right, right, cv::Size(3, 3), 0);
}

// Time a disparity computation similar to opencv_bm_gpu.cpp
double timeDisparityComputation(const cv::Mat& left, const cv::Mat& right, int num_disp, int block_size) {
    #ifdef HAVE_OPENCV_CUDA
    cv::cuda::GpuMat d_left(left), d_right(right), d_disp;
    
    auto bm = cv::cuda::createStereoBM(num_disp, block_size);
    bm->setPreFilterCap(31);
    bm->setUniquenessRatio(15);
    
    auto start = std::chrono::high_resolution_clock::now();
    bm->compute(d_left, d_right, d_disp);
    cv::cuda::Stream::Null().waitForCompletion();
    auto end = std::chrono::high_resolution_clock::now();
    
    return std::chrono::duration<double>(end - start).count();
    #else
    throw std::runtime_error("OpenCV built without CUDA support");
    #endif
}

// Test performance with different disparity ranges
void testPerformanceScaling() {
    #ifdef HAVE_OPENCV_CUDA
    cv::Mat left, right;
    createRealisticTestData(left, right, 1024, 1024);
    
    std::cout << "Performance test with 1024x1024 images:" << std::endl;
    
    // Test different disparity ranges
    std::vector<int> disparity_ranges = {32, 64, 128};
    std::vector<double> times;
    
    for (int num_disp : disparity_ranges) {
        double elapsed = timeDisparityComputation(left, right, num_disp, 21);
        times.push_back(elapsed);
        std::cout << "  " << num_disp << " disparities: " << elapsed << " seconds" << std::endl;
    }
    
    // Check that time scales reasonably with disparity range
    for (size_t i = 1; i < times.size(); i++) {
        double ratio = times[i] / times[i-1];
        assertLess(ratio, 3.0, "Time should not increase too dramatically with disparity range");
        assertGreater(ratio, 1.0, "Time should increase with disparity range");
    }
    #else
    std::cout << "Skipping performance test - OpenCV built without CUDA support" << std::endl;
    #endif
}

// Test effect of tile size on performance
void testTilePerformance() {
    #ifdef HAVE_OPENCV_CUDA
    cv::Mat left, right;
    createRealisticTestData(left, right, 2048, 2048);
    
    std::cout << "Tile size performance test:" << std::endl;
    
    std::vector<int> tile_sizes = {256, 512, 1024, 2048};
    std::vector<double> total_times;
    
    for (int tile_size : tile_sizes) {
        int overlap = std::min(32, tile_size / 8);
        
        auto start_total = std::chrono::high_resolution_clock::now();
        
        for (int y = 0; y < left.rows; y += tile_size - overlap) {
            for (int x = 0; x < left.cols; x += tile_size - overlap) {
                int tile_w = std::min(tile_size, left.cols - x);
                int tile_h = std::min(tile_size, left.rows - y);
                
                cv::Rect roi(x, y, tile_w, tile_h);
                cv::Mat l_tile = left(roi);
                cv::Mat r_tile = right(roi);
                
                // Normalize tiles
                cv::Mat l_tile_norm = robustNormalize(l_tile);
                cv::Mat r_tile_norm = robustNormalize(r_tile);
                
                // Skip actual disparity calculation to make this test faster
                // This is just to measure the overhead of different tile sizes
            }
        }
        
        auto end_total = std::chrono::high_resolution_clock::now();
        double total_time = std::chrono::duration<double>(end_total - start_total).count();
        total_times.push_back(total_time);
        
        std::cout << "  Tile size " << tile_size << ": " << total_time << " seconds" << std::endl;
    }
    
    // Smaller tiles should have more overhead
    assertGreater(total_times[0], total_times[1], "Smaller tiles should have more overhead");
    #else
    std::cout << "Skipping tile performance test - OpenCV built without CUDA support" << std::endl;
    #endif
}

// Test robustNormalize performance
void testNormalizePerformance() {
    // Create large float matrix
    cv::Mat large_float(2048, 2048, CV_32F);
    cv::randu(large_float, cv::Scalar(0), cv::Scalar(1000));
    
    // Add some NaN values
    for (int i = 0; i < large_float.rows; i += 100) {
        for (int j = 0; j < large_float.cols; j += 100) {
            large_float.at<float>(i, j) = std::numeric_limits<float>::quiet_NaN();
        }
    }
    
    // Measure performance
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat normalized = robustNormalize(large_float);
    auto end = std::chrono::high_resolution_clock::now();
    
    double elapsed = std::chrono::duration<double>(end - start).count();
    std::cout << "robustNormalize (2048x2048): " << elapsed << " seconds" << std::endl;
    
    // Basic performance check
    assertLess(elapsed, 1.0, "Normalization should be reasonably fast");
}

#ifdef RUN_STANDALONE
int main() {
    TestRunner runner;
    runner.runTest("PerformanceScaling", testPerformanceScaling);
    runner.runTest("TilePerformance", testTilePerformance);
    runner.runTest("NormalizePerformance", testNormalizePerformance);
    runner.printSummary();
    return runner.getFailureCount();
}
#endif