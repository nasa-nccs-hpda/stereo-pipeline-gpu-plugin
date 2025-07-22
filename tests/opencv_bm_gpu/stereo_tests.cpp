// stereo_tests.cpp
#include "test_utils.h"
#include "opencv_bm_gpu_functions.h"
#include <opencv2/opencv.hpp>
#include <opencv2/cudastereo.hpp>
#include <gdal_priv.h>
#include <cpl_conv.h>
#include <filesystem>
#include <limits>
#include <cmath>

// Helper function to create test images
void createTestImages(cv::Mat& left, cv::Mat& right, int width = 256, int height = 256) {
    left = cv::Mat(height, width, CV_8U);
    right = cv::Mat(height, width, CV_8U);
    
    // Create a simple pattern with known disparity
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            left.at<uchar>(y, x) = (x % 32 < 16) ? 255 : 0; // Vertical bars
            
            // Shift pattern by 10 pixels in right image
            int shift = 10;
            int shifted_x = x - shift;
            if (shifted_x >= 0 && shifted_x < width) {
                right.at<uchar>(y, shifted_x) = (x % 32 < 16) ? 255 : 0;
            }
        }
    }
}

// Helper function to simulate GPU disparity computation
cv::Mat computeGPUDisparity(const cv::Mat& left, const cv::Mat& right, 
                          int num_disp = 64, int block_size = 21) {
    #ifdef HAVE_OPENCV_CUDA
    // Create GPU mats
    cv::cuda::GpuMat d_left(left), d_right(right), d_disp;
    
    // Create and configure StereoBM
    auto bm = cv::cuda::createStereoBM(num_disp, block_size);
    bm->setTextureThreshold(10);
    bm->setPreFilterCap(31);
    bm->setUniquenessRatio(15);
    bm->setSpeckleWindowSize(100);
    bm->setSpeckleRange(32);
    bm->setDisp12MaxDiff(1);
    
    // Compute disparity
    bm->compute(d_left, d_right, d_disp);
    
    // Download result
    cv::Mat disp_raw;
    d_disp.download(disp_raw);
    cv::Mat disp_f;
    disp_raw.convertTo(disp_f, CV_32F, -1.0/16.0);
    
    return disp_f;
    #else
    throw std::runtime_error("OpenCV built without CUDA support");
    #endif
}

// Test robust normalization function from opencv_bm_gpu.cpp
void testRobustNormalize() {
    // Create test data with known range
    cv::Mat testData(10, 10, CV_32F);
    for (int i = 0; i < testData.rows; i++) {
        for (int j = 0; j < testData.cols; j++) {
            testData.at<float>(i, j) = i * 10 + j;  // Values from 0 to 99
        }
    }
    
    cv::Mat normalized = robustNormalize(testData);
    
    // Check normalized range
    assertEqual(static_cast<int>(normalized.at<uchar>(0, 0)), 0, "Min value should be 0");
    assertEqual(static_cast<int>(normalized.at<uchar>(9, 9)), 255, "Max value should be 255");
    assertEqual(normalized.type(), CV_8U, "Output type should be CV_8U");
    
    // Test with constant data
    cv::Mat constantData = cv::Mat::ones(5, 5, CV_32F) * 42.0f;
    cv::Mat constantNorm = robustNormalize(constantData);
    assertEqual(constantNorm.at<uchar>(0, 0), 0, "Constant data should normalize to zero");
    
    // Test with NaN values
    cv::Mat nanData = testData.clone();
    nanData.at<float>(3, 3) = std::numeric_limits<float>::quiet_NaN();
    nanData.at<float>(3, 4) = std::numeric_limits<float>::quiet_NaN();
    cv::Mat nanNorm = robustNormalize(nanData);
    assertEqual(nanNorm.at<uchar>(0, 0), 0, "Min value should still be 0");
}

// Test GeoTIFF I/O functionality from opencv_bm_gpu.cpp
void testGeoTIFFIO() {
    // Create test data
    cv::Mat test_data(50, 50, CV_32F);
    for (int i = 0; i < test_data.rows; i++) {
        for (int j = 0; j < test_data.cols; j++) {
            test_data.at<float>(i, j) = i * j / 100.0f;
        }
    }
    
    // Set some values to NaN
    for (int i = 10; i < 15; i++) {
        for (int j = 10; j < 15; j++) {
            test_data.at<float>(i, j) = std::numeric_limits<float>::quiet_NaN();
        }
    }
    
    // Save to GeoTIFF using the function from opencv_bm_gpu.cpp
    std::string filename = "test_geotiff.tif";
    SaveGeoTIFF(filename, test_data);
    
    // Verify file exists
    assertTrue(std::filesystem::exists(filename), "Output file should exist");
    
    // Read back using GDAL
    GDALAllRegister();
    GDALDataset* read_ds = (GDALDataset*)GDALOpen(filename.c_str(), GA_ReadOnly);
    assertNotNull(read_ds, "Should be able to open saved file");
    
    // Check dimensions
    assertEqual(read_ds->GetRasterXSize(), test_data.cols, "Width should match");
    assertEqual(read_ds->GetRasterYSize(), test_data.rows, "Height should match");
    
    // Read data back
    cv::Mat read_data(test_data.rows, test_data.cols, CV_32F);
    read_ds->GetRasterBand(1)->RasterIO(
        GF_Read, 0, 0, read_data.cols, read_data.rows,
        read_data.ptr<float>(), read_data.cols, read_data.rows,
        GDT_Float32, 0, 0
    );
    GDALClose(read_ds);
    
    // Compare data (sample points)
    for (int i = 0; i < test_data.rows; i += 5) {
        for (int j = 0; j < test_data.cols; j += 5) {
            float original = test_data.at<float>(i, j);
            float read = read_data.at<float>(i, j);
            
            if (std::isnan(original)) {
                assertTrue(std::isnan(read), "NaN values should be preserved");
            } else {
                assertEqual(original, read, 1e-6, "Data values should match");
            }
        }
    }
    
    // Clean up
    std::remove(filename.c_str());
}

// Test disparity computation (simulating what opencv_bm_gpu does)
void testDisparityComputation() {
    #ifdef HAVE_OPENCV_CUDA
    cv::Mat left, right;
    createTestImages(left, right);
    
    cv::Mat disparity = computeGPUDisparity(left, right);
    
    // Check that we have valid data
    cv::Mat valid_mask = disparity == disparity; // Not NaN
    int valid_count = cv::countNonZero(valid_mask);
    int total_pixels = disparity.rows * disparity.cols;
    
    assertGreater(static_cast<double>(valid_count), total_pixels * 0.3, "Should have at least 30% valid disparities");
    
    // Check central region for approximate disparity
    cv::Rect central_rect(50, 50, 150, 150);
    cv::Mat central_region = disparity(central_rect);
    cv::Mat central_mask = valid_mask(central_rect);
    
    if (cv::countNonZero(central_mask) > 0) {
        cv::Scalar mean_disp = cv::mean(central_region, central_mask);
        // We expect negative disparity values (convention)
        assertLess(mean_disp[0], 0, "Mean disparity should be negative");
        assertGreater(std::abs(mean_disp[0]), 5.0, "Disparity magnitude should be reasonable");
    }
    #else
    std::cout << "Skipping CUDA test - OpenCV built without CUDA support" << std::endl;
    #endif
}

// Test tiled processing logic similar to opencv_bm_gpu.cpp
void testTiledProcessing() {
    #ifdef HAVE_OPENCV_CUDA
    // Create larger test images
    cv::Mat big_left, big_right;
    createTestImages(big_left, big_right, 512, 512);
    
    // Process with tiling (similar to opencv_bm_gpu.cpp)
    int tile_size = 256;
    int overlap = 32;
    
    cv::Mat disp_sum = cv::Mat::zeros(big_left.size(), CV_32F);
    cv::Mat disp_count = cv::Mat::zeros(big_left.size(), CV_32F);
    
    for (int y = 0; y < big_left.rows; y += tile_size - overlap) {
        for (int x = 0; x < big_left.cols; x += tile_size - overlap) {
            int tile_w = std::min(tile_size, big_left.cols - x);
            int tile_h = std::min(tile_size, big_left.rows - y);
            
            cv::Rect roi(x, y, tile_w, tile_h);
            cv::Mat l_tile = big_left(roi);
            cv::Mat r_tile = big_right(roi);
            
            // Normalize tiles as opencv_bm_gpu.cpp does
            cv::Mat l_tile_norm = robustNormalize(l_tile);
            cv::Mat r_tile_norm = robustNormalize(r_tile);
            
            // Compute disparity for tile
            cv::Mat disp_tile = computeGPUDisparity(l_tile_norm, r_tile_norm);
            
            // Accumulate results
            for (int yy = 0; yy < tile_h; ++yy) {
                for (int xx = 0; xx < tile_w; ++xx) {
                    float v = disp_tile.at<float>(yy, xx);
                    if (std::isnan(v)) continue;
                    int gy = y + yy;
                    int gx = x + xx;
                    disp_sum.at<float>(gy, gx) += v;
                    disp_count.at<float>(gy, gx) += 1.0f;
                }
            }
        }
    }
    
    cv::Mat final_disp = disp_sum / disp_count;
    final_disp.setTo(std::numeric_limits<float>::quiet_NaN(), disp_count == 0);
    
    // Verify results
    cv::Mat valid_mask = final_disp == final_disp; // Not NaN
    int valid_count = cv::countNonZero(valid_mask);
    assertGreater(static_cast<double>(valid_count), big_left.rows * big_left.cols * 0.4, 
                 "Tiled processing should produce reasonable coverage");
    #else
    std::cout << "Skipping tiled processing test - OpenCV built without CUDA support" << std::endl;
    #endif
}

#ifdef RUN_STANDALONE
int main() {
    TestRunner runner;
    runner.runTest("RobustNormalize", testRobustNormalize);
    runner.runTest("DisparityComputation", testDisparityComputation);
    runner.runTest("GeoTIFFIO", testGeoTIFFIO);
    runner.runTest("TiledProcessing", testTiledProcessing);
    runner.printSummary();
    return runner.getFailureCount();
}
#endif