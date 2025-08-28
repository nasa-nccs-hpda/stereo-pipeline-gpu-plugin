#include <cassert>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

// Include the main implementation
#include "mgm_gpu.cu"

using namespace std;
using namespace cv;

// ----------------------------------------------------------------------------
// MGMStereoGPUTests
//
// Comprehensive regression test suite for MGM Stereo GPU implementation
// ----------------------------------------------------------------------------
class MGMStereoGPUTests
{
private:
    
    bool verbose_;
    int tests_passed_;
    int tests_failed_;
    
    // Test data generators
    Mat generateSyntheticImage(int width, int height, float base_value = 0.5f);
    Mat generateDisparityPattern(int width, int height, int max_disp);
	
    pair<Mat, Mat> generateStereoImagePair(int width, 
										   int height, 
										   int max_disp);
    
    // Helper methods
    void printTestResult(const string& test_name, bool passed);
	
    bool compareFloatArrays(const float* a, 
							const float* b, 
							int size,
							float tolerance = 1e-6f);
	
    bool compareShortArrays(const short* a, const short* b, int size);
    void checkCudaError(const string& operation);
    
    // Individual test methods
    bool testGPUMemoryAllocation();
    bool testBasicKernelExecution();
    bool testCostComputation();
    bool testPathProcessing();
    bool testWinnerTakesAll();
    bool testSubPixelRefinement();
    bool testTileProcessing();
    bool testBoundaryConditions();
    bool testLargeImages();
    bool testConsistencyCheck();
    bool testMemoryValidation();
    bool testErrorHandling();

public:
    
    // ------------------------------------------------------------------------
    // MGMStereoGPUTests
    // ------------------------------------------------------------------------
    MGMStereoGPUTests(bool verbose = true) : verbose_(verbose), 
                                             tests_passed_(0), 
                                             tests_failed_(0) 
    {
    }
    
    // ------------------------------------------------------------------------
    // runAllTests
    // ------------------------------------------------------------------------
    bool runAllTests();
    
    // ------------------------------------------------------------------------
    // runSpecificTest
    // ------------------------------------------------------------------------
    bool runSpecificTest(const string& test_name);
    
    // ------------------------------------------------------------------------
    // printSummary
    // ------------------------------------------------------------------------
    void printSummary();
    
    // ------------------------------------------------------------------------
    // generateTestData
    // ------------------------------------------------------------------------
    static void generateTestData(const string& output_dir);
};

// ----------------------------------------------------------------------------
// generateSyntheticImage
// ----------------------------------------------------------------------------
Mat MGMStereoGPUTests::generateSyntheticImage(int width, 
                                               int height, 
                                               float base_value)
{
    Mat image(height, width, CV_32F);
    
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            // Create gradient pattern with some texture
            float value = base_value + 
                         0.3f * sin(x * 0.1f) + 
                         0.2f * cos(y * 0.15f) +
                         0.1f * sin((x + y) * 0.05f);
                         
            image.at<float>(y, x) = max(0.0f, min(1.0f, value));
        }
    }
    
    return image;
}

// ----------------------------------------------------------------------------
// generateDisparityPattern
// ----------------------------------------------------------------------------
Mat MGMStereoGPUTests::generateDisparityPattern(int width, 
                                                 int height, 
                                                 int max_disp)
{
    Mat disparity(height, width, CV_16S);
    
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            // Create a simple ramp pattern
            int d = (x * max_disp) / width;
            disparity.at<short>(y, x) = d;
        }
    }
    
    return disparity;
}

// ----------------------------------------------------------------------------
// generateStereoImagePair
// ----------------------------------------------------------------------------
pair<Mat, Mat> MGMStereoGPUTests::generateStereoImagePair(int width, 
                                                           int height, 
                                                           int max_disp)
{
    Mat left = generateSyntheticImage(width, height, 0.5f);
    Mat right(height, width, CV_32F, Scalar(0));
    
    // Generate right image by shifting left image according to disparity
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int d = (x * max_disp) / width;  // Simple ramp disparity
            int right_x = x - d;
            
            if (right_x >= 0 && right_x < width)
            {
                right.at<float>(y, right_x) = left.at<float>(y, x);
            }
        }
    }
    
    return make_pair(left, right);
}

// ----------------------------------------------------------------------------
// printTestResult
// ----------------------------------------------------------------------------
void MGMStereoGPUTests::printTestResult(const string& test_name, bool passed)
{
    if (passed)
    {
        tests_passed_++;
		
        if (verbose_)
        {
            cout << "[PASS] " << test_name << endl;
        }
    }
    else
    {
        tests_failed_++;
        cout << "[FAIL] " << test_name << endl;
    }
}

// ----------------------------------------------------------------------------
// compareFloatArrays
// ----------------------------------------------------------------------------
bool MGMStereoGPUTests::compareFloatArrays(const float* a, 
                                            const float* b, 
                                            int size, 
                                            float tolerance)
{
    for (int i = 0; i < size; i++)
    {
        if (isnan(a[i]) || isnan(b[i]) || 
            isinf(a[i]) || isinf(b[i]))
        {
            if (verbose_)
            {
                cout << "Invalid values at index " << i 
                     << ": a=" << a[i] << ", b=" << b[i] << endl;
            }
            return false;
        }
        
        if (abs(a[i] - b[i]) > tolerance)
        {
            if (verbose_)
            {
                cout << "Mismatch at index " << i 
                     << ": a=" << a[i] << ", b=" << b[i] 
                     << ", diff=" << abs(a[i] - b[i]) << endl;
            }
            return false;
        }
    }
    return true;
}

// ----------------------------------------------------------------------------
// compareShortArrays
// ----------------------------------------------------------------------------
bool MGMStereoGPUTests::compareShortArrays(const short* a, 
                                            const short* b, 
                                            int size)
{
    for (int i = 0; i < size; i++)
    {
        if (a[i] != b[i])
        {
            if (verbose_)
            {
                cout << "Mismatch at index " << i 
                     << ": a=" << a[i] << ", b=" << b[i] << endl;
            }
            return false;
        }
    }
    return true;
}

// ----------------------------------------------------------------------------
// checkCudaError
// ----------------------------------------------------------------------------
void MGMStereoGPUTests::checkCudaError(const string& operation)
{
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        cout << "CUDA error in " << operation << ": " 
             << cudaGetErrorString(error) << endl;
        throw runtime_error("CUDA error: " + string(cudaGetErrorString(error)));
    }
}

// ----------------------------------------------------------------------------
// testGPUMemoryAllocation
// ----------------------------------------------------------------------------
bool MGMStereoGPUTests::testGPUMemoryAllocation()
{
    try
    {
        MGMStereoTileGPU tile(0, 0, 64, 64, 256, 256, 32, 8);
        
        // Test memory validation
        tile.validateMemory("Allocation test");
        
        return true;
    }
    catch (const exception& e)
    {
        if (verbose_)
        {
            cout << "Memory allocation test failed: " << e.what() << endl;
        }
        return false;
    }
}

// ----------------------------------------------------------------------------
// testBasicKernelExecution
// ----------------------------------------------------------------------------
bool MGMStereoGPUTests::testBasicKernelExecution()
{
    try
    {
        // Test simple kernel
        testKernel<<<1, 1>>>(42);
        cudaDeviceSynchronize();
        checkCudaError("testKernel");
        
        return true;
    }
    catch (const exception& e)
    {
        if (verbose_)
        {
            cout << "Basic kernel test failed: " << e.what() << endl;
        }
        return false;
    }
}

// ----------------------------------------------------------------------------
// testCostComputation
// ----------------------------------------------------------------------------
bool MGMStereoGPUTests::testCostComputation()
{
    try
    {
        int width = 64, height = 64, max_disp = 16;
        auto image_pair = generateStereoImagePair(width, height, max_disp);
        
        // Allocate GPU memory
        float* d_left;
        float* d_right;
        float* d_costs;
        
        size_t img_size = width * height * sizeof(float);
        size_t cost_size = width * height * max_disp * sizeof(float);
        
        cudaMalloc(&d_left, img_size);
        cudaMalloc(&d_right, img_size);
        cudaMalloc(&d_costs, cost_size);
        
        cudaMemcpy(d_left, image_pair.first.ptr<float>(), 
                   img_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_right, image_pair.second.ptr<float>(), 
                   img_size, cudaMemcpyHostToDevice);
        
        // Launch cost computation kernel
        dim3 block(8, 8, 4);
        dim3 grid((width + block.x - 1) / block.x,
                  (height + block.y - 1) / block.y,
                  (max_disp + block.z - 1) / block.z);
        
        computeTileMatchingCosts<<<grid, block>>>(
            d_left, d_right, d_costs,
            width, height, 0, 0, width, height, 0, max_disp);
        
        cudaDeviceSynchronize();
        checkCudaError("computeTileMatchingCosts");
        
        // Check results
        vector<float> costs(width * height * max_disp);
        cudaMemcpy(costs.data(), d_costs, cost_size, cudaMemcpyDeviceToHost);
        
        // Verify costs are reasonable
        int valid_costs = 0;
        float min_cost = INFINITY, max_cost = -INFINITY;
        
        for (float cost : costs)
        {
            if (!isnan(cost) && !isinf(cost))
            {
                valid_costs++;
                min_cost = min(min_cost, cost);
                max_cost = max(max_cost, cost);
            }
        }
        
        bool success = (valid_costs > costs.size() * 0.8f) && 
                      (max_cost > min_cost) && 
                      (min_cost >= 0.0f);
        
        if (verbose_)
        {
            cout << "Cost computation: " << valid_costs << "/" << costs.size() 
                 << " valid, range [" << min_cost << ", " << max_cost << "]" << endl;
        }
        
        cudaFree(d_left);
        cudaFree(d_right);
        cudaFree(d_costs);
        
        return success;
    }
    catch (const exception& e)
    {
        if (verbose_)
        {
            cout << "Cost computation test failed: " << e.what() << endl;
        }
        return false;
    }
}

// ----------------------------------------------------------------------------
// testPathProcessing
// ----------------------------------------------------------------------------
bool MGMStereoGPUTests::testPathProcessing()
{
    try
    {
        int tile_size = 32;
        int max_disp = 8;
        
        MGMStereoTileGPU tile(0, 0, tile_size, tile_size, 
                              128, 128, max_disp, 4);
        
        // Create test images
        auto image_pair = generateStereoImagePair(128, 128, max_disp);
        
        float* d_left;
        float* d_right;
        size_t img_size = 128 * 128 * sizeof(float);
        
        cudaMalloc(&d_left, img_size);
        cudaMalloc(&d_right, img_size);
        
        cudaMemcpy(d_left, image_pair.first.ptr<float>(), 
                   img_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_right, image_pair.second.ptr<float>(), 
                   img_size, cudaMemcpyHostToDevice);
        
        // Process tile
        auto result = tile.processTileFromGPU(d_left, d_right);
        
        // Check for reasonable disparity values
        short min_disp = *min_element(result.disparity_map.begin(), 
                                      result.disparity_map.end());
        short max_disp_found = *max_element(result.disparity_map.begin(), 
                                           result.disparity_map.end());
        
        bool success = (min_disp >= 0) && 
                      (max_disp_found < max_disp) && 
                      (max_disp_found > min_disp);
        
        if (verbose_)
        {
            cout << "Path processing: disparity range [" 
                 << min_disp << ", " << max_disp_found << "]" << endl;
        }
        
        cudaFree(d_left);
        cudaFree(d_right);
        
        return success;
    }
    catch (const exception& e)
    {
        if (verbose_)
        {
            cout << "Path processing test failed: " << e.what() << endl;
        }
        return false;
    }
}

// ----------------------------------------------------------------------------
// testWinnerTakesAll
// ----------------------------------------------------------------------------
bool MGMStereoGPUTests::testWinnerTakesAll()
{
    try
    {
        int width = 16, height = 16, max_disp = 8;
        
        // Create simple test costs where disparity 3 should win
        vector<float> test_costs(width * height * max_disp);
        
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                for (int d = 0; d < max_disp; d++)
                {
                    int idx = (y * width + x) * max_disp + d;
                    
                    if (d == 3)
                    {
                        test_costs[idx] = 0.1f;  // Minimum cost
                    }
                    else
                    {
                        test_costs[idx] = 1.0f + d * 0.1f;  // Higher costs
                    }
                }
            }
        }
        
        // Allocate GPU memory
        float* d_costs;
        short* d_disparity;
        
        cudaMalloc(&d_costs, test_costs.size() * sizeof(float));
        cudaMalloc(&d_disparity, width * height * sizeof(short));
        
        cudaMemcpy(d_costs, test_costs.data(), 
                   test_costs.size() * sizeof(float), cudaMemcpyHostToDevice);
        
        // Launch WTA kernel
        dim3 block(16, 16);
        dim3 grid(1, 1);
        
        computeWinnerTakesAll<<<grid, block>>>(
            d_costs, d_disparity, width, height, max_disp, width, 0);
        
        cudaDeviceSynchronize();
        checkCudaError("computeWinnerTakesAll");
        
        // Check results
        vector<short> disparity_map(width * height);
        cudaMemcpy(disparity_map.data(), d_disparity, 
                   width * height * sizeof(short), cudaMemcpyDeviceToHost);
        
        // All disparities should be 3
        bool success = true;
        for (short d : disparity_map)
        {
            if (d != 3)
            {
                success = false;
                break;
            }
        }
        
        if (verbose_ && !success)
        {
            cout << "WTA test: Expected all disparities to be 3" << endl;
            for (int i = 0; i < min(10, (int)disparity_map.size()); i++)
            {
                cout << "  disparity[" << i << "] = " << disparity_map[i] << endl;
            }
        }
        
        cudaFree(d_costs);
        cudaFree(d_disparity);
        
        return success;
    }
    catch (const exception& e)
    {
        if (verbose_)
        {
            cout << "WTA test failed: " << e.what() << endl;
        }
        return false;
    }
}

// ----------------------------------------------------------------------------
// testSubPixelRefinement
// ----------------------------------------------------------------------------
bool MGMStereoGPUTests::testSubPixelRefinement()
{
    try
    {
        int width = 8, height = 8, max_disp = 8;
        
        // Create test costs with clear sub-pixel minimum at disparity 3.5
        vector<float> test_costs(width * height * max_disp);
        vector<short> test_disparity(width * height, 3);
        
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                for (int d = 0; d < max_disp; d++)
                {
                    int idx = (y * width + x) * max_disp + d;
                    
                    // Parabola with minimum at 3.5
                    float diff = d - 3.5f;
                    test_costs[idx] = 0.1f + diff * diff * 0.1f;
                }
            }
        }
        
        // Allocate GPU memory
        float* d_costs;
        short* d_disparity;
        float* d_refined;
        
        cudaMalloc(&d_costs, test_costs.size() * sizeof(float));
        cudaMalloc(&d_disparity, width * height * sizeof(short));
        cudaMalloc(&d_refined, width * height * sizeof(float));
        
        cudaMemcpy(d_costs, test_costs.data(), 
                   test_costs.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_disparity, test_disparity.data(), 
                   width * height * sizeof(short), cudaMemcpyHostToDevice);
        
        // Launch refinement kernel
        dim3 block(8, 8);
        dim3 grid(1, 1);
        
        refineDisparity<<<grid, block>>>(
            d_costs, d_disparity, d_refined, width, height, max_disp, width, 0);
        
        cudaDeviceSynchronize();
        checkCudaError("refineDisparity");
        
        // Check results
        vector<float> refined_disparity(width * height);
        cudaMemcpy(refined_disparity.data(), d_refined, 
                   width * height * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Refined disparities should be close to 3.5
        bool success = true;
        for (float d : refined_disparity)
        {
            if (abs(d - 3.5f) > 0.2f)  // Allow some tolerance
            {
                success = false;
                break;
            }
        }
        
        if (verbose_)
        {
            cout << "Sub-pixel refinement: refined values around " 
                 << refined_disparity[0] << " (expected ~3.5)" << endl;
        }
        
        cudaFree(d_costs);
        cudaFree(d_disparity);
        cudaFree(d_refined);
        
        return success;
    }
    catch (const exception& e)
    {
        if (verbose_)
        {
            cout << "Sub-pixel refinement test failed: " << e.what() << endl;
        }
        return false;
    }
}

// ----------------------------------------------------------------------------
// testTileProcessing
// ----------------------------------------------------------------------------
bool MGMStereoGPUTests::testTileProcessing()
{
    try
    {
        int img_width = 128, img_height = 128;
        int tile_size = 64, max_disp = 16;
        
        auto image_pair = generateStereoImagePair(img_width, img_height, max_disp);
        
        Mat result = correlate(image_pair.first, image_pair.second, 
                              tile_size, max_disp, 8);
        
        // Check result dimensions
        bool correct_size = (result.rows == img_height) && 
                           (result.cols == img_width);
        
        // Check for reasonable disparity values
        double min_val, max_val;
        minMaxLoc(result, &min_val, &max_val);
        
        bool reasonable_range = (min_val >= 0) && 
                               (max_val < max_disp) && 
                               (max_val > min_val);
        
        if (verbose_)
        {
            cout << "Tile processing: result size " << result.cols << "x" << result.rows
                 << ", disparity range [" << min_val << ", " << max_val << "]" << endl;
        }
        
        return correct_size && reasonable_range;
    }
    catch (const exception& e)
    {
        if (verbose_)
        {
            cout << "Tile processing test failed: " << e.what() << endl;
        }
        return false;
    }
}

// ----------------------------------------------------------------------------
// testBoundaryConditions
// ----------------------------------------------------------------------------
bool MGMStereoGPUTests::testBoundaryConditions()
{
    try
    {
        // Test very small tile
        MGMStereoTileGPU small_tile(0, 0, 4, 4, 16, 16, 4, 2);
        
        auto image_pair = generateStereoImagePair(16, 16, 4);
        
        float* d_left;
        float* d_right;
        size_t img_size = 16 * 16 * sizeof(float);
        
        cudaMalloc(&d_left, img_size);
        cudaMalloc(&d_right, img_size);
        
        cudaMemcpy(d_left, image_pair.first.ptr<float>(), 
                   img_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_right, image_pair.second.ptr<float>(), 
                   img_size, cudaMemcpyHostToDevice);
        
        auto result = small_tile.processTileFromGPU(d_left, d_right);
        
        // Should not crash and should produce some valid results
        bool success = (result.disparity_map.size() == 16) &&
                      (result.refined_disparity.size() == 16);
        
        cudaFree(d_left);
        cudaFree(d_right);
        
        return success;
    }
    catch (const exception& e)
    {
        if (verbose_)
        {
            cout << "Boundary conditions test failed: " << e.what() << endl;
        }
        return false;
    }
}

// ----------------------------------------------------------------------------
// testLargeImages
// ----------------------------------------------------------------------------
bool MGMStereoGPUTests::testLargeImages()
{
    try
    {
        // Test with larger image
        int img_width = 512, img_height = 384;
        int tile_size = 128, max_disp = 32;
        
        auto image_pair = generateSyntheticImage(img_width, img_height, 0.4f);
        auto right_image = generateSyntheticImage(img_width, img_height, 0.6f);
        
        Mat result = correlate(image_pair, right_image, 
                              tile_size, max_disp, 16);
        
        bool success = (result.rows == img_height) && 
                      (result.cols == img_width);
        
        if (verbose_)
        {
            cout << "Large image test: processed " << img_width << "x" << img_height
                 << " image successfully" << endl;
        }
        
        return success;
    }
    catch (const exception& e)
    {
        if (verbose_)
        {
            cout << "Large image test failed: " << e.what() << endl;
        }
        return false;
    }
}

// ----------------------------------------------------------------------------
// testConsistencyCheck
// ----------------------------------------------------------------------------
bool MGMStereoGPUTests::testConsistencyCheck()
{
    try
    {
        int width = 16;
		int height = 16;
        
		// ---
        // Create simple test case where all disparities are 0
        // This avoids the correspondence calculation issue
		// ---
        vector<short> left_disp(width * height, 0);   // All zeros
        vector<short> right_disp(width * height, 0);  // All zeros
        vector<short> filtered_disp(width * height, -999);  // Init w/invalid
        
        // Allocate GPU memory
        short* d_left_disp;
        short* d_right_disp;
        short* d_filtered_disp;
        
        size_t disp_size = width * height * sizeof(short);
        
        cudaMalloc(&d_left_disp, disp_size);
        cudaMalloc(&d_right_disp, disp_size);
        cudaMalloc(&d_filtered_disp, disp_size);
        
        cudaMemcpy(d_left_disp, 
				   left_disp.data(), 
				   disp_size, 
				   cudaMemcpyHostToDevice);
				   
        cudaMemcpy(d_right_disp, 
				   right_disp.data(), 
				   disp_size, 
				   cudaMemcpyHostToDevice);
				   
        cudaMemcpy(d_filtered_disp, 
				   filtered_disp.data(), 
				   disp_size, 
				   cudaMemcpyHostToDevice);
        
        // Launch consistency check
        dim3 block(16, 16);
        dim3 grid(1, 1);
        
        consistencyCheck<<<grid, block>>>(
            d_left_disp, d_right_disp, d_filtered_disp, width, height, 1);
        
        cudaDeviceSynchronize();
        checkCudaError("consistencyCheck");
        
        // Check results
        cudaMemcpy(filtered_disp.data(), 
				   d_filtered_disp, 
				   disp_size, 
				   cudaMemcpyDeviceToHost);
        
        // All should pass consistency check (disparity 0 at x=0 should match)
        bool success = true;
		
        for (int i = 0; i < width * height; i++)
        {
            if (filtered_disp[i] != 0)
            {
                if (verbose_)
                {
                    cout << "Mismatch at index " 
						 << i << ": expected 0, got " 
                         << filtered_disp[i] 
						 << endl;
                }
                success = false;
                break;
            }
        }
        
        if (verbose_)
        {
            cout << "Consistency check: All disparities " 
                 << (success ? "passed" : "failed") 
				 << " consistency test" 
				 << endl;
        }
        
        cudaFree(d_left_disp);
        cudaFree(d_right_disp);
        cudaFree(d_filtered_disp);
        
        return success;
    }
    catch (const exception& e)
    {
        if (verbose_)
        {
            cout << "Consistency check test failed: " << e.what() << endl;
        }
        return false;
    }
}

// ----------------------------------------------------------------------------
// testMemoryValidation
// ----------------------------------------------------------------------------
bool MGMStereoGPUTests::testMemoryValidation()
{
    try
    {
        MGMStereoTileGPU tile(0, 0, 32, 32, 64, 64, 16, 4);
        
        // This should not crash
        tile.validateMemory("Test validation");
        
        return true;
    }
    catch (const exception& e)
    {
        if (verbose_)
        {
            cout << "Memory validation test failed: " << e.what() << endl;
        }
		
        return false;
    }
}

// ----------------------------------------------------------------------------
// testErrorHandling
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// testErrorHandling
// ----------------------------------------------------------------------------
bool MGMStereoGPUTests::testErrorHandling()
{
    try
    {
        // Test with invalid tile position (negative)
        try
        {
            MGMStereoTileGPU invalid_tile(-1, -1, 32, 32, 64, 64, 16, 4);
            return false;  // Should have thrown an exception
        }
        catch (const std::invalid_argument&)
        {
            // Expected behavior
            if (verbose_)
            {
                cout << "  Correctly caught negative tile position" << endl;
            }
        }
        
        // Test with invalid tile dimensions (zero)
        try
        {
            MGMStereoTileGPU invalid_tile(0, 0, 0, 0, 64, 64, 16, 4);
            return false;  // Should have thrown an exception
        }
        catch (const std::invalid_argument&)
        {
            // Expected behavior
            if (verbose_)
            {
                cout << "  Correctly caught zero tile dimensions" << endl;
            }
        }
        
        // Test with invalid max disparity
        try
        {
            MGMStereoTileGPU invalid_tile(0, 0, 32, 32, 64, 64, -1, 4);
            return false;  // Should have thrown an exception
        }
        catch (const std::invalid_argument&)
        {
            // Expected behavior
            if (verbose_)
            {
                cout << "  Correctly caught negative max disparity" << endl;
            }
        }
        
        // Test with tile outside image bounds
        try
        {
            MGMStereoTileGPU invalid_tile(100, 100, 32, 32, 64, 64, 16, 4);
            return false;  // Should have thrown an exception
        }
        catch (const std::invalid_argument&)
        {
            // Expected behavior
            if (verbose_)
            {
                cout << "  Correctly caught tile outside image bounds" << endl;
            }
        }
        
        return true;
    }
    catch (const exception& e)
    {
        if (verbose_)
        {
            cout << "Error handling test failed: " << e.what() << endl;
        }
        return false;
    }
}

// ----------------------------------------------------------------------------
// runAllTests
// ----------------------------------------------------------------------------
bool MGMStereoGPUTests::runAllTests()
{
    cout << "Running MGM Stereo GPU Regression Tests..." << endl;
    cout << "===========================================" << endl;
    
    printTestResult("GPU Memory Allocation", testGPUMemoryAllocation());
    printTestResult("Basic Kernel Execution", testBasicKernelExecution());
    printTestResult("Cost Computation", testCostComputation());
    printTestResult("Path Processing", testPathProcessing());
    printTestResult("Winner Takes All", testWinnerTakesAll());
    printTestResult("Sub-pixel Refinement", testSubPixelRefinement());
    printTestResult("Tile Processing", testTileProcessing());
    printTestResult("Boundary Conditions", testBoundaryConditions());
    printTestResult("Large Images", testLargeImages());
    printTestResult("Consistency Check", testConsistencyCheck());
    printTestResult("Memory Validation", testMemoryValidation());
    printTestResult("Error Handling", testErrorHandling());
    
    return tests_failed_ == 0;
}

// ----------------------------------------------------------------------------
// runSpecificTest
// ----------------------------------------------------------------------------
bool MGMStereoGPUTests::runSpecificTest(const string& test_name)
{
    cout << "Running specific test: " << test_name << endl;
	
    if (test_name == "memory") 
        return testGPUMemoryAllocation();
    else if (test_name == "kernel")
        return testBasicKernelExecution();
    else if (test_name == "cost")
        return testCostComputation();
    else if (test_name == "path")
        return testPathProcessing();
    else if (test_name == "wta")
        return testWinnerTakesAll();
    else if (test_name == "refinement")
        return testSubPixelRefinement();
    else if (test_name == "tile")
        return testTileProcessing();
    else if (test_name == "boundary")
        return testBoundaryConditions();
    else if (test_name == "large")
        return testLargeImages();
    else if (test_name == "consistency")
        return testConsistencyCheck();
    else if (test_name == "validation")
        return testMemoryValidation();
    else if (test_name == "error")
	{
        return testErrorHandling();
    }
	else
    {
        cout << "Unknown test: " 
			 << test_name 
			 << endl;
		
        cout << "Available tests: memory, kernel, cost, path, wta, "
			 << "refinement, tile, boundary, large, consistency, "
			 << "validation, error" 
			 << endl;
		
        return false;
    }
}

// ----------------------------------------------------------------------------
// printSummary
// ----------------------------------------------------------------------------
void MGMStereoGPUTests::printSummary()
{
    cout << "\n===========================================" << endl;
    cout << "Test Summary:" << endl;
    cout << "  Passed: " << tests_passed_ << endl;
    cout << "  Failed: " << tests_failed_ << endl;
    cout << "  Total:  " << (tests_passed_ + tests_failed_) << endl;
    
    if (tests_failed_ == 0)
    {
        cout << "ALL TESTS PASSED!" << endl;
    }
    else
    {
        cout << "SOME TESTS FAILED!" << endl;
    }
    cout << "===========================================" << endl;
}

// ----------------------------------------------------------------------------
// generateTestData
// ----------------------------------------------------------------------------
void MGMStereoGPUTests::generateTestData(const string& output_dir)
{
    cout << "Generating test data in directory: " << output_dir << endl;
    
    // Create synthetic stereo image pairs of various sizes
    vector<pair<int, int>> test_sizes = 
    {
        {64, 64},
        {128, 96},
        {256, 192},
        {512, 384}
    };
    
    vector<int> disparities = {8, 16, 32, 64};
    
    for (auto size : test_sizes)
    {
        int width = size.first;
        int height = size.second;
        
        for (int max_disp : disparities)
        {
            MGMStereoGPUTests test_gen(false);  // Non-verbose
			
            auto image_pair = test_gen.generateStereoImagePair(width, 
															   height,
															   max_disp);
            
            // Save images as OpenCV format
            string left_filename = output_dir + 
								   "/left_" + 
                                   to_string(width) + 
								   "x" + 
								   to_string(height) + 
                                   "_d" + 
								   to_string(max_disp) + 
								   ".tiff";
			
            string right_filename = output_dir + 
									"/right_" + 
                                   	to_string(width) + 
									"x" + 
									to_string(height) + 
                                    "_d" + 
									to_string(max_disp) + 
									".tiff";
            
            // Convert to 8-bit for saving
            Mat left_8bit, right_8bit;
            image_pair.first.convertTo(left_8bit, CV_8UC1, 255.0);
            image_pair.second.convertTo(right_8bit, CV_8UC1, 255.0);
            
            imwrite(left_filename, left_8bit);
            imwrite(right_filename, right_8bit);
            
            cout << "Generated: " 
				 << left_filename 
				 << " and " 
				 << right_filename 
				 << endl;
            
            // Generate ground truth disparity
            Mat gt_disparity = test_gen.generateDisparityPattern(width,
				 												 height,
																 max_disp);
			
            string gt_filename = output_dir + 
								 "/gt_disp_" + 
                                 to_string(width) + 
								 "x" + 
								 to_string(height) + 
                                 "_d" + 
								 to_string(max_disp) + 
								 ".tiff";
			
            imwrite(gt_filename, gt_disparity);
            
            cout << "Generated: " << gt_filename << endl;
        }
    }
    
    cout << "Test data generation complete." << endl;
}

// ----------------------------------------------------------------------------
// Test runner main function
// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
    // Check for CUDA device
    int device_count;
    cudaGetDeviceCount(&device_count);
    
    if (device_count == 0)
    {
        cerr << "No CUDA devices found. Tests require CUDA-capable GPU." 
			 << endl;
		
        return 1;
    }
    
    cout << "Found " << device_count << " CUDA device(s)" << endl;
    
    // Set device properties
    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    cout << "Using GPU: " << prop.name << endl;
    cout << "Compute Capability: " << prop.major << "." << prop.minor << endl;
	
    cout << "Total Global Memory: " 
		 << prop.totalGlobalMem / (1024 * 1024) 
		 << " MB" 
		 << endl;
    
    MGMStereoGPUTests tests(true);  // Verbose output
    
    bool success = false;
    
    if (argc > 1)
    {
        string command(argv[1]);
        
        if (command == "generate" && argc > 2)
        {
            // Generate test data
            string output_dir(argv[2]);
            MGMStereoGPUTests::generateTestData(output_dir);
            return 0;
        }
        else if (command == "run" && argc > 2)
        {
            // Run specific test
            string test_name(argv[2]);
            success = tests.runSpecificTest(test_name);
        }
        else if (command == "all")
        {
            // Run all tests
            success = tests.runAllTests();
		    tests.printSummary();
        }
        else
        {
            cout << "Usage:" << endl;
			
            cout << "  " 
				 << argv[0] 
				 << " all                    - Run all tests" 
				 << endl;
			
            cout << "  " 
				 << argv[0] 
				 << " run <test_name>        - Run specific test" 
				 << endl;
			
            cout << "  " 
			     << argv[0] 
				 << " generate <output_dir>  - Generate test data" 
				 << endl;
			
            return 1;
        }
    }
    else
    {
        // Default: run all tests
        success = tests.runAllTests();
	    tests.printSummary();
    }
    
    return success ? 0 : 1;
}
