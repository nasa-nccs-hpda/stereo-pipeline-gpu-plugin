#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <gdal_priv.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

#ifdef COMPILE_FOR_TESTING
// When compiling for testing, make functions available for linking
// but don't include main()
#define SKIP_MAIN
#endif

// ----------------------------------------------------------------------------
// MGM StereoTileGPU
// ----------------------------------------------------------------------------
class MGMStereoTileGPU 
{

private:

    // Tile dimensions and position
    int tile_x_;
    int tile_y_;
    int tile_width_;
    int tile_height_;
    int max_disparity_;
    int num_paths_;
    float P1_;
    float P2_;
    int border_size_;
    
    // Host memory
    vector<float> h_aggregated_costs_;
    vector<short> h_disparity_map_;
    vector<float> h_refined_disparity_;
    
    // Device memory pointers
    float* d_tile_costs_;
    float* d_aggregated_costs_;
    short* d_disparity_map_;
    float* d_refined_disparity_;
    
    // Image dimensions (full image)
    int img_width_;
    int img_height_;
    
    // Padded tile dimensions
    int padded_width_;
    int padded_height_;
    
    float* d_path_costs_;

public:

    // ------------------------------------------------------------------------
    // MGMStereoTileGPU
    // ------------------------------------------------------------------------
    MGMStereoTileGPU(int tile_x, 
                     int tile_y, 
                     int tile_width, 
                     int tile_height,
                     int img_width, 
                     int img_height, 
                     int max_disparity, 
                     int border_size = 16, 
                     int num_paths = 8): tile_x_(tile_x), 
                                         tile_y_(tile_y), 
                                         tile_width_(tile_width), 
                                         tile_height_(tile_height),
                                         img_width_(img_width), 
                                         img_height_(img_height),
                                         max_disparity_(max_disparity), 
                                         border_size_(border_size),
                                         num_paths_(num_paths), 
                                         // P1_(10.0f),
                                         // P2_(120.0f)
                                         P1_(8.0f), 
                                         P2_(32.0f) 
    {
	    // Validate input parameters
	    if (tile_x < 0 || tile_y < 0)
	    {
	        throw std::invalid_argument("Tile position cannot be negative");
	    }
    
	    if (tile_width <= 0 || tile_height <= 0)
	    {
	        throw std::invalid_argument("Tile dimensions must be positive");
	    }
    
	    if (img_width <= 0 || img_height <= 0)
	    {
	        throw std::invalid_argument("Image dimensions must be positive");
	    }
    
	    if (max_disparity <= 0)
	    {
	        throw std::invalid_argument("Max disparity must be positive");
	    }
    
	    if (border_size < 0)
	    {
	        throw std::invalid_argument("Border size cannot be negative");
	    }
    
	    if (num_paths <= 0 || num_paths > 8)
	    {
	        throw std::invalid_argument("Number of paths must be between "
										"1 and 8");
	    }
    
	    if (tile_x >= img_width || tile_y >= img_height)
	    {
	        throw std::invalid_argument("Tile position is outside "
										"image bounds");
	    }
    
	    if (tile_x + tile_width > img_width || 
			tile_y + tile_height > img_height)
	    {
	        throw std::invalid_argument("Tile extends beyond image bounds");
	    }
    
        // Calculate padded dimensions
        padded_width_ = tile_width_ + 2 * border_size_;
        padded_height_ = tile_height_ + 2 * border_size_;
        
        allocateGPUMemory();
        
        // Allocate host result buffers
        h_aggregated_costs_.resize(tile_width_ * tile_height_ *
				 					max_disparity_);
		
        h_disparity_map_.resize(tile_width_ * tile_height_);
        h_refined_disparity_.resize(tile_width_ * tile_height_);

        // Check GPU limits
        int device;
        cudaGetDevice(&device);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
    }
    
    // ------------------------------------------------------------------------
    // ~MGMStereoTileGPU
    // ------------------------------------------------------------------------
    ~MGMStereoTileGPU() 
    {
        freeGPUMemory();
    }
    
    // ------------------------------------------------------------------------
    // allocateGPUMemory
    // ------------------------------------------------------------------------
    void allocateGPUMemory() 
    {
        // Tile cost volumes (padded)
        size_t tile_cost_size = padded_width_ * 
                               padded_height_ * 
                               max_disparity_ * 
                               sizeof(float);
        
        cudaError_t err1 = cudaMalloc(&d_tile_costs_, tile_cost_size);
        cudaError_t err2 = cudaMalloc(&d_path_costs_, tile_cost_size);
        cudaError_t err3 = cudaMalloc(&d_aggregated_costs_, tile_cost_size);
    
        if (err1 != cudaSuccess) 
        {
            cout << "ERROR allocating d_tile_costs_: " 
                 << cudaGetErrorString(err1) 
                 << endl;
        }
        
        if (err2 != cudaSuccess) 
        {
            cout << "ERROR allocating d_path_costs_: " 
                 << cudaGetErrorString(err2) 
                 << endl;
        }
        
        if (err3 != cudaSuccess) 
        {
            cout << "ERROR allocating d_aggregated_costs_: " 
                 << cudaGetErrorString(err3) 
                 << endl;
        }
        
        // Disparity maps (core tile only)
        size_t tile_pixels = tile_width_ * tile_height_;
        
        cudaError_t err4 = cudaMalloc(&d_disparity_map_, 
                                      tile_pixels * sizeof(short));
                                      
        cudaError_t err5 = cudaMalloc(&d_refined_disparity_, 
                                      tile_pixels * sizeof(float));
    
        if (err4 != cudaSuccess) 
        {
            cout << "ERROR allocating d_disparity_map_: " 
                 << cudaGetErrorString(err4) 
                 << endl;
        }
        
        if (err5 != cudaSuccess) 
        {
            cout << "ERROR allocating d_refined_disparity_: " 
                 << cudaGetErrorString(err5) 
                 << endl;
        }
    
        cout << "GPU memory allocation complete." << endl;
    }
        
    // ------------------------------------------------------------------------
    // freeGPUMemory
    // ------------------------------------------------------------------------
    void freeGPUMemory() 
    {
        cudaFree(d_tile_costs_);
        cudaFree(d_aggregated_costs_);
        cudaFree(d_disparity_map_);
        cudaFree(d_refined_disparity_);
        cudaFree(d_path_costs_);
    }
    
    // ------------------------------------------------------------------------
    // validateMemory
    // ------------------------------------------------------------------------
    void validateMemory(const std::string& stage) 
    {
        cout << "=== Memory Validation " << stage << " ===" << endl;
        
        // Validate tile costs memory
        float test_pattern[4] = {1.23f, 4.56f, 7.89f, 0.12f};
        
        size_t size = padded_width_ * 
                      padded_height_ * 
                      max_disparity_ * 
                      sizeof(float);
        
        cudaMemcpy(d_tile_costs_, 
                   test_pattern, 
                   4 * sizeof(float), 
                   cudaMemcpyHostToDevice);
                   
        cudaDeviceSynchronize();
        float read_pattern[4];
        
        cudaMemcpy(read_pattern, 
                   d_tile_costs_, 
                   4 * sizeof(float), 
                   cudaMemcpyDeviceToHost);
        
        bool valid = (read_pattern[0] == test_pattern[0] && 
                      read_pattern[1] == test_pattern[1] &&
                      read_pattern[2] == test_pattern[2] && 
                      read_pattern[3] == test_pattern[3]);
        
        cout << "d_tile_costs_ validation: " 
             << (valid ? "PASSED" : "FAILED") 
             << endl;
        
        // Check GPU memory status
        size_t free_bytes;
        size_t total_bytes;
        cudaMemGetInfo(&free_bytes, &total_bytes);
        
        cout << "GPU Memory: " 
             << free_bytes / (1024*1024) 
             << " MB free of " 
             << total_bytes / (1024*1024)
             << " MB total" 
             << endl;
        
        // Check for CUDA errors
        cudaError_t error = cudaGetLastError();
        
        cout << "CUDA Status: " 
             << (error == cudaSuccess ? "OK" : cudaGetErrorString(error))
             << endl;
        
        cout << "=============================================" << endl;
    }

    // ------------------------------------------------------------------------
    // TileResult
    // ------------------------------------------------------------------------
    struct TileResult 
    {
        vector<float> aggregated_costs;
        vector<short> disparity_map;
        vector<float> refined_disparity;
        
        TileResult(int width, int height, int max_disp) 
        {
            aggregated_costs.resize(width * height * max_disp);
            disparity_map.resize(width * height);
            refined_disparity.resize(width * height);
        }
    };
    
    // ------------------------------------------------------------------------
    // processPathSafe
    // ------------------------------------------------------------------------
    void processPathSafe(int dx, int dy);
    
    // ------------------------------------------------------------------------
    // processDiagonalSafe
    // ------------------------------------------------------------------------
    void processDiagonalSafe(int dx, int dy);
    
    // ------------------------------------------------------------------------
    // processTile
    // ------------------------------------------------------------------------
    TileResult processTileFromGPU(float* d_left_image, float* d_right_image);
};

// ----------------------------------------------------------------------------
// computeTileMatchingCosts
//
// CUDA kernel to compute initial matching costs for a tile
// ----------------------------------------------------------------------------
__global__ void computeTileMatchingCosts(const float* left_img, 
                                         const float* right_img,
                                         float* tile_costs,
                                         int img_width, 
                                         int img_height,
                                         int tile_x, 
                                         int tile_y, 
                                         int tile_width, 
                                         int tile_height,
                                         int border_size, 
                                         int max_disp) 
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.z * blockDim.z + threadIdx.z;
    
    int padded_width = tile_width + 2 * border_size;
    int padded_height = tile_height + 2 * border_size;
    
    if (tx >= padded_width || ty >= padded_height || d >= max_disp) return;
    
    // Calculate the tile index BEFORE any other operations
    int tile_idx = (ty * padded_width + tx) * max_disp + d;
    int max_tile_idx = padded_width * padded_height * max_disp - 1;
    
    // CRITICAL: Check for buffer overrun
    if (tile_idx < 0 || tile_idx > max_tile_idx) 
    {
        printf("ERROR: Buffer overrun! tile_idx=%d, max_allowed=%d\n",     
               tile_idx, 
               max_tile_idx);
               
        printf("  tx=%d, ty=%d, d=%d, padded=(%dx%d), max_disp=%d\n", 
               tx, ty, d, padded_width, padded_height, max_disp);
               
        return; // Don't write anything
    }

    // Convert local tile coordinates to global image coordinates
    int global_x = tile_x - border_size + tx;
    int global_y = tile_y - border_size + ty;
    
    // Check if we're within image bounds
    if (global_x < 0 || global_x >= img_width || 
        global_y < 0 || global_y >= img_height ||
        global_x - d < 0) 
    {
        tile_costs[tile_idx] = INFINITY;
        
        // Debug: Report boundary conditions
        if (tx == 0 && ty == 0 && d < 3) 
        {
            printf("BOUNDARY: global_x=%d, global_y=%d, d=%d, "
				   "img_size=(%dx%d)\n",
                   global_x, global_y, d, img_width, img_height);
        }
        return;
    }
    
    // Compute matching cost
    int left_idx = global_y * img_width + global_x;
    int right_idx = global_y * img_width + (global_x - d);
    
    // Bounds check for safety
    if (left_idx >= img_width * img_height || 
        right_idx >= img_width * img_height ||
        left_idx < 0 || right_idx < 0) 
    {
        tile_costs[tile_idx] = INFINITY;
        return;
    }
    
    float left_val = left_img[left_idx];
    float right_val = right_img[right_idx];
    float cost = fabsf(left_val - right_val);
    tile_costs[tile_idx] = cost;

	// Debug #1
	if (tx == 0 && ty == 0 && d == 0)
	{
	    float min_cost = INFINITY;
	    float max_cost = -INFINITY;
	    int count = 0;
    
	    for (int i = 0; i < 100; i++)
	    {
	        if (i >= padded_width * padded_height * max_disp) break;
	        float cost = tile_costs[i];
	        if (!isinf(cost) && !isnan(cost))
	        {
	            min_cost = fminf(min_cost, cost);
	            max_cost = fmaxf(max_cost, cost);
	            count++;
	        }
	    }
    
	    printf("COST STATS: Range %.3f to %.3f, Valid count: %d/100\n", 
	           min_cost, max_cost, count);
	}
	// End Debug #1
}

// ----------------------------------------------------------------------------
// processRowSafe
//
// Process a single row of the cost volume, parallelizing within the row
// ----------------------------------------------------------------------------
__global__ void processRowSafe(
    const float* initial_costs, 
    float* path_costs,
    int padded_width, 
    int padded_height, 
    int max_disp,
    int row, 
    int dx, 
    int dy, 
    float P1, 
    float P2) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int d = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= padded_width || d >= max_disp) 
    {
        return;
    }
    
    int y = row;
    int idx = (y * padded_width + x) * max_disp + d;
    int prev_x = x - dx;
    int prev_y = y - dy;
    
    // For first pixel in path or boundary, just use initial cost
    if (prev_x < 0 || prev_x >= padded_width || 
        prev_y < 0 || prev_y >= padded_height) 
    {
        path_costs[idx] = initial_costs[idx];
        return;
    }
    
    // Get current cost
    float curr_cost = initial_costs[idx];
    
    // Find minimum cost from previous pixel
    float min_prev_cost = INFINITY;
    float min_all_prev = INFINITY;
    
    for (int prev_d = 0; prev_d < max_disp; prev_d++) 
    {
        int prev_idx = (prev_y * padded_width + prev_x) * max_disp + prev_d;
        float prev_cost = path_costs[prev_idx];
        
        min_all_prev = fminf(min_all_prev, prev_cost);
        
        float penalty = 0.0f;
        
        if (prev_d == d) 
        {
            penalty = 0.0f;
        } 
        else if (abs(prev_d - d) == 1) 
        {
            penalty = P1;
        } 
        else 
        {
            penalty = P2;
        }
        
        min_prev_cost = fminf(min_prev_cost, prev_cost + penalty);
    }
    
    // SGM aggregation
    path_costs[idx] = curr_cost + min_prev_cost - min_all_prev;
}

// ----------------------------------------------------------------------------
// processColumnSafe
//
// Process a single column of the cost volume
// ----------------------------------------------------------------------------
__global__ void processColumnSafe(
    const float* initial_costs, 
    float* path_costs,
    int padded_width, 
    int padded_height, 
    int max_disp,
    int col, 
    int dx, 
    int dy, 
    float P1, 
    float P2) 
{
    int y = blockIdx.x * blockDim.x + threadIdx.x;
    int d = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (y >= padded_height || d >= max_disp) 
    {
        return;
    }
    
    int x = col;
    int idx = (y * padded_width + x) * max_disp + d;
    int prev_x = x - dx;
    int prev_y = y - dy;
    
    // For first pixel in path or boundary, just use initial cost
    if (prev_x < 0 || prev_x >= padded_width || 
        prev_y < 0 || prev_y >= padded_height) 
    {
        path_costs[idx] = initial_costs[idx];
        return;
    }
    
    // Get current cost
    float curr_cost = initial_costs[idx];
    
    // Find minimum cost from previous pixel
    float min_prev_cost = INFINITY;
    float min_all_prev = INFINITY;
    
    for (int prev_d = 0; prev_d < max_disp; prev_d++) 
    {
        int prev_idx = (prev_y * padded_width + prev_x) * max_disp + prev_d;
        float prev_cost = path_costs[prev_idx];
        
        min_all_prev = fminf(min_all_prev, prev_cost);
        
        float penalty = 0.0f;
        
        if (prev_d == d) 
        {
            penalty = 0.0f;
        } 
        else if (abs(prev_d - d) == 1) 
        {
            penalty = P1;
        } 
        else 
        {
            penalty = P2;
        }
        
        min_prev_cost = fminf(min_prev_cost, prev_cost + penalty);
    }
    
    // SGM aggregation
    path_costs[idx] = curr_cost + min_prev_cost - min_all_prev;
}

// ----------------------------------------------------------------------------
// processDiagonalWave
//
// Process one wave of diagonal direction
// ----------------------------------------------------------------------------
__global__ void processDiagonalWave(const float* initial_costs, 
    								float* path_costs,
    								int padded_width, 
    								int padded_height, 
    								int max_disp,
    								int wave, 
    								int dx, 
    								int dy, 
    								float P1, 
    								float P2) 
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int d = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (d >= max_disp) return;
    
    // Calculate coordinates based on direction - THIS WAS WRONG
    int x, y;
    
    if (dx > 0 && dy > 0) 
    {
        // Top-left to bottom-right: (1,1)
        x = tx;
        y = wave - tx;
    } 
    else if (dx > 0 && dy < 0) 
    {
        // Bottom-left to top-right: (1,-1) 
        x = tx;
        y = (padded_height - 1) - (wave - tx);
    } 
    else if (dx < 0 && dy > 0) 
    {
        // Top-right to bottom-left: (-1,1)
        x = (padded_width - 1) - tx;
        y = wave - tx;
    } 
    else // dx < 0 && dy < 0
    {
        // Bottom-right to top-left: (-1,-1) - FIX THIS CASE
        x = (padded_width - 1) - tx;
        y = (padded_height - 1) - (wave - tx);
    }
    
    // CRITICAL: Add bounds checking
    if (x < 0 || x >= padded_width || y < 0 || y >= padded_height) 
    {
        return;
    }
    
    // Add wave bounds checking
    if (wave <= 0 || wave >= (padded_width + padded_height - 1))
    {
        return;
    }
    
    int idx = (y * padded_width + x) * max_disp + d;
    int prev_x = x - dx;
    int prev_y = y - dy;
    
    // For first pixel in path or boundary, just use initial cost
    if (prev_x < 0 || prev_x >= padded_width || 
        prev_y < 0 || prev_y >= padded_height) 
    {
        path_costs[idx] = initial_costs[idx];
        return;
    }
    
    float curr_cost = initial_costs[idx];
    
    // Validate current cost
    if (isnan(curr_cost) || isinf(curr_cost))
    {
        path_costs[idx] = 1.0f; // Default cost
        return;
    }
    
    float min_prev_cost = INFINITY;
    float min_all_prev = INFINITY;
    
    for (int prev_d = 0; prev_d < max_disp; prev_d++) 
    {
        int prev_idx = (prev_y * padded_width + prev_x) * max_disp + prev_d;
        float prev_cost = path_costs[prev_idx];
        
        // Validate previous cost
        if (isnan(prev_cost) || isinf(prev_cost))
        {
            continue; // Skip invalid costs
        }
        
        min_all_prev = fminf(min_all_prev, prev_cost);
        
        float penalty = 0.0f;
        if (prev_d == d) 
        {
            penalty = 0.0f;
        } 
        else if (abs(prev_d - d) == 1) 
        {
            penalty = P1;
        } 
        else 
        {
            penalty = P2;
        }
        
        min_prev_cost = fminf(min_prev_cost, prev_cost + penalty);
    }
    
    // Validate final result
    if (isinf(min_prev_cost) || isinf(min_all_prev))
    {
        path_costs[idx] = curr_cost; // Just use current cost
    }
    else
    {
        float result = curr_cost + min_prev_cost - min_all_prev;
        path_costs[idx] = isnan(result) ? curr_cost : result;
    }
}

// ----------------------------------------------------------------------------
// accumulatePathCosts
//
// Add path costs to aggregated costs
// ----------------------------------------------------------------------------
__global__ void accumulatePathCosts(const float* path_costs, 
    								float* aggregated_costs,
    								int padded_width, 
    								int padded_height, 
    								int max_disp)
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    int tz = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (tx >= padded_width || ty >= padded_height || tz >= max_disp) 
    {
        return;
    }
    
    int idx = (ty * padded_width + tx) * max_disp + tz;
    float path_cost = path_costs[idx];
    
    // Only accumulate valid costs
    if (!isnan(path_cost) && !isinf(path_cost))
    {
        atomicAdd(&aggregated_costs[idx], path_cost);
    }
}

// ----------------------------------------------------------------------------
// normalizeCosts
//
// Kernel to normalize aggregated costs
// ----------------------------------------------------------------------------
__global__ void normalizeCosts(float* aggregated_costs, 
                              int tile_width, int tile_height, int max_disp,
                              int padded_width, int border_size, int num_paths) 
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (tx >= tile_width || ty >= tile_height || d >= max_disp) return;
    
    // Convert to padded coordinates
    int padded_tx = tx + border_size;
    int padded_ty = ty + border_size;
    
    int idx = (padded_ty * padded_width + padded_tx) * max_disp + d;
    aggregated_costs[idx] /= num_paths;
}

// ----------------------------------------------------------------------------
// computeWinnerTakesAll
//
// CUDA kernel for winner-takes-all disparity computation
// ----------------------------------------------------------------------------
__global__ void computeWinnerTakesAll(const float* aggregated_costs, 
                                      short* disparity_map,
                                      int tile_width, 
                                      int tile_height, 
                                      int max_disp,
                                      int padded_width, 
                                      int border_size) 
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (tx >= tile_width || ty >= tile_height) return;
    
    // Convert to padded coordinates for cost lookup
    int padded_tx = tx + border_size;
    int padded_ty = ty + border_size;
    
    float min_cost = INFINITY;
    short best_disp = 0;
    
    // Find disparity with minimum cost
    for (int d = 0; d < max_disp; d++) 
    {
        int cost_idx = (padded_ty * padded_width + padded_tx) * max_disp + d;
        float cost = aggregated_costs[cost_idx];
        
        // Debug the first few pixels
        if (tx < 2 && ty < 2)
        {
            printf("Pixel(%d,%d) d=%d: cost=%.3f\n", tx, ty, d, cost);
        }
        
        if (!isnan(cost) && !isinf(cost) && cost < min_cost) 
        {
            min_cost = cost;
            best_disp = d;
        }
    }
    
    // Debug winner
    if (tx < 2 && ty < 2)
    {
        printf("WTA result for (%d,%d): disp=%d with cost=%.3f\n", 
               tx, ty, best_disp, min_cost);
    }
    
    // Store in tile coordinates
    int tile_idx = ty * tile_width + tx;
    disparity_map[tile_idx] = best_disp;
}

// ----------------------------------------------------------------------------
// refineDisparity
//
// CUDA kernel for sub-pixel disparity refinement
// ----------------------------------------------------------------------------
__global__ void refineDisparity(
    const float* aggregated_costs, const short* disparity_map, 
    float* refined_disparity,
    int tile_width, int tile_height, int max_disp,
    int padded_width, int border_size) 
{
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (tx >= tile_width || ty >= tile_height) return;
    
    int tile_idx = ty * tile_width + tx;
    short d = disparity_map[tile_idx];
    
    // Skip if at the boundaries of disparity range
    if (d <= 0 || d >= max_disp - 1) 
    {
        refined_disparity[tile_idx] = d;
        return;
    }
    
    // Convert to padded coordinates for cost lookup
    int padded_tx = tx + border_size;
    int padded_ty = ty + border_size;
    
    // Get costs for d-1, d, d+1
    int idx_prev = (padded_ty * padded_width + padded_tx) * max_disp + (d - 1);
    int idx_curr = (padded_ty * padded_width + padded_tx) * max_disp + d;
    int idx_next = (padded_ty * padded_width + padded_tx) * max_disp + (d + 1);
    
    float c_prev = aggregated_costs[idx_prev];
    float c_curr = aggregated_costs[idx_curr];
    float c_next = aggregated_costs[idx_next];
    
    // Parabolic interpolation
    float denom = 2.0f * (c_prev + c_next - 2.0f * c_curr);
    if (fabsf(denom) > 1e-5f) 
    {
        float offset = (c_prev - c_next) / denom;
        refined_disparity[tile_idx] = d + offset;
    } 
    else 
    {
        refined_disparity[tile_idx] = d;
    }
}

// ----------------------------------------------------------------------------
// consistencyCheck
//
// CUDA kernel for left-right consistency check
// ----------------------------------------------------------------------------
__global__ void consistencyCheck(
    const short* left_disparity, const short* right_disparity,
    short* filtered_disparity, int width, int height, int threshold) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    short d_left = left_disparity[idx];
    
    // Check corresponding pixel in right image
    int right_x = x - d_left;
    if (right_x < 0 || right_x >= width) 
    {
        filtered_disparity[idx] = -1; // Invalid
        return;
    }
    
    int right_idx = y * width + right_x;
    short d_right = right_disparity[right_idx];
    
    // Check consistency
    if (abs(d_left - d_right) <= threshold) 
    {
        filtered_disparity[idx] = d_left;
    } 
    else 
    {
        filtered_disparity[idx] = -1; // Invalid
    }
}

// ----------------------------------------------------------------------------
// processPathSafe
//
// Safe method to process a path direction
// ----------------------------------------------------------------------------
void MGMStereoTileGPU::processPathSafe(int dx, int dy) 
{
    // Initialize path costs with initial costs
    cudaMemcpy(d_path_costs_, 
               d_tile_costs_,
               padded_width_ * padded_height_ * max_disparity_ * sizeof(float),
               cudaMemcpyDeviceToDevice);
    
    // Determine processing order
    if (dx == 0) 
    {
        // Vertical direction - process row by row
        int start_row = (dy > 0) ? 1 : padded_height_ - 2;
        int end_row = (dy > 0) ? padded_height_ : -1;
        int step = (dy > 0) ? 1 : -1;
        
        for (int row = start_row; row != end_row; row += step) 
        {
            dim3 block(16, 16);
            dim3 grid((padded_width_ + block.x - 1) / block.x,
                     (max_disparity_ + block.y - 1) / block.y);
            
            processRowSafe<<<grid, block>>>(
                d_tile_costs_, 
                d_path_costs_,
                padded_width_, 
                padded_height_, 
                max_disparity_,
                row, 
                dx, 
                dy, 
                P1_, 
                P2_);
				
			cudaDeviceSynchronize();
        }
    } 
    else if (dy == 0) 
    {
        // Horizontal direction - process column by column
        int start_col = (dx > 0) ? 1 : padded_width_ - 2;
        int end_col = (dx > 0) ? padded_width_ : -1;
        int step = (dx > 0) ? 1 : -1;
        
        for (int col = start_col; col != end_col; col += step) 
        {
            dim3 block(16, 16);
            dim3 grid((padded_height_ + block.x - 1) / block.x,
                     (max_disparity_ + block.y - 1) / block.y);
            
            processColumnSafe<<<grid, block>>>(
                d_tile_costs_, 
                d_path_costs_,
                padded_width_, 
                padded_height_, 
                max_disparity_,
                col, 
                dx, 
                dy, 
                P1_, 
                P2_);
        }
    }
    else 
    {
        // Diagonal direction - process diagonal by diagonal
        processDiagonalSafe(dx, dy);
    }
    
    // Accumulate results
    dim3 acc_block(8, 8, 8);
    dim3 acc_grid((padded_width_ + acc_block.x - 1) / acc_block.x,
                  (padded_height_ + acc_block.y - 1) / acc_block.y,
                  (max_disparity_ + acc_block.z - 1) / acc_block.z);
    
    accumulatePathCosts<<<acc_grid, acc_block>>>(
        d_path_costs_, 
        d_aggregated_costs_,
        padded_width_, 
        padded_height_, 
        max_disparity_);
	
	// Debug #2
	float* debug_buffer = new float[100];
	
	cudaMemcpy(debug_buffer, 
			   d_path_costs_, 
			   100 * sizeof(float), 
			   cudaMemcpyDeviceToHost);

	float min_path = INFINITY;
	float max_path = -INFINITY;
	int valid_count = 0;

	for (int i = 0; i < 100; i++)
	{
	    if (!isinf(debug_buffer[i]) && !isnan(debug_buffer[i]))
	    {
	        min_path = std::min(min_path, debug_buffer[i]);
	        max_path = std::max(max_path, debug_buffer[i]);
	        valid_count++;
	    }
	}

	cout << "PATH COSTS for direction (" << dx << "," << dy 
	     << "): Range " << min_path << " to " << max_path
	     << ", Valid: " << valid_count << "/100" << endl;

	delete[] debug_buffer;
	// End Debug #2
}

// ----------------------------------------------------------------------------
// processDiagonalSafe
//
// Process diagonal directions safely
// ----------------------------------------------------------------------------
void MGMStereoTileGPU::processDiagonalSafe(int dx, int dy) 
{
    // For diagonal processing, we process wave by wave
    int num_waves = padded_width_ + padded_height_ - 1;
    
    for (int wave = 1; wave < num_waves; wave++) 
    {
        dim3 block(16, 16);
        dim3 grid((padded_width_ + block.x - 1) / block.x,
                 (max_disparity_ + block.y - 1) / block.y);
        
        processDiagonalWave<<<grid, block>>>(
            d_tile_costs_, 
            d_path_costs_,
            padded_width_, 
            padded_height_, 
            max_disparity_,
            wave, 
            dx, 
            dy, 
            P1_, 
            P2_);
    }
}

// ----------------------------------------------------------------------------
// processTileFromGPU
//
// Main method to process a tile
// ----------------------------------------------------------------------------
MGMStereoTileGPU::TileResult MGMStereoTileGPU::processTileFromGPU(
    float* d_left_image, 
    float* d_right_image) 
{
    // Step 1: Compute initial matching costs for the tile
    dim3 cost_block(8, 8, 8);
	
    dim3 cost_grid((padded_width_ + cost_block.x - 1) / cost_block.x,
                   (padded_height_ + cost_block.y - 1) / cost_block.y,
                   (max_disparity_ + cost_block.z - 1) / cost_block.z);

    computeTileMatchingCosts<<<cost_grid, cost_block>>>(
        d_left_image, 
        d_right_image, 
        d_tile_costs_,
        img_width_, 
        img_height_,
        tile_x_, 
        tile_y_, 
        tile_width_, 
        tile_height_,
        border_size_, 
        max_disparity_);
        
    cudaDeviceSynchronize();

    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    
    if (error != cudaSuccess) 
    {
       cerr << "\nCUDA kernel error: " << cudaGetErrorString(error) << endl;
    }

    // Initialize aggregated costs to zero
    cudaMemset(d_aggregated_costs_, 
               0, 
               padded_width_ * padded_height_ * max_disparity_ *
				   sizeof(float));
    
    // Step 2: Process all 8 path directions safely
    int directions[8][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, 
                            {1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    
    for (int path = 0; path < num_paths_; path++) 
    {
        processPathSafe(directions[path][0], directions[path][1]);
        cudaDeviceSynchronize();
    }

    // Step 3: Normalize costs
    dim3 norm_block(8, 8, 8);
	
    dim3 norm_grid((tile_width_ + norm_block.x - 1) / norm_block.x,
                   (tile_height_ + norm_block.y - 1) / norm_block.y,
                   (max_disparity_ + norm_block.z - 1) / norm_block.z);
    
    normalizeCosts<<<norm_grid, norm_block>>>(
        d_aggregated_costs_, 
        tile_width_, 
        tile_height_, 
        max_disparity_,
        padded_width_, 
        border_size_, 
        num_paths_);
    
    // Step 4: Winner-takes-all disparity computation
    dim3 wta_block(16, 16);
	
    dim3 wta_grid((tile_width_ + wta_block.x - 1) / wta_block.x,
                  (tile_height_ + wta_block.y - 1) / wta_block.y);

	// Debug #3
	float* debug_agg_costs = new float[100];
	
	cudaMemcpy(debug_agg_costs, 
			   d_aggregated_costs_, 100 * sizeof(float), 
			   cudaMemcpyDeviceToHost);

	float min_agg = INFINITY;
	float max_agg = -INFINITY;
	int valid_agg = 0;

	for (int i = 0; i < 100; i++)
	{
	    if (!isinf(debug_agg_costs[i]) && !isnan(debug_agg_costs[i]))
	    {
	        min_agg = std::min(min_agg, debug_agg_costs[i]);
	        max_agg = std::max(max_agg, debug_agg_costs[i]);
	        valid_agg++;
	    }
	}

	cout << "AGGREGATED COSTS: Range " << min_agg << " to " << max_agg 
	     << ", Valid: " << valid_agg << "/100" << endl;

	delete[] debug_agg_costs;
	// End Debug #3
					  
    computeWinnerTakesAll<<<wta_grid, wta_block>>>(
        d_aggregated_costs_,
        d_disparity_map_,
        tile_width_, 
        tile_height_,
        max_disparity_,
        padded_width_, 
        border_size_);

	// Debug #5
	short debug_disp[100];
	
	cudaMemcpy(debug_disp, 
			   d_disparity_map_, 100 * sizeof(short), 
			   cudaMemcpyDeviceToHost);

	int zero_count = 0;
	int max_found_disp = 0;
	
	for (int i = 0; i < 100; i++)
	{
	    if (debug_disp[i] == 0) zero_count++;
	    max_found_disp = std::max(max_found_disp, (int)debug_disp[i]);
	}

	cout << "DISPARITY MAP: " << zero_count << "/100 zeros, max value: " 
	     << max_found_disp << endl;
	// End Debug #5

    // Step 5: Sub-pixel refinement
    refineDisparity<<<wta_grid, wta_block>>>(
        d_aggregated_costs_,
        d_disparity_map_,
        d_refined_disparity_,
        tile_width_, 
        tile_height_, 
        max_disparity_,
        padded_width_, 
        border_size_);
    
    // Step 6: Copy results back to host
    vector<float> temp_costs(padded_width_ * padded_height_ * max_disparity_);
    
    cudaMemcpy(temp_costs.data(), 
               d_aggregated_costs_,
               temp_costs.size() * sizeof(float), 
               cudaMemcpyDeviceToHost);
    
    // Create result structure
    TileResult result(tile_width_, tile_height_, max_disparity_);
    
    int result_idx = 0;
    
    for (int ty = 0; ty < tile_height_; ty++) 
    {
        for (int tx = 0; tx < tile_width_; tx++) 
        {
            for (int d = 0; d < max_disparity_; d++) 
            {
                int padded_tx = tx + border_size_;
                int padded_ty = ty + border_size_;
                
                int padded_idx = (padded_ty * padded_width_ + padded_tx) *
                                 max_disparity_ + d;
                
                result.aggregated_costs[result_idx++] = temp_costs[padded_idx];
            }
        }
    }
    
    // Copy disparity maps
    cudaMemcpy(result.disparity_map.data(), 
               d_disparity_map_,
               tile_width_ * tile_height_ * sizeof(short), 
               cudaMemcpyDeviceToHost);
    
    cudaMemcpy(result.refined_disparity.data(), 
               d_refined_disparity_,
               tile_width_ * tile_height_ * sizeof(float),
               cudaMemcpyDeviceToHost);
    
    return result;
}

// ----------------------------------------------------------------------------
// checkGPUMemoryState
// ----------------------------------------------------------------------------
void checkGPUMemoryState(const std::string& checkpoint) 
{
    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    
    cout << "=== GPU Memory at " << checkpoint << " ===" << endl;
    cout << "Free memory: " << free_bytes / (1024*1024) << " MB" << endl;
    cout << "Total memory: " << total_bytes / (1024*1024) << " MB" << endl;
    
    cout << "Used memory: " 
         << (total_bytes - free_bytes) / (1024*1024) 
         << " MB" 
         << endl;
    
    // Check for any CUDA errors
    cudaError_t error = cudaGetLastError();
    
    if (error != cudaSuccess) 
    {
        cout << "CUDA error present: " << cudaGetErrorString(error) << endl;
    } 
    else 
    {
        cout << "No CUDA errors" << endl;
    }
    cout << "=================================" << endl;
}

// ----------------------------------------------------------------------------
// testKernel
// ----------------------------------------------------------------------------
__global__ void testKernel(int test_param) 
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid == 0) 
    {
        printf("Test kernel called with param: %d\n", test_param);
    }
}

// ----------------------------------------------------------------------------
// correlate
// ----------------------------------------------------------------------------
Mat correlate(const Mat& left, 
              const Mat& right, 
              int tile_size, 
              int max_disparity, 
              int border_size) 
{
    int img_width = left.cols;
    int img_height = left.rows;
    
    // Allocate GPU memory for full images ONCE
    float* d_left_image;
    float* d_right_image;
    size_t img_size = img_width * img_height * sizeof(float);
    cudaMalloc(&d_left_image, img_size);
    cudaMalloc(&d_right_image, img_size);
    
    // Copy images to GPU once.
    cudaMemcpy(d_left_image, 
               left.ptr<float>(), 
               img_size, 
               cudaMemcpyHostToDevice);
               
    cudaMemcpy(d_right_image, 
               right.ptr<float>(), 
               img_size, 
               cudaMemcpyHostToDevice);
  
    int num_tiles_x = (img_width + tile_size - 1) / tile_size;
    int num_tiles_y = (img_height + tile_size - 1) / tile_size;
    
    cout << "Processing " 
         << num_tiles_x 
         << "x" 
         << num_tiles_y 
         << " tiles on GPU..." 
         << endl;
    
    // Create full disparity maps for visualization
    Mat full_disparity(img_height, img_width, CV_16S, Scalar(0));
    Mat full_refined(img_height, img_width, CV_32F, Scalar(0.0f));
    
    for (int ty = 0; ty < num_tiles_y; ty++) 
    {
        for (int tx = 0; tx < num_tiles_x; tx++) 
        {
            int tile_x = tx * tile_size;
            int tile_y = ty * tile_size;
            int tile_width = min(tile_size, img_width - tile_x);
            int tile_height = min(tile_size, img_height - tile_y);
            
            // Calculate actual border size based on tile position
            int left_border = min(border_size, tile_x);
            int top_border = min(border_size, tile_y);
            
            int right_border = min(border_size, 
                                   img_width - (tile_x + tile_width));
                                   
            int bottom_border = min(border_size, 
                                    img_height - (tile_y + tile_height));
            
            // Use minimum border for simplicity
            int actual_border = min({left_border, 
                                     top_border, 
                                     right_border, 
                                     bottom_border});
            
            cout << "Tile (" 
                 << tile_x 
                 << "," 
                 << tile_y 
                 << ") using border: " 
                 << actual_border 
                 << endl;
            
            cout << "Testing simple kernel call..." << endl;

            testKernel<<<1, 1>>>(42);
            cudaDeviceSynchronize();

            checkGPUMemoryState("After test kernel");
            
            cudaError_t test_error = cudaGetLastError();

            if (test_error != cudaSuccess) 
            {
                cerr << "Even simple kernel failed: " 
                     << cudaGetErrorString(test_error) 
                     << endl;
            } 
            else 
            {
                cout << "Simple kernel succeeded" << endl;
            }

            MGMStereoTileGPU tile(tile_x,
                                  tile_y,
                                  tile_width,
                                  tile_height,
                                  img_width,
                                  img_height,
                                  max_disparity,
                                  actual_border);

            tile.validateMemory("Before processing");

            MGMStereoTileGPU::TileResult result =
                tile.processTileFromGPU(d_left_image, d_right_image);

            tile.validateMemory("After processing");

            // Copy tile results to full image
            for (int y = 0; y < tile_height; y++)
            {
                for (int x = 0; x < tile_width; x++)
                {
                    int tile_idx = y * tile_width + x;
                    int img_x = tile_x + x;
                    int img_y = tile_y + y;

                    if (img_x < img_width && img_y < img_height)
                    {
                        full_disparity.at<short>(img_y, img_x) =
                            result.disparity_map[tile_idx];

                        full_refined.at<float>(img_y, img_x) =
                            result.refined_disparity[tile_idx];
                    }
                }
            }

            // Print statistics for this tile
            short min_disp = *min_element(result.disparity_map.begin(),
                                          result.disparity_map.end());

            short max_disp = *max_element(result.disparity_map.begin(),
                                          result.disparity_map.end());

            cout << "  Tile disparity range: "
                 << min_disp
                 << " to "
                 << max_disp
                 << endl;
        }
    }
    
    return full_refined;
}

// ----------------------------------------------------------------------------
// saveGeoTIFF
// ----------------------------------------------------------------------------
void saveGeoTIFF(const string& filename, const Mat& data) 
{
    GDALAllRegister();
    GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("GTiff");
    
    if (!driver) 
    {
        cerr << "Could not get GTiff driver" << endl;
        return;
    }

    GDALDataset* ds = driver->Create(filename.c_str(), 
                                     data.cols, 
                                     data.rows, 
                                     1, 
                                     GDT_Byte, 
                                     nullptr);

    if (!ds) 
    {
        cerr << "Could not create output dataset" << endl;
        return;
    }
    
    ds->GetRasterBand(1)->RasterIO(GF_Write, 
                                   0,
                                   0, 
                                   data.cols,
                                   data.rows,
                                   (void*)data.ptr<float>(),
                                   data.cols,
                                   data.rows,
                                   GDT_Float32,
                                   0,
                                   0);
                                   
    ds->GetRasterBand(1)->SetNoDataValue(numeric_limits<float>::quiet_NaN());
    ds->FlushCache();
    GDALClose(ds);
    cout << "Saved: " << filename << endl;
}

// ----------------------------------------------------------------------------
// validateImage
// ----------------------------------------------------------------------------
void validateImage(const Mat& IMAGE)
{
    int nan_count = 0;
    int inf_count = 0;
    int zero_count = 0;
    float min_val = 9999.0;
    float max_val = 0.0;
    
    for (int y = 0; y < IMAGE.rows; y++) 
    {
        for (int x = 0; x < IMAGE.cols; x++) 
        {
            float val = IMAGE.at<float>(y, x);
            
            if (isnan(val)) 
            {
                nan_count++;
            } 
            else if (isinf(val)) 
            {
                inf_count++;
            } 
            else if (val == 0)
            {
                zero_count++;
            }
            else 
            {
                min_val = std::min(min_val, val);
                max_val = std::max(max_val, val);
            }
        }
    }
}

// ----------------------------------------------------------------------------
// runCorrelator
//
// Regression testing needs a main function which would conflict with this
// program's main.  Move the business logic from the main to here.
//
// int main(int argc, char** argv)
// ----------------------------------------------------------------------------
int runCorrelator(int argc, char** argv)
{
    // Process command-line options.
    if (argc < 4) 
    {
        cerr << "Usage:\n"
             << argv[0]
             << " [options] left.tif right.tif output_disparity.tif\n\n"
             << "Options:\n"
             << "  -num_disp <int>\n"
             << "  -tile <int> (default 1024)\n"
             << "  -overlap <int> (default 64)\n";
             
        return 1;
    }

    // Default parameters
    int num_disp = 64;
    int tile_size = 1024;
    int overlap = 64;

    int argi = 1;
    
    cout << "argc: " << argc << endl;
    
    while (argi < argc - 3) 
    {
        string key(argv[argi]);
        string val(argv[argi+1]);
        
        if (key == "-num_disp") num_disp = stoi(val);
        else if (key == "-tile") tile_size = stoi(val);
        else if (key == "-overlap") overlap = stoi(val);
        else 
        {
            cerr << "Unknown option: " << key << endl;
            return 1;
        }
        
        argi += 2;
    }

    string left_path(argv[argc-3]);
    string right_path(argv[argc-2]);
    string out_path(argv[argc-1]);

    cout << "Running GPU StereoBM with parameters:\n"
         << "  num_disp=" << num_disp
         << ", tile=" << tile_size
         << ", overlap=" << overlap << endl;

    GDALAllRegister();
    GDALDataset* l_ds = (GDALDataset*)GDALOpen(left_path.c_str(), 
                                               GA_ReadOnly);

    GDALDataset* r_ds = (GDALDataset*)GDALOpen(right_path.c_str(),
                                               GA_ReadOnly);
    
    if (!l_ds || !r_ds) 
    {
        cerr << "Failed to open input images." << endl;
        return 1;
    }

    int width = l_ds->GetRasterXSize();
    int height = l_ds->GetRasterYSize();
    cout << "Raster size: " << width << " x " << height << endl;

    Mat l_f(height, width, CV_32F);
    Mat r_f(height, width, CV_32F);
    
    l_ds->GetRasterBand(1)->RasterIO(GF_Read,
                                     0,
                                     0,
                                     width,
                                     height,
                                     l_f.ptr(),
                                     width,
                                     height,
                                     GDT_Float32,
                                     0,
                                     0);
                                     
    r_ds->GetRasterBand(1)->RasterIO(GF_Read,
                                     0,
                                     0,
                                     width,
                                     height,
                                     r_f.ptr(),
                                     width,
                                     height,
                                     GDT_Float32,
                                     0,
                                     0);
                                     
     validateImage(l_f);
     validateImage(r_f);

#ifdef HAVE_OPENCV_CUDA

    Mat final_disp = correlate(l_f, r_f, tile_size, num_disp, overlap);
    saveGeoTIFF(out_path, final_disp);

#else
    
    cerr << "OpenCV built without CUDA support!" << endl;
    return 1;

#endif

    GDALClose(l_ds);
    GDALClose(r_ds);

    return 0;
}
    
// ----------------------------------------------------------------------------
// main
// ----------------------------------------------------------------------------
#ifndef SKIP_MAIN
int main(int argc, char** argv) 
{
    return runCorrelator(argc, argv);
}
#endif	  