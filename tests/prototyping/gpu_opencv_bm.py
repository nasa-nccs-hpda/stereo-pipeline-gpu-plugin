import cv2
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import time

"""
    } else if (alg_name == "opencv_sgbm") {
      
      default_opts = std::string("-mode sgbm -block_size 3 -P1 8 -P2 32 -prefilter_cap 63 ") +
        "-uniqueness_ratio 10 -speckle_size 100 -speckle_range 32 -disp12_diff 1";
"""

def read_raster_as_gray(path):
    with rasterio.open(path) as src:
        image = src.read(1)
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        return image.astype(np.uint8)

# Load rasters
left_img = read_raster_as_gray("left.tif")
right_img = read_raster_as_gray("right.tif")

# Set matching parameters
num_disparities = 64  # must be divisible by 16
block_size = 15       # must be odd

####################################
# CPU StereoBM
####################################
stereo_cpu = cv2.StereoBM_create(
    numDisparities=num_disparities,
    blockSize=block_size
)

start_cpu = time.time()
disparity_cpu = stereo_cpu.compute(left_img, right_img).astype(np.float32) / 16.0
end_cpu = time.time()
cpu_time = end_cpu - start_cpu

####################################
# GPU StereoBM
####################################
# Upload to GPU
left_gpu = cv2.cuda_GpuMat()
right_gpu = cv2.cuda_GpuMat()
left_gpu.upload(left_img)
right_gpu.upload(right_img)

# Create GPU matcher
stereo_gpu = cv2.cuda.createStereoBM(
    numDisparities=num_disparities,
    blockSize=block_size
)

# Warm-up GPU
_ = stereo_gpu.compute(left_gpu, right_gpu)

start_gpu = time.time()
disparity_gpu = stereo_gpu.compute(left_gpu, right_gpu)
disparity_gpu = disparity_gpu.download().astype(np.float32) / 16.0
end_gpu = time.time()
gpu_time = end_gpu - start_gpu

####################################
# Results
####################################
print(f"CPU StereoBM Time: {cpu_time:.4f} seconds")
print(f"GPU StereoBM Time: {gpu_time:.4f} seconds")

# Display side-by-side
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
axs[0].imshow(disparity_cpu, cmap='plasma')
axs[0].set_title(f"CPU StereoBM ({cpu_time:.3f}s)")
axs[0].axis("off")
axs[1].imshow(disparity_gpu, cmap='plasma')
axs[1].set_title(f"GPU StereoBM ({gpu_time:.3f}s)")
axs[1].axis("off")
plt.tight_layout()
plt.show()
