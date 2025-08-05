# SPGPU - Stereo Pipeline GPU Plugin

Stereo Pipeline GPU Plugin implementation.

## Proposal Summary

The diverse pool of current ASP stereo correlators has been extensively optimized to take 
advantage of state-of-the-art CPU architectures. While effective, CPU-based implementations 
still have limitations in processing speed and scalability when compared to their Graphics 
Processing Unit (GPU) counterparts.

We will create GPU-based versions of the most-used ASP stereo correlators, ensuring that
these new GPU correlators will:
  - handle large datasets and high-resolution images efficiently
  - maintain or improve the relative accuracy of current stereo correlators
  - be easily deployed both on-premises and to commercial cloud systems

Correlators we will support:
  - Block Matching correlator
  - Semi-Global Matching (SGM, Hirschmüller 2011)
  - More Global Matching (MGM, Facciolo et al. 2015)

Tasks for each correlator:
  - include the addition of built-in regression testing through automated continuous integration across Earth and Planetary datasets
  - publish the software containers for enhanced portability
  - update the documentation of the module as it integrates into the overall ASP workflow
  - a comprehensive performance report comparing the GPU and CPU implementations,
  
In the event we find an operational GPU implementation for these algorithms (e.g. OpenCV libraries),
the team will proceed to assess, benchmark, and adapt existing implementation(s) following the 
established ASP "plug-in" requirements. If these existing implementations do not provide the expected 
accuracy or computational performance, the team will develop the GPU-based acceleration module following 
the currently optimized algorithms in ASP.

This [site](https://stereopipeline.readthedocs.io/en/latest/next_steps.html#stereo-alg-overview) has
the summary of the current correlator algorithms available in ASP.

## Adding new algorithms to ASP

ASP makes it possible for anybody to add their own algorithm to be used for stereo 
correlation without having to recompile ASP itself.

Any such algorithm must be a program to be invoked as:

```bash
myprog <options> left_image.tif right_image.tif \
  output_disparity.tif
```

Here, as often assumed in the computer vision community, left_image.tif and right_image.tif are small image clips with epipolar alignment applied to them, so that the epipolar lines are horizontal and the resulting disparity only need to be searched in the x direction (along each row). The images must have the same size. (ASP will take care of preparing these images.)

The images must be in the TIF format, with pixel values being of the float type, and no-data pixels being set to NaN. The output disparity is expected to satisfy the same assumptions and be of dimensions equal to those of the input images.

The options passed to this program are expected to have no other characters except letters, numbers, space, period, underscore, plus, minus, and equal signs. Each option must have exactly one value, such as:

```bash
-opt1 val1 -opt2 val2 -opt3 val3
```

(More flexible options, including boolean ones, so with no value, may be implemented going forward.)

Such a program, say named myprog, should be copied to the location:

```bash
plugins/stereo/myprog/bin/myprog
```

relative to the ASP top-level directory, with any libraries in:

```bash
plugins/stereo/myprog/lib
```

Then, add a line to the file:

```bash
plugins/stereo/plugin_list.txt
```

in the ASP top-level directory, in the format:

```bash
myprog plugins/stereo/myprog/bin/myprog plugins/stereo/myprog/lib
```

The entries here are the program name (in lowercase), path to the program, and path to any libraries apart from those shipped with ASP (the last entry is optional).

Then, ASP can invoke this program by calling it, for example, as:

```bash
parallel_stereo --alignment-method local_epipolar \
  --stereo-algorithm "myprog <options>"           \
  <images> <cameras> <output prefix>
```

The program will be called for each pair of locally aligned tiles obtained from these input images, with one subdirectory for each such pair of inputs. That subdirectory will also have the output disparity produced by the program. All such disparities will be read back by ASP, blended together, then ASP will continue with the steps of disparity filtering and triangulation.

It may be helpful to visit one of such subdirectories, examine the stereo_corr log file which will show how precisely the program was called, and also look at its input image tiles and output disparity stored there. Note such auxiliary data is removed by default, unless parallel_stereo is called with the option --keep-only unchanged (Section 16.52).

## GPU Implementation

Given the above explanation, we will deliver a container that provides:
  - the NVIDIA drivers
  - the plugin binaries
  - the plugin txt files modified
  - documentation

## Dependencies

### Conda Environment

We might consider enabling the use of conda environments with cudatoolkit
installed as part of this work, together with the asp conda package.
This work is still TBD.

### Container

There are two containers being built from this repository:

- docker://nasanccs/spgpu-baselibs: contains all the backend dependencies
- docker://nasanccs/spgpu: contains the compiled correlators

Downloading the production container with the deployed correlators:

```bash
singularity build --sandbox /lscratch/jacaraba/container/spgpu docker://nasanccs/spgpu:latest
```

Downloading the container for development:

```bash
singularity build --sandbox /lscratch/jacaraba/container/spgpu docker://nasanccs/spgpu-baselibs:latest
```

If using Singularity and you want to shell into the container:

```bash
singularity shell --nv -B $NOBACKUP,/explore/nobackup/people,/explore/nobackup/projects,/panfs/ccds02/nobackup/people /lscratch/jacaraba/container/spgpu
```

## Directory Configurations

The following are example files we will test our workflows with.
Missing lunar and mars data within the examples.

| Dataset    | Explore Cloud Path                                              |
| ---------- | --------------------------------------------------------------- |
| WorldView  | /explore/nobackup/projects/ilab/projects/ASP_GPU/data/worldview |
| HiRISE     | /explore/nobackup/projects/ilab/projects/ASP_GPU/data/hirise    |
| BlackSky   | /explore/nobackup/projects/ilab/projects/ASP_GPU/data/blacksky  |

### WorldView Data

WorldView data, provided by Maxar, refers to high-resolution satellite imagery collected
by the WorldView constellation of satellites. These satellites, particularly WorldView-1,
WorldView-2, WorldView-3, and WorldView-Legion, offer very high-resolution imagery with
diverse spectral capabilities and frequent revisit times.

#### Download the Data

stereo_dirs:

```bash
'/explore/nobackup/projects/ilab/projects/ASP_GPU/data/WV02_20160623_10300100577C7E00_1030010058580000'
'/explore/nobackup/projects/ilab/projects/ASP_GPU/data/WV01_20130825_1020010024E78600_10200100241E6200'
'/explore/nobackup/projects/ilab/projects/ASP_GPU/data/WV03_20160616_104001001EBDB400_104001001E13F600'
```

where:

```bash
disparity_map_regex: 'out-F.tif'
stereo_pair_regex: '*r100_*m.tif'
lowres_dsm_regex: 'out-DEM_24m.tif'
midres_dsm_regex: 'out-DEM_4m.tif'
highres_dsm_regex: 'out-DEM_1m.tif'
```

#### ASP Run

```bash
parallel_stereo -t rpc       \
  --stereo-algorithm asp_mgm \
  --subpixel-mode 9          \
  left.tif right.tif         \
  results/run
```

or

```bash
parallel_stereo -t rpc       \
  --stereo-algorithm asp_mgm \
  --subpixel-mode 9          \
  left.tif right.tif         \
  left.xml right.xml         \
  results/run
```

To only run the disparity calculation portion:

```bash
parallel_stereo               \
  --correlator-mode           \
  --stereo-algorithm asp_mgm  \
  --subpixel-mode 9           \
  run/run-L.tif run/run-R.tif \
  run_corr/run
```

Evaluating the correlation with:

```bash
corr_eval --prefilter-mode 0 --kernel-size 5 5 --metric ncc \
  run/run-L.tif run/run-R.tif run/run-RD.tif run/run
```

#### SPGPU Run

```bash
TBD with the new plugin
```

### Mars Reconnaissance Orbiter HiRISE Data

Example taken from [ASP Documentation](https://stereopipeline.readthedocs.io/en/latest/examples/hirise.html).

HiRISE is one of the most challenging cameras to use when making 3D models because HiRISE exposures can be 
several gigabytes each. Working with this data requires patience as it will take time.

#### Download the Data

```bash
wget -r -l1 -np \
  "http://hirise-pds.lpl.arizona.edu/PDS/EDR/ESP/ORB_029400_029499/ESP_029421_2300/" \
  -A "*RED*IMG"
```

#### ASP Run

The ASP website has details on how to run this portion. Update this section
with those changes.

```bash
TBD
```

#### SPGPU Run

```bash
TBD with the new plugin
```

### BlackSky

Example #1 Location:

```bash
/explore/nobackup/projects/ilab/projects/ASP_GPU/data/blacksky/large_tile_output_for_correlator/asp_local_align-0_0_3193_5807/0_0_3193_5807-left-aligned-tile.tif
/explore/nobackup/projects/ilab/projects/ASP_GPU/data/blacksky/large_tile_output_for_correlator/asp_local_align-0_0_3193_5807/0_0_3193_5807-right-aligned-tile.tif
/explore/nobackup/projects/ilab/projects/ASP_GPU/data/blacksky/large_tile_output_for_correlator/asp_local_align-0_0_3193_5807/0_0_3193_5807-aligned-disparity.tif
```

### GPU Correlators

Compilation Example (from inside the container):

```bash
g++ /explore/nobackup/people/jacaraba/development/stereo-pipeline-gpu-plugin/tests/prototyping/gpu_opencv_bm.cpp -o /explore/nobackup/people/jacaraba/development/stereo-pipeline-gpu-plugin/tests/prototyping/gpu_opencv_bm   -I/usr/local/include/opencv4   -I/usr/include/gdal   -L/usr/local/lib   -lopencv_core   -lopencv_imgproc   -lopencv_highgui   -lopencv_calib3d   -lopencv_cudaimgproc   -lopencv_cudastereo   -lopencv_imgcodecs  -lgdal -DHAVE_OPENCV_CUDA=1
```

Some example execution from initial testing:

```bash
/explore/nobackup/people/jacaraba/development/stereo-pipeline-gpu-plugin/tests/prototyping/gpu_opencv_bm /explore/nobackup/projects/ilab/projects/ASP_GPU/data/blacksky/large_tile_output_for_correlator/asp_local_align-0_0_3193_5807/0_0_3193_5807-left-aligned-tile.tif /explore/nobackup/projects/ilab/projects/ASP_GPU/data/blacksky/large_tile_output_for_correlator/asp_local_align-0_0_3193_5807/0_0_3193_5807-right-aligned-tile.tif

/explore/nobackup/people/jacaraba/development/stereo-pipeline-gpu-plugin/tests/prototyping/gpu_opencv_bm /explore/nobackup/projects/ilab/projects/ASP_GPU/data/blacksky/large_tile_output_for_correlator/asp_local_align-0_0_3193_5807/0_0_3193_5807-left-aligned-tile.tif /explore/nobackup/projects/ilab/projects/ASP_GPU/data/blacksky/large_tile_output_for_correlator/asp_local_align-0_0_3193_5807/0_0_3193_5807-right-aligned-tile.tif \
  64 \
  21 \
  10 \
  31 \
  15 \
  100 \
  32 \
  1
```

Example output:

```bash
Singularity> /explore/nobackup/people/jacaraba/development/stereo-pipeline-gpu-plugin/tests/prototyping/gpu_opencv_bm /explore/nobackup/projects/ilab/projects/ASP_GPU/data/blacksky/large_tile_output_for_correlator/asp_local_align-0_0_3193_5807/0_0_3193_5807-left-aligned-tile.tif /explore/nobackup/projects/ilab/projects/ASP_GPU/data/blacksky/large_tile_output_for_correlator/asp_local_align-0_0_3193_5807/0_0_3193_5807-right-aligned-tile.tif
Loaded rasters with size: 3473 x 5958
Warning: No geotransform found, using identity transform.
CPU StereoBM time: 0.0892171 seconds
Saved GeoTIFF: disparity_cpu.tif
GPU StereoBM time: 0.0241288 seconds
Saved GeoTIFF: disparity_gpu.tif
```

## Analyzing StereoPipeline Correlation Algorithms Implementation

This section is work in progress.

### BlockMatching

The OpenCV blockmatching algorithm is called from [stereo_corr.cc](https://github.com/NeoGeographyToolkit/StereoPipeline/blob/master/src/asp/Tools/stereo_corr.cc). The function [call_opencv_bm_or_sgbm](https://github.com/NeoGeographyToolkit/StereoPipeline/blob/master/src/asp/Core/LocalAlignment.cc) calls OpenCV.

Over the course of this implementation, we created a custom **GPU-based disparity map generator** leveraging **OpenCV’s CUDA StereoBM** functionality. Our goal was to replicate the behavior of NASA’s **Ames Stereo Pipeline (ASP)** `opencv_bm` correlator, which internally uses OpenCV’s CPU implementation, but instead harness GPU acceleration to process large epipolar-aligned image tiles faster.

The workflow involved several steps. First, we built a C++ program capable of ingesting two input rasters (float GeoTIFFs with NaN nodata), normalizing their pixel intensity ranges robustly to 8-bit images suitable for StereoBM, and computing disparities on the GPU. We implemented a careful handling of nodata, ensuring any zero disparities produced by the GPU (which typically indicate invalid matches) were converted to `NaN` in the output. To maintain consistency with ASP’s convention, we also flipped the sign of the disparities to match the negative scale expected in downstream processing.

A key challenge we encountered was that the **raw GPU disparities had significantly smaller magnitudes** compared to the CPU outputs (e.g., -0.2 vs. -6.5 for similar regions). This discrepancy likely arises because the GPU implementation uses different internal normalization and scoring heuristics compared to the CPU version. To bridge this gap, we introduced a **scaling factor**, computed as the ratio between a reference CPU mean disparity and the GPU mean, thereby aligning the value ranges and improving interpretability when blending or mosaicking these outputs with CPU-generated disparity maps.

By combining robust normalization, clear invalid pixel handling, sign flipping, and scaling, we achieved a reproducible GPU disparity workflow that produces outputs similar in character to ASP’s standard opencv\_bm correlator but can run significantly faster on modern NVIDIA GPUs. This approach can be integrated into ASP’s plugin framework for large-scale stereo pipelines requiring GPU acceleration.

Example compilation:

```bash
g++ /explore/nobackup/people/jacaraba/development/stereo-pipeline-gpu-plugin/spgpu/correlator/opencv_bm_gpu.cpp -o /explore/nobackup/people/jacaraba/development/stereo-pipeline-gpu-plugin/spgpu/correlator/opencv_bm_gpu   -I/usr/local/include/opencv4   -I/usr/include/gdal   -L/usr/local/lib   -lopencv_core   -lopencv_imgproc   -lopencv_highgui   -lopencv_calib3d   -lopencv_cudaimgproc   -lopencv_cudastereo   -lopencv_imgcodecs  -lgdal -DHAVE_OPENCV_CUDA=1
```

Example execution:

```bash
/explore/nobackup/people/jacaraba/development/stereo-pipeline-gpu-plugin/spgpu/correlator/opencv_bm_gpu   -num_disp 64   -block_size 21   -texture_thresh 10   -prefilter_cap 31   -uniqueness_ratio 15   -speckle_size 100   -speckle_range 32   -disp12_diff 1   /explore/nobackup/projects/ilab/projects/ASP_GPU/data/blacksky/large_tile_output_for_correlator/asp_local_align-0_0_3193_5807/0_0_3193_5807-left-aligned-tile.tif /explore/nobackup/projects/ilab/projects/ASP_GPU/data/blacksky/large_tile_output_for_correlator/asp_local_align-0_0_3193_5807/0_0_3193_5807-right-aligned-tile.tif disparity.tif
```

### SGM

In this workflow, we implemented a GPU-accelerated stereo disparity estimation pipeline using OpenCV’s CUDA StereoSGM. The process begins by reading the left and right input rasters with GDAL and normalizing them tile by tile. Tiling is critical because it allows processing of large rasters that would otherwise exceed GPU memory limits. Each tile is optionally padded and passed to the StereoSGM algorithm, which computes the local disparity map on the GPU with subpixel precision. To handle overlapping tiles, we aggregate disparities into a running sum and count matrices, which we average after all tiles are processed to produce a smoothly blended output.

A key improvement we introduced is robust nodata masking: prior to stereo matching, we create a mask of valid pixels (i.e., those not containing NaNs) and explicitly track where the input data is invalid. After computing the disparity map, we set disparities to NaN in any pixel where the input was nodata or where the disparity computation itself failed. This ensures that no artificial disparity values “leak” into empty regions—a common problem when using block matching algorithms that cannot distinguish between valid content and uninitialized pixels. Finally, the blended disparity raster is saved as a GeoTIFF with NaNs explicitly marked as nodata, ensuring clean, reliable outputs ready for further analysis or visualization.

Example compilation:

```bash
g++ /explore/nobackup/people/jacaraba/development/stereo-pipeline-gpu-plugin/spgpu/correlator/opencv_sgm_gpu.cpp -o /explore/nobackup/people/jacaraba/development/stereo-pipeline-gpu-plugin/spgpu/correlator/opencv_sgm_gpu   -I/usr/local/include/opencv4   -I/usr/include/gdal   -L/usr/local/lib   -lopencv_core   -lopencv_imgproc   -lopencv_highgui   -lopencv_calib3d   -lopencv_cudaimgproc   -lopencv_cudastereo   -lopencv_imgcodecs  -lgdal -DHAVE_OPENCV_CUDA=1
```

Example run:

```bash
/explore/nobackup/people/jacaraba/development/stereo-pipeline-gpu-plugin/spgpu/correlator/opencv_sgm_gpu -num_disp 128 -mode hh4 -min_disp 0 -p1 10 -p2 120 -uniqueness 1 /explore/nobackup/projects/ilab/projects/ASP_GPU/data/blacksky/large_tile_output_for_correlator/asp_local_align-0_0_3193_5807/0_0_3193_5807-left-aligned-tile.tif /explore/nobackup/projects/ilab/projects/ASP_GPU/data/blacksky/large_tile_output_for_correlator/asp_local_align-0_0_3193_5807/0_0_3193_5807-right-aligned-tile.tif disparity_sgm.tif
```

### StereoBeliefPropagation

Example compilation:

```bash
g++ /explore/nobackup/people/jacaraba/development/stereo-pipeline-gpu-plugin/spgpu/correlator/opencv_bp_gpu.cpp -o /explore/nobackup/people/jacaraba/development/stereo-pipeline-gpu-plugin/spgpu/correlator/opencv_bp_gpu   -I/usr/local/include/opencv4   -I/usr/include/gdal   -L/usr/local/lib   -lopencv_core   -lopencv_imgproc   -lopencv_highgui   -lopencv_calib3d   -lopencv_cudaimgproc   -lopencv_cudastereo   -lopencv_imgcodecs  -lgdal -DHAVE_OPENCV_CUDA=1
```

Example run:

```bash
/explore/nobackup/people/jacaraba/development/stereo-pipeline-gpu-plugin/spgpu/correlator/opencv_bp_gpu -num_disp 128 -iters 5 -levels 4 /explore/nobackup/projects/ilab/projects/ASP_GPU/data/blacksky/large_tile_output_for_correlator/asp_local_align-0_0_3193_5807/0_0_3193_5807-left-aligned-tile.tif /explore/nobackup/projects/ilab/projects/ASP_GPU/data/blacksky/large_tile_output_for_correlator/asp_local_align-0_0_3193_5807/0_0_3193_5807-right-aligned-tile.tif disparity_bp.tif
```

### StereoConstantSpaceBP

Example compilation:

```bash
g++ /explore/nobackup/people/jacaraba/development/stereo-pipeline-gpu-plugin/spgpu/correlator/opencv_csbp_gpu.cpp -o /explore/nobackup/people/jacaraba/development/stereo-pipeline-gpu-plugin/spgpu/correlator/opencv_csbp_gpu   -I/usr/local/include/opencv4   -I/usr/include/gdal   -L/usr/local/lib   -lopencv_core   -lopencv_imgproc   -lopencv_highgui   -lopencv_calib3d   -lopencv_cudaimgproc   -lopencv_cudastereo   -lopencv_imgcodecs  -lgdal -DHAVE_OPENCV_CUDA=1
```

Example run:

```bash
/explore/nobackup/people/jacaraba/development/stereo-pipeline-gpu-plugin/spgpu/correlator/opencv_csbp_gpu -num_disp 128 -iters 5 /explore/nobackup/projects/ilab/projects/ASP_GPU/data/blacksky/large_tile_output_for_correlator/asp_local_align-0_0_3193_5807/0_0_3193_5807-left-aligned-tile.tif /explore/nobackup/projects/ilab/projects/ASP_GPU/data/blacksky/large_tile_output_for_correlator/asp_local_align-0_0_3193_5807/0_0_3193_5807-right-aligned-tile.tif disparity_csbp.tif
```

## Running GPU Correlators as Plugins

### CPU Version

```bash
/opt/StereoPipeline/bin/stereo_corr --correlator-mode --stereo-algorithm "opencv_bm_gpu -block_size 21 -texture_thresh 10 -prefilter_cap 31 -uniqueness_ratio 15 -speckle_size 100 -speckle_range 32 -disp12_diff 1" --alignment-method local_epipolar /explore/nobackup/projects/ilab/projects/ASP_GPU/data/blacksky/large_tile_output_for_correlator/asp_local_align-0_0_3193_5807/0_0_3193_5807-left-aligned-tile.tif /explore/nobackup/projects/ilab/projects/ASP_GPU/data/blacksky/large_tile_output_for_correlator/asp_local_align-0_0_3193_5807/0_0_3193_5807-right-aligned-tile.tif  disparity_bm_cpu --skip-low-res-disparity-comp --corr-seed-mode 1 --sgm-collar-size 0 --corr-tile-size 5807 --threads 1 --trans-crop-win 0 0 3193 5807
```

### GPU Version

```bash
/opt/StereoPipeline/bin/stereo_corr -t rpc --stereo-algorithm "opencv_bm_gpu -block_size 21 -texture_thresh 10 -prefilter_cap 31 -uniqueness_
ratio 15 -speckle_size 100 -speckle_range 32 -disp12_diff 1" --alignment-method local_epipolar /explore/nobackup/projects/ilab/projects/ASP_GPU/data/blacksky/large_tile_output_for_correlator/asp_local_align-0_0_3193_5807/0_0_3193_5807-left-aligned-tile.tif /explore/nobackup/projects/ilab/projects/ASP_GPU/data/blacksky/large_tile_output_for_correlator/asp_local_align-0_0_3193_5807/0_0_3193_5807-right-aligned-tile.tif  disparity_bm_gpu.tif --skip-low-res-disparity-comp --corr-seed-mode 1 --sgm-collar-size 0 --corr-tile-size 5807 --threads 1 --trans-crop-win 0 0 3193 5807
```