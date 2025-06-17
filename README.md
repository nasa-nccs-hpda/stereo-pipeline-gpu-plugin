# Stereo Pipeline GPU Plugin

Stereo Pipeline GPU Plugin implementation.

## Directory Configurations

The following are example files we will test our workflows with.
Missing lunar and mars data within the examples.

stereo_dirs:

  - '/explore/nobackup/projects/ilab/projects/ASP_GPU/data/WV02_20160623_10300100577C7E00_1030010058580000'
  - '/explore/nobackup/projects/ilab/projects/ASP_GPU/data/WV01_20130825_1020010024E78600_10200100241E6200'
  - '/explore/nobackup/projects/ilab/projects/ASP_GPU/data/WV03_20160616_104001001EBDB400_104001001E13F600'

where:

  - disparity_map_regex: 'out-F.tif'
  - stereo_pair_regex: '*r100_*m.tif'
  - lowres_dsm_regex: 'out-DEM_24m.tif'
  - midres_dsm_regex: 'out-DEM_4m.tif'
  - highres_dsm_regex: 'out-DEM_1m.tif'

## Example ASP Run

## Example GPU Run

## Benchmark Numbers
