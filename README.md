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

## Dependencies

### Conda Environment

```bash
```

### Container

```bash
```

## Directory Configurations

The following are example files we will test our workflows with.
Missing lunar and mars data within the examples.

### WorldView Data

WorldView data, provided by Maxar, refers to high-resolution satellite imagery collected
by the WorldView constellation of satellites. These satellites, particularly WorldView-1,
WorldView-2, WorldView-3, and WorldView-Legion, offer very high-resolution imagery with
diverse spectral capabilities and frequent revisit times.

#### Download the Data

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

#### ASP Run

```bash
```

#### SPGPU Run

```bash
```

### Mars Data

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

```bash
```

#### SPGPU Run

```bash
```

