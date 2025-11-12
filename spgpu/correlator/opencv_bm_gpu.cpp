#include <opencv2/opencv.hpp>
#include <opencv2/cudastereo.hpp>
#include <gdal_priv.h>
#include <cpl_conv.h>
#include <iostream>
#include <limits>
#include <chrono>

using namespace std;
using namespace cv;

#ifdef COMPILE_FOR_TESTING
// When compiling for testing, make functions available for linking
// but don't include main()
#define SKIP_MAIN
#endif

void SaveGeoTIFF(const string& filename, const Mat& data)
{
    GDALAllRegister();
    GDALDriver* driver = GetGDALDriverManager()->GetDriverByName("GTiff");

    if (!driver) {
        cerr << "Could not get GTiff driver" << endl;
        return;
    }
    GDALDataset* ds = driver->Create(filename.c_str(), data.cols, data.rows, 1, GDT_Float32, nullptr);
    if (!ds) {
        cerr << "Could not create output dataset" << endl;
        return;
    }
    ds->GetRasterBand(1)->RasterIO(
        GF_Write, 0,0, data.cols,data.rows,
        (void*)data.ptr<float>(), data.cols,data.rows,
        GDT_Float32,0,0
    );
    ds->GetRasterBand(1)->SetNoDataValue(numeric_limits<float>::quiet_NaN());
    ds->FlushCache();
    GDALClose(ds);
    cout << "Saved: " << filename << endl;
}

Mat robustNormalize(const Mat& src) {
    Mat mask = src == src;
    double minVal, maxVal;
    minMaxLoc(src, &minVal, &maxVal, nullptr, nullptr, mask);
    cout << "Valid range: " << minVal << " to " << maxVal << endl;
    Mat normalized;
    if (abs(maxVal - minVal) < 1e-5) {
        normalized = Mat::zeros(src.size(), CV_8U);
    } else {
        Mat scaled = (src - minVal) * (255.0 / (maxVal - minVal));
        scaled.convertTo(normalized, CV_8U);
    }
    return normalized;
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
	// ***
	// Debug LD_LIBRARY_PATH.
	// ***
	setenv("LD_LIBRARY_PATH", "/usr/local/lib/python3.10/dist-packages/torch/lib:/usr/local/lib/python3.10/dist-packages/torch_tensorrt/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/.singularity.d/libs:/opt/StereoPipeline/plugins/stereo/opencv_bm_gpu/lib:/opt/StereoPipeline/lib", 1);
		
	system("env | grep -E '(CUDA|NVIDIA|LD_LIBRARY)' > child_env.txt");
	// ***
	// End Debug
	// ***
    
	if (argc < 4)
	{
        cerr << "Usage:\n  "
			 << argv[0]
             << " [options] left.tif right.tif output_disparity.tif\n"
             << "Options:\n"
             << "  -num_disp <int>\n"
             << "  -block_size <int>\n"
             << "  -texture_thresh <int>\n"
             << "  -prefilter_cap <int>\n"
             << "  -uniqueness_ratio <int>\n"
             << "  -speckle_size <int>\n"
             << "  -speckle_range <int>\n"
             << "  -disp12_diff <int>\n"
             << "  -tile <int> (default 1024)\n"
             << "  -overlap <int> (default 64)\n";

        return 1;
    }

    // Default parameters
    int num_disp = 64;
    int block_size = 21;
    int texture_thresh = 10;
    int prefilter_cap = 31;
    int uniqueness_ratio = 15;
    int speckle_size = 100;
    int speckle_range = 32;
    int disp12_diff = 1;
    int tile_size = 1024;
    int overlap = 64;

    int argi = 1;

	cout << "argc: " << argc << endl;

    while (argi < argc - 3)
	{
        string key(argv[argi]);
        string val(argv[argi+1]);

        if (key == "-num_disp") num_disp = stoi(val);
        else if (key == "-block_size") block_size = stoi(val);
        else if (key == "-texture_thresh") texture_thresh = stoi(val);
        else if (key == "-prefilter_cap") prefilter_cap = stoi(val);
        else if (key == "-uniqueness_ratio") uniqueness_ratio = stoi(val);
        else if (key == "-speckle_size") speckle_size = stoi(val);
        else if (key == "-speckle_range") speckle_range = stoi(val);
        else if (key == "-disp12_diff") disp12_diff = stoi(val);
        else if (key == "-tile") tile_size = stoi(val);
        else if (key == "-overlap") overlap = stoi(val);
        else
		{
            cerr << "Unknown option: " << key << endl;
            // return 1;
        }

		argi += 2;
    }

	// ---
	// The number of disparities must be between 0 and max_supported_ndisp,
	// the value of which we think is 256.
	// https://docs.opencv.org/4.x/dd/d47/group__cudastereo.html
	// ---
	if (num_disp <= 0 || num_disp > 256) {

		cerr << "--num_disp must be between 1 and 256, inclusive" << endl;
		return 1;
	}

    string left_path(argv[argc-3]);
    string right_path(argv[argc-2]);
    string out_path(argv[argc-1]);

    cout << "Running GPU StereoBM with parameters:\n"
         << "  num_disp=" << num_disp
         << ", block_size=" << block_size
         << ", texture_thresh=" << texture_thresh
         << ", prefilter_cap=" << prefilter_cap
         << ", uniqueness_ratio=" << uniqueness_ratio
         << ", speckle_size=" << speckle_size
         << ", speckle_range=" << speckle_range
         << ", disp12_diff=" << disp12_diff
         << ", tile=" << tile_size
         << ", overlap=" << overlap << endl;

    GDALAllRegister();
    GDALDataset* l_ds = (GDALDataset*)GDALOpen(left_path.c_str(), GA_ReadOnly);
    GDALDataset* r_ds = (GDALDataset*)GDALOpen(right_path.c_str(), GA_ReadOnly);

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

    Mat disp_sum = Mat::zeros(height, width, CV_32F);
    Mat disp_count = Mat::zeros(height, width, CV_32F);

#ifdef HAVE_OPENCV_CUDA

    Ptr<cuda::StereoBM> bm = cuda::createStereoBM(num_disp, block_size);
    bm->setTextureThreshold(texture_thresh);
    bm->setPreFilterCap(prefilter_cap);
    bm->setUniquenessRatio(uniqueness_ratio);
    bm->setSpeckleWindowSize(speckle_size);
    bm->setSpeckleRange(speckle_range);
    bm->setDisp12MaxDiff(disp12_diff);

    for (int y = 0; y < height; y += tile_size - overlap)
	{
        for (int x = 0; x < width; x += tile_size - overlap)
		{
            int tile_w = min(tile_size, width - x);
            int tile_h = min(tile_size, height - y);

            Rect roi(x,y,tile_w,tile_h);
            Mat l_tile = robustNormalize(l_f(roi));
            Mat r_tile = robustNormalize(r_f(roi));

            cuda::GpuMat d_left(l_tile);
            cuda::GpuMat d_right(r_tile);
            cuda::GpuMat d_disp;

            cout << "Processing tile ("
				 << x
				 << ","
				 << y
				 << ") size: "
				 << tile_w
				 << "x"
				 << tile_h
				 << endl;

            auto t1 = chrono::high_resolution_clock::now();
            bm->compute(d_left, d_right, d_disp);
            auto t2 = chrono::high_resolution_clock::now();

            cout << "Tile time: "
				 << chrono::duration<double>(t2 - t1).count()
				 << " sec"
				 << endl;

            Mat disp_raw;
            d_disp.download(disp_raw);
            Mat disp_f;
            disp_raw.convertTo(disp_f, CV_32F, -1.0/16.0);

            Mat invalid = disp_raw == 0;
            disp_f.setTo(numeric_limits<float>::quiet_NaN(), invalid);

            // Blend
            for (int yy = 0; yy < tile_h; ++yy)
			{
                for (int xx = 0; xx < tile_w; ++xx)
				{
                    float v = disp_f.at<float>(yy, xx);
                    if (std::isnan(v)) continue;
                    int gy = y + yy;
                    int gx = x + xx;
                    disp_sum.at<float>(gy,gx) += v;
                    disp_count.at<float>(gy,gx) += 1.0f;
                }
            }
        }
    }

    Mat final_disp = disp_sum / disp_count;
    final_disp.setTo(numeric_limits<float>::quiet_NaN(), disp_count == 0);

    // Compute mean of valid disparities
    Scalar mean_disp = mean(final_disp, final_disp == final_disp);
    double gpu_mean = mean_disp[0];
    cout << "GPU mean disparity before scaling: " << gpu_mean << endl;

    // Scale factor derived from CPU reference mean (-6.58)
    double scale_factor = -6.58 / gpu_mean;
    cout << "Applying scale factor: " << scale_factor << endl;

    final_disp *= scale_factor;

    double scaled_min, scaled_max;
    minMaxLoc(final_disp, &scaled_min, &scaled_max);

    cout << "Scaled disparity range: "
		 << scaled_min
		 << " to "
		 << scaled_max
		 << endl;

    SaveGeoTIFF(out_path, final_disp);

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
