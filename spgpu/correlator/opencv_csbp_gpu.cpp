#include <opencv2/opencv.hpp>
#include <opencv2/cudastereo.hpp>
#include <gdal_priv.h>
#include <cpl_conv.h>
#include <iostream>
#include <limits>
#include <chrono>
#include <vector>

using namespace std;
using namespace cv;

#ifdef COMPILE_FOR_TESTING
// When compiling for testing, make functions available for linking
// but don't include main()
#define SKIP_MAIN
#endif

void SaveGeoTIFF(const string& filename, const Mat& data) {
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
        GF_Write, 0, 0, data.cols, data.rows,
        (void*)data.ptr<float>(), data.cols, data.rows,
        GDT_Float32, 0, 0
    );
    ds->GetRasterBand(1)->SetNoDataValue(numeric_limits<float>::quiet_NaN());
    ds->FlushCache();
    GDALClose(ds);
    cout << "Saved: " << filename << endl;
}

Mat robustNormalize(const Mat& src, const Mat& mask) {
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
    if (argc < 4) {
        cerr << "Usage:\n"
             << argv[0]
             << " [options] left.tif right.tif output_disparity.tif\n\n"
             << "Options:\n"
             << "  -num_disp <int>          (default: 128)\n"
             << "  -iters <int>             (default: 5)\n";
        return 1;
    }

    // Parameters
    int num_disp = 128;
    int iters = 5;
    int tileSize = 1024;
    int overlap = 64;

    // Parse options
    int argi = 1;
    while (argi < argc - 3) {
        string key(argv[argi]);
        if (key == "-num_disp") num_disp = stoi(argv[++argi]);
        else if (key == "-iters") iters = stoi(argv[++argi]);
        else { cerr << "Unknown option: " << key << endl; return 1; }
        argi++;
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
	
    string left_path(argv[argc - 3]);
    string right_path(argv[argc - 2]);
    string out_path(argv[argc - 1]);

    cout << "Running GPU StereoConstantSpaceBP with parameters:\n"
         << "  num_disp=" << num_disp
         << ", iters=" << iters << endl;

    GDALAllRegister();
    GDALDataset* l_ds = (GDALDataset*)GDALOpen(left_path.c_str(), GA_ReadOnly);
    GDALDataset* r_ds = (GDALDataset*)GDALOpen(right_path.c_str(), GA_ReadOnly);
    if (!l_ds || !r_ds) {
        cerr << "Failed to open input images." << endl;
        return 1;
    }

    int width = l_ds->GetRasterXSize();
    int height = l_ds->GetRasterYSize();
    cout << "Raster size: " << width << " x " << height << endl;

    Mat l_f(height, width, CV_32F), r_f(height, width, CV_32F);
    l_ds->GetRasterBand(1)->RasterIO(GF_Read, 0,0,width,height, l_f.ptr(), width,height, GDT_Float32,0,0);
    r_ds->GetRasterBand(1)->RasterIO(GF_Read, 0,0,width,height, r_f.ptr(), width,height, GDT_Float32,0,0);

    Mat disp_sum = Mat::zeros(height, width, CV_32F);
    Mat disp_count = Mat::zeros(height, width, CV_32F);

#ifdef HAVE_OPENCV_CUDA
    Ptr<cuda::StereoConstantSpaceBP> csbp = cuda::createStereoConstantSpaceBP(num_disp, iters);

    for (int y = 0; y < height; y += tileSize - overlap) {
        for (int x = 0; x < width; x += tileSize - overlap) {
            int tileWidth = min(tileSize, width - x);
            int tileHeight = min(tileSize, height - y);

            Rect roi(x, y, tileWidth, tileHeight);
            Mat l_tile_f = l_f(roi);
            Mat r_tile_f = r_f(roi);

            Mat mask_l = (l_tile_f == l_tile_f);
            Mat mask_r = (r_tile_f == r_tile_f);
            Mat valid_mask = mask_l & mask_r;
            int valid_pixels = countNonZero(valid_mask);

            if (valid_pixels < 100) {
                cout << "Skipping tile (" << x << "," << y << ") due to insufficient valid pixels." << endl;
                continue;
            }

            Mat l_tile = robustNormalize(l_tile_f, valid_mask);
            Mat r_tile = robustNormalize(r_tile_f, valid_mask);

            cuda::GpuMat d_left(l_tile), d_right(r_tile), d_disp;

            cout << "Processing tile (" << x << "," << y << ") size: " << tileWidth << "x" << tileHeight << endl;
            auto t1 = chrono::high_resolution_clock::now();
            csbp->compute(d_left, d_right, d_disp);
            auto t2 = chrono::high_resolution_clock::now();
            cout << "Tile time: " << chrono::duration<double>(t2 - t1).count() << " sec" << endl;

            Mat disp_raw;
            d_disp.download(disp_raw);
            Mat disp_f;
            disp_raw.convertTo(disp_f, CV_32F, -1.0 / 16.0);

            Mat invalid = disp_raw == 0;
            disp_f.setTo(numeric_limits<float>::quiet_NaN(), invalid);

            disp_f.setTo(numeric_limits<float>::quiet_NaN(), ~valid_mask);

            for (int yy = 0; yy < tileHeight; ++yy) {
                for (int xx = 0; xx < tileWidth; ++xx) {
                    float val = disp_f.at<float>(yy, xx);
                    if (std::isnan(val)) continue;
                    int global_y = y + yy;
                    int global_x = x + xx;
                    disp_sum.at<float>(global_y, global_x) += val;
                    disp_count.at<float>(global_y, global_x) += 1.0f;
                }
            }
        }
    }

    Mat final_disp = disp_sum / disp_count;
    final_disp.setTo(numeric_limits<float>::quiet_NaN(), disp_count == 0);
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
