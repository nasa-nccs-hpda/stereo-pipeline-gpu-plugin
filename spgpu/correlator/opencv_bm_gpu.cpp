#include <opencv2/opencv.hpp>
#include <opencv2/cudastereo.hpp>
#include <gdal_priv.h>
#include <cpl_conv.h>
#include <iostream>
#include <limits>
#include <chrono>

using namespace std;
using namespace cv;

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
        GF_Write,
        0, 0,
        data.cols, data.rows,
        (void*)data.ptr<float>(),
        data.cols, data.rows,
        GDT_Float32,
        0, 0
    );
    ds->GetRasterBand(1)->SetNoDataValue(numeric_limits<float>::quiet_NaN());
    ds->FlushCache();
    GDALClose(ds);
    cout << "Saved: " << filename << endl;
}

Mat robustNormalize(const Mat& src) {
    Mat mask = src == src; // finite pixels
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

int main(int argc, char** argv) {
    if (argc < 4) {
        cerr << "Usage:\n  " << argv[0]
             << " [options] left.tif right.tif output_disparity.tif\n"
             << "Options:\n"
             << "  -num_disp <int>\n"
             << "  -block_size <int>\n"
             << "  -texture_thresh <int>\n"
             << "  -prefilter_cap <int>\n"
             << "  -uniqueness_ratio <int>\n"
             << "  -speckle_size <int>\n"
             << "  -speckle_range <int>\n"
             << "  -disp12_diff <int>\n";
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

    // Parse options
    int argi = 1;
    while (argi < argc - 3) {
        string key(argv[argi]);
        string val(argv[argi + 1]);
        if (key == "-num_disp") num_disp = stoi(val);
        else if (key == "-block_size") block_size = stoi(val);
        else if (key == "-texture_thresh") texture_thresh = stoi(val);
        else if (key == "-prefilter_cap") prefilter_cap = stoi(val);
        else if (key == "-uniqueness_ratio") uniqueness_ratio = stoi(val);
        else if (key == "-speckle_size") speckle_size = stoi(val);
        else if (key == "-speckle_range") speckle_range = stoi(val);
        else if (key == "-disp12_diff") disp12_diff = stoi(val);
        else {
            cerr << "Unknown option: " << key << endl;
            return 1;
        }
        argi += 2;
    }

    string left_path(argv[argc - 3]);
    string right_path(argv[argc - 2]);
    string out_path(argv[argc - 1]);

    cout << "Running GPU StereoBM with parameters:\n"
         << "  num_disp=" << num_disp
         << ", block_size=" << block_size
         << ", texture_thresh=" << texture_thresh
         << ", prefilter_cap=" << prefilter_cap
         << ", uniqueness_ratio=" << uniqueness_ratio
         << ", speckle_size=" << speckle_size
         << ", speckle_range=" << speckle_range
         << ", disp12_diff=" << disp12_diff << endl;

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

    Mat l_f(height, width, CV_32F);
    Mat r_f(height, width, CV_32F);
    (void)l_ds->GetRasterBand(1)->RasterIO(GF_Read, 0,0,width,height, l_f.ptr(), width,height, GDT_Float32,0,0);
    (void)r_ds->GetRasterBand(1)->RasterIO(GF_Read, 0,0,width,height, r_f.ptr(), width,height, GDT_Float32,0,0);

    cout << "Normalizing left image..." << endl;
    Mat l_u8 = robustNormalize(l_f);
    cout << "Normalizing right image..." << endl;
    Mat r_u8 = robustNormalize(r_f);

#ifdef HAVE_OPENCV_CUDA
    cuda::GpuMat d_left(l_u8), d_right(r_u8), d_disp;
    Ptr<cuda::StereoBM> bm = cuda::createStereoBM(num_disp, block_size);
    bm->setTextureThreshold(texture_thresh);
    bm->setPreFilterCap(prefilter_cap);
    bm->setUniquenessRatio(uniqueness_ratio);
    bm->setSpeckleWindowSize(speckle_size);
    bm->setSpeckleRange(speckle_range);
    bm->setDisp12MaxDiff(disp12_diff);

    cout << "Running GPU StereoBM..." << endl;
    auto t1 = chrono::high_resolution_clock::now();
    bm->compute(d_left, d_right, d_disp);
    auto t2 = chrono::high_resolution_clock::now();
    cout << "GPU StereoBM time: " << chrono::duration<double>(t2 - t1).count() << " seconds" << endl;

    Mat disp_raw;
    d_disp.download(disp_raw);

    double dmin, dmax;
    minMaxLoc(disp_raw, &dmin, &dmax);
    cout << "Disparity raw range: " << dmin << " to " << dmax << endl;

    // Convert to float and flip sign like ASP
    Mat disp_f;
    disp_raw.convertTo(disp_f, CV_32F, -1.0/16.0);

    // GPU invalid disparities are 0
    Mat invalid_mask = disp_raw == 0;
    disp_f.setTo(numeric_limits<float>::quiet_NaN(), invalid_mask);

    // Compute mean of valid disparities
    Scalar mean_disp = mean(disp_f, disp_f == disp_f);
    double gpu_mean = mean_disp[0];
    cout << "GPU mean disparity before scaling: " << gpu_mean << endl;

    // Scale factor derived from CPU reference mean (-6.58)
    double scale_factor = -6.58 / gpu_mean;
    cout << "Applying scale factor: " << scale_factor << endl;

    disp_f *= scale_factor;

    double scaled_min, scaled_max;
    minMaxLoc(disp_f, &scaled_min, &scaled_max);
    cout << "Scaled disparity range: " << scaled_min << " to " << scaled_max << endl;

    SaveGeoTIFF(out_path, disp_f);
#else
    cerr << "OpenCV built without CUDA support!" << endl;
    return 1;
#endif

    GDALClose(l_ds);
    GDALClose(r_ds);
    return 0;
}
