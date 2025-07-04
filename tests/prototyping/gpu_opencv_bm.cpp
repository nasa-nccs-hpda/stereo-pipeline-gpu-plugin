#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudastereo.hpp>
#include <gdal_priv.h>
#include <cpl_conv.h>
#include <iostream>
#include <limits>

using namespace std;
using namespace cv;

void SaveGeoTIFF(const string &filename, const Mat &data, const char *projection, double *geotransform)
{
    GDALAllRegister();
    const char *format = "GTiff";
    GDALDriver *driver = GetGDALDriverManager()->GetDriverByName(format);
    if (!driver)
    {
        cerr << "Could not get GDAL GTiff driver." << endl;
        return;
    }

    GDALDataset *output = driver->Create(filename.c_str(), data.cols, data.rows, 1, GDT_Float32, nullptr);
    if (!output)
    {
        cerr << "Could not create output dataset." << endl;
        return;
    }

    output->SetProjection(projection);
    output->SetGeoTransform(geotransform);

    output->GetRasterBand(1)->RasterIO(GF_Write, 0, 0, data.cols, data.rows,
                                       (void *)data.ptr<float>(), data.cols, data.rows, GDT_Float32, 0, 0);

    GDALClose(output);
    cout << "Saved GeoTIFF: " << filename << endl;
}

int main(int argc, char **argv)
{
    if (argc < 9)
    {
        cout << "Usage: " << argv[0] << " left.tif right.tif num_disp block_size texture_thresh prefilter_cap uniqueness_ratio speckle_size speckle_range disp12_diff" << endl;
        return -1;
    }

    string left_path = argv[1];
    string right_path = argv[2];

    int num_disp = stoi(argv[3]);
    int block_size = stoi(argv[4]);
    int texture_thresh = stoi(argv[5]);
    int prefilter_cap = stoi(argv[6]);
    int uniqueness_ratio = stoi(argv[7]);
    int speckle_size = stoi(argv[8]);
    int speckle_range = stoi(argv[9]);
    int disp12_diff = stoi(argv[10]);

    GDALAllRegister();

    GDALDataset *left_ds = (GDALDataset *)GDALOpen(left_path.c_str(), GA_ReadOnly);
    GDALDataset *right_ds = (GDALDataset *)GDALOpen(right_path.c_str(), GA_ReadOnly);
    if (!left_ds || !right_ds)
    {
        cerr << "Could not open input rasters." << endl;
        return -1;
    }

    int width = left_ds->GetRasterXSize();
    int height = left_ds->GetRasterYSize();
    cout << "Loaded rasters with size: " << width << " x " << height << endl;

    Mat tmp_left(height, width, CV_32F);
    Mat tmp_right(height, width, CV_32F);

    GDALRasterBand *left_band = left_ds->GetRasterBand(1);
    GDALRasterBand *right_band = right_ds->GetRasterBand(1);

    left_band->RasterIO(GF_Read, 0, 0, width, height, tmp_left.ptr(), width, height, GDT_Float32, 0, 0);
    right_band->RasterIO(GF_Read, 0, 0, width, height, tmp_right.ptr(), width, height, GDT_Float32, 0, 0);

    // NaN mask
    Mat nan_mask_left = tmp_left != tmp_left;
    Mat nan_mask_right = tmp_right != tmp_right;
    tmp_left.setTo(0, nan_mask_left);
    tmp_right.setTo(0, nan_mask_right);

    double minL, maxL, minR, maxR;
    minMaxLoc(tmp_left, &minL, &maxL);
    minMaxLoc(tmp_right, &minR, &maxR);

    cout << "Left image valid range: " << minL << " to " << maxL << endl;
    cout << "Right image valid range: " << minR << " to " << maxR << endl;

    // Histogram stretch to 0-255
    Mat left8u, right8u;
    tmp_left.convertTo(left8u, CV_8U, 255.0 / (maxL - minL), -255.0 * minL / (maxL - minL));
    tmp_right.convertTo(right8u, CV_8U, 255.0 / (maxR - minR), -255.0 * minR / (maxR - minR));

    minMaxLoc(left8u, &minL, &maxL);
    minMaxLoc(right8u, &minR, &maxR);
    cout << "Final left normalized range: " << minL << " to " << maxL << endl;
    cout << "Final right normalized range: " << minR << " to " << maxR << endl;

    // Save debug inputs
    imwrite("left_normalized.png", left8u);
    imwrite("right_normalized.png", right8u);

    // CPU StereoBM
    Ptr<StereoBM> stereo_cpu = StereoBM::create(num_disp, block_size);
    stereo_cpu->setTextureThreshold(texture_thresh);
    stereo_cpu->setPreFilterCap(prefilter_cap);
    stereo_cpu->setUniquenessRatio(uniqueness_ratio);
    stereo_cpu->setSpeckleWindowSize(speckle_size);
    stereo_cpu->setSpeckleRange(speckle_range);
    stereo_cpu->setDisp12MaxDiff(disp12_diff);

    auto t1 = chrono::high_resolution_clock::now();
    Mat disp_cpu;
    stereo_cpu->compute(left8u, right8u, disp_cpu);
    auto t2 = chrono::high_resolution_clock::now();
    cout << "CPU StereoBM time: " << chrono::duration<double>(t2 - t1).count() << " seconds" << endl;

    Mat disp_cpu_f;
    disp_cpu.convertTo(disp_cpu_f, CV_32F, 1.0 / 16.0);
    disp_cpu_f.setTo(numeric_limits<float>::quiet_NaN(), disp_cpu_f < 0);

    SaveGeoTIFF("disparity_cpu.tif", disp_cpu_f, left_ds->GetProjectionRef(), left_ds->GetGeoTransform());

#ifdef HAVE_OPENCV_CUDA
    // GPU StereoBM
    cuda::GpuMat d_left(left8u);
    cuda::GpuMat d_right(right8u);
    cuda::GpuMat d_disp;

    Ptr<cuda::StereoBM> stereo_gpu = cuda::createStereoBM(num_disp, block_size);
    stereo_gpu->setTextureThreshold(texture_thresh);
    stereo_gpu->setPreFilterCap(prefilter_cap);
    stereo_gpu->setUniquenessRatio(uniqueness_ratio);
    stereo_gpu->setSpeckleWindowSize(speckle_size);
    stereo_gpu->setSpeckleRange(speckle_range);
    stereo_gpu->setDisp12MaxDiff(disp12_diff);

    t1 = chrono::high_resolution_clock::now();
    stereo_gpu->compute(d_left, d_right, d_disp);
    t2 = chrono::high_resolution_clock::now();
    cout << "GPU StereoBM time: " << chrono::duration<double>(t2 - t1).count() << " seconds" << endl;

    Mat disp_gpu;
    d_disp.download(disp_gpu);
    Mat disp_gpu_f;
    disp_gpu.convertTo(disp_gpu_f, CV_32F, 1.0 / 16.0);
    disp_gpu_f.setTo(numeric_limits<float>::quiet_NaN(), disp_gpu_f < 0);

    SaveGeoTIFF("disparity_gpu.tif", disp_gpu_f, left_ds->GetProjectionRef(), left_ds->GetGeoTransform());
#else
    cout << "OpenCV CUDA not available. Skipping GPU StereoBM." << endl;
#endif

    GDALClose(left_ds);
    GDALClose(right_ds);
    return 0;
}
