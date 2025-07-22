// opencv_bm_gpu_functions.h
#ifndef OPENCV_BM_GPU_FUNCTIONS_H
#define OPENCV_BM_GPU_FUNCTIONS_H

#include <opencv2/opencv.hpp>
#include <string>

// Declare functions from opencv_bm_gpu.cpp that we want to test
cv::Mat robustNormalize(const cv::Mat& src);
void SaveGeoTIFF(const std::string& filename, const cv::Mat& data);

#endif // OPENCV_BM_GPU_FUNCTIONS_H
