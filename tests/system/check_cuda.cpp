#include <opencv2/core/cuda.hpp>
#include <iostream>

int main() {
    int n = cv::cuda::getCudaEnabledDeviceCount();
    std::cout << "CUDA devices detected: " << n << std::endl;
    return 0;
}
