#include <opencv2/opencv.hpp>
#include <opencv2/cudastereo.hpp>
using namespace cv;

int main() {
    Mat left(512, 512, CV_8U, Scalar(128));
    Mat right(512, 512, CV_8U, Scalar(128));
    cuda::GpuMat d_left(left), d_right(right), d_disp;

    Ptr<cuda::StereoSGM> sgm = cuda::createStereoSGM(0, 128, 10, 120, 1, cuda::StereoSGM::MODE_HH4);
    sgm->compute(d_left, d_right, d_disp);

    Mat h_disp;
    d_disp.download(h_disp);
    std::cout << "Done, disparity type: " << h_disp.type() << std::endl;
}


//g++ /explore/nobackup/people/jacaraba/development/stereo-pipeline-gpu-plugin/tests/system/check_opencv_sgm.cpp -o /explore/nobackup/people/jacaraba/development/stereo-pipeline-gpu-plugin/tests/system/check_opencv_sgm   -I/usr/local/include/opencv4   -I/usr/include/gdal   -L/usr/local/lib   -lopencv_core   -lopencv_imgproc   -lopencv_highgui   -lopencv_calib3d   -lopencv_cudaimgproc   -lopencv_cudastereo   -lopencv_imgcodecs  -lgdal -DHAVE_OPENCV_CUDA=1