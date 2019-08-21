#include <iostream>
#include <glow/glbase.h>
#include <glow/glutil.h>

#include <glow/GlBuffer.h>
#include <glow/GlFramebuffer.h>
#include <glow/GlProgram.h>
#include <glow/GlVertexArray.h>
#include <glow/ScopedBinder.h>

#include <algorithm>
#include <random>
#include <vector>
#include <opencv2/opencv.hpp>
#include <glow/GlSampler.h>

#include <glow/GlCapabilities.h>
#include <glow/GlState.h>
#include <glow/util/X11OffscreenContext.h>

#include "timer.h"
using namespace glow;

//
cv::Mat3b rgb_method0(const cv::Mat& image ) {
    uint32_t width = image.cols, height = image.rows;

    std::vector<float> values(4 * width * height, 0);
    for (auto i = 0; i < image.rows; i++)
        for (auto j = 0; j < image.cols; j++) {
            float r = image.at<cv::Vec3b>(i, j)[0];
            float g = image.at<cv::Vec3b>(i, j)[1];
            float b = image.at<cv::Vec3b>(i, j)[2];

            values[4 * (i * width + j)] = (float) r;
            values[4 * (i * width + j) + 1] = (float) g;
            values[4 * (i * width + j) + 2] = (float) b;
            values[4 * (i * width + j) + 3] = 1;
        }

    GlTexture input{width, height, TextureFormat::RGBA_FLOAT};

    input.assign(PixelFormat::RGBA, PixelType::FLOAT, &values[0]);


    std::vector<vec4> data;
    input.download(data);

    cv::Mat out_image(height,width, CV_8UC3);
    for (int i = 0; i < width* height; i++) {
        int x = i % width;
        int y = i / width;
        out_image.at<cv::Vec3b>(y,x)[0] =   data[i].x ;
        out_image.at<cv::Vec3b>(y,x)[1] =   data[i].y ;
        out_image.at<cv::Vec3b>(y,x)[2] =   data[i].z ;
    }

    return out_image;

}


cv::Mat rgb_method1(const cv::Mat& image ) {
    uint32_t width = image.cols, height = image.rows;

   cv::Mat float_image;
   image.convertTo(float_image, CV_32FC3);

    GlTexture input{width, height, TextureFormat::RGBA_FLOAT};

    input.assign(PixelFormat::RGB, PixelType::FLOAT, float_image.ptr());


    std::vector<vec4> data;
    input.download(data);

    cv::Mat out_image1(height,width, CV_32FC4);
    memcpy(out_image1.ptr(), data.data(), height*width*4* sizeof(float));

    cv::Mat out_image2(height,width, CV_8UC3);
    out_image1.convertTo(out_image2, CV_8UC3);
    return out_image2;

}


cv::Mat gray_method0(const cv::Mat& image ) {
    uint32_t width = image.cols, height = image.rows;

    cv::Mat float_image;
    image.convertTo(float_image, CV_32FC1);

    GlTexture input{width, height, TextureFormat::R_FLOAT};

    input.assign(PixelFormat::R, PixelType::FLOAT, float_image.ptr());


    std::vector<float> data;
    input.download(data);

    cv::Mat out_image(height,width, CV_8UC1);
    for (int i = 0; i < width* height; i++) {
        int x = i % width;
        int y = i / width;
        out_image.at<uchar>(y,x) =   data[i];
//        out_image.at<cv::Vec3b>(y,x)[1] =   data[i].y ;
//        out_image.at<cv::Vec3b>(y,x)[2] =   data[i].z ;
    }

    return out_image;

}




int main(int argc, char** argv) {
    // init window
    glow::X11OffscreenContext ctx(3, 3);  // OpenGl context
    glow::inititializeGLEW();

    _CheckGlError(__FILE__, __LINE__);
    //  std::cout << "On entry: " << GlState::queryAll() << std::endl;
    std::string image_file = "/home/pang/Documents/000000_11.png";
    cv::Mat image = cv::imread(image_file, CV_LOAD_IMAGE_COLOR);
    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, CV_RGB2GRAY);




    cv::Mat3b rgb_0 = rgb_method0(image);
    cv::Mat rgb_1 = rgb_method1(image);
    cv::Mat gray0 = gray_method0(gray_image);


    cv::imshow("out_image0", rgb_0);
    cv::imshow("out_image1", rgb_1);
    cv::imshow("gray0", gray0);
    cv::waitKey(10000);

    return 0;
}