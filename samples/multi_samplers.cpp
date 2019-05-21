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

void imageConvolution(const cv::Mat& input_rgb_image, cv::Mat& output_rgb_image, int unRadius) {
    //Convolution
    float fSum = 0.0f;
    int unTotal = 0;
    auto unHeight = input_rgb_image.rows;
    auto unWidth = input_rgb_image.cols;
    auto unChannel = input_rgb_image.channels();

    output_rgb_image = cv::Mat(unHeight,  unWidth, CV_8UC3);
    for(int i=0; i<unHeight; i++)
        for(int j=0; j<unWidth; j++)
            for(int k=0; k<unChannel; k++){
                for(int ii=i-unRadius; ii<=i+unRadius; ii++)
                    for(int jj=j-unRadius; jj<=j+unRadius; jj++){
                        if(ii>=0 && jj>=0 && ii<unHeight && jj<unWidth){
                            fSum += input_rgb_image.at<cv::Vec3b>(ii,jj)[k];
                            unTotal++;
                        }
                    }
                output_rgb_image.at<cv::Vec3b>(i,j)[k] = fSum / (float)unTotal;
                unTotal = 0;
                fSum = 0.0f;
            }
    return;

}
int main(int argc, char** argv) {
    // init window
    glow::X11OffscreenContext ctx(3,3);  // OpenGl context
    glow::inititializeGLEW();

    float convolution_radius = 5;
    _CheckGlError(__FILE__, __LINE__);
    //  std::cout << "On entry: " << GlState::queryAll() << std::endl;
    std::string image_file = "/home/pang/Documents/lenna.png";
    cv::Mat image = cv::imread(image_file, CV_LOAD_IMAGE_COLOR);
    uint32_t width = image.cols, height = image.rows;

    std::vector<float> values(3 * width * height, 0);
    for(auto i = 0 ; i < image.rows; i++)
        for(auto j = 0; j < image.cols; j++) {
            float r = image.at<cv::Vec3b>(i,j)[0];
            float g = image.at<cv::Vec3b>(i,j)[1];
            float b = image.at<cv::Vec3b>(i,j)[2];

            values[3 * (i * width +j)] = (float)r;
            values[3 * (i * width +j) + 1] = (float)g;
            values[3 * (i * width +j) + 2] = (float)b;
        }

    cv::Mat cpu_smoothed_image;
    Timer cpu_timer, gpu_timer;
    cpu_timer.start();
    imageConvolution(image, cpu_smoothed_image, convolution_radius);
    cpu_timer.stop();
    std::cout << "cpu Convolution: " << cpu_timer.elapsedMilliseconds() << "ms"<< std::endl;


    gpu_timer.start();
    GlFramebuffer fbo(width, height);

    _CheckGlError(__FILE__, __LINE__);

    GlTexture input{width, height, TextureFormat::RGB_FLOAT};

    input.assign(PixelFormat::RGB, PixelType::FLOAT, &values[0]);

    GlTexture output{width, height, TextureFormat::RGB_FLOAT};
    GlRenderbuffer rbo(width, height, RenderbufferFormat::DEPTH_STENCIL);

    CheckGlError();
    fbo.attach(FramebufferAttachment::COLOR0, output);
    fbo.attach(FramebufferAttachment::DEPTH_STENCIL, rbo);
    CheckGlError();

    GlProgram program;
    program.attach(GlShader::fromFile(ShaderType::VERTEX_SHADER, "/home/pang/suma_ws/src/glow/samples/shader/empty.vert"));
    program.attach(GlShader::fromFile(ShaderType::GEOMETRY_SHADER, "/home/pang/suma_ws/src/glow/samples/shader/quad.geom"));
    program.attach(GlShader::fromFile(ShaderType::FRAGMENT_SHADER, "/home/pang/suma_ws/src/glow/samples/shader/convolution.frag"));
    program.link();

    program.setUniform(GlUniform<float>("fRadius", convolution_radius));
    program.setUniform(GlUniform<float>("nWidth", image.rows));
    program.setUniform(GlUniform<float>("nHeight", image.cols));

    GlSampler sampler;
    sampler.setMagnifyingOperation(TexMagOp::NEAREST);
    sampler.setMinifyingOperation(TexMinOp::NEAREST);

    GlVertexArray vao_no_points;

    fbo.bind();
    vao_no_points.bind();
    glActiveTexture(GL_TEXTURE0);
    input.bind();
    program.bind();

    sampler.bind(0);

    glDisable(GL_DEPTH_TEST);
    glViewport(0, 0, width, height);

    glDrawArrays(GL_POINTS, 0, 1);

    program.release();
    input.release();
    vao_no_points.release();
    fbo.release();

    sampler.release(0);

    glEnable(GL_DEPTH_TEST);

    gpu_timer.stop();
    std::cout << "gpu Convolution : " << gpu_timer.elapsedMilliseconds() << "ms" << std::endl;

    std::vector<vec4> data;
    output.download(data);

    cv::Mat out_image(height,width, CV_8UC3);
    for (int i = 0; i < width* height; i++) {
        int x = i % width;
        int y = i / width;
        out_image.at<cv::Vec3b>(y,x)[0] =   data[i].x ;
        out_image.at<cv::Vec3b>(y,x)[1] =   data[i].y ;
        out_image.at<cv::Vec3b>(y,x)[2] =   data[i].z ;
    }

    cv::imshow("image", image);
    cv::imshow("out_image", out_image);
    cv::imshow("cpu_smooth_image", cpu_smoothed_image);
    cv::waitKey(10000);

    return 0;
}