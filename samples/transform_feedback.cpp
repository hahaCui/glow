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

struct GrayValue {
    int gray;
};
int main(int argc, char** argv) {
    // init window
    glow::X11OffscreenContext ctx(3,3);  // OpenGl context
    glow::inititializeGLEW();

    //  std::cout << "On entry: " << GlState::queryAll() << std::endl;
    std::string image_file = "/home/pang/Documents/lenna.png";
    cv::Mat image = cv::imread(image_file, CV_LOAD_IMAGE_COLOR);
    uint32_t width = image.cols, height = image.rows;

    GlBuffer<vec4> pixel_buffer{BufferTarget::ARRAY_BUFFER, BufferUsage::STATIC_DRAW};
    GlBuffer<vec4> color_buffer{BufferTarget::ARRAY_BUFFER, BufferUsage::STATIC_DRAW};

    std::vector<vec4> pixels;
    std::vector<vec4> colors;



    std::vector<float> values0( width * height, 0);
    std::vector<float> values1( width * height, 0);
    std::vector<float> values2( width * height, 0);
    for(auto i = 0 ; i < image.rows; i++)
        for(auto j = 0; j < image.cols; j++) {
            float r = image.at<cv::Vec3b>(i,j)[0];
            float g = image.at<cv::Vec3b>(i,j)[1];
            float b = image.at<cv::Vec3b>(i,j)[2];

            colors.push_back(vec4(r,g,b,1));

            vec4 v;
            v.x = 2.0f * (float(j + 0.5f) / float(width)) - 1.0f;
            v.y = 2.0f * (float(i + 0.5f) / float(height)) - 1.0f;
            v.z = 0;
            v.w = 0;
            pixels.push_back(v);
        }


    pixel_buffer.assign(pixels);
    color_buffer.assign(colors);

    GlFramebuffer fbo(width, height);

//    ASSERT_NO_THROW(_CheckGlError(__FILE__, __LINE__));

    GlTexture output{width, height, TextureFormat::RGBA_FLOAT};
    GlRenderbuffer rbo(width, height, RenderbufferFormat::DEPTH_STENCIL);

    fbo.attach(FramebufferAttachment::COLOR0, output);
    CheckGlError();
    fbo.attach(FramebufferAttachment::DEPTH_STENCIL, rbo);
    CheckGlError();

    glow::GlTransformFeedback rgb2gray_feedback;

    std::vector<std::string> gray_varyings{
            "gray"
    };
    glow::GlBuffer<GrayValue> gray_values{glow::BufferTarget::ARRAY_BUFFER,
                             glow::BufferUsage::DYNAMIC_DRAW};
    gray_values.reserve(width* height);
    rgb2gray_feedback.attach(gray_varyings, gray_values);

    GlProgram program;
    program.attach(GlShader::fromFile(ShaderType::VERTEX_SHADER, "/home/pang/suma_ws/src/glow/samples/shader/rgb_to_gray.vert"));
    program.attach(GlShader::fromFile(ShaderType::FRAGMENT_SHADER, "/home/pang/suma_ws/src/glow/samples/shader/passthrough.frag"));
    //program.attach(rgb2gray_feedback);
    program.link();



    GlVertexArray vao;
    // 1. set
    vao.setVertexAttribute(0, pixel_buffer, 4, AttributeType::FLOAT, false, 4 * sizeof(float), nullptr);
    vao.setVertexAttribute(1, color_buffer, 4, AttributeType::FLOAT, false, 4 * sizeof(float), nullptr);
    // 2. enable
    vao.enableVertexAttribute(0);
    vao.enableVertexAttribute(1);

    glDisable(GL_DEPTH_TEST);

    fbo.bind();
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    glViewport(0, 0, width, height);
//    rgb2gray_feedback.bind();
    program.bind();
    vao.bind();

//    rgb2gray_feedback.begin(TransformFeedbackMode::POINTS);
    glDrawArrays(GL_POINTS, 0, pixel_buffer.size());
//    gray_values.resize(rgb2gray_feedback.end());


    vao.release();
    program.release();
//    rgb2gray_feedback.release();
    fbo.release();

    glEnable(GL_DEPTH_TEST);


    // retrieve result
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

    cv::imshow("out_image", out_image);
    cv::waitKey(10000);

    return 0;
}