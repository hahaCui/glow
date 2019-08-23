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

int main(int argc, char** argv) {
    // init window
    glow::X11OffscreenContext ctx(3,3);  // OpenGl context
    glow::inititializeGLEW();

    std::string image_file = "/home/pang/disk/dataset/kitti/00/image_0/000000.png";

    cv::Mat image = cv::imread(image_file, CV_LOAD_IMAGE_COLOR);

    uint32_t width = image.cols;
    uint32_t height = image.rows;

    //  std::cout << "On entry: " << GlState::queryAll() << std::endl;
    GlFramebuffer fbo(width, height);

    GlTexture output{width, height, TextureFormat::RGBA_FLOAT};
    GlRenderbuffer rbo(width, height, RenderbufferFormat::DEPTH_STENCIL);

    fbo.attach(FramebufferAttachment::COLOR0, output);
    CheckGlError();
    fbo.attach(FramebufferAttachment::DEPTH_STENCIL, rbo);
    CheckGlError();

    // input
    cv::Mat float_image;
    image.convertTo(float_image, CV_32FC3);
    GlTexture input_texture{width, height, TextureFormat::RGBA_FLOAT};
    input_texture.assign(PixelFormat::RGB, PixelType::FLOAT, float_image.ptr());

    GlProgram program;
    program.attach(GlShader::fromFile(ShaderType::VERTEX_SHADER, "/home/pang/suma_ws/src/glow/samples/shader/sampler.vert"));
    program.attach(GlShader::fromFile(ShaderType::FRAGMENT_SHADER, "/home/pang/suma_ws/src/glow/samples/shader/sampler.frag"));
    program.link();

    program.setUniform(GlUniform<int32_t>("input_texture", 0));

    GlBuffer<vec4> pixel_buffer{BufferTarget::ARRAY_BUFFER, BufferUsage::STATIC_DRAW};

    std::vector<vec4> pixels;
    // Pay attention to coordinates
    for (uint32_t i = 0; i < height ; ++i) {
        for (uint32_t j = 0; j < width ; ++j) {
            vec4 v;
            v.x = 2.0f * (float(j + 0.5f) / float(width)) - 1.0f;
            v.y = 2.0f * (float(i + 0.5f) / float(height)) - 1.0f;
            v.z = float(j ) / float(width);
            v.w = float(i ) / float(height);
            pixels.push_back(v);
        }
    }

    pixel_buffer.assign(pixels);

    GlVertexArray vao;
    // 1. set
    vao.setVertexAttribute(0, pixel_buffer, 4, AttributeType::FLOAT, false, 4 * sizeof(float), nullptr);
    // 2. enable
    vao.enableVertexAttribute(0);

    GlSampler sampler;
    sampler.setMagnifyingOperation(TexMagOp::NEAREST);
    sampler.setMinifyingOperation(TexMinOp::NEAREST);




    glDisable(GL_DEPTH_TEST);

    fbo.bind();
    sampler.bind(0);
    glActiveTexture(GL_TEXTURE0);
    input_texture.bind();

    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    glViewport(0, 0, width, height);
    program.bind();
    vao.bind();

    glDrawArrays(GL_POINTS, 0, pixel_buffer.size());

    vao.release();
    program.release();
    fbo.release();

    sampler.release(0);
    glActiveTexture(GL_TEXTURE0);
    input_texture.release();


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