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

    //  std::cout << "On entry: " << GlState::queryAll() << std::endl;
    uint32_t width = 640, height = 480;
    GlFramebuffer fbo(width, height);

//    ASSERT_NO_THROW(_CheckGlError(__FILE__, __LINE__));

    GlTexture output{width, height, TextureFormat::RGBA_FLOAT};
    GlRenderbuffer rbo(width, height, RenderbufferFormat::DEPTH_STENCIL);

    fbo.attach(FramebufferAttachment::COLOR0, output);
    CheckGlError();
    fbo.attach(FramebufferAttachment::DEPTH_STENCIL, rbo);
    CheckGlError();

    GlProgram program;
    program.attach(GlShader::fromFile(ShaderType::VERTEX_SHADER, "/home/pang/suma_ws/src/glow/samples/shader/passthrough1.vert"));
    program.attach(GlShader::fromFile(ShaderType::FRAGMENT_SHADER, "/home/pang/suma_ws/src/glow/samples/shader/passthrough.frag"));
    program.link();

    GlBuffer<vec4> pixel_buffer{BufferTarget::ARRAY_BUFFER, BufferUsage::STATIC_DRAW};
    GlBuffer<vec4> color_buffer{BufferTarget::ARRAY_BUFFER, BufferUsage::STATIC_DRAW};

    std::vector<vec4> pixels;
    std::vector<vec4> colors;


    pixels.push_back(vec4(0.50f, 0.50f, 0.0f,1.0f));
    pixels.push_back(vec4(-0.50f, 0.50f, 0.0f,1.0f));
    pixels.push_back(vec4(0.50f,-0.50f, 0.0f,1.0f));


    pixel_buffer.assign(pixels);
//    color_buffer.assign(colors);

    GlVertexArray vao;
    // 1. set
    vao.setVertexAttribute(0, pixel_buffer, 4, AttributeType::FLOAT, false, 4 * sizeof(float), nullptr);
//    vao.setVertexAttribute(1, color_buffer, 4, AttributeType::FLOAT, false, 4 * sizeof(float), nullptr);
    // 2. enable
    vao.enableVertexAttribute(0);
//    vao.enableVertexAttribute(1);

    glDisable(GL_DEPTH_TEST);

    fbo.bind();
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    glViewport(0, 0, width, height);
    program.bind();
    vao.bind();

//    glDrawArrays(GL_POINTS, 0, pixel_buffer.size());
    glDrawArrays(GL_TRIANGLES, 0, pixel_buffer.size());

    vao.release();
    program.release();
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