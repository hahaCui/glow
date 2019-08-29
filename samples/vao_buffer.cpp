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
//    GlFramebuffer fbo(width, height);
    GLuint FramebufferName = 0;
    glGenFramebuffers(1, &FramebufferName);
    glBindFramebuffer(GL_FRAMEBUFFER, FramebufferName);

//    ASSERT_NO_THROW(_CheckGlError(__FILE__, __LINE__));

    GlTexture output0{width, height, TextureFormat::RGBA_FLOAT};
    GlTexture output1{width, height, TextureFormat::RGBA_FLOAT};
    GlRenderbuffer rbo(width, height, RenderbufferFormat::DEPTH_STENCIL);

//    fbo.attach(FramebufferAttachment::COLOR0, output);
//    CheckGlError();
//    fbo.attach(FramebufferAttachment::DEPTH_STENCIL, rbo);
//    CheckGlError();
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rbo.id());
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, output0.id(), 0);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, output1.id(), 0);

    // Set the list of draw buffers.
    GLenum DrawBuffers[2] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1};
    glDrawBuffers(2, DrawBuffers); // "1" is the size of DrawBuffers


    GlProgram program;
    program.attach(GlShader::fromFile(ShaderType::VERTEX_SHADER, "/home/pang/suma_ws/src/glow/samples/shader/ndc.vert"));
    program.attach(GlShader::fromFile(ShaderType::FRAGMENT_SHADER, "/home/pang/suma_ws/src/glow/samples/shader/ndc.frag"));
    program.link();

    GlBuffer<vec4> pixel_buffer{BufferTarget::ARRAY_BUFFER, BufferUsage::STATIC_DRAW};
    GlBuffer<vec4> color_buffer{BufferTarget::ARRAY_BUFFER, BufferUsage::STATIC_DRAW};

    std::vector<vec4> pixels;
    std::vector<vec4> colors;
    for (uint32_t i = 0; i < height ; ++i) {
        for (uint32_t j = 0; j < width ; ++j) {
            vec4 v;
            v.x = 2.0f * (float(j + 0.5f) / float(width)) - 1.0f;
            v.y = 2.0f * (float(i + 0.5f) / float(height)) - 1.0f;
            v.z = 0;
            v.w = 0;
            pixels.push_back(v);

            v.x = (float)j / width * 255;
            v.y = (float)i / height * 255;
            v.z = 0;
            v.w = 0;
            colors.push_back(v);
        }
    }

    pixel_buffer.assign(pixels);
    color_buffer.assign(colors);

    GlVertexArray vao;
    // 1. set
    vao.setVertexAttribute(0, pixel_buffer, 4, AttributeType::FLOAT, false, 4 * sizeof(float), nullptr);
    vao.setVertexAttribute(1, color_buffer, 4, AttributeType::FLOAT, false, 4 * sizeof(float), nullptr);
    // 2. enable
    vao.enableVertexAttribute(0);
    vao.enableVertexAttribute(1);

    glDisable(GL_DEPTH_TEST);

//    fbo.bind();
    glBindFramebuffer(GL_FRAMEBUFFER, FramebufferName);
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    glViewport(0, 0, width, height);
    program.bind();
    vao.bind();

    glDrawArrays(GL_POINTS, 0, pixel_buffer.size());

    vao.release();
    program.release();
//    fbo.release();
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glEnable(GL_DEPTH_TEST);


    // retrieve result
    std::vector<vec4> data0, data1;
    output0.download(data0);
    output1.download(data1);

    cv::Mat out_image0(height,width, CV_8UC3);
    cv::Mat out_image1(height,width, CV_8UC3);
    for (int i = 0; i < width* height; i++) {
        int x = i % width;
        int y = i / width;
        out_image0.at<cv::Vec3b>(y,x)[0] =   data0[i].x ;
        out_image0.at<cv::Vec3b>(y,x)[1] =   data0[i].y ;
        out_image0.at<cv::Vec3b>(y,x)[2] =   data0[i].z ;

        out_image1.at<cv::Vec3b>(y,x)[0] =   data1[i].x ;
        out_image1.at<cv::Vec3b>(y,x)[1] =   data1[i].y ;
        out_image1.at<cv::Vec3b>(y,x)[2] =   data1[i].z ;
    }

    cv::imshow("out_image0", out_image0);
    cv::imshow("out_image1", out_image1);
    cv::waitKey(10000);

    return 0;
}