#include <iostream>
#include <glow/glbase.h>
#include <glow/glutil.h>

#include <glow/GlBuffer.h>
#include <glow/GlFramebuffer.h>
#include <glow/GlProgram.h>
#include <glow/GlVertexArray.h>
#include <glow/ScopedBinder.h>
#include <glow/GlTransformFeedback.h>

#include <algorithm>
#include <random>
#include <vector>
#include <glow/GlSampler.h>
#include <opencv2/opencv.hpp>

#include <glow/GlCapabilities.h>
#include <glow/GlState.h>
#include <glow/util/X11OffscreenContext.h>

#include "types.h"
using namespace glow;



int main(int argc, char** argv) {
    // init window
    glow::X11OffscreenContext ctx(3,3);  // OpenGl context
    glow::inititializeGLEW();

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
    program.attach(GlShader::fromFile(ShaderType::VERTEX_SHADER, "/home/pang/suma_ws/src/glow/samples/shader/passthrough.vert"));
    program.attach(GlShader::fromFile(ShaderType::FRAGMENT_SHADER, "/home/pang/suma_ws/src/glow/samples/shader/passthrough.frag"));
    program.link();

    GlBuffer<vec4> buffer{BufferTarget::ARRAY_BUFFER, BufferUsage::STATIC_DRAW};

    std::vector<vec4> pixels;
    for (uint32_t i = 0; i < width; i+=10) {
        for (uint32_t j = 0; j < height; ++j) {
            vec4 v;
            v.x = 2.0f * (float(i + 0.5f) / float(width)) - 1.0f;
            v.y = 2.0f * (float(j + 0.5f) / float(height)) - 1.0f;
            v.z = i;
            v.w = j;
            pixels.push_back(v);
        }
    }

    buffer.assign(pixels);

    GlVertexArray vao;
    vao.setVertexAttribute(0, buffer, 4, AttributeType::FLOAT, false, 4 * sizeof(float), nullptr);
    vao.enableVertexAttribute(0);

    glDisable(GL_DEPTH_TEST);

    fbo.bind();
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    glViewport(0, 0, width, height);
    program.bind();
    vao.bind();

    glDrawArrays(GL_POINTS, 0, buffer.size());

    vao.release();
    program.release();
    fbo.release();

    glEnable(GL_DEPTH_TEST);

    std::vector<vec4> values;
    output.download(values);


//
    cv::Mat out_image(height,width, CV_8UC3);
    for (int i = 0; i < width* height; i++) {
        int x = i % width;
        int y = i / width;
        out_image.at<cv::Vec3b>(y,x)[0] =   values[i].x ;
        out_image.at<cv::Vec3b>(y,x)[1] =   values[i].y ;
        out_image.at<cv::Vec3b>(y,x)[2] =   values[i].z ;
    }

    cv::imshow("out_image", out_image);
    cv::waitKey(10000);

    return 0;
}