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
    std::string image_file = "/home/pang/Documents/lenna.jpeg";
    cv::Mat image = cv::imread(image_file, CV_LOAD_IMAGE_COLOR);
    uint32_t width = image.cols, height = image.rows;

    GlFramebuffer fbo(width, height);

    glow::GlBuffer<vec4> surfels{glow::BufferTarget::ARRAY_BUFFER,
                                    glow::BufferUsage::DYNAMIC_DRAW};  // feedback stores updated surfels inside surfels_.

    glow::GlBuffer<vec4> extractBuffer_{glow::BufferTarget::ARRAY_BUFFER, glow::BufferUsage::DYNAMIC_DRAW};
    glow::GlProgram extractProgram_;
    glow::GlTransformFeedback extractFeedback_;

    std::vector<vec4> vec;
    vec.push_back(vec4(1,0,1,0));
    vec.push_back(vec4(1,1,1,1));
    vec.push_back(vec4(1,-1,0, 1));
    surfels.assign(vec);
    std::cout << "surfels: " << surfels.size() << std::endl;

    std::vector<std::string> varyings{
            "position_out",
    };
    surfels.reserve(1000);
    extractFeedback_.attach(varyings, surfels);

    glow::GlVertexArray vao_surfels_;
    // now we can set the vertex attributes. (the "shallow copy" of surfels now contains the correct id.
    vao_surfels_.setVertexAttribute(0, surfels, 4, AttributeType::FLOAT, false, sizeof(vec4),
                                    reinterpret_cast<GLvoid*>(0));


    extractProgram_.attach(GlShader::fromFile(ShaderType::VERTEX_SHADER, "/home/pang/suma_ws/src/glow/samples/shader/extract_surfels.vert"));
    extractProgram_.attach(GlShader::fromFile(ShaderType::FRAGMENT_SHADER, "/home/pang/suma_ws/src/glow/samples/shader/empty.frag"));
    extractProgram_.attach(extractFeedback_);
    extractProgram_.link();


    extractFeedback_.bind();
    extractProgram_.bind();

    glEnable(GL_RASTERIZER_DISCARD);

    extractProgram_.bind();
    extractFeedback_.bind();
    vao_surfels_.bind();



    extractFeedback_.begin(TransformFeedbackMode::POINTS);
    glDrawArrays(GL_POINTS, 0, surfels.size());
    uint32_t extractedSize = extractFeedback_.end();

    extractBuffer_.resize(extractedSize);

    vao_surfels_.release();
    extractFeedback_.release();
    extractProgram_.release();


    std::vector<vec4> download_surfels;
    extractBuffer_.get(download_surfels);
    std::cout << "download_surfels: " << download_surfels.size() << std::endl;

    for (auto i : download_surfels) {
        std::cout << i.x << " " << i.y << " " <<  i.z << " " << i.w << std::endl;
    }

    glDisable(GL_RASTERIZER_DISCARD);

    return 0;
}