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

    glow::GlBuffer<vec4> input_vec{glow::BufferTarget::ARRAY_BUFFER,
                                    glow::BufferUsage::DYNAMIC_DRAW};  // feedback stores updated input_vec inside input_vec.

    glow::GlBuffer<vec4> extractBuffer{glow::BufferTarget::ARRAY_BUFFER, glow::BufferUsage::DYNAMIC_DRAW};
    glow::GlProgram extractProgram;
    glow::GlTransformFeedback extractFeedback;

    std::vector<vec4> vec;
    vec.push_back(vec4(1,0,1,0));
    vec.push_back(vec4(1,1,1,1));
    vec.push_back(vec4(1,-1,0, 1));
    input_vec.assign(vec);
    std::cout << "input_vec: " << input_vec.size() << std::endl;

    std::vector<std::string> varyings{
            "position_out",
    };
    extractBuffer.reserve(100);
    extractFeedback.attach(varyings, extractBuffer);

    glow::GlVertexArray vao_input_vec;
    // now we can set the vertex attributes. (the "shallow copy" of input_vec now contains the correct id.
    vao_input_vec.setVertexAttribute(0, input_vec, 4, AttributeType::FLOAT, false, sizeof(vec4),
                                    reinterpret_cast<GLvoid*>(0));


    extractProgram.attach(GlShader::fromFile(ShaderType::VERTEX_SHADER, "/home/pang/suma_ws/src/glow/samples/shader/extract_vec.vert"));
    extractProgram.attach(GlShader::fromFile(ShaderType::FRAGMENT_SHADER, "/home/pang/suma_ws/src/glow/samples/shader/empty.frag"));
    extractProgram.attach(extractFeedback);
    extractProgram.link();


    extractFeedback.bind();
    extractProgram.bind();

    glEnable(GL_RASTERIZER_DISCARD);

    extractProgram.bind();
    extractFeedback.bind();
    vao_input_vec.bind();



    extractFeedback.begin(TransformFeedbackMode::POINTS);
    glDrawArrays(GL_POINTS, 0, input_vec.size());
    uint32_t extractedSize = extractFeedback.end();

    extractBuffer.resize(extractedSize);

    std::vector<vec4> download_input_vec;
    download_input_vec.reserve(3);
    extractBuffer.get(download_input_vec);
    std::cout << "download_input_vec: " << download_input_vec.size() << std::endl;

    for (auto i : download_input_vec) {
        std::cout << i.x << " " << i.y << " " <<  i.z << " " << i.w << std::endl;
    }
    
    vao_input_vec.release();
    extractFeedback.release();
    extractProgram.release();
    
    glDisable(GL_RASTERIZER_DISCARD);

    return 0;
}