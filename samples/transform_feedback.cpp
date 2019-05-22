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

struct Surfel {
public:
    Surfel() {}
    Surfel(float _x, float _y, float _z):
        x(_x), y(_y), z(_z){}
    float x, y, z;
    float radius;
    float nx, ny, nz;
    float confidence;

    int32_t timestamp;
    float color, weight, count;
};

int main(int argc, char** argv) {
    // init window
    glow::X11OffscreenContext ctx(3,3);  // OpenGl context
    glow::inititializeGLEW();

    //  std::cout << "On entry: " << GlState::queryAll() << std::endl;
    std::string image_file = "/home/pang/Documents/lenna.png";
    cv::Mat image = cv::imread(image_file, CV_LOAD_IMAGE_COLOR);
    uint32_t width = image.cols, height = image.rows;

    GlFramebuffer fbo(width, height);

    glow::GlBuffer<Surfel> surfels{glow::BufferTarget::ARRAY_BUFFER,
                                    glow::BufferUsage::DYNAMIC_DRAW};  // feedback stores updated surfels inside surfels_.

    glow::GlBuffer<Surfel> extractBuffer_{glow::BufferTarget::ARRAY_BUFFER, glow::BufferUsage::DYNAMIC_DRAW};
    glow::GlProgram extractProgram_;
    glow::GlTransformFeedback extractFeedback_;

    std::vector<Surfel> surfel_vec;
    surfel_vec.push_back(Surfel(1,0,1));
    surfel_vec.push_back(Surfel(1,1,1));
    surfel_vec.push_back(Surfel(1,-1,1));
    surfels.assign(surfel_vec);
    std::cout << "surfels: " << surfels.size() << std::endl;

    std::vector<std::string> surfel_varyings{
            "sfl_position_radius", "sfl_normal_confidence", "sfl_timestamp", "sfl_color_weight_count",
    };
    surfels.reserve(1000);
    extractFeedback_.attach(surfel_varyings, surfels);

    glow::GlVertexArray vao_surfels_;
    // now we can set the vertex attributes. (the "shallow copy" of surfels now contains the correct id.
    vao_surfels_.setVertexAttribute(0, surfels, 4, AttributeType::FLOAT, false, sizeof(Surfel),
                                    reinterpret_cast<GLvoid*>(0));
    vao_surfels_.setVertexAttribute(1, surfels, 4, AttributeType::FLOAT, false, sizeof(Surfel),
                                    reinterpret_cast<GLvoid*>(4 * sizeof(GLfloat)));
    vao_surfels_.setVertexAttribute(2, surfels, 1, AttributeType::INT, false, sizeof(Surfel),
                                    reinterpret_cast<GLvoid*>(8 * sizeof(GLfloat)));
    vao_surfels_.setVertexAttribute(3, surfels, 3, AttributeType::FLOAT, false, sizeof(Surfel),
                                    reinterpret_cast<GLvoid*>(offsetof(Surfel, color)));

    extractProgram_.attach(GlShader::fromFile(ShaderType::VERTEX_SHADER, "shader/extract_surfels.vert"));
    extractProgram_.attach(GlShader::fromFile(ShaderType::GEOMETRY_SHADER, "shader/copy_surfels.geom"));
    extractProgram_.attach(GlShader::fromFile(ShaderType::FRAGMENT_SHADER, "shader/empty.frag"));
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

    glDisable(GL_RASTERIZER_DISCARD);

    return 0;
}