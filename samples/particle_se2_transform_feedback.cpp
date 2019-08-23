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
#include <glow/util/GLContext.h>

#include "timer.h"
using namespace glow;

std::vector<vec2> genrateLaserScan(int beam_cnt, int max_range) {
    std::vector<vec2> meas;
    double delta_theta = M_PI*2 / beam_cnt;
    for (int i = 0; i < beam_cnt; i++) {
        double theta = i * delta_theta;
        double range = abs(sin(theta)) * max_range;

        vec2 pt((range*sin(theta)), (range*cos(theta)));
        meas.push_back(pt);
    }
    return meas;
}

std::vector<Eigen::Vector3d> generateSE2TransformationHypothesis(int cnt) {
    std::vector<Eigen::Vector3d> hypothesis;
    double delta_theta = M_PI*2 / cnt;
    for (int i = 0; i < cnt; i++) {
        double theta = i * delta_theta;
        double range = abs(sin(theta)) * 250;  //   In order to visualize the transformations

        Eigen::Vector3d pt(abs(range*sin(theta)), abs(range*cos(theta)), theta);
        hypothesis.push_back(pt);
    }
    return hypothesis;
}

Eigen::Matrix2d rot(const double theta) {
    Eigen::Matrix2d rotation;
    double c = cos(theta);
    double s = sin(theta);
    rotation << c, -s, s, c;
    return rotation;
}

void transfrom_cpu(const std::vector<vec2>& pts,
                   const std::vector<Eigen::Vector3d>& se2_transformations) {
    for (auto se2:se2_transformations) {
        Eigen::Matrix2d R = rot(se2(2));
        for (auto pt : pts) {
            auto transformed = R*Eigen::Vector2d(pt.x, pt.y) + se2.head<2>();
        }
    }
}

int main(int argc, char** argv) {

    /// simulate
    int beam_cnt = 360;
    int max_range = 200;

    int se2_particles_cnt = 3000;

    std::vector<vec2> laserMeas = genrateLaserScan(beam_cnt, max_range);
    std::vector<Eigen::Vector3d> se2_particles = generateSE2TransformationHypothesis(se2_particles_cnt);
    std::cout << "generate meas: " << laserMeas.size() << std::endl;
    std::cout << "generate se2 : " << se2_particles.size() << std::endl;


    Timer cpu_timer, gpu_timer;

    /// CPU
    cpu_timer.start();
    //transfrom_cpu(laserMeas, se2_particles);
    cpu_timer.stop();
    std::cout << "cpu transform: " << cpu_timer.elapsedMilliseconds() << "ms"<< std::endl;


    // init window
//    glow::X11OffscreenContext ctx(3,3);  // OpenGl context
//    glow::inititializeGLEW();

    GLAutoContext glAutoContext;

    glow::GlBuffer<vec4> input_vec{glow::BufferTarget::ARRAY_BUFFER,
                                    glow::BufferUsage::DYNAMIC_DRAW};  // feedback stores updated input_vec inside input_vec.

    glow::GlBuffer<vec4> extractBuffer{glow::BufferTarget::ARRAY_BUFFER, glow::BufferUsage::DYNAMIC_DRAW};
    glow::GlProgram extractProgram;
    glow::GlTransformFeedback extractFeedback;

    std::vector<vec4> vec;

    for (auto se2 : se2_particles) {
        vec4 p;
        p.x = se2(0);
        p.y = se2(1);
        p.z = se2(2);
        p.w = 0;
        vec.push_back(p);

    }
    input_vec.assign(vec);
    std::cout << "input_vec: " << input_vec.size() << std::endl;

    std::vector<std::string> varyings{
            "result_out",
    };
    extractBuffer.reserve(10000);
    extractFeedback.attach(varyings, extractBuffer);

    glow::GlVertexArray vao_input_vec;
    // now we can set the vertex attributes. (the "shallow copy" of input_vec now contains the correct id.
    vao_input_vec.setVertexAttribute(0, input_vec, 4, AttributeType::FLOAT, false, sizeof(vec4),
                                    reinterpret_cast<GLvoid*>(0));


    extractProgram.attach(GlShader::fromFile(ShaderType::VERTEX_SHADER, "/home/pang/suma_ws/src/glow/samples/shader/particle_se2_transform_feedback.vert"));
    extractProgram.attach(GlShader::fromFile(ShaderType::FRAGMENT_SHADER, "/home/pang/suma_ws/src/glow/samples/shader/empty.frag"));
    extractProgram.attach(extractFeedback);
    extractProgram.link();

    extractProgram.setUniform(GlUniform<std::vector<vec2>>("laser_points", laserMeas));

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
    download_input_vec.reserve(10000);
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