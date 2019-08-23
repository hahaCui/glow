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


#include <time.h>
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

    int se2_particles_cnt = 300000;

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



    /// GPU
    // init window
//    glow::X11OffscreenContext ctx(3,3);  // OpenGl context
//    glow::inititializeGLEW();

    GLAutoContext glAutoContext;
//    std::cout << "On entry: " << GlState::queryAll() << std::endl;
    uint32_t width = 600, height = 500;  // 3000
    GlFramebuffer fbo(width, height);

//    ASSERT_NO_THROW(_CheckGlError(__FILE__, __LINE__));

    gpu_timer.start();


    GlTexture output{width, height, TextureFormat::RGBA_FLOAT};
    GlRenderbuffer rbo(width, height, RenderbufferFormat::DEPTH_STENCIL);

    fbo.attach(FramebufferAttachment::COLOR0, output);
    CheckGlError();
    fbo.attach(FramebufferAttachment::DEPTH_STENCIL, rbo);
    CheckGlError();

    GlProgram program;
    program.attach(GlShader::fromFile(ShaderType::VERTEX_SHADER, "/home/pang/suma_ws/src/glow/samples/shader/transform_se2.vert"));
    program.attach(GlShader::fromFile(ShaderType::FRAGMENT_SHADER, "/home/pang/suma_ws/src/glow/samples/shader/transform_se2.frag"));
    program.link();

    GlBuffer<vec4> pixel_buffer{BufferTarget::ARRAY_BUFFER, BufferUsage::STATIC_DRAW};
    GlBuffer<vec4> particle_buffer{BufferTarget::ARRAY_BUFFER, BufferUsage::STATIC_DRAW};

    std::vector<vec4> pixels;
    std::vector<vec4> particles;
    for (uint32_t i = 0; i < height; ++i) {
        for (uint32_t j = 0; j < width; ++j) {
            vec4 v;
            v.x = 2.0f * (float(j + 0.5f) / float(width)) - 1.0f;
            v.y = 2.0f * (float(i + 0.5f) / float(height)) - 1.0f;
            v.z = 0;
            v.w = 0;
            pixels.push_back(v);

            int index = i * width + j;
            v.x = se2_particles.at(index)[0];
            v.y = se2_particles.at(index)[1];
            v.z = se2_particles.at(index)[2];
            v.w = 0;
            particles.push_back(v);
        }
    }

    pixel_buffer.assign(pixels);
    particle_buffer.assign(particles);

    GlVertexArray vao;
    // 1. set
    vao.setVertexAttribute(0, pixel_buffer, 4, AttributeType::FLOAT, false, 4 * sizeof(float), nullptr);
    vao.setVertexAttribute(1, particle_buffer, 4, AttributeType::FLOAT, false, 4 * sizeof(float), nullptr);
    // 2. enable
    vao.enableVertexAttribute(0);
    vao.enableVertexAttribute(1);


    program.setUniform(GlUniform<std::vector<vec2>>("laser_points", laserMeas));

    glDisable(GL_DEPTH_TEST);

    fbo.bind();
    glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    glViewport(0, 0, width, height);
    program.bind();
    vao.bind();

    glDrawArrays(GL_POINTS, 0, pixel_buffer.size());

    vao.release();
    program.release();
    fbo.release();

    glEnable(GL_DEPTH_TEST);

    gpu_timer.stop();
    std::cout << std::endl;
    std::cout << "gpu transform: " << gpu_timer.elapsedMilliseconds() << "ms"<< std::endl;


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
    cv::waitKey(1000);

    return 0;
}