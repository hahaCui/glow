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

#include <time.h>
#include "timer.h"
using namespace glow;


std::vector<Eigen::Vector2d> genrateLaserScan(int beam_cnt, int max_range) {
    std::vector<Eigen::Vector2d> meas;
    double delta_theta = M_PI*2 / beam_cnt;
    for (int i = 0; i < beam_cnt; i++) {
        double theta = i * delta_theta;
        double range = abs(sin(theta)) * max_range;

        Eigen::Vector2d pt(range*sin(theta), range*cos(theta));
        meas.push_back(pt);
    }
    return meas;
};

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

void transfrom_cpu(const std::vector<Eigen::Vector2d>& pts,
        const std::vector<Eigen::Vector3d>& se2_transformations) {
    for (auto se2:se2_transformations) {
        Eigen::Matrix2d R = rot(se2(2));

        for (auto pt : pts) {
            auto transformed = R*pt + se2.head<2>();
        }
    }
}

int main(int argc, char** argv) {

    /// simulate
    int beam_cnt = 1440;
    int max_range = 30;

    int se2_particles_cnt = 300000;

    std::vector<Eigen::Vector2d> laserMeas = genrateLaserScan(beam_cnt, max_range);
    std::vector<Eigen::Vector3d> se2_particles = generateSE2TransformationHypothesis(se2_particles_cnt);
    std::cout << "generate meas: " << laserMeas.size() << std::endl;
    std::cout << "generate se2 : " << se2_particles.size() << std::endl;




    Timer cpu_timer, gpu_timer;

    /// CPU
//    cpu_timer.start();
//    transfrom_cpu(laserMeas, se2_particles);
//    cpu_timer.stop();
//    std::cout << "cpu transform: " << cpu_timer.elapsedMilliseconds() << "ms"<< std::endl;



    /// GPU
    // init window
    glow::X11OffscreenContext ctx(3,3);  // OpenGl context
    glow::inititializeGLEW();

    //  std::cout << "On entry: " << GlState::queryAll() << std::endl;
    uint32_t width = 600, height = 500;  // 3000
    GlFramebuffer fbo(width, height);

//    ASSERT_NO_THROW(_CheckGlError(__FILE__, __LINE__));

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
    GlBuffer<vec4> color_buffer{BufferTarget::ARRAY_BUFFER, BufferUsage::STATIC_DRAW};

    std::vector<vec4> pixels;
    std::vector<vec4> colors;
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


    // todo: pass simulated laser points use glUniform3fv

    std::vector<vec2> point_vec;
    point_vec.push_back(vec2(255,0));
    point_vec.push_back(vec2(0,255));

    program.setUniform(GlUniform<std::vector<vec2>>("laser_points", point_vec));

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