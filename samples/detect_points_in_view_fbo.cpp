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



std::vector<vec3> loadLidarPoints(const std::string& bin_file ) {
    // load point cloud
    std::fstream input(bin_file, std::ios::in | std::ios::binary);
    if(!input.good()){
        std::cerr << "Could not read file: " << bin_file << std::endl;
        exit(EXIT_FAILURE);
    }
    input.seekg(0, std::ios::beg);

    std::vector<vec3> points;

    int i;
    for (i=0; input.good() && !input.eof(); i++) {
        vec3 pt;
        float intensity;
        input.read((char *) &pt.x, 3*sizeof(float));
        input.read((char *) &intensity, sizeof(float));
        points.push_back(pt);
    }
    input.close();
//    std::cout << "Read KTTI point cloud with " << i << " points" << std::endl;
    return points;
}


int main(int argc, char** argv) {

    std::string image_file = "/home/pang/disk/dataset/kitti/00/image_0/000000.png";
    std::string lidarscan_file = "/home/pang/disk/dataset/kitti/00/velodyne/000000.bin";

    cv::Mat image = cv::imread(image_file, CV_LOAD_IMAGE_COLOR);
//    cv::imshow("image", image);
//    cv::waitKey(3000);
    std::vector<vec3> lidar_points = loadLidarPoints(lidarscan_file);


    uint32_t width = image.cols;
    uint32_t height = image.rows;
    std::cout << "lidar_points: " << lidar_points.size() << std::endl;


    Eigen::Matrix4f T_cam_lidar;
    T_cam_lidar <<4.276802385584e-04, -9.999672484946e-01, -8.084491683471e-03, -1.198459927713e-02,
            -7.210626507497e-03, 8.081198471645e-03, -9.999413164504e-01, -5.403984729748e-02,
            9.999738645903e-01, 4.859485810390e-04, -7.206933692422e-03, -2.921968648686e-01,
            0,0,0,1;

    float fx = 7.188560000000e+02;
    float fy = 7.188560000000e+02;
    float cx = 6.071928000000e+02;
    float cy = 1.852157000000e+02;

    vec4 intrinsic(fx, fy, cx, cy);
    vec2 image_wh(width, height);

    // init window
    glow::X11OffscreenContext ctx(3,3);  // OpenGl context
    glow::inititializeGLEW();

    //  std::cout << "On entry: " << GlState::queryAll() << std::endl;
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
    program.attach(GlShader::fromFile(ShaderType::VERTEX_SHADER, "/home/pang/suma_ws/src/glow/samples/shader/detect_in_view_fbo.vert"));
    program.attach(GlShader::fromFile(ShaderType::FRAGMENT_SHADER, "/home/pang/suma_ws/src/glow/samples/shader/detect_in_view_fbo.frag"));
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

            v.x = (float)image.at<cv::Vec3b>(i,j)[0];
            v.y = (float)image.at<cv::Vec3b>(i,j)[1];
            v.z = (float)image.at<cv::Vec3b>(i,j)[2];
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