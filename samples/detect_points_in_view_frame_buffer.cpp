#include <iostream>
#include <fstream>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
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
#include <glow/GlUniform.h>
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

struct PointInView {
    float x,y,z;
    float r,g,b;
    float u,v, isInView;
};
int main() {
    std::string image_file = "/home/pang/disk/dataset/kitti/00/image_0/000000.png";
    std::string lidarscan_file = "/home/pang/disk/dataset/kitti/00/velodyne/000000.bin";

    cv::Mat image = cv::imread(image_file, CV_LOAD_IMAGE_COLOR);
//    cv::imshow("image", image);
//    cv::waitKey(3000);
    std::vector<vec3> lidar_points = loadLidarPoints(lidarscan_file);


    uint32_t image_width = image.cols;
    uint32_t image_height = image.rows;
    std::cout << "lidar_points: " << lidar_points.size() << std::endl;
//    for (auto i : lidar_points) {
//        std::cout << i.x << " " << i.y << " " << i.z << std::endl;
//    }


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
    vec2 image_wh(image_width, image_height);


    // init window
    glow::X11OffscreenContext ctx(3,3);  // OpenGl context
    glow::inititializeGLEW();


    Timer gpu_timer;
    gpu_timer.start();


    // input
    cv::Mat float_image;
    image.convertTo(float_image, CV_32FC3);
    GlTexture input_texture{image_width, image_height, TextureFormat::RGBA_FLOAT};
    input_texture.assign(PixelFormat::RGB, PixelType::FLOAT, float_image.ptr());

    glow::GlBuffer<vec3> input_vec{glow::BufferTarget::ARRAY_BUFFER,
                                   glow::BufferUsage::DYNAMIC_DRAW};  // feedback stores updated input_vec inside input_vec.
    input_vec.assign(lidar_points);

    glow::GlVertexArray vao_input_vec;
    // now we can set the vertex attributes. (the "shallow copy" of input_vec now contains the correct id.
    vao_input_vec.setVertexAttribute(0, input_vec, 3, AttributeType::FLOAT, false, sizeof(vec3),
                                     reinterpret_cast<GLvoid*>(0));


    // output
    GlFramebuffer fbo(image_width, image_height);

//    ASSERT_NO_THROW(_CheckGlError(__FILE__, __LINE__));

    GlTexture output{image_width, image_height, TextureFormat::RGBA_FLOAT};
    GlRenderbuffer rbo(image_width, image_height, RenderbufferFormat::DEPTH_STENCIL);

    fbo.attach(FramebufferAttachment::COLOR0, output);
    CheckGlError();
    fbo.attach(FramebufferAttachment::DEPTH_STENCIL, rbo);
    CheckGlError();

    glow::GlProgram extractProgram;
    extractProgram.attach(GlShader::fromFile(ShaderType::VERTEX_SHADER, "/home/pang/suma_ws/src/glow/samples/shader/detect_in_view_frame_buffer.vert"));
    extractProgram.attach(GlShader::fromFile(ShaderType::FRAGMENT_SHADER, "/home/pang/suma_ws/src/glow/samples/shader/detect_in_view_frame_buffer.frag"));
    extractProgram.link();

    extractProgram.setUniform(GlUniform<int32_t>("input_texture", 0));

    extractProgram.setUniform(GlUniform<Eigen::Matrix4f>("T_cam_lidar", T_cam_lidar));
    extractProgram.setUniform(GlUniform<vec2>("image_wh", image_wh));
    extractProgram.setUniform(GlUniform<vec4>("intrinsic", intrinsic));


    GlSampler sampler;
    sampler.setMagnifyingOperation(TexMagOp::NEAREST);
    sampler.setMinifyingOperation(TexMinOp::NEAREST);



    extractProgram.bind();
    fbo.bind();
    sampler.bind(0);

    vao_input_vec.bind();
    glActiveTexture(GL_TEXTURE0);
    input_texture.bind();


    glDrawArrays(GL_POINTS, 0, input_vec.size());

    vao_input_vec.release();
    extractProgram.release();
    fbo.release();
    sampler.release(0);
    glActiveTexture(GL_TEXTURE0);
    input_texture.release();


    gpu_timer.stop();

    // retrieve result
    std::vector<vec4> data;
    output.download(data);

    cv::Mat out_image(image_height,image_width, CV_8UC3);
    for (int i = 0; i < image_height* image_width; i++) {
        int x = i % image_width;
        int y = i / image_width;
        out_image.at<cv::Vec3b>(y,x)[0] =   data[i].x ;
        out_image.at<cv::Vec3b>(y,x)[1] =   data[i].y ;
        out_image.at<cv::Vec3b>(y,x)[2] =   data[i].z ;
    }

    cv::imshow("out_image", out_image);
    cv::waitKey(10000);



    return 0;
}