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
#include <opencv2/opencv.hpp>
#include <glow/GlSampler.h>

#include <glow/GlCapabilities.h>
#include <glow/GlState.h>
#include <glow/util/X11OffscreenContext.h>

#include "timer.h"
using namespace glow;



struct Point3f
{
public:
    // Point3f Public Methods
    Point3f()
            : vec(0.f, 0.f, 0.f, 1.f) //, x(vec[0]), y(vec[1]), z(vec[2])
    {

    }

    Point3f(float xx, float yy, float zz)
            : vec(xx, yy, zz, 1.f) //, x(vec[0]), y(vec[1]), z(vec[2])
    {
        assert(!HasNaNs());
    }

    Point3f(const Point3f& p)
            : vec(p.vec[0], p.vec[1], p.vec[2], 1.0f) //, x(vec[0]), y(vec[1]), z(vec[2])
    {
        assert(!p.HasNaNs());
    }


    inline const float& x() const
    {
        return vec[0];
    }

    inline float& x()
    {
        return vec[0];
    }

    inline const float& y() const
    {
        return vec[1];
    }

    inline float& y()
    {
        return vec[1];
    }

    inline const float& z() const
    {
        return vec[2];
    }

    inline float& z()
    {
        return vec[2];
    }

    bool HasNaNs() const
    {
        return std::isnan(vec[0]) || std::isnan(vec[1]) || std::isnan(vec[2]);
    }

    bool operator==(const Point3f &p) const
    {

        return vec == p.vec;
    }

    bool operator!=(const Point3f &p) const
    {
        return vec != p.vec;
    }

    friend std::ostream& operator<<(std::ostream& out, const Point3f& p)
    {
        out.width(4);
        out.precision(3);
        out << p.vec[0] << ", " << p.vec[1] << ", " << p.vec[2];
        return out;
    }
    // Point3f Public Data
    Eigen::Vector4f vec;
//    float& x, &y, &z;
};



int main(int argc, char** argv) {
    // init window
    glow::X11OffscreenContext ctx(3,3);  // OpenGl context
    glow::inititializeGLEW();

    float convolution_radius = 5;
    _CheckGlError(__FILE__, __LINE__);
    //  std::cout << "On entry: " << GlState::queryAll() << std::endl;
    std::string image_file = "/home/pang/Documents/lenna.png";
    cv::Mat image = cv::imread(image_file, CV_LOAD_IMAGE_COLOR);
    uint32_t width = image.cols, height = image.rows;
    std::cout << "image: " << width << " " << height << std::endl;

    std::vector<float> values(3 * width * height, 0);
    for(auto i = 0 ; i < image.rows; i++)
        for(auto j = 0; j < image.cols; j++) {
            float r = image.at<cv::Vec3b>(i,j)[0];
            float g = image.at<cv::Vec3b>(i,j)[1];
            float b = image.at<cv::Vec3b>(i,j)[2];

            values[3 * (i * width +j)] = (float)r;
            values[3 * (i * width +j) + 1] = (float)g;
            values[3 * (i * width +j) + 2] = (float)b;
        }


    glow::GlBuffer<Point3f> points{glow::BufferTarget::ARRAY_BUFFER, glow::BufferUsage::DYNAMIC_READ};
    std::vector<Point3f> vertices;
    vertices.push_back(Point3f(0.5f,  0.5f, 0.0f));
    vertices.push_back(Point3f(0.5f, -0.5f, 0.0f));
    vertices.push_back(Point3f(-0.5f, -0.5f, 0.0f));
    vertices.push_back(Point3f(-0.5f,  0.5f, 0.0f));
    points.assign(vertices);
    glow::GlVertexArray vao_points;
    vao_points.setVertexAttribute(0, points, 4, AttributeType::FLOAT, false, 4 * sizeof(float), nullptr);
    vao_points.enableVertexAttribute(0);



    glow::GlBuffer<Point3f> tex_coord{glow::BufferTarget::ARRAY_BUFFER, glow::BufferUsage::DYNAMIC_READ};
    std::vector<Point3f> tex_coord_vec;
    tex_coord_vec.push_back(Point3f(1.0f, 1.0f, 0.0f));
    tex_coord_vec.push_back(Point3f(1.0f, 0.0f, 0.0f));
    tex_coord_vec.push_back(Point3f(0.0f, 0.0f, 0.0f));
    tex_coord_vec.push_back(Point3f(0.0f, 1.0f, 0.0f));
    tex_coord.assign(tex_coord_vec);
    glow::GlVertexArray vao_tex_coord;
    vao_tex_coord.setVertexAttribute(0, tex_coord, 4, AttributeType::FLOAT, false, 4 * sizeof(float), nullptr);
    vao_tex_coord.enableVertexAttribute(1);




    GlFramebuffer fbo(width, height);

    _CheckGlError(__FILE__, __LINE__);

    GlTexture input{width, height, TextureFormat::RGB_FLOAT};

    input.assign(PixelFormat::RGB, PixelType::FLOAT, &values[0]);

    GlTexture output{width, height, TextureFormat::RGB_FLOAT};
    GlRenderbuffer rbo(width, height, RenderbufferFormat::DEPTH_STENCIL);

    CheckGlError();
    fbo.attach(FramebufferAttachment::COLOR0, output);
    fbo.attach(FramebufferAttachment::DEPTH_STENCIL, rbo);
    CheckGlError();



    GlProgram program;
    program.attach(GlShader::fromFile(ShaderType::VERTEX_SHADER, "/home/pang/suma_ws/src/glow/samples/shader/scale.vert"));
    program.attach(GlShader::fromFile(ShaderType::FRAGMENT_SHADER, "/home/pang/suma_ws/src/glow/samples/shader/texture.frag"));
    program.link();

    GlSampler sampler;
    sampler.setMagnifyingOperation(TexMagOp::NEAREST);
    sampler.setMinifyingOperation(TexMinOp::NEAREST);

//    GlVertexArray vao_no_points;

    fbo.bind();
    vao_points.bind();
    vao_tex_coord.bind();
    glActiveTexture(GL_TEXTURE0);
    input.bind();
    program.bind();

    sampler.bind(0);

    glDisable(GL_DEPTH_TEST);
    glViewport(0, 0, width, height);

    glDrawArrays(GL_POINTS, 0, 1);

    program.release();
    input.release();
    vao_tex_coord.release();
    vao_points.release();
    fbo.release();

    sampler.release(0);

    glEnable(GL_DEPTH_TEST);

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

    cv::imshow("image", image);
    cv::imshow("out_image", out_image);
    cv::waitKey(10000);

    return 0;
}