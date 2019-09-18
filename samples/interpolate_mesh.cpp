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
#include "triangle/triangle.h"

#include "timer.h"

using namespace glow;


std::vector<vec4> loadLidarPoints(const std::string& bin_file ) {
    // load point cloud
    std::fstream input(bin_file, std::ios::in | std::ios::binary);
    if(!input.good()){
        std::cerr << "Could not read file: " << bin_file << std::endl;
        exit(EXIT_FAILURE);
    }
    input.seekg(0, std::ios::beg);

    std::vector<vec4> points;

    int i;
    for (i=0; input.good() && !input.eof(); i++) {
        vec4 pt;
        float intensity;
        input.read((char *) &pt.x, 3*sizeof(float));
        input.read((char *) &intensity, sizeof(float));
        pt.w = 1;
        points.push_back(pt);
    }
    input.close();
//    std::cout << "Read KTTI point cloud with " << i << " points" << std::endl;
    return points;
}

struct PointInView {
    float x,y,z;
    float r,g,b;
    float u,v;
};

inline void bilinearWeights(float x, float y, float* w00, float* w01,
                            float* w10, float* w11) {
    int x_floor = static_cast<int>(x);
    int y_floor = static_cast<int>(y);

    float dx = x - x_floor;
    float dy = y - y_floor;

    /* Compute rectangles using only 1 multiply (taken from LSD-SLAM). */
    *w11 = dx * dy;
    *w01 = dx - *w11;
    *w10 = dy - *w11;
    *w00 = 1.0f - dx - dy + *w11;

    return;
}


template <typename ChannelType, typename RetType>
inline RetType bilinearInterp(uint32_t rows, uint32_t cols, std::size_t step,
                              const void* data, float x, float y) {


    int x_floor = static_cast<int>(x);
    int y_floor = static_cast<int>(y);

    float w00, w01, w10, w11;
    bilinearWeights(x, y, &w00, &w01, &w10, &w11);

    const uint8_t* datab =
            &(static_cast<const uint8_t*>(data))[y_floor * step + x_floor * sizeof(ChannelType)];

    return w00 * (*reinterpret_cast<const ChannelType*>(datab)) +
           w01 * (*reinterpret_cast<const ChannelType*>(datab + sizeof(ChannelType))) +
           w10 * (*reinterpret_cast<const ChannelType*>(datab + step)) +
           w11 * (*reinterpret_cast<const ChannelType*>(datab + sizeof(ChannelType) + step));
}



template <typename ChannelType, typename RetType>
inline RetType bilinearInterp(const cv::Mat& img, float x, float y) {
    return bilinearInterp<ChannelType, RetType>(img.rows, img.cols, img.step,
                                                img.data, x, y);
}



inline int getPointsInCameraView(const std::vector<vec4>& cloud,
                                 const cv::Mat& image,
                                 const Eigen::Matrix4d& T_cam_lidar,
                                 const Eigen::Matrix<double,3,4>& camera_intrinsic,
                                 std::vector<vec4>& in_view_cloud,
                                 std::vector<Eigen::Vector3d>& uv_with_depth) {


    Eigen::Matrix<double,3,4> project_matix = camera_intrinsic*T_cam_lidar;

    uv_with_depth.clear();
    int in_view_cnt = 0;
    for (int i = 0; i < cloud.size(); i++) {
        auto  p3d = cloud.at(i);
        Eigen::Vector4d p4d(p3d.x, p3d.y,p3d.z,1.0);
        Eigen::Vector4d p4d_in_camera_frame = T_cam_lidar* p4d;
        Eigen::Vector3d p3d_in_camera = project_matix*p4d;
        if (p3d_in_camera(2) < 5 || p3d_in_camera(2) > 70) continue;

        Eigen::Vector3d uv(p3d_in_camera(0)/p3d_in_camera(2),
                           p3d_in_camera(1)/p3d_in_camera(2), p3d_in_camera(2));

        if (uv(0) >=0 && uv(0) <= image.cols && uv(1) >=0 && uv(1) <= image.rows) {
            in_view_cnt ++;
            vec4 pointXyzi;
            pointXyzi.x = (float)p4d_in_camera_frame[0];
            pointXyzi.y = (float)p4d_in_camera_frame[1];
            pointXyzi.z = (float)p4d_in_camera_frame[2];

            auto intensity = bilinearInterp<uchar,float>(image, float(uv(0)),
                                                         float(uv(1)));
            pointXyzi.w = intensity;

            uv_with_depth.push_back(uv);

            in_view_cloud.push_back(pointXyzi);
        }
    }

    return in_view_cnt;
}

using Vertex = cv::Point2f;
using Triangle = cv::Vec3i;
using Edge = cv::Vec2i;

/**
 * \brief Class that implements Delaunay triangulation.
 */
class Delaunay  {
public:
    Delaunay() = default;
    ~Delaunay() = default;

    Delaunay(const Delaunay& rhs) = delete;
    Delaunay& operator=(const Delaunay& rhs) = delete;

    Delaunay(Delaunay&& rhs) = default;
    Delaunay& operator=(Delaunay&& rhs) = default;


    void triangulate(const std::vector<Vertex>& support,
                     std::vector<Triangle>* triangles) {

        // input/output structure for triangulation
        struct triangulateio in;
        int32_t k;

        // inputs
        in.numberofpoints = support.size();
        in.pointlist = (float*)malloc(in.numberofpoints*2*sizeof(float)); // NOLINT
        k = 0;
        for (int32_t i = 0; i < support.size(); i++) {
            in.pointlist[k++] = support[i].x;
            in.pointlist[k++] = support[i].y;
        }
        in.numberofpointattributes = 0;
        in.pointattributelist      = NULL;
        in.pointmarkerlist         = NULL;
        in.numberofsegments        = 0;
        in.numberofholes           = 0;
        in.numberofregions         = 0;
        in.regionlist              = NULL;

        // outputs
        out_.pointlist              = NULL;
        out_.pointattributelist     = NULL;
        out_.pointmarkerlist        = NULL;
        out_.trianglelist           = NULL;
        out_.triangleattributelist  = NULL;
        out_.neighborlist           = NULL;
        out_.segmentlist            = NULL;
        out_.segmentmarkerlist      = NULL;
        out_.edgelist               = NULL;
        out_.edgemarkerlist         = NULL;

        // do triangulation (z=zero-based, n=neighbors, Q=quiet, B=no boundary markers)
        char parameters[] = "zneQB";
        ::triangulate(parameters, &in, &out_, NULL);
        free(in.pointlist);

        getTriangles(triangles);
        getNeighbors();
        getEdges();
        cleanup();

        return;
    }

    void triangulate(const std::vector<Vertex>& vertices) {
        triangulate(vertices, &triangles_);
        return;
    }

    void cleanup() {
        // free memory used for triangulation
        free(out_.pointlist);
        free(out_.trianglelist);
        free(out_.edgelist);
        free(out_.neighborlist);

        out_.pointlist = NULL;
        out_.trianglelist = NULL;
        out_.edgelist = NULL;
        out_.neighborlist = NULL;

        return;
    }

    void getTriangles(std::vector<Triangle>* triangles) {
        // put resulting triangles into vector tri
        triangles->resize(out_.numberoftriangles);
        int k = 0;
        for (int32_t i = 0; i < out_.numberoftriangles; i++) {
            (*triangles)[i] = Triangle(out_.trianglelist[k],
                                       out_.trianglelist[k+1],
                                       out_.trianglelist[k+2]);
            k+=3;
        }
        return;
    }

    void getNeighbors() {
        // put neighboring triangles into vector tri
        neighbors_.resize(out_.numberoftriangles);
        int k = 0;
        for (int32_t i = 0; i < out_.numberoftriangles; i++) {
            neighbors_[i] = Triangle(out_.neighborlist[k],
                                     out_.neighborlist[k+1],
                                     out_.neighborlist[k+2]);
            k+=3;
        }
        return;
    }

    void getEdges()  {
        // put resulting edges into vector
        edges_.resize(out_.numberofedges);
        int k = 0;
        for (int32_t i = 0; i < out_.numberofedges; i++) {
            edges_[i] = Edge(out_.edgelist[k], out_.edgelist[k+1]);
            k+=2;
        }
        return;
    }

    // Accessors.
    const std::vector<Triangle>& triangles() const { return triangles_; }
    const std::vector<Edge>& edges() const { return edges_; }
    const std::vector<Triangle>& neighbors() const { return neighbors_; }

private:


    struct triangulateio out_;

    std::vector<Triangle> triangles_;
    std::vector<Triangle> neighbors_;
    std::vector<Edge> edges_;
};


namespace utils {

    inline int min3(int x, int y, int z) {
        return x < y ? (x < z ? x : z) : (y < z ? y : z);
    }

    inline int max3(int x, int y, int z) {
        return x > y ? (x > z ? x : z) : (y > z ? y : z);
    }

    struct Edge {
        static const int stepXSize = 4;
        static const int stepYSize = 1;

        // __m128 is the SSE 128-bit packed float type (4 floats).
        __m128 oneStepX;
        __m128 oneStepY;

        __m128 init(const cv::Point &v0, const cv::Point &v1,
                    const cv::Point &origin) {
            // Edge setup
            float A = v1.y - v0.y;
            float B = v0.x - v1.x;
            float C = v1.x * v0.y - v0.x * v1.y;

            // Step deltas
            // __m128i y = _mm_set1_ps(x) sets y[0..3] = x.
            oneStepX = _mm_set1_ps(A * stepXSize);
            oneStepY = _mm_set1_ps(B * stepYSize);

            // x/y values for initial pixel block
            // NOTE: Set operations have arguments in reverse order!
            // __m128 y = _mm_set_epi32(x3, x2, x1, x0) sets y0 = x0, etc.
            __m128 x = _mm_set_ps(origin.x + 3, origin.x + 2, origin.x + 1, origin.x);
            __m128 y = _mm_set1_ps(origin.y);

            // Edge function values at origin
            // A*x + B*y + C.
            __m128 A4 = _mm_set1_ps(A);
            __m128 B4 = _mm_set1_ps(B);
            __m128 C4 = _mm_set1_ps(C);

            return _mm_add_ps(_mm_add_ps(_mm_mul_ps(A4, x), _mm_mul_ps(B4, y)), C4);
        }
    };

    void DrawShadedTriangleBarycentric(cv::Point p1, cv::Point p2, cv::Point p3,
                                       float v1, float v2, float v3, cv::Mat *img) {
        // Compute triangle bounding box
        int xmin = min3(p1.x, p2.x, p3.x);
        int ymin = min3(p1.y, p2.y, p3.y);
        int xmax = max3(p1.x, p2.x, p3.x);
        int ymax = max3(p1.y, p2.y, p3.y);

        cv::Point p(xmin, ymin);
        Edge e12, e23, e31;

        // __m128 is the SSE 128-bit packed float type (4 floats).
        __m128 w1_row = e23.init(p2, p3, p);
        __m128 w2_row = e31.init(p3, p1, p);
        __m128 w3_row = e12.init(p1, p2, p);

        // Values as 4 packed floats.
        __m128 v14 = _mm_set1_ps(v1);
        __m128 v24 = _mm_set1_ps(v2);
        __m128 v34 = _mm_set1_ps(v3);

        // Rasterize
        for (p.y = ymin; p.y <= ymax; p.y += Edge::stepYSize) {
            // Determine barycentric coordinates
            __m128 w1 = w1_row;
            __m128 w2 = w2_row;
            __m128 w3 = w3_row;

            for (p.x = xmin; p.x <= xmax; p.x += Edge::stepXSize) {
                // If p is on or inside all edges, render pixel.
                __m128 zero = _mm_set1_ps(0.0f);

                // (w1 >= 0) && (w2 >= 0) && (w3 >= 0)
                // mask tells whether we should set the pixel.
                __m128 mask = _mm_and_ps(_mm_cmpge_ps(w1, zero),
                                         _mm_and_ps(_mm_cmpge_ps(w2, zero),
                                                    _mm_cmpge_ps(w3, zero)));

                // w1 + w2 + w3
                __m128 norm = _mm_add_ps(w1, _mm_add_ps(w2, w3));

                // v1*w1 + v2*w2 + v3*w3 / norm
                __m128 vals = _mm_div_ps(_mm_add_ps(_mm_mul_ps(v14, w1),
                                                    _mm_add_ps(_mm_mul_ps(v24, w2),
                                                               _mm_mul_ps(v34, w3))), norm);

                // Grab original data.  We need to use different store/load functions if
                // the address is not aligned to 16-bytes.
                uint32_t addr = sizeof(float) * (p.y * img->cols + p.x);
                if (addr % 16 == 0) {
                    float *img_ptr = reinterpret_cast<float *>(&(img->data[addr]));
                    __m128 data = _mm_load_ps(img_ptr);

                    // Set values using mask.
                    // If mask is true, use vals, otherwise use data.
                    __m128 res = _mm_or_ps(_mm_and_ps(mask, vals), _mm_andnot_ps(mask, data));
                    _mm_store_ps(img_ptr, res);
                } else {
                    // Address is not 16-byte aligned. Need to use special functions to load/store.
                    float *img_ptr = reinterpret_cast<float *>(&(img->data[addr]));
                    __m128 data = _mm_loadu_ps(img_ptr);

                    // Set values using mask.
                    // If mask is true, use vals, otherwise use data.
                    __m128 res = _mm_or_ps(_mm_and_ps(mask, vals), _mm_andnot_ps(mask, data));
                    _mm_storeu_ps(img_ptr, res);
                }

                // One step to the right.
                w1 = _mm_add_ps(w1, e23.oneStepX);
                w2 = _mm_add_ps(w2, e31.oneStepX);
                w3 = _mm_add_ps(w3, e12.oneStepX);
            }

            // Row step.
            w1_row = _mm_add_ps(w1_row, e23.oneStepY);
            w2_row = _mm_add_ps(w2_row, e31.oneStepY);
            w3_row = _mm_add_ps(w3_row, e12.oneStepY);
        }

        return;
    }

}
void interpolateMesh(const std::vector<Triangle>& triangles,
                     const std::vector<cv::Point2f>& vertices,
                     const std::vector<float>& values,
                     const std::vector<bool>& vtx_validity,
                     const std::vector<bool>& tri_validity,
                     cv::Mat* img) {
    for (int ii = 0; ii < triangles.size(); ++ii) {
        if (tri_validity[ii] &&
            vtx_validity[triangles[ii][0]] && vtx_validity[triangles[ii][1]] &&
            vtx_validity[triangles[ii][2]]) {
            // Triangle spits out points in clockwise order, but drawing function
            // expects CCW.
            utils::DrawShadedTriangleBarycentric(vertices[triangles[ii][2]],
                                                 vertices[triangles[ii][1]],
                                                 vertices[triangles[ii][0]],
                                                 values[triangles[ii][2]],
                                                 values[triangles[ii][1]],
                                                 values[triangles[ii][0]],
                                                 img);
        }
    }

    return;
}
// helper to check and display for shader linker error
bool check_program_link_status(GLuint obj) {
    GLint status;
    glGetProgramiv(obj, GL_LINK_STATUS, &status);
    if(status == GL_FALSE) {
        GLint length;
        glGetProgramiv(obj, GL_INFO_LOG_LENGTH, &length);
        std::vector<char> log(length);
        glGetProgramInfoLog(obj, length, &length, &log[0]);
        std::cerr << &log[0];
        return false;
    }
    return true;
}
int main() {
    std::string image_file = "/home/pang/data/dataset/kitti/00/image_0/000001.png";
    std::string lidarscan_file = "/home/pang/data/dataset/kitti/00/velodyne/000001.bin";

    cv::Mat image = cv::imread(image_file, CV_LOAD_IMAGE_COLOR);
//    cv::imshow("image", image);
//    cv::waitKey(3000);
    std::vector<vec4> lidar_points = loadLidarPoints(lidarscan_file);


    uint32_t image_width = image.cols;
    uint32_t image_height = image.rows;
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
    vec2 image_wh(image_width, image_height);
    Eigen::Matrix<double,3,4,Eigen::RowMajor> intrinsic_mat;
    intrinsic_mat << 7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, 0.000000000000e+00,
            0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 0.000000000000e+00,
            0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00;

    std::vector<vec4> in_view_cloud;
    std::vector<Eigen::Vector3d> uv_with_depth;
    cv::Mat gray_image;
    cv::cvtColor(image, gray_image, CV_RGB2GRAY);
    auto cnt = getPointsInCameraView(lidar_points,
                                     gray_image,
                                     T_cam_lidar.cast<double>(),
                                     intrinsic_mat,
                                     in_view_cloud,
                                     uv_with_depth);


    std::cout << "lidar_points: " << lidar_points.size() << std::endl;
    std::cout << "in view points: " << in_view_cloud.size() << std::endl;




    std::vector<Vertex> support;

    std::vector<float> values;
    cv::Mat show_image(gray_image.rows, gray_image.cols, CV_8UC1);
    for (int i = 0; i < in_view_cloud.size(); i++) {
        int u = uv_with_depth.at(i)(0);
        int v = uv_with_depth.at(i)(1);
        show_image.at<uchar >(v, u) = in_view_cloud.at(i).w;

        support.push_back(cv::Point2f(u,v));
        values.push_back(in_view_cloud.at(i).w);
    }




    Timer delaunay_timer;
    delaunay_timer.start();
    std::vector<Triangle> triangles;
    Delaunay delaunay;
    delaunay.triangulate(support, &triangles);
    delaunay_timer.stop();

    std::cout << "delaunay: " << delaunay_timer.elapsedMilliseconds() << std::endl;

//    for (auto trs: triangles) {
//        std::cout << trs[0] << " " << trs[1] << " " << trs[2] << std::endl;
//    }



    // interpolate mesh
    const std::vector<bool> vtx_validity(support.size(), true);
    const std::vector<bool> tri_validity(triangles.size(), true);


    cv::Mat1f interpolated_image(image.rows, image.cols, CV_32FC1);
    interpolateMesh(triangles,
                     support,
                     values,
                     vtx_validity,
                      tri_validity,
                    &interpolated_image);



    cv::Mat1b interpolated_image_char(interpolated_image.rows, interpolated_image.cols, CV_8UC1);
    interpolated_image.convertTo(interpolated_image_char, CV_8UC1);


//    cv::imshow("show_image", show_image);
//    cv::imshow("interpolated_image_char", interpolated_image_char);
//    cv::waitKey(2000000);


    // init window
    glow::X11OffscreenContext ctx(3,3);  // OpenGl context
    glow::inititializeGLEW();


    int width = image.cols;
    int height = image.rows;

    // must constaint this line, otherwise, there will be a linking error
    GlFramebuffer fbo1(width, height);

    // shader source code
    std::string vertex_source =
            "#version 330\n"
            "layout(location = 0) in vec4 vposition;\n"
            "layout(location = 1) in vec4 vcolor;\n"
            "out vec4 fcolor;\n"
            "void main() {\n"
            "   fcolor = vcolor;\n"
            "   gl_Position = vposition;\n"
            "}\n";

    std::string fragment_source =
            "#version 330\n"
            "in vec4 fcolor;\n"
            "layout(location = 0) out vec4 FragColor;\n"
            "void main() {\n"
            "   FragColor = fcolor;\n"
            "}\n";

    // program and shader handles
    GLuint shader_program, vertex_shader, fragment_shader;

    // we need these to properly pass the strings
    const char *source;
    int length;

    // create and compiler vertex shader
    vertex_shader = glCreateShader(GL_VERTEX_SHADER);
    source = vertex_source.c_str();
    length = vertex_source.size();
    glShaderSource(vertex_shader, 1, &source, &length);
    glCompileShader(vertex_shader);
//    if(!check_shader_compile_status(vertex_shader)) {
//
//        return 1;
//    }

    // create and compiler fragment shader
    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
    source = fragment_source.c_str();
    length = fragment_source.size();
    glShaderSource(fragment_shader, 1, &source, &length);
    glCompileShader(fragment_shader);
//    if(!check_shader_compile_status(fragment_shader)) {
//
//        return 1;
//    }

    // create program
    shader_program = glCreateProgram();

    // attach shaders
    glAttachShader(shader_program, vertex_shader);
    glAttachShader(shader_program, fragment_shader);

    // link the program and check for errors
    glLinkProgram(shader_program);
    check_program_link_status(shader_program);

    // vao and vbo handle
    GLuint vao, vbo, ibo;

    // generate and bind the vao
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);

    // generate and bind the vertex buffer object
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    // data for a fullscreen quad
    GLfloat vertexData[] = {
            //  X     Y     Z           R     G     B
            0.5f, 0.5f, 0.0f,       1.0f, 0.0f, 0.0f, // vertex 0
            -0.5f, 0.5f, 0.0f,       0.0f, 1.0f, 0.0f, // vertex 1
            0.5f,-0.5f, 0.0f,       0.0f, 0.0f, 1.0f, // vertex 2
            -0.5f,-0.5f, 0.0f,       1.0f, 0.0f, 0.0f, // vertex 3
    }; // 4 vertices with 6 components (floats) each

    // fill with data
    glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat)*4*6, vertexData, GL_STATIC_DRAW);


    // set up generic attrib pointers
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6*sizeof(GLfloat), (char*)0 + 0*sizeof(GLfloat));

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6*sizeof(GLfloat), (char*)0 + 3*sizeof(GLfloat));


    // generate and bind the index buffer object
    glGenBuffers(1, &ibo);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);

    GLuint indexData[] = {
            0,1,2, // first triangle
            2,1,3, // second triangle
    };

    // fill with data
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint)*2*3, indexData, GL_STATIC_DRAW);


    // texture handle
    GLuint texture;

    // generate texture
    glGenTextures(1, &texture);

    // bind the texture
    glBindTexture(GL_TEXTURE_2D, texture);

    // set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    // set texture content
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);


    // framebuffer handle
    GLuint fbo;

    // generate framebuffer
    glGenFramebuffers(1, &fbo);

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);





    // clear first
    glClear(GL_COLOR_BUFFER_BIT);

    // use the shader program
    glUseProgram(shader_program);

    // bind the vao
    glBindVertexArray(vao);

    // draw
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);


    std::vector<unsigned char> data;
    //output.download(data);
    glBindTexture(GL_TEXTURE_2D, texture);
    data.resize(4 * width * height);
    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, reinterpret_cast<GLvoid*>(&data[0]));
    cv::Mat gpu_image(height, width, CV_8UC4, &data[0]);
    cv::imshow("gpu_image", gpu_image);
    cv::waitKey(2000);



    // check for errors
    GLenum error = glGetError();

    // delete the created objects
    glDeleteVertexArrays(1, &vao);
    glDeleteBuffers(1, &vbo);
    glDeleteBuffers(1, &ibo);

    glDetachShader(shader_program, vertex_shader);
    glDetachShader(shader_program, fragment_shader);
    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);
    glDeleteProgram(shader_program);






    return 0;
}