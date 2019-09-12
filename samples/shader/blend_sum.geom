#version 330 core

layout(points) in;
layout(points, max_vertices = 1) out;

out CostInformation
{
  bool valid;
  vec3 C1p;
  vec3 C0p;

  vec2 uv1;
  vec2 uv0;
} gs_in[];

out vec4 tex_color;

const int half_patch_size = 1;
uniform mat4 T_L0_L1;
uniform mat4 T_Cam_Lidar;
uniform mat4 T_Lidar_Cam;

uniform vec4 intrinsic;

uniform sampler2D cur_texture;
uniform sampler2D last_texture;

out vec3 point_in_view_xyz;
out vec3 point_in_view_rgb;
out vec2 point_in_view_uv;

mat3 skew(vec3 xi)
{
  mat3 m = mat3(0.0, xi.z, -xi.y, -xi.z, 0.0, -xi.x, xi.y -xi.x, 0.0);

  return m;
}

void main()
{
  if(gs_in[0].valid)
  {

//    Eigen::Vector3d C_point_cur_ =
//depth_ref[i] * Eigen::Vector3d((px_ref[i][0] - cx) / fx, (px_ref[i][1] - cy) / fy, 1);
//Eigen::Vector3d L_point_cur = T_lidar_cam.topLeftCorner(3,3) * C_point_cur_ + T_lidar_cam.topRightCorner(3,1);
//Eigen::Vector3d L_point_last = T21.topLeftCorner(3,3) * L_point_cur + T21.topRightCorner(3,1);
//Eigen::Vector3d C_point_last = T_cam_lidar.topLeftCorner(3,3) * L_point_last + T_cam_lidar.topRightCorner(3,1);

    vec4 C1p_homo = vec4(gs_in[0].C1p, 1);
    vec4 L1p_homo = T_Lidar_Cam * C1p_homo;
    vec4 L0p_homo = T_L0_L1 * L1p_homo;
    vec4 C0p_homo = T_Cam_Lidar * L0p_homo;
    vec3 C0p = C0p_homo.xyz;

    float u = gs_in[0].uv0.x;
    float v = gs_in[0].uv0.y;


//    projection[i] = Eigen::Vector2d(u, v);
    float X = C0p.x;
    float Y = C0p.y;
    float Z = C0p.z;
    float Z2 = Z * Z;
    float Z_inv = 1.0 / Z;
    float Z2_inv = Z_inv * Z_inv;



      // todo: single pixel

//            double error = GetPixelValue(img1, px_ref[i][0] + x, px_ref[i][1] + y) -
//                          GetPixelValue(img2, u + x, v + y);
      vec2 tex_coords1 = vec2((gs_in[0].uv1.x + x)/wh.x, (gs_in[0].uv1.y + y)/wh.y);
      vec2 tex_coords0 = vec2((gs_in[0].uv0.x + x)/wh.x, (gs_in[0].uv0.y + y)/wh.h);
      float error = texture(cur_texture, tex_coords1) - texture(last_texture, tex_coords0);
//            Matrix26d J_pixel_xi;
//            Eigen::Matrix<double,2,3, Eigen::RowMajor> J_pixel_q;
//            Eigen::Matrix<double,3,6, Eigen::RowMajor> J_q_xi;

      // todo: check wether is column major
      mat3 aug_J_pixel_q = mat3(fx * Z_inv, 0, 0,
                               0, fy * Z_inv, 0,
                               -fx * X * Z2_inv, -fy * Y * Z2_inv, 0);

      mat3 R = mat3(T_Cam_Lidar);
      mat3 J_q_xi_translation = R;
      mat3 J_q_xi_rotation = R * skew(L0p_homo.xyz);



//
//            Eigen::Vector2d J_img_pixel;
//
//            J_pixel_q << fx * Z_inv, 0, -fx * X * Z2_inv,
//            0, fy * Z_inv, -fy * Y * Z2_inv;
//
//            J_q_xi << T_cam_lidar.topLeftCorner(3,3), - T_cam_lidar.topLeftCorner(3,3) * skew(L_point_last);
//
        float grad_x = texture(last_texture, vec2(gs_in[0].uv0.x + x, gs_in[0].uv0.y + y))
                     - texture(last_texture, vec2(gs_in[0].uv0.x + x, gs_in[0].uv0.y + y));
        float grad_y = 0.0;
//        vec2 J_img_pixel = vec2(
//        0.5 * (GetPixelValue(img2, u + 1 + x, v + y) - GetPixelValue(img2, u - 1 + x, v + y)),
//        0.5 * (GetPixelValue(img2, u + x, v + 1 + y) - GetPixelValue(img2, u + x, v - 1 + y))
//        );
//
//            // total jacobian
//            Vector6d J = -1.0 * (J_img_pixel.transpose() * J_pixel_q * J_q_xi).transpose();
//
//            hessian += J * J.transpose();
//            bias += -error * J;
//            cost_tmp += error * error;





    // todo  if valid
//    gl_Position = gs_in[0].position;
//    tex_color = gs_in[0].rgb;
//
//    point_in_view_xyz = gs_in[0].xyz;
//    point_in_view_rgb = gs_in[0].rgb.xyz;
//    point_in_view_uv = gs_in[0].uv;
//
//    EmitVertex();
//    EndPrimitive();
  }
}