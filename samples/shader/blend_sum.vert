#version 330 core

layout (location = 0) in vec3 input_C1p;
layout (location = 1) in vec3 input_rgb1;
layout (location = 2) in vec2 input_uv1;

uniform sampler2D last_texture;

uniform mat4 T_C0_C1;
uniform mat4 T_Cam_Lidar;
uniform vec2 wh;
uniform vec4 intrinsic;

out CostInformation
{
  bool valid;
  vec3 C1p;
  vec3 C0p;
  vec3 rgb1;
  vec3 rgb0;
  vec2 uv1;
  vec2 uv0;
} vs_out;

out vec4 color;

void main()
{

  vec4 C1p_homo = vec4(input_C1p, 1);
  vec4 C0p_homo = T_C0_C1* C1p_homo;
  vec3 C0p = C0p_homo.xyz;

  //    position_out = Cp;

  float inv_z = 1.0/ C0p.z;

  float u = intrinsic.x * C0p.x * inv_z + intrinsic.z;
  float v = intrinsic.y * C0p.y * inv_z + intrinsic.w;

  if ( C0p.z < 0 || u < 0 || u > wh.x || v < 0 || v > wh.y) {
    vs_out.valid = false;


  } else {
    vs_out.valid = true;
    float nu = 2.0f * (float(u + 0.5f) / float(wh.x)) - 1.0f;
    float nv = 2.0f * (float(v + 0.5f) / float(wh.y)) - 1.0f;
    gl_Position = vec4(nu, nv, 0, 1.0);


//    vec2 normal_coords = vec2(2.0f * (float(u + 0.5f) / float(wh.x)) - 1.0f,
//    2.0f * (float(v + 0.5f) / float(wh.y)) - 1.0f);
//    vs_out.position = vec4(normal_coords, 0, 1.0);
    vec2 tex_coords = vec2(u/float(wh.x), (float(v) / float(wh.y)));
////    vs_out.rgb = texture(input_texture, tex_coords);
//
//    vs_out.xyz = C0p;
//    vs_out.uv = vec2(u,v);
//    color = vec4(input_rgb,1.0);

    vs_out.C1p = C1p_homo.xyz;
    vs_out.C0p = C0p.xyz;
    vs_out.rgb1 = input_rgb1;
    vs_out.rgb0 = texture(last_texture, tex_coords).xyz;
    vs_out.uv1 = input_uv1;
    vs_out.uv0 = vec2(u,v);

    color = texture(last_texture, tex_coords);
  }

}