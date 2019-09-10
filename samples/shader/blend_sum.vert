#version 330 core

layout (location = 0) in vec3 input_Cp;
layout (location = 1) in vec3 input_rgb;
layout (location = 2) in vec3 input_uv;

uniform sampler2D last_texture;

uniform mat4 T_cam_lidar;
uniform vec2 wh;
uniform vec4 intrinsic;

out Element
{
  bool valid;
  vec4 position;
  vec4 rgb;
  vec3 xyz;
  vec2 uv;
} vs_out;

out vec4 color;
void main()
{

  vec3 Cp = input_Cp.xyz;

  //    position_out = Cp;

  float inv_z = 1.0/ Cp.z;

  float u = intrinsic.x * Cp.x * inv_z + intrinsic.z;
  float v = intrinsic.y * Cp.y * inv_z + intrinsic.w;

  if ( Cp.z < 0 || u < 0 || u > wh.x || v < 0 || v > wh.y) {
    vs_out.valid = false;


  } else {
    vs_out.valid = true;
    float nu = 2.0f * (float(u + 0.5f) / float(wh.x)) - 1.0f;
    float nv = 2.0f * (float(v + 0.5f) / float(wh.y)) - 1.0f;
    gl_Position = vec4(nu, nv, 0, 1.0);
    vec2 normal_coords = vec2(2.0f * (float(u + 0.5f) / float(wh.x)) - 1.0f,
    2.0f * (float(v + 0.5f) / float(wh.y)) - 1.0f);
    vs_out.position = vec4(normal_coords, 0, 1.0);
    vec2 tex_coords = vec2(u/float(wh.x), (float(v) / float(wh.y)));
//    vs_out.rgb = texture(input_texture, tex_coords);

    vs_out.xyz = Cp;
    vs_out.uv = vec2(u,v);
//    color = vec4(input_rgb,1.0);

    color = texture(last_texture, tex_coords);
  }

}