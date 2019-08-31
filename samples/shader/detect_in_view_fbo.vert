#version 330 core

layout (location = 0) in vec4 position;
//layout (location = 1) in vec4 color;
out vec4 coords;
uniform sampler2D input_texture;
uniform vec2 wh;
uniform mat4 T_cam_lidar;
uniform vec4 intrinsic;

out Element
{
  bool valid;
  vec4 position;
  vec4 rgb;
} vs_out;



void main()
{
  float u = position.x;
  float v = position.y;
  if (u >0 && u < wh.x && v > 0 && v < wh.y) {
    vs_out.valid = true;

    vec2 normal_coords = vec2(2.0f * (float(u + 0.5f) / float(wh.x)) - 1.0f,
                              2.0f * (float(v + 0.5f) / float(wh.y)) - 1.0f);
    vs_out.position = vec4(normal_coords, 0, 1.0);
    vec2 tex_coords = vec2(u/float(wh.x), (float(v) / float(wh.y)));
    vs_out.rgb = texture(input_texture, tex_coords);
  } else {
    vs_out.valid = false;
  }
  
}