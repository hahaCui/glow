#version 330 core

layout (location = 0) in vec4 position;
//layout (location = 1) in vec4 color;
out vec4 coords;
uniform sampler2D input_texture;
uniform vec2 wh;


out Element
{
  bool valid;
  vec4 position;
  vec4 rgb;
} vs_out;



void main()
{
  if (position.x >0 && position.x < wh.x && position.y > 0 && position.y < wh.y) {
    vs_out.valid = true;

    vec2 normal_coords = vec2(2.0f * (float(position.x + 0.5f) / float(wh.x)) - 1.0f,
    2.0f * (float(position.y + 0.5f) / float(wh.y)) - 1.0f);
    vs_out.position = vec4(normal_coords, 0, 1.0);
    vec2 tex_coords = vec2(position.x/float(wh.x), (float(position.y) / float(wh.y)));
    vs_out.rgb = texture(input_texture, tex_coords);
  } else {
    vs_out.valid = false;

    vec2 normal_coords = vec2(2.0f * (float(position.x + 0.5f) / float(wh.x)) - 1.0f,
    2.0f * (float(position.y + 0.5f) / float(wh.y)) - 1.0f);
    vs_out.position = vec4(normal_coords, 0, 1.0);
    vec2 tex_coords = vec2(position.x/float(wh.x), (float(position.y) / float(wh.y)));
    vs_out.rgb = texture(input_texture, tex_coords);
  }
  
}