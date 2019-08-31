#version 330 core

layout (location = 0) in vec4 position;
layout (location = 1) in vec4 color;
out vec4 coords;

uniform vec2 wh;
void main()
{
  vec2 normal_coords = vec2(2.0f * (float(position.x + 0.5f) / float(wh.x)) - 1.0f,
                     2.0f * (float(position.y + 0.5f) / float(wh.y)) - 1.0f);
  gl_Position = vec4(normal_coords, 0, 1.0);
  coords = color;
  
}