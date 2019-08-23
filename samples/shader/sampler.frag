#version 330 core

in vec2 coords;


uniform sampler2D input_texture;
out vec4 color;

void main()
{
  color = vec4(texture(input_texture, coords).rgb, 1);
}