#version 330 core

layout(location = 0) out vec4 Frag0;
layout(location = 1) out vec4 Frag1;

in vec4 tex_color;
//out vec4 color;

void main()
{
  Frag0 = vec4(tex_color.x,0,0,0);
  Frag1 = vec4(tex_color.x, 0,0,0) ;
}