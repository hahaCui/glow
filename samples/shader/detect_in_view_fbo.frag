#version 330 core

layout(location = 0) out vec4 Frag0;
layout(location = 1) out vec4 Frag1;

in vec4 coords;
//out vec4 color;

void main()
{
  Frag0 = coords;
  Frag1 = vec4(255 - coords.x, coords.yzw) ;
}