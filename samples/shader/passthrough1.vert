#version 330 core

layout (location = 0) in vec4 position;

void main()
{
  gl_Position = vec4(position.x, position.y, 0, 1.0);
}