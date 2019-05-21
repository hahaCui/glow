#version 330 core

layout (location = 0) in vec4 position;
layout (location = 1) in vec4 color;
out vec4 coords;

out int gray;

void main()
{
  gl_Position = vec4(position.x, position.y, 0, 1.0);
  coords = color;
  float r = color.x;
  float g = color.y;
  float b = color.z;

  gray = int((r *299 + g*587 + b*114 + 500) / 1000);
}