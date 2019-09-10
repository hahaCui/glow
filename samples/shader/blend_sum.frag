#version 330

in vec4 color;
out vec4 point_in_view_rgbd;

void main()
{
  point_in_view_rgbd = color;
}