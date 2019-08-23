#version 330 core

layout (location = 0) in vec4 position;
out vec2 coords;

void main()
{
  if (position.x > -0.5 && position.x < 0.5) {
    gl_Position = vec4(position.x,position.y, 0, 1.0);

  } else {
    gl_Position = vec4(0,0, 0, 1.0);

  }
  coords = vec2(position.z, position.w);
  
}