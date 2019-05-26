#version 330 core

layout (location = 0) in vec4 position;


out vec4 position_out;

void main()
{
    position_out = position;
}