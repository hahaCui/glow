#version 330 core
layout (location = 0) in  vec4 position; // x, y, z, 1;
layout (location = 1) in vec4 texCoord;

out vec2 TexCoord;

void main()
{
    gl_Position = vec4(position);
    TexCoord = texCoord.rg;
}