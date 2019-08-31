#version 330 core

layout(points) in;
layout(points, max_vertices = 1) out;

in Element
{
    bool valid;
    vec4 position;
    vec4 rgb;
} gs_in[];

out vec4 tex_color;

void main()
{
    if(gs_in[0].valid)
    {
        gl_Position = gs_in[0].position;
        tex_color = gs_in[0].rgb;


        EmitVertex();
        EndPrimitive();
    }
}