#version 330 core

layout(points) in;
layout(points, max_vertices = 1) out;

in Element
{
    bool valid;
    vec4 position;
    vec4 rgb;
    vec3 xyz;
    vec2 uv;
} gs_in[];

out vec4 tex_color;


out vec3 point_in_view_xyz;
out vec3 point_in_view_rgb;
out vec2 point_in_view_uv;

void main()
{
    if(gs_in[0].valid)
    {
        gl_Position = gs_in[0].position;
        tex_color = gs_in[0].rgb;

        point_in_view_xyz = gs_in[0].xyz;
        point_in_view_rgb = gs_in[0].rgb.xyz;
        point_in_view_uv = gs_in[0].uv;

        EmitVertex();
        EndPrimitive();
    }
}