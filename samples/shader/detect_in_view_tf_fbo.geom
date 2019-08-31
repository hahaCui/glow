#version 330 core

layout(points) in;
layout(points, max_vertices = 1) out;

uniform vec2 image_wh;
in Element
{
    bool valid;
    vec3 xyz;
    vec3 rgb;
    vec2 uv;
} gs_in[];

out vec3 point_in_view_xyz;
out vec3 point_in_view_rgb;
out vec2 point_in_view_uv;

void main()
{
    if(gs_in[0].valid)
    {

        point_in_view_xyz = gs_in[0].xyz;
        point_in_view_rgb = gs_in[0].rgb;
        point_in_view_uv = gs_in[0].uv;

        vec2 coords;
        coords.x = 2.0f * (float(point_in_view_uv.x + 0.5f) / (image_wh.x)) - 1.0f;
        coords.y = 2.0f * (float(point_in_view_uv.y + 0.5f) / (image_wh.y)) - 1.0f;
        gl_Position = vec4(coords, 0,0);

        EmitVertex();
        EndPrimitive();
    }
}