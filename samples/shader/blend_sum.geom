#version 330 core

layout(points) in;
layout(points, max_vertices = 1) out;

out CostInformation
{
  bool valid;
  vec3 C1p;
  vec3 C0p;
  vec3 rgb1;
  vec3 rgb0;
  vec2 uv1;
  vec2 uv0;
} gs_in[];

out vec4 tex_color;


out vec3 point_in_view_xyz;
out vec3 point_in_view_rgb;
out vec2 point_in_view_uv;

void main()
{
  if(gs_in[0].valid)
  {
    // todo  if valid
//    gl_Position = gs_in[0].position;
//    tex_color = gs_in[0].rgb;
//
//    point_in_view_xyz = gs_in[0].xyz;
//    point_in_view_rgb = gs_in[0].rgb.xyz;
//    point_in_view_uv = gs_in[0].uv;
//
//    EmitVertex();
//    EndPrimitive();
  }
}