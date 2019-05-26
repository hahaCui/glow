#version 330 core

layout(points) in;
layout(points, max_vertices = 1) out;

out SURFEL
{
  bool valid;
  vec4 position;

} gs_in[];

out vec4 position_out;

void main()
{
  if(gs_in[0].valid)
  {
    position_out = gs_in[0].position;

    EmitVertex();
    EndPrimitive();
  }
}