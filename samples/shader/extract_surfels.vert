#version 330 core

layout (location = 0) in vec4 position_radius;
layout (location = 1) in vec4 normal_confidence;
layout (location = 2) in int in_timestamp;
layout (location = 3) in vec3 surfel_color_weight_count;

uniform samplerBuffer poseBuffer;
uniform vec2 submap_center;
uniform float submap_extent;

out SURFEL 
{
  bool valid;
  vec4 position_radius;
  vec4 normal_confidence;
  int timestamp;
  vec3 color_weight_count;
} vs_out;



void main()
{


    vs_out.valid = true;
    vs_out.position_radius = position_radius;
    vs_out.normal_confidence = normal_confidence;
    vs_out.timestamp = in_timestamp;
    vs_out.color_weight_count = surfel_color_weight_count;

}