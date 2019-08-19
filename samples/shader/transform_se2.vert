#version 330 core

layout (location = 0) in vec4 position;
layout (location = 1) in vec4 se3_particle;


uniform vec2 laser_points[2];

out vec4 color_rgb;

void main()
{
  gl_Position = vec4(position.x, position.y, 0, 1.0);


  // for visualize se2 transformations
//  color_rgb = vec4(se3_particle.x < 255? se3_particle.x: 255,
//                  se3_particle.y < 255? se3_particle.y: 255,
//                    se3_particle.z, 0);


    color_rgb = vec4(laser_points[0], 0, 0);
  
}