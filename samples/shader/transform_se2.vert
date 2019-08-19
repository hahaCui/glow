#version 330 core

layout (location = 0) in vec4 position;
layout (location = 1) in vec4 se3_particle;


uniform vec2 laser_points[360];

out vec4 color_rgb;

void main()
{
  gl_Position = vec4(position.x, position.y, 0, 1.0);


  // for visualize se2 transformations
//  color_rgb = vec4(se3_particle.x < 255? se3_particle.x: 255,
//                  se3_particle.y < 255? se3_particle.y: 255,
//                    se3_particle.z, 0);


    float c = cos(se3_particle.z);
    float s = sin(se3_particle.z);



    for (int i = 0; i < 360; i ++) {
        // for debug
//        if (laser_points[i].x > 180  && laser_points[i].y > 0 ) {
//            color_rgb = vec4(laser_points[i], 0, 0);
//            break;
//        }

        float xx = c*laser_points[i].x - s * laser_points[i].y + se3_particle.x;
        float yy = s*laser_points[i].x + c * laser_points[i].y + se3_particle.y;
    }

  
}