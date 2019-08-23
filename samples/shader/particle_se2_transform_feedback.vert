#version 330 core

layout (location = 0) in vec4 se3_particle;

uniform vec2 laser_points[360];

out vec4 result_out;

void main()
{

    float c = cos(se3_particle.z);
    float s = sin(se3_particle.z);


    float xx;
    float yy;
    for (int i = 0; i < 360; i ++) {


        xx = c*laser_points[i].x - s * laser_points[i].y + se3_particle.x;
        yy = s*laser_points[i].x + c * laser_points[i].y + se3_particle.y;
    }

    // todo: finish result
    // just for debug
    result_out = vec4(xx, yy, xx, yy);
}