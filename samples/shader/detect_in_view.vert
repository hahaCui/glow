#version 330 core

layout (location = 0) in vec3 position;


uniform mat4 T_cam_lidar;
uniform vec2 image_wh;
uniform vec4 intrinsic;

out Element
{
    bool valid;
    vec4 xyz_intensity;
    vec2 uv;
} vs_out;


out vec3 position_out;

void main()
{
    position_out = vec3(T_cam_lidar[0][0], T_cam_lidar[1][1], T_cam_lidar[2][2]);

    
}