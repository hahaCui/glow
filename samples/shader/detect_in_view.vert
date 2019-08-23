#version 330 core

layout (location = 0) in vec3 Lp;


uniform mat4 T_cam_lidar;
uniform vec2 image_wh;
uniform vec4 intrinsic;

out vec4 point_in_view_xyz_i;
out vec3 point_in_view_uv_isInView;

void main()
{
    vec4 homo_Lp = vec4(Lp, 1);
    vec4 homo_Cp = T_cam_lidar*homo_Lp;
    vec3 Cp = homo_Cp.xyz;

//    position_out = Cp;

    float inv_z = 1.0/ Cp.z;

    float u = intrinsic.x * Cp.x * inv_z + intrinsic.z;
    float v = intrinsic.y * Cp.y * inv_z + intrinsic.w;

    if ( Cp.z < 0 || u < 0 || u > image_wh.x || v < 0 || v > image_wh.y) {
        point_in_view_uv_isInView = vec3(u,v,0);
    } else {
        point_in_view_uv_isInView = vec3(u,v,1);
        point_in_view_xyz_i = vec4(Cp, 0);
    }

}