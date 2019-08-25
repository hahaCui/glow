#version 330 core

layout (location = 0) in vec3 Lp;

uniform sampler2D input_texture;
uniform mat4 T_cam_lidar;
uniform vec2 image_wh;
uniform vec4 intrinsic;


out Element
{
    bool valid;
    vec3 xyz;
    vec3 rgb;
    vec2 uv;
} vs_out;


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
        vs_out.valid = false;
    } else {
        // todo: bi-interpolatation
        highp int x_floor = int(u);
        highp int y_floor = int(v);

        float dx = u - x_floor;
        float dy = v - y_floor;

        /* Compute rectangles using only 1 multiply (taken from LSD-SLAM). */
        float w11 = dx * dy;
        float w01 = dx - w11;
        float w10 = dy - w11;
        float w00 = 1.0f - dx - dy + w11;


        vec3 rgb00 = texture(input_texture, vec2(x_floor/image_wh.x, y_floor/image_wh.y)).rgb;
        vec3 rgb01 = texture(input_texture, vec2((x_floor + 1)/image_wh.x, y_floor/image_wh.y)).rgb;
        vec3 rgb10 = texture(input_texture, vec2(x_floor/image_wh.x, (y_floor + 1)/image_wh.y)).rgb;
        vec3 rgb11 = texture(input_texture, vec2((x_floor + 1)/image_wh.x, (y_floor + 1)/image_wh.y)).rgb;

        vec2 coords = vec2(u/image_wh.x, v/image_wh.y);
        vs_out.valid = true;
        vs_out.rgb = w00*rgb00 + w01*rgb01 + w10*rgb10 + w11*rgb11;
        vs_out.xyz = Cp;
        vs_out.uv = vec2(u, v);
    }

}