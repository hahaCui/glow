#version 330 core

layout (location = 0) in vec3 Lp;

uniform sampler2D input_texture;

uniform mat4 T_cam_lidar;
uniform vec2 wh;
uniform vec4 intrinsic;

out Element
{
    bool valid;
    vec4 position;
    vec4 rgb;
    vec3 xyz;
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

    if ( Cp.z < 0 || u < 0 || u > wh.x || v < 0 || v > wh.y) {
        vs_out.valid = false;


    } else {
        vs_out.valid = true;

        vec2 normal_coords = vec2(2.0f * (float(u + 0.5f) / float(wh.x)) - 1.0f,
        2.0f * (float(v + 0.5f) / float(wh.y)) - 1.0f);
        vs_out.position = vec4(normal_coords, 0, 1.0);
        vec2 tex_coords = vec2(u/float(wh.x), (float(v) / float(wh.y)));
        vs_out.rgb = texture(input_texture, tex_coords);

        vs_out.xyz = Cp;
        vs_out.uv = vec2(u,v);


    }

}