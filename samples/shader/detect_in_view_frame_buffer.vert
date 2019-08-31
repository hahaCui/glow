#version 330 core

layout (location = 0) in vec3 Lp;

uniform sampler2D input_texture;

uniform mat4 T_cam_lidar;
uniform vec2 image_wh;
uniform vec4 intrinsic;

out vec4 tex_color;

out vec3 point_in_view_xyz;
out vec3 point_in_view_rgb;
out vec3 point_in_view_uv_in_view;

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
//        gl_Position = vec4(0,0,0,1);
        vec2 texCoords = vec2(0,0);
        tex_color = vec4(0,0,0,1);

        point_in_view_xyz = Cp;
        point_in_view_rgb = vec3(0,0,0);
        point_in_view_uv_in_view = vec3(0,0,-1);


    } else {
        float normal_u = 2.0f * (float(u ) / float(image_wh.x)) - 1.0f;
        float normal_v = 2.0f * (float(v ) / float(image_wh.y)) - 1.0f;
        gl_Position = vec4(normal_u, normal_v, 0, 1);
        vec2 texCoords = vec2(u/image_wh.x, v/image_wh.y);

        tex_color = vec4(texture(input_texture, texCoords).rgb, 1);

        point_in_view_xyz = vec3(normal_u, normal_v, 1);
        point_in_view_rgb = tex_color.rgb;
        point_in_view_uv_in_view = vec3(u,v,1);


    }

}