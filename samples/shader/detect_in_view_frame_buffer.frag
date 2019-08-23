#version 330

in float depth;
in vec2 texCoords;

uniform sampler2D input_texture;
out vec4 point_in_view_rgbd;


void main()
{
    if (depth < 0) {
        point_in_view_rgbd = vec4(0,0,0,0);
    } else {
        point_in_view_rgbd = vec4(texture(input_texture, texCoords).rgb, depth);
    }
}