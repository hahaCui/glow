#version 330

in vec2 texCoords;

uniform sampler2D tex_R;
uniform sampler2D tex_G;
uniform sampler2D tex_B;

out vec3 color;


void main()
{
  vec2 coords = vec2(texCoords.x, texCoords.y);
  float r = texture(tex_R, coords).r;
  float g = texture(tex_G, coords).r;
  float b = texture(tex_B, coords).r;
  color = vec3(r,g,b);
}