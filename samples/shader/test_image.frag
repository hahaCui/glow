#version 330

in vec2 texCoords;

uniform sampler2D tex_input;

uniform float fRadius;
uniform float nWidth;
uniform float nHeight;

out vec3 color;

void main()
{
  vec2 coords = vec2(texCoords.x, texCoords.y);


  vec3 fSum = vec3(0.0, 0.0, 0.0);		//Sum of the neighborhood.
  vec3 fTotal = vec3(0.0, 0.0, 0.0);		//NoPoints in the neighborhood.
  vec3 vec3Result = vec3(0.0, 0.0, 0.0);	//Output vector to replace the current texture.

  	//Neighborhood summation.
  //plus 1.0 for the '0.5 effect'.
  	for (float ii = coords.x - fRadius; ii < coords.x + fRadius + 0.5; ii += 1.0) {
      for (float jj = coords.y - fRadius; jj <= coords.y + fRadius + 0.5; jj += 1.0) {
        if (ii >= 0.0 && jj >= 0.0 && ii < nWidth && jj < nHeight) {
          fSum += texture(tex_input, vec2(ii, jj)).rgb;
          fTotal += vec3(1.0, 1.0, 1.0);
        }
      }
    }
  	vec3Result = fSum / fTotal;

  //vec2 new_coord = vec2(coords.x + fRadius, coords.y);
  vec3 tex = texture(tex_input, coords.xy).rgb;

  color = vec3(tex.xyz);
}