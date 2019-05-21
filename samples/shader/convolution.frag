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
	float normalized_pixel_x = 1.0 / nWidth;
	float normalized_pixel_y = 1.0 / nHeight;
	float normalized_radius_x = fRadius / nWidth;
	float normalized_radius_y = fRadius / nHeight;


	vec3 fSum = vec3(0.0, 0.0, 0.0);		//Sum of the neighborhood.
	vec3 fTotal = vec3(0.0, 0.0, 0.0);		//NoPoints in the neighborhood.
	vec3 vec3Result = vec3(0.0, 0.0, 0.0);	//Output vector to replace the current texture.

	//Neighborhood summation.
	for (float ii = coords.x - normalized_radius_x; ii < coords.x + normalized_radius_x ; ii += normalized_pixel_x) {
		for (float jj = coords.y - normalized_radius_y; jj <= coords.y + normalized_radius_y; jj += normalized_pixel_y) {
			if (ii >= 0.0 && jj >= 0.0 && ii < 1.0 && jj < 1.0) {
				vec2 new_pos = vec2(ii, jj);
				fSum += texture(tex_input, new_pos).rgb;
				fTotal += vec3(1.0, 1.0, 1.0);
			}
		}
	}
	vec3Result = fSum / fTotal;
	color = vec3(vec3Result);
}