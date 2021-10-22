#version 330

// Input vertex attributes (from vertex shader)
in vec2 fragTexCoord;
in vec4 fragColor;

// Input uniform values
uniform sampler2D texture0;

// Output fragment color
out vec4 finalColor;

// TODO pass these in when/if we have multiple source tile sizes in the texture.
const float texture_section_w = 128.0;
const float texture_section_h = 128.0;

void main()
{
    ivec2 tSize = textureSize(texture0, 0);
    vec2 iResolution = vec2(tSize);

    // Based off of https://www.shadertoy.com/view/MlB3D3

    // Mapping into correct space for conditioning
    vec2 unilateralCoord = vec2(
        fragTexCoord.x * (iResolution.x / texture_section_w),
        fragTexCoord.y * (iResolution.y / texture_section_h)
    );

    vec2 pixel = unilateralCoord * iResolution;
    // Conditioning the UVs
    // emulate point sampling
    vec2 uv = floor(pixel) + 0.5;

    // subpixel aa algorithm (COMMENT OUT TO COMPARE WITH POINT SAMPLING)
    uv += 1.0 - clamp((1.0 - fract(pixel)), 0.0, 1.0);

    // Unmapping back into texture space after conditioning
    vec2 conditionedUnilateralCoord = uv / iResolution;

    vec2 conditionedFragTexCoord = vec2(
        conditionedUnilateralCoord.x / (iResolution.x / texture_section_w),
        conditionedUnilateralCoord.y / (iResolution.y / texture_section_h)
    );

    vec4 color = texture2D(texture0, conditionedFragTexCoord);

    finalColor = color*fragColor;
}
