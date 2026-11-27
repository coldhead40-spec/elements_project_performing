// GLSL TOP "reveal"  -- black&white painting + four element clips, revealed through the mask.
// Inputs: 0 base (desaturated painting)  1 air clip  2 earth clip  3 fire clip  4 water clip  5 stamp mask
out vec4 fragColor;

void main()
{
    vec2 uv   = vUV.st;
    vec4 col  = texture(sTD2DInputs[0], uv);
    vec4 m    = texture(sTD2DInputs[5], uv);                        // r,g,b,a = air, earth, fire, water

    col = mix(col, texture(sTD2DInputs[1], uv), smoothstep(0.0, 1.0, m.r));
    col = mix(col, texture(sTD2DInputs[2], uv), smoothstep(0.0, 1.0, m.g));
    col = mix(col, texture(sTD2DInputs[3], uv), smoothstep(0.0, 1.0, m.b));
    col = mix(col, texture(sTD2DInputs[4], uv), smoothstep(0.0, 1.0, m.a));

    fragColor = TDOutputSwizzle(vec4(col.rgb, 1.0));
}
