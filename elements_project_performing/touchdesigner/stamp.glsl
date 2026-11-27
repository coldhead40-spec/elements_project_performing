// GLSL TOP "stamp"  -- paints a decaying 4-channel mask (R=air G=earth B=fire A=water).
// Input 0 : Feedback TOP whose "Target TOP" is this very GLSL TOP (classic feedback loop).
// Vectors page (names must match):
//   uWeights  = (air, earth, fire, water)        from OSC In CHOP  (0..1)
//   uHands    = (lw_x, lw_y, rw_x, rw_y)         from OSC In CHOP  (0..1, y up)
//   uBrush    = (radius, decay)                  e.g. (0.10, 0.985); radius can be driven by /speed
uniform vec4 uWeights;
uniform vec4 uHands;
uniform vec2 uBrush;
out vec4 fragColor;

void main()
{
    vec2  uv     = vUV.st;
    float aspect = uTD2DInfos[0].res.z / uTD2DInfos[0].res.w;      // width / height
    vec2  p      = vec2(uv.x * aspect, uv.y);
    vec2  hl     = vec2(uHands.x * aspect, uHands.y);
    vec2  hr     = vec2(uHands.z * aspect, uHands.w);

    float b = max(1.0 - smoothstep(0.0, uBrush.x, distance(p, hl)),
                  1.0 - smoothstep(0.0, uBrush.x, distance(p, hr)));

    vec4 prev = texture(sTD2DInputs[0], uv) * uBrush.y;             // slow fade = the painting "remembers"
    fragColor = TDOutputSwizzle(max(prev, uWeights * b));
}
