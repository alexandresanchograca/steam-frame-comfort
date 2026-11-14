#version 450
// Window-overlay content for one eye: the selected source eye shown flat,
// with a feathered vignette mask in alpha. Nasal width applies toward the nose
// (+X for the left eye, -X for the right); temple width applies outward.
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outputColor;
layout(set = 0, binding = 0) uniform sampler2D source;
layout(push_constant) uniform Parameters {
    vec4 crop;     // source uv offset (xy) and size (zw)
    vec4 mask;     // temple width, nasal width, height, feather
    vec4 style;    // shape (0 ellipse, 1 rectangle), strength, eye (0 left, 1 right), encode sRGB
    float probeStep; // 0 for the overlay, 1 / cell width for fingerprints
} p;

vec3 encodeSrgb(vec3 linear) {
    vec3 low = linear * 12.92;
    vec3 high = 1.055 * pow(max(linear, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055;
    return mix(high, low, lessThanEqual(linear, vec3(0.0031308)));
}

void main() {
    vec2 centered = uv * 2.0 - 1.0;
    bool nasal = p.style.z < 0.5 ? centered.x > 0.0 : centered.x < 0.0;
    vec2 normalized = centered / vec2(max(nasal ? p.mask.y : p.mask.x, 1e-4), max(p.mask.z, 1e-4));
    float radius = p.style.x < 0.5 ? length(normalized) : max(abs(normalized.x), abs(normalized.y));
    float edge = clamp(p.mask.w, max(fwidth(radius), 1e-4), 1.0);
    float visible = 1.0 - smoothstep(1.0 - edge, 1.0, radius);
    // Stagger the samples within their cells so thin features moving along
    // either axis cross more than 16 distinct sample coordinates per eye.
    vec2 probeOffset = (vec2(uv.y, 1.0 - uv.x) - 0.5) * p.probeStep;
    vec2 at = p.crop.xy + (uv + probeOffset) * p.crop.zw;
    vec3 color = texture(source, at).rgb;
    if (p.style.w > 0.5) color = encodeSrgb(color);
    outputColor = vec4(color, mix(1.0, visible, clamp(p.style.y, 0.0, 1.0)));
}
