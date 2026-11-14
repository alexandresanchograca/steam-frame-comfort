#version 450
// Colours are gamma encoded in the immutable snapshot images and decoded once
// before the refresh blend. The single straight alpha is independent of blend.
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 outputColor;
layout(set = 0, binding = 0) uniform sampler2D oldSnapshot;
layout(set = 0, binding = 1) uniform sampler2D newSnapshot;
struct Capture {
    vec4 rows[3]; // inverse eye transform rows xyz/translation
    vec4 projection;
};
struct Draw {
    vec4 faceCenter; // world xyz, shell half-size
    vec4 faceRight;  // world xyz, per-eye face resolution
    vec4 faceUp;
    Capture capture[2];
    vec4 displayEye[3]; // eye origin with head-aligned axes
    vec4 mask;   // half width tangent, half height tangent, feather, rectangle
    vec4 style;  // strength, entry (opacity), refresh weight, grid enabled
    vec4 look;   // linear background, linear lines, thickness metres, spacing
    vec4 dimensions; // old packed width/height, new packed width/height
    vec4 effects; // blink enabled, blink line brightness (linear), grid over ring, picture scale (linear)
};
layout(std140, set = 0, binding = 2) uniform Parameters { Draw draws[12]; } p;
layout(push_constant) uniform Selection { uint drawIndex; } selection;

vec3 decodeSrgb(vec3 gamma) {
    return mix(pow((gamma + 0.055) / 1.055, vec3(2.4)), gamma / 12.92,
               lessThanEqual(gamma, vec3(0.04045)));
}
vec3 encodeSrgb(vec3 linear) {
    return mix(1.055 * pow(max(linear, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055,
               linear * 12.92, lessThanEqual(linear, vec3(0.0031308)));
}
vec3 eyePoint(vec4 rows[3], vec3 world) {
    vec4 point = vec4(world, 1.0);
    return vec3(dot(rows[0], point), dot(rows[1], point), dot(rows[2], point));
}
// pictureScale dims the captured picture only; the gap fill keeps its look.
vec3 sampleCapture(sampler2D image, Capture capture, vec3 world, vec3 gap,
                   float eye, vec2 dimensions, float pictureScale) {
    if (dimensions.x < 2.0) return gap; // no snapshot (Plain, Grid): gap fill only
    vec3 point = eyePoint(capture.rows, world);
    if (point.z >= -1e-8) return gap;
    vec2 tangent = vec2(point.x, -point.y) / -point.z;
    vec4 f = capture.projection;
    vec2 at = (tangent - f.xz) / (f.yw - f.xz);
    if (any(lessThan(at, vec2(0.0))) || any(greaterThan(at, vec2(1.0)))) return gap;
    // A packed texture must never linearly filter across its eye boundary.
    at = clamp(at, vec2(1.0 / dimensions.x, 0.5 / dimensions.y),
                    vec2(1.0 - 1.0 / dimensions.x, 1.0 - 0.5 / dimensions.y));
    return decodeSrgb(texture(image, vec2((at.x + eye) * 0.5, at.y)).rgb) * pictureScale;
}
// Both eyes use the same opening angles about head-forward. Fading changes
// only the periphery's opacity.
float peripheralAlpha(Draw d, vec3 world) {
    float level = d.style.x * d.style.y;
    if (level == 0.0) return 0.0;
    vec3 point = eyePoint(d.displayEye, world);
    if (point.z >= -1e-8) return level;
    vec2 normalized = point.xy / -point.z / d.mask.xy;
    float radius = d.mask.w > 0.5 ? max(abs(normalized.x), abs(normalized.y)) : length(normalized);
    if (d.mask.z < 1e-6) return radius >= 1.0 ? level : 0.0;
    return level * smoothstep(1.0 - d.mask.z, 1.0, radius);
}
void main() {
    Draw d = p.draws[selection.drawIndex];
    vec2 coordinates = vec2(2.0 * uv.x - 1.0, 1.0 - 2.0 * uv.y) * d.faceCenter.w;
    vec3 world = d.faceCenter.xyz + d.faceRight.xyz * coordinates.x + d.faceUp.xyz * coordinates.y;
    // Room-fixed grid lines through the face centre, anti-aliased over one texel.
    float lines = 0.0;
    bool blink = d.effects.x > 0.5;
    bool gridOverRing = d.effects.z > 0.5;
    if (d.style.w > 0.5 || blink || gridOverRing) {
        float texel = 2.0 * d.faceCenter.w / d.faceRight.w;
        float halfThickness = max(d.look.z, texel) * 0.5;
        vec2 distance = abs(coordinates - round(coordinates / d.look.w) * d.look.w);
        vec2 coverage = clamp((halfThickness - distance) / texel + 0.5, 0.0, 1.0);
        lines = max(coverage.x, coverage.y);
    }
    vec3 gap = vec3(d.style.w > 0.5 ? mix(d.look.x, d.look.y, lines) : d.look.x);
    float eye = float(selection.drawIndex & 1u);
    // Sample only the snapshots that contribute: one outside a cross-fade,
    // and one during a blink. The weight is uniform for the whole draw.
    float weight = d.style.z;
    vec3 colour;
    if (blink) {
        // Refresh blink: the old snapshot dips to grid lines on black, the new
        // one takes over at the darkest point, then it clears. The dip is
        // sin^2, so it starts and ends without a step and the swap is unseen.
        vec3 shown = weight < 0.5 ? sampleCapture(oldSnapshot, d.capture[0], world, gap, eye, d.dimensions.xy, d.effects.w)
                                  : sampleCapture(newSnapshot, d.capture[1], world, gap, eye, d.dimensions.zw, d.effects.w);
        float dip = sin(3.14159265358979 * weight);
        float dark = gridOverRing ? 0.0 : d.effects.y * lines;
        colour = mix(shown, vec3(dark), dip * dip);
    } else if (weight <= 0.0) {
        colour = sampleCapture(oldSnapshot, d.capture[0], world, gap, eye, d.dimensions.xy, d.effects.w);
    } else if (weight >= 1.0) {
        colour = sampleCapture(newSnapshot, d.capture[1], world, gap, eye, d.dimensions.zw, d.effects.w);
    } else {
        colour = mix(sampleCapture(oldSnapshot, d.capture[0], world, gap, eye, d.dimensions.xy, d.effects.w),
                     sampleCapture(newSnapshot, d.capture[1], world, gap, eye, d.dimensions.zw, d.effects.w), weight);
    }
    if (gridOverRing) colour = mix(colour, vec3(d.look.y), lines);
    outputColor = vec4(encodeSrgb(colour), peripheralAlpha(d, world));
}
