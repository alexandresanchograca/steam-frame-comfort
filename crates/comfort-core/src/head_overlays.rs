//! Straight-alpha RGBA textures for head-relative overlays.
use std::f32::consts::PI;

pub const CROSSHAIR_TEXTURE_SIZE: usize = 256;
pub const CROSSHAIR_TEXTURE_DEG: f32 = 8.0;
pub const NOSE_EYE_SIZE: usize = 320;
pub const NOSE_TEXTURE_DEG: f32 = 90.0;
pub const NOSE_DISTANCE_M: f32 = 1.0;

pub fn overlay_width_m(view_deg: f32, distance_m: f32) -> f32 {
    2.0 * distance_m * (view_deg * PI / 360.0).tan()
}

fn smooth_coverage(value: f32) -> f32 {
    let x = value.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

fn byte(value: f32) -> u8 {
    (value * 255.0 + 0.5) as u8
}

pub fn crosshair_pixels(size_deg: f32, opacity: f32) -> Vec<u8> {
    const SIZE: usize = CROSSHAIR_TEXTURE_SIZE;
    let pixel_angle = CROSSHAIR_TEXTURE_DEG / SIZE as f32;
    let tangent = overlay_width_m(CROSSHAIR_TEXTURE_DEG, 0.5);
    let angles: [f32; SIZE] = std::array::from_fn(|index| {
        (((2.0 * (index as f32 + 0.5) / SIZE as f32 - 1.0) * tangent).atan() * 180.0 / PI).abs()
    });
    let half_width = (size_deg * 0.06).clamp(0.035, 0.18) * 0.5;
    let half_segment = (size_deg * 0.5 - half_width).max(0.0);
    let outline = (pixel_angle * 1.25).max(half_width);
    let mut rgba = Vec::with_capacity(SIZE * SIZE * 4);
    for ay in angles {
        for ax in angles {
            let horizontal = (ax - half_segment).max(0.0).hypot(ay);
            let vertical = ax.hypot((ay - half_segment).max(0.0));
            let distance = horizontal.min(vertical) - half_width;
            let white = smooth_coverage(0.5 - distance / pixel_angle);
            let coverage = smooth_coverage(0.5 - (distance - outline) / pixel_angle);
            let grey = if coverage > 0.0 {
                byte((white / coverage).min(1.0))
            } else {
                0
            };
            rgba.extend([grey, grey, grey, byte(coverage * opacity)]);
        }
    }
    rgba
}

pub fn nose_pixels(scale: f32, opacity: f32) -> Vec<u8> {
    const SIZE: usize = NOSE_EYE_SIZE;
    const WIDTH: usize = SIZE * 2;
    let mut rgba = vec![0; WIDTH * SIZE * 4];
    for y in 0..SIZE {
        // Tilt the bridge toward the centre as the shape broadens.
        let v = (2.0 * (y as f32 + 0.5) / SIZE as f32 - 1.0 - 0.62) / (0.30 * scale);
        for x in 0..SIZE {
            let u = (2.0 * (x as f32 + 0.5) / SIZE as f32 - 1.0 - 0.45) / (0.20 * scale);
            let offset = u - 0.22 * v;
            let profile = (offset / (1.0 + 0.2 * v.clamp(-1.0, 1.0))).hypot(v);
            let coverage = smooth_coverage((1.0 - profile) / 0.24);
            if coverage <= 0.0 {
                continue;
            }
            let highlight = smooth_coverage(0.6 - 0.8 * offset);
            let pixel = [
                byte(0.31 + 0.18 * highlight),
                byte(0.24 + 0.15 * highlight),
                byte(0.22 + 0.12 * highlight),
                byte(coverage * opacity),
            ];
            for x in [x, WIDTH - 1 - x] {
                let at = (y * WIDTH + x) * 4;
                rgba[at..at + 4].copy_from_slice(&pixel);
            }
        }
    }
    rgba
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlays_are_straight_alpha_and_the_nose_is_mirrored() {
        let crosshair = crosshair_pixels(1.0, 0.5);
        let centre = (CROSSHAIR_TEXTURE_SIZE * (CROSSHAIR_TEXTURE_SIZE / 2)
            + CROSSHAIR_TEXTURE_SIZE / 2)
            * 4;
        assert_eq!(&crosshair[centre..centre + 4], &[254, 254, 254, 128]);
        assert_eq!(&crosshair[..4], &[0; 4]);
        let nose = nose_pixels(1.0, 0.6);
        for row in nose.chunks_exact(NOSE_EYE_SIZE * 2 * 4) {
            assert!(row.chunks_exact(4).eq(row.rchunks_exact(4)));
        }
        assert!(nose.chunks_exact(4).any(|pixel| pixel[3] == 153));
    }
}
