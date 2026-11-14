//! Pixel data shared by the live scenes and the synthetic cadence producer.

pub(super) const CODE_CELLS: usize = 48;
#[cfg(any(feature = "scene-fixtures", test))]
pub(super) const ORIGIN: u32 = 16;

#[cfg(any(feature = "scene-fixtures", test))]
pub(super) fn code_cell_size(width: u32, height: u32) -> Option<u32> {
    (width > 40 && height > 38).then(|| 16.min((width - 32) / 8).min((height - 32) / 6))
}

pub(super) fn fill_gray(bytes: &mut [u8], brightness: u8) {
    for rgba in bytes.chunks_exact_mut(4) {
        rgba.copy_from_slice(&[brightness, brightness, brightness, 255]);
    }
}

pub(super) fn write_code(bytes: &mut [u8], cell: usize, frame: u32) {
    let cell_bytes = cell * cell * 4;
    let code = u64::from(frame) << 16 | 0xa55a;
    for (bit, pixels) in bytes[..CODE_CELLS * cell_bytes]
        .chunks_exact_mut(cell_bytes)
        .enumerate()
    {
        fill_gray(pixels, if code & (1 << bit) == 0 { 8 } else { 240 });
    }
    fill_gray(
        &mut bytes[CODE_CELLS * cell_bytes..],
        (24 + frame.wrapping_mul(73) % 208) as u8,
    );
}

// Aperiodic texture with a Frame-like 80/512 stereo disparity and distinct
// eye colors. Both eyes must match after compensating for that shift.
pub(super) fn stereo_pixel(x: u32, y: u32, width: u32, height: u32, eye: usize) -> [u8; 4] {
    let x = (u64::from(x) * 512 / u64::from(width)) as u32 + eye as u32 * 80;
    let y = (u64::from(y) * 512 / u64::from(height)) as u32;
    let cell = |x: u32, y: u32| {
        let value = x.wrapping_mul(2_246_822_519) ^ y.wrapping_mul(3_266_489_917);
        (value ^ (value >> 15)).wrapping_mul(2_246_822_519) >> 24
    };
    let blend = |a, b, fraction| (a * (32 - fraction) + b * fraction) / 32;
    let a = blend(cell(x / 32, y / 32), cell(x / 32 + 1, y / 32), x % 32);
    let b = blend(
        cell(x / 32, y / 32 + 1),
        cell(x / 32 + 1, y / 32 + 1),
        x % 32,
    );
    let full = (24 + blend(a, b, y % 32) * 9 / 10) as u8;
    if eye == 0 {
        [full, full / 2, full / 8, 255]
    } else {
        [full / 8, full / 2, full, 255]
    }
}
