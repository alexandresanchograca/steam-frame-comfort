use serde_json::{Value, json};

pub fn snapshot(pixels: &[u8], extent: [u32; 2], source: [u32; 2]) -> Value {
    let [width, height] = extent.map(|value| value as usize);
    let row_bytes = width * 8;
    if width == 0
        || height == 0
        || pixels.len() != row_bytes * height
        || source[0] <= 40
        || source[1] <= 38
    {
        return json!({"snapshotAudit":false,"invalidExtent":true});
    }
    let cell = 16.min((source[0] - 32) / 8).min((source[1] - 32) / 6);
    let [left, right] = [0, width].map(|offset| {
        let mut bits = 0_u64;
        for bit in 0..48 {
            let x = ((16 + bit % 8 * cell) * 2 + cell) as usize * width / (2 * source[0] as usize);
            let y = ((16 + bit / 8 * cell) * 2 + cell) as usize * height / (2 * source[1] as usize);
            let at = y * row_bytes + (x + offset) * 4;
            let rgb = &pixels[at..at + 3];
            let low = *rgb.iter().min()?;
            let high = *rgb.iter().max()?;
            if high - low > 32 {
                return None;
            }
            let average = rgb.iter().map(|&value| u32::from(value)).sum::<u32>() / 3;
            match average {
                0..=80 => {}
                176.. => bits |= 1 << bit,
                _ => return None,
            }
        }
        ((bits & 0xffff) == 0xa55a).then_some((bits >> 16) as u32)
    });
    let at = height * 3 / 4 * row_bytes;
    let row = &pixels[at..at + row_bytes];
    let tint = |eye: &[u8]| {
        eye.chunks_exact(4)
            .map(|rgba| i64::from(rgba[0]) - i64::from(rgba[2]))
            .sum::<i64>()
    };
    let colours = tint(&row[..width * 4]) > 0 && tint(&row[width * 4..]) < 0;
    json!({"snapshotAudit":left.is_some() && left == right && colours,
        "leftFrame":left,"rightFrame":right,"correctColours":colours})
}

#[cfg(test)]
mod tests {
    use super::snapshot;
    use serde_json::Value;

    const SOURCE: [u32; 2] = [256, 256];

    fn coded_snapshot(extent: [u32; 2], ids: [u32; 2], sync: u16) -> Vec<u8> {
        let [width, height] = extent;
        let mut pixels = Vec::with_capacity((width * height * 8) as usize);
        let codes = ids.map(|id| u64::from(id) << 16 | u64::from(sync));
        for y in 0..height {
            for x in 0..2 * width {
                let eye = usize::from(x >= width);
                let source_x = (x % width) * SOURCE[0] / width;
                let source_y = y * SOURCE[1] / height;
                let rgba = if (16..144).contains(&source_x) && (16..112).contains(&source_y) {
                    let bit = (source_x - 16) / 16 + 8 * ((source_y - 16) / 16);
                    let luma = if codes[eye] & (1 << bit) == 0 { 8 } else { 240 };
                    [luma, luma, luma, 255]
                } else if eye == 0 {
                    [200, 40, 20, 255]
                } else {
                    [20, 40, 200, 255]
                };
                pixels.extend(rgba);
            }
        }
        pixels
    }

    #[test]
    fn audit_decodes_packed_eyes_when_downsampled() {
        let extent = [128, 64];
        for id in [0, 1, u32::MAX, 0x8000_0000, 0x1234_5678, 0xa55a_5aa5] {
            let same = coded_snapshot(extent, [id, id], 0xa55a);
            let audit = snapshot(&same, extent, SOURCE);
            assert_eq!(audit["snapshotAudit"], true);
            assert_eq!(audit["leftFrame"], id);
            assert_eq!(audit["rightFrame"], id);

            let mixed = coded_snapshot(extent, [id, id.wrapping_add(1)], 0xa55a);
            let audit = snapshot(&mixed, extent, SOURCE);
            assert_eq!(audit["snapshotAudit"], false);
            assert_eq!(audit["rightFrame"], id.wrapping_add(1));
        }
    }

    #[test]
    fn audit_rejects_wrong_sync_ambiguous_cells_and_truncated_pixels() {
        let extent = SOURCE;
        let wrong_sync = coded_snapshot(extent, [42; 2], 0x5aa5);
        assert_eq!(
            snapshot(&wrong_sync, extent, SOURCE)["leftFrame"],
            Value::Null
        );
        let mut valid = coded_snapshot(extent, [42; 2], 0xa55a);
        let at = (24 * 512 + 24) * 4;
        for rgb in [[128; 3], [240, 8, 8]] {
            valid[at..at + 3].copy_from_slice(&rgb);
            let audit = snapshot(&valid, extent, SOURCE);
            assert_eq!(audit["snapshotAudit"], false);
            assert_eq!(audit["leftFrame"], Value::Null);
            assert_eq!(audit["rightFrame"], 42);
        }
        valid.pop();
        assert_eq!(snapshot(&valid, extent, SOURCE)["invalidExtent"], true);
    }
}
