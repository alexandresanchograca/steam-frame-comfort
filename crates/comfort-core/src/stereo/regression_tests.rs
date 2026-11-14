use super::*;

fn texture(frame: u32, x: i32, y: i32) -> u8 {
    let bits = (x as u32).wrapping_mul(2_246_822_519)
        ^ (y as u32).wrapping_mul(3_266_489_917)
        ^ frame.wrapping_mul(668_265_263);
    ((bits ^ (bits >> 15)).wrapping_mul(2_246_822_519) >> 24) as u8
}

fn thumbnail(pixel: impl Fn(i32, i32) -> u8) -> LumaThumbnail {
    std::array::from_fn(|index| {
        pixel(
            (index % THUMBNAIL_SIZE) as i32,
            (index / THUMBNAIL_SIZE) as i32,
        )
    })
}

fn measure(first: &LumaThumbnail, second: &LumaThumbnail) -> StereoShift {
    stereo_shift(&StereoFeatures::new(first), &StereoFeatures::new(second))
}

#[test]
fn stereo_evidence_survives_lens_masks_and_shared_frame_code() {
    for frame in 1..=4 {
        let lens = |right: bool, frame| {
            thumbnail(|x, y| {
                let dx = f64::from(x - if right { 36 } else { 27 }) / 29.0;
                let dy = f64::from(y - 34) / 34.0;
                if dx * dx + dy * dy > 1.0 {
                    0
                } else {
                    let background = if y < 29 { 35 } else { 195 };
                    let shift = if right { 0 } else { 10 };
                    background + texture(frame, x - shift, y) / 8
                }
            })
        };
        let left = lens(false, frame);
        let right = lens(true, frame);
        for (shift, expected) in [
            (measure(&left, &right), 10.0),
            (measure(&right, &left), -10.0),
        ] {
            assert!(
                shift.strong() && (shift.fraction - expected / 64.0).abs() < 0.01,
                "{shift:?}"
            );
        }
        assert!(!measure(&left, &lens(true, frame + 123)).strong());

        let coded = |shift| {
            thumbnail(|x, y| {
                if (2..18).contains(&x) && (2..14).contains(&y) {
                    if (x / 2 + y / 2 + frame as i32) % 2 == 0 {
                        8
                    } else {
                        240
                    }
                } else {
                    texture(frame, x - shift, y)
                }
            })
        };
        let shift = measure(&coded(10), &coded(0));
        assert!(
            shift.strong() && (shift.fraction - 10.0 / 64.0).abs() < 0.01,
            "{shift:?}"
        );
        assert!(!measure(&coded(10), &coded(10)).strong());
    }
}

#[test]
fn stereo_evidence_rejects_blank_periodic_and_half_empty_images() {
    let horizon = thumbnail(|_, y| if y < 29 { 40 } else { 220 });
    let bars = |shift| thumbnail(|x, _| if (x + shift) % 8 < 4 { 40 } else { 220 });
    let half = |shift| thumbnail(|x, y| if y < 32 { texture(1, x - shift, y) } else { 0 });
    assert!(!measure(&[0; THUMBNAIL_SIZE * THUMBNAIL_SIZE], &horizon).strong());
    assert!(!measure(&horizon, &horizon).strong());
    assert!(!measure(&bars(0), &bars(10)).strong());
    assert!(!measure(&half(10), &half(0)).strong());
}

#[test]
fn competing_packings_and_old_slots_cannot_establish_eye_order() {
    let left = thumbnail(|x, y| texture(1, x - 10, y));
    let right = thumbnail(|x, y| texture(1, x, y));
    let shift = measure(&right, &left);
    assert!(shift.strong() && shift.fraction < 0.0);
    let sample = |packing, signature, new_frame| LayoutSample {
        packing,
        shift,
        signatures: [signature, signature],
        new_frame,
    };
    let mut verifier = LayoutVerifier::default();
    for index in 0..10 {
        verifier.observe(&[sample(Packing::Array, 0, true)]);
        verifier.observe(&[sample(Packing::Array, index, false)]);
        assert!(verifier.proven().is_none());
    }
    for index in 1..=5 {
        verifier.observe(&[
            sample(Packing::Array, index, true),
            sample(Packing::LeftQuadrants, index, true),
        ]);
        assert!(verifier.proven().is_none());
    }
    for index in 1..=3 {
        verifier.observe(&[sample(Packing::Array, index, true)]);
        assert!(verifier.proven().is_none());
    }
    verifier.observe(&[sample(Packing::Array, 4, true)]);
    assert_eq!(
        verifier.proven(),
        Some(Layout {
            packing: Packing::Array,
            swapped: true
        })
    );
}

const PACKED_LAYOUTS: [(Packing, [[usize; 2]; 2], [usize; 2]); 8] = [
    (Packing::SideBySide, [[0, 0], [1, 0]], [2, 1]),
    (Packing::TopBottom, [[0, 0], [0, 1]], [1, 2]),
    (Packing::LeftQuadrants, [[0, 0], [0, 1]], [2, 2]),
    (Packing::RightQuadrants, [[1, 0], [1, 1]], [2, 2]),
    (Packing::TopQuadrants, [[0, 0], [1, 0]], [2, 2]),
    (Packing::BottomQuadrants, [[0, 1], [1, 1]], [2, 2]),
    (Packing::MainDiagonal, [[0, 0], [1, 1]], [2, 2]),
    (Packing::OtherDiagonal, [[1, 0], [0, 1]], [2, 2]),
];

#[test]
fn competing_crops_distinguish_every_packed_layout_in_both_eye_orders() {
    const N: usize = THUMBNAIL_SIZE;
    for (packing, cells, extent) in PACKED_LAYOUTS {
        let [width, height] = extent.map(|tiles| tiles * N);
        for swapped in [false, true] {
            let mut verifier = LayoutVerifier::default();
            let shifts = if swapped { [0, 10] } else { [10, 0] };
            for frame in 1..=4 {
                // Paint independent integer cells, then sample the production crops.
                let mut atlas = vec![32; width * height];
                for ([column, row], shift) in cells.into_iter().zip(shifts) {
                    let pixels = thumbnail(|x, y| texture(frame, x - shift, y));
                    for (y, pixels) in pixels.chunks_exact(N).enumerate() {
                        let start = (row * N + y) * width + column * N;
                        atlas[start..start + N].copy_from_slice(pixels);
                    }
                }
                let samples = PACKED_LAYOUTS.map(|(candidate, _, _)| {
                    let [first, second] = candidate.eye_regions().map(|region| {
                        let [left, top, span_x, span_y] = region.crop.map(f64::from);
                        thumbnail(|x, y| {
                            let sx = ((left + span_x * (f64::from(x) + 0.5) / N as f64)
                                * width as f64) as usize;
                            let sy = ((top + span_y * (f64::from(y) + 0.5) / N as f64)
                                * height as f64) as usize;
                            atlas[sy * width + sx]
                        })
                    });
                    LayoutSample {
                        packing: candidate,
                        shift: measure(&first, &second),
                        signatures: [u64::from(frame); 2],
                        new_frame: true,
                    }
                });
                verifier.observe(&samples);
            }
            assert_eq!(verifier.proven(), Some(Layout { packing, swapped }));
        }
    }
}

#[test]
fn stereo_shift_accepts_parallax_and_rejects_different_frames() {
    let view = |frame: u32, right: bool| {
        thumbnail(|x, y| {
            let dx = f64::from(x - if right { 36 } else { 27 }) / 31.0;
            let dy = f64::from(y - 33) / 36.0;
            if dx * dx + dy * dy > 1.0 {
                return 0;
            }
            let floor = y >= 32;
            let shift = if right { 0 } else { 10 + (y - 32).max(0) / 3 };
            let scene = u16::from(texture(frame + 1000 * u32::from(floor), x - shift, y));
            let noise = u16::from(texture(frame * 31 + if right { 7 } else { 3 }, x, y));
            (20 + ((4 * scene + noise) / 5) * 200 / 255) as u8
        })
    };
    for frame in 1..=4 {
        let left = view(frame, false);
        let right = view(frame, true);
        let shift = measure(&left, &right);
        assert!(
            shift.strong() && (shift.fraction - 10.0 / 64.0).abs() < 0.02,
            "{shift:?}"
        );
        assert!((0.4..0.7).contains(&shift.score), "{shift:?}");
        assert!(!measure(&left, &view(frame + 50, true)).strong());
    }
}

#[test]
fn learns_permuted_eye_slots_at_supported_refresh_rates() {
    for hz in [72, 90, 120, 144] {
        let appropriate_delays = if hz <= 90 {
            &[0, 750_000, 2_000_000][..]
        } else {
            &[0, 750_000][..]
        };
        for &delay in appropriate_delays {
            let learner = timing_fixture(hz, delay, false, false);
            assert_eq!(
                learner.proven(),
                [[0, 4], [1, 5], [2, 3]],
                "{hz} Hz, {delay} ns"
            );
        }
        assert_eq!(
            timing_fixture(hz, 750_000, false, true).proven(),
            [[0, 4], [1, 5], [2, 3]]
        );
        let changing = timing_fixture(hz, 750_000, true, false);
        assert!(changing.proven().is_empty() && changing.conflicts > 0);
        assert!(
            timing_fixture(hz, 4_000_000, false, false)
                .proven()
                .is_empty()
        );
    }
    for delay in [2_600_000, 2_920_000, 3_200_000] {
        for jitter in [false, true] {
            assert!(
                timing_fixture(144, delay, false, jitter)
                    .proven()
                    .is_empty(),
                "{delay} ns, jitter {jitter}"
            );
        }
    }
}

fn timing_fixture(hz: u64, delay: u64, changing_partner: bool, jitter: bool) -> PairTimingLearner {
    let right_slots = [4, 5, 3];
    let period = 1_000_000_000 / hz;
    let mut learner = PairTimingLearner::default();
    let mut signatures = [0; 6];
    let (mut left_frame, mut right_frame) = (0, 0);
    let mut now = 1_000_000;
    let mut steps = [850_000, 1_100_000, 1_900_000, 930_000].into_iter().cycle();
    while now <= 30 * period {
        while left_frame * period + 3_999_000 <= now {
            signatures[(left_frame % 3) as usize] = left_frame + 1;
            left_frame += 1;
        }
        while right_frame * period + 3_999_000 + delay <= now {
            let offset = if changing_partner { right_frame / 3 } else { 0 };
            signatures[right_slots[((right_frame + offset) % 3) as usize]] = right_frame + 1;
            right_frame += 1;
        }
        learner.observe(now, &signatures);
        now += if jitter {
            steps.next().unwrap()
        } else {
            1_000_000
        };
    }
    learner
}
