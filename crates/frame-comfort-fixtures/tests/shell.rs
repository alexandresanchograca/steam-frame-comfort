//! Per-texel checks of the room shell shader (`restframe.frag`) on any Vulkan
//! device. A 65 px face puts the room-fixed line through the face centre on
//! a texel centre, so line and gap pixels are exact.
#![cfg(all(target_os = "linux", feature = "vulkan-fixtures"))]

use comfort_core::geometry::{Aperture, Face, Pose, Projection, Shell};
use comfort_runtime::vulkan::{
    Capture, Context, ContextOptions, Gap, ShellParams, ShellRenderer, Snapshot,
    fixture::{keep_driver_loaded, painted_snapshot},
};
use std::sync::Arc;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

const SIZE: usize = 65;
/// The vertical line through the face centre, 3 m below head height.
const LINE: (usize, usize) = (32, 56);
/// One metre left of that line, between lines.
const GAP: (usize, usize) = (24, 56);
/// Straight ahead, inside the opening.
const CENTRE: (usize, usize) = (32, 32);
/// 0.2 linear, sRGB encoded.
const LINE_SHADE: u8 = 124;
/// 0.125 linear, sRGB encoded.
const GRID_SHADE: u8 = 99;
/// The snapshot's red (left eye) and blue (right eye) channel.
const PICTURE: u8 = 128;

fn context() -> Result<Arc<Context>> {
    let context = Context::new(ContextOptions::default())?;
    keep_driver_loaded(&context);
    Ok(context)
}

fn snapshot(context: &Arc<Context>) -> Result<Snapshot> {
    painted_snapshot(context, 16, [[0.5, 0.0, 0.0, 1.0], [0.0, 0.0, 0.5, 1.0]])
}

fn base<'a>() -> ShellParams<'a> {
    ShellParams {
        shell: Shell::default(),
        display_eyes: [Pose::default(); 2],
        aperture: Aperture {
            half_width_tan: 10_f64.to_radians().tan(),
            half_height_tan: 10_f64.to_radians().tan(),
            ..Aperture::default()
        },
        gap: Gap {
            grid: false,
            background: 0.0,
            lines: 0.2,
            thickness_cm: 2.0,
            spacing: 2.0,
        },
        refresh_weight: 1.0,
        faces: 1 << Face::Front as u32,
        ..ShellParams::default()
    }
}

fn game(snapshot: &Snapshot) -> ShellParams<'_> {
    let eyes = [(Pose::default(), Projection::default()); 2];
    ShellParams {
        captures: Some([Capture { snapshot, eyes }, Capture { snapshot, eyes }]),
        ..base()
    }
}

struct Face65(Vec<u8>);

impl Face65 {
    fn pixel(&self, eye: usize, (x, y): (usize, usize)) -> [u8; 4] {
        let at = (y * SIZE * 2 + eye * SIZE + x) * 4;
        self.0[at..at + 4].try_into().expect("four channels")
    }
}

fn render(context: &Arc<Context>, params: ShellParams<'_>) -> Result<Face65> {
    let mut shell = ShellRenderer::new(Arc::clone(context), SIZE as u32)?;
    shell.render(params, 0)?;
    Ok(Face65(shell.read_face(0, Face::Front)?))
}

#[track_caller]
fn assert_rgb(actual: [u8; 4], expected: [u8; 3]) {
    assert_eq!(actual[3], 255, "periphery alpha in {actual:?}");
    for (channel, (&a, &e)) in actual.iter().zip(&expected).enumerate() {
        assert!(
            a.abs_diff(e) <= 2,
            "channel {channel}: {actual:?} vs {expected:?}"
        );
    }
}

fn picture(eye: usize) -> [u8; 3] {
    if eye == 0 {
        [PICTURE, 0, 0]
    } else {
        [0, 0, PICTURE]
    }
}

#[test]
#[ignore = "requires a Vulkan device"]
fn plain_is_black_around_a_clear_opening() -> Result<()> {
    let context = context()?;
    let face = render(&context, base())?;
    for eye in 0..2 {
        assert_eq!(face.pixel(eye, CENTRE)[3], 0, "the opening shows the game");
        assert_rgb(face.pixel(eye, GAP), [0; 3]);
        assert_rgb(face.pixel(eye, LINE), [0; 3]);
    }
    Ok(())
}

#[test]
#[ignore = "requires a Vulkan device"]
fn grid_draws_its_shade_and_lines() -> Result<()> {
    let context = context()?;
    let mut params = base();
    params.gap.grid = true;
    params.gap.background = 0.125;
    let face = render(&context, params)?;
    for eye in 0..2 {
        assert_rgb(face.pixel(eye, GAP), [GRID_SHADE; 3]);
        assert_rgb(face.pixel(eye, LINE), [LINE_SHADE; 3]);
    }
    Ok(())
}

#[test]
#[ignore = "requires a Vulkan device"]
fn grid_over_ring_draws_lines_on_plain() -> Result<()> {
    let context = context()?;
    let face = render(
        &context,
        ShellParams {
            grid_over_ring: true,
            ..base()
        },
    )?;
    for eye in 0..2 {
        assert_rgb(face.pixel(eye, GAP), [0; 3]);
        assert_rgb(face.pixel(eye, LINE), [LINE_SHADE; 3]);
    }
    Ok(())
}

#[test]
#[ignore = "requires a Vulkan device"]
fn game_shows_each_eye_and_lines_only_with_grid_over_ring() -> Result<()> {
    let context = context()?;
    let snapshot = snapshot(&context)?;
    let plain = render(&context, game(&snapshot))?;
    let lined = render(
        &context,
        ShellParams {
            grid_over_ring: true,
            ..game(&snapshot)
        },
    )?;
    for eye in 0..2 {
        assert_rgb(plain.pixel(eye, GAP), picture(eye));
        assert_rgb(plain.pixel(eye, LINE), picture(eye));
        assert_rgb(lined.pixel(eye, GAP), picture(eye));
        assert_rgb(lined.pixel(eye, LINE), [LINE_SHADE; 3]);
    }
    Ok(())
}

#[test]
#[ignore = "requires a Vulkan device"]
fn a_capture_of_part_of_the_display_leaves_the_rest_to_the_gap() -> Result<()> {
    // A game field covering only the right half of each display eye: the
    // snapshot holds that part and the shell fills the rest with the gap.
    let context = context()?;
    let snapshot = snapshot(&context)?;
    let part = Projection::default().restrict([0.5, 0.0, 1.0, 1.0]);
    let eyes = [(Pose::default(), part); 2];
    let face = render(
        &context,
        ShellParams {
            captures: Some(
                [Capture {
                    snapshot: &snapshot,
                    eyes,
                }; 2],
            ),
            ..base()
        },
    )?;
    // GAP is left of straight ahead; its mirror image is right of it.
    let right = (SIZE - 1 - GAP.0, GAP.1);
    for eye in 0..2 {
        assert_rgb(face.pixel(eye, GAP), [0; 3]);
        assert_rgb(face.pixel(eye, right), picture(eye));
    }
    Ok(())
}

#[test]
#[ignore = "requires a Vulkan device"]
fn refresh_blink_dips_to_black_and_keeps_grid_over_ring_lines() -> Result<()> {
    let context = context()?;
    let snapshot = snapshot(&context)?;
    // The darkest point of the dip, where the snapshots swap.
    let darkest = |grid_over_ring| ShellParams {
        blink: true,
        refresh_weight: 0.5,
        grid_over_ring,
        ..game(&snapshot)
    };
    let without = render(&context, darkest(false))?;
    let with = render(&context, darkest(true))?;
    for eye in 0..2 {
        assert_rgb(without.pixel(eye, GAP), [0; 3]);
        assert_rgb(without.pixel(eye, LINE), [0; 3]);
        assert_rgb(with.pixel(eye, GAP), [0; 3]);
        assert_rgb(with.pixel(eye, LINE), [LINE_SHADE; 3]);
    }
    Ok(())
}

#[test]
#[ignore = "requires a Vulkan device"]
fn dim_darkens_the_picture_but_not_the_lines() -> Result<()> {
    let context = context()?;
    let snapshot = snapshot(&context)?;
    let face = render(
        &context,
        ShellParams {
            dim: 0.5,
            grid_over_ring: true,
            ..game(&snapshot)
        },
    )?;
    // 0.5 dims linear light by 0.5^2.2: 128 encodes to 61.
    let dimmed = |eye| picture(eye).map(|value| if value == 0 { 0 } else { 61 });
    for eye in 0..2 {
        assert_rgb(face.pixel(eye, GAP), dimmed(eye));
        assert_rgb(face.pixel(eye, LINE), [LINE_SHADE; 3]);
    }
    Ok(())
}

// Each eye and each quadrant has a distinct colour to catch wrong eye,
// wrong matrix row and flipped image axes. A separate CPU lookup uses the
// capture projections and bilinear sampling of the packed eye coordinates.
const TILES: [[[u8; 3]; 4]; 2] = [
    [[180, 10, 30], [90, 80, 10], [60, 150, 10], [220, 60, 30]],
    [[10, 60, 180], [10, 20, 200], [30, 140, 80], [30, 80, 220]],
];

#[test]
#[ignore = "requires a Vulkan device"]
fn all_faces_match_translated_and_canted_capture_reference() -> Result<()> {
    use comfort_core::geometry::{Quaternion, Vec3};
    use comfort_runtime::vulkan::fixture::tiled_snapshot;

    const EYE_SIZE: u32 = 16;
    let context = context()?;
    let snapshot = tiled_snapshot(
        &context,
        EYE_SIZE,
        TILES.map(|eye| {
            eye.map(|[r, g, b]| [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0, 1.0])
        }),
    )?;
    let rotate = |axis: Vec3, angle: f64| Quaternion {
        w: (angle / 2.0).cos(),
        axis: axis * (angle / 2.0).sin(),
    };
    let head = Pose {
        position: Vec3(0.24, -0.17, 0.5),
        rotation: rotate(Vec3(0.0, 1.0, 0.0), 0.81)
            .then_local(rotate(Vec3(1.0, 0.0, 0.0), -0.31))
            .then_local(rotate(Vec3(0.0, 0.0, 1.0), 0.2)),
    };
    let capture = Capture {
        snapshot: &snapshot,
        eyes: [-1.0, 1.0].map(|side| {
            (
                head.compose(Pose {
                    position: Vec3(side * 0.035, 0.0, 0.0),
                    rotation: rotate(Vec3(0.0, 1.0, 0.0), side * 0.13),
                }),
                Projection {
                    left: -1.3 + side * 0.14,
                    right: 0.94 + side * 0.1,
                    top: -0.81,
                    bottom: 1.1,
                },
            )
        }),
    };
    let shell = Shell {
        pose: Pose {
            position: Vec3(0.12, 0.20, -0.15),
            rotation: rotate(Vec3(0.0, 1.0, 0.0), -0.18),
        },
        ..Shell::default()
    };
    let aperture = Aperture {
        half_width_tan: 0.64,
        half_height_tan: 0.52,
        strength: 0.86,
        entry: 0.71,
        ..Default::default()
    };
    let params = ShellParams {
        shell,
        aperture,
        captures: Some([capture; 2]),
        gap: Gap {
            background: 0.125,
            ..base().gap
        },
        faces: 0x3f,
        ..base()
    };
    let mut renderer = ShellRenderer::new(context, SIZE as u32)?;
    let mut displayed = [0; 3];
    for display in [Pose::default(), head] {
        let display_eyes = capture
            .eyes
            .map(|(eye, _)| display.head_aligned_eye(eye.position));
        renderer.render(
            ShellParams {
                display_eyes,
                ..params
            },
            0,
        )?;
        for face in Face::ALL {
            let pixels = Face65(renderer.read_face(0, face)?);
            for (eye, (display_eye, (capture_eye, projection))) in
                display_eyes.into_iter().zip(capture.eyes).enumerate()
            {
                for y in 0..SIZE {
                    for x in 0..SIZE {
                        let world = shell
                            .face_point(
                                face,
                                [
                                    (x as f64 + 0.5) / SIZE as f64,
                                    (y as f64 + 0.5) / SIZE as f64,
                                ],
                            )
                            .unwrap();
                        let alpha = aperture.peripheral_alpha(display_eye, world).unwrap();
                        let actual = pixels.pixel(eye, (x, y));
                        let sample = projection.project(capture_eye, world);
                        let expected = sample.map_or([GRID_SHADE; 3], |[u, v]| {
                            let blend =
                                |at: f64| ((at - 0.5) * f64::from(EYE_SIZE) + 0.5).clamp(0.0, 1.0);
                            let [r, d] = [blend(u), blend(v)];
                            let mix =
                                |a: u8, b: u8, t: f64| f64::from(a) * (1.0 - t) + f64::from(b) * t;
                            std::array::from_fn(|channel| {
                                let colours = TILES[eye].map(|c| c[channel]);
                                (mix(colours[0], colours[1], r) * (1.0 - d)
                                    + mix(colours[2], colours[3], r) * d)
                                    .round() as u8
                            })
                        });
                        let expected = [
                            expected[0],
                            expected[1],
                            expected[2],
                            (alpha * 255.0).round() as u8,
                        ];
                        assert!(
                            actual
                                .into_iter()
                                .zip(expected)
                                .all(|(a, b)| a.abs_diff(b) <= 2),
                            "{face:?} eye {eye} at ({x}, {y}): {actual:?} != {expected:?}"
                        );
                        displayed[0] += usize::from(alpha == 0.0);
                        displayed[1] += usize::from(alpha > 0.0 && sample.is_some());
                        displayed[2] += usize::from(alpha > 0.0 && sample.is_none());
                    }
                }
            }
        }
        // SAFETY: The local readbacks are complete and no external consumer exists.
        unsafe { renderer.retire(0)? };
    }
    assert!(displayed.into_iter().all(|count| count > 0));
    Ok(())
}

#[test]
#[ignore = "requires a Vulkan device"]
fn shell_leases_retire_rebind_and_render_selected_faces() -> Result<()> {
    let context = context()?;
    let first = snapshot(&context)?;
    let second = painted_snapshot(&context, 16, [[0.0, 0.5, 0.0, 1.0]; 2])?;
    let eyes = [(Pose::default(), Projection::default()); 2];
    let old = Capture {
        snapshot: &first,
        eyes,
    };
    let new = Capture {
        snapshot: &second,
        eyes,
    };
    let mut renderer = ShellRenderer::new(context, SIZE as u32)?;
    let mut params = ShellParams {
        captures: Some([old, new]),
        refresh_weight: 0.0,
        faces: 0x3f,
        ..base()
    };
    assert!(
        renderer
            .render(
                ShellParams {
                    faces: 0x40,
                    ..params
                },
                0
            )
            .is_err()
    );
    assert!(
        renderer
            .render(
                ShellParams {
                    refresh_weight: 1.1,
                    ..params
                },
                0
            )
            .is_err()
    );
    assert!(
        renderer
            .render(
                ShellParams {
                    gap: Gap {
                        spacing: 0.0,
                        ..params.gap
                    },
                    ..params
                },
                0
            )
            .is_err()
    );
    renderer.render(params, 0)?;
    assert!(renderer.render(params, 0).is_err());
    let full = Face::ALL.map(|face| renderer.read_face(0, face).expect("all faces rendered"));
    let mask = (1 << Face::Front as u32) | (1 << Face::Left as u32);
    renderer.render(
        ShellParams {
            faces: mask,
            ..params
        },
        1,
    )?;
    for (index, face) in Face::ALL.into_iter().enumerate() {
        if mask & (1 << index) != 0 {
            assert_eq!(renderer.read_face(1, face)?, full[index]);
        } else {
            assert!(renderer.output_image(1, face).is_none());
            assert!(renderer.read_face(1, face).is_err());
        }
    }
    // SAFETY: There are no external consumers.
    unsafe { renderer.retire(0)? };
    assert!(renderer.output_image(0, Face::Front).is_none());
    params.refresh_weight = 1.0;
    renderer.render(params, 0)?;
    let green = renderer.read_face(0, Face::Front)?;
    let expected = Face65(green.clone()).pixel(0, GAP);
    assert_rgb(expected, [0, 128, 0]);
    // SAFETY: There are no external consumers.
    unsafe { renderer.retire(0)? };
    params.captures = Some([new, old]);
    renderer.render(params, 0)?;
    assert_eq!(renderer.read_face(0, Face::Front)?, full[0]);

    let front = |renderer: &mut ShellRenderer, set| -> Result<Vec<u8>> {
        renderer.read_face(set, Face::Front)
    };
    params.faces = 1 << Face::Front as u32;
    params.refresh_weight = 0.5;
    renderer.render(params, 2)?;
    let blend = Face65(front(&mut renderer, 2)?);
    // Half energy in each of the red and green channels, in sRGB.
    assert_rgb(blend.pixel(0, GAP), [92, 92, 0]);
    assert!(
        blend
            .0
            .iter()
            .skip(3)
            .step_by(4)
            .eq(green.iter().skip(3).step_by(4))
    );
    // SAFETY: There are no external consumers.
    unsafe { renderer.retire(2)? };
    params.aperture.entry = 0.0;
    renderer.render(params, 2)?;
    assert!(
        front(&mut renderer, 2)?
            .iter()
            .skip(3)
            .step_by(4)
            .all(|&alpha| alpha == 0)
    );
    // SAFETY: There are no external consumers.
    unsafe { renderer.retire(2)? };
    params.aperture.entry = 1.0;
    params.aperture.strength = 0.0;
    renderer.render(params, 2)?;
    assert!(
        front(&mut renderer, 2)?
            .iter()
            .skip(3)
            .step_by(4)
            .all(|&alpha| alpha == 0)
    );
    Ok(())
}
