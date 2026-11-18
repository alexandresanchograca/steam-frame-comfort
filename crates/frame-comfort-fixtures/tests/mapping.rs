//! The snapshot copy against independently computed eye mappings: marked
//! eyes and corners, and landmarks at known view directions, through the
//! compositor's texture bounds as source selection applies them.
//!
//! The failing cases reproduce two games measured on the Frame: Asgard's
//! Wrath 2 renders each eye with a horizontally symmetric field that the
//! compositor crops, and Vader Immortal renders through OpenGL ES, bottom row
//! first. Mapping either as a whole upright image misplaces the rim.
#![cfg(all(target_os = "linux", feature = "vulkan-fixtures"))]

use comfort_core::{
    geometry::Projection,
    source::{Eye, FULL_FIELD},
};
use comfort_runtime::vulkan::{
    Context, ContextOptions, ImportedImage, SnapshotRenderer, SourceView,
    fixture::{keep_driver_loaded, uploaded_source},
};
use std::sync::Arc;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
type Rgba = [u8; 4];

const LEFT: Rgba = [200, 0, 0, 255];
const RIGHT: Rgba = [0, 0, 200, 255];
const TOP_LEFT: Rgba = [255, 255, 0, 255];
const TOP_RIGHT: Rgba = [0, 255, 0, 255];
const BOTTOM_LEFT: Rgba = [0, 255, 255, 255];
const BOTTOM_RIGHT: Rgba = [255, 0, 255, 255];
const LANDMARK: Rgba = [0, 255, 0, 255];
const BLACK: Rgba = [0, 0, 0, 255];

/// The Frame's left and right eye projections, measured with
/// `--probe-projection`.
const DISPLAY: [Projection; 2] = [
    Projection {
        left: -1.665,
        right: 1.213,
        top: -1.172,
        bottom: 1.727,
    },
    Projection {
        left: -1.2187,
        right: 1.654,
        top: -1.172,
        bottom: 1.727,
    },
];

fn context() -> Result<Arc<Context>> {
    let context = Context::new(ContextOptions::default())?;
    keep_driver_loaded(&context);
    Ok(context)
}

/// Texels from a function of the normalised texel centre, top row first.
fn picture(width: u32, height: u32, texel: impl Fn(f32, f32) -> Rgba) -> Vec<Rgba> {
    (0..height)
        .flat_map(|y| (0..width).map(move |x| (x, y)))
        .map(|(x, y)| {
            texel(
                (x as f32 + 0.5) / width as f32,
                (y as f32 + 0.5) / height as f32,
            )
        })
        .collect()
}

/// An eye colour with distinct corner blocks, so any flip or rotation and
/// the eye itself are visible.
fn marked(eye: usize) -> impl Fn(f32, f32) -> Rgba {
    move |u, v| match (u < 0.2, u > 0.8, v < 0.2, v > 0.8) {
        (true, _, true, _) => TOP_LEFT,
        (_, true, true, _) => TOP_RIGHT,
        (true, _, _, true) => BOTTOM_LEFT,
        (_, true, _, true) => BOTTOM_RIGHT,
        _ => [LEFT, RIGHT][eye],
    }
}

/// As an OpenGL producer stores it: bottom row first.
fn bottom_up(width: u32, height: u32, texels: Vec<Rgba>) -> Vec<Rgba> {
    let mut rows: Vec<_> = texels.chunks_exact(width as usize).collect();
    rows.reverse();
    assert_eq!(rows.len(), height as usize);
    rows.concat()
}

fn whole(id: u64, layer: u32) -> Eye {
    Eye {
        id,
        layout: 5,
        crop: [0.0, 0.0, 1.0, 1.0],
        layer,
        flip: [false; 2],
        field: FULL_FIELD,
    }
}

struct Snapshot {
    pixels: Vec<u8>,
    extent: [u32; 2],
}

impl Snapshot {
    fn pixel(&self, eye: usize, x: u32, y: u32) -> Rgba {
        let [width, _] = self.extent;
        let at = ((y * width * 2 + eye as u32 * width + x) * 4) as usize;
        self.pixels[at..at + 4].try_into().expect("four channels")
    }

    /// The landmark's centre column in one row, from its green coverage.
    fn landmark_column(&self, eye: usize, y: u32, near: f32, radius: f32) -> Option<f32> {
        let [width, _] = self.extent;
        let (mut weight, mut sum) = (0.0, 0.0);
        for x in 0..width {
            let centre = x as f32 + 0.5;
            if (centre - near).abs() > radius {
                continue;
            }
            let [r, g, b, _] = self.pixel(eye, x, y);
            let green = (f32::from(g) - f32::from(r.max(b))).max(0.0);
            weight += green;
            sum += green * centre;
        }
        (weight > 0.0).then(|| sum / weight)
    }
}

fn capture(
    context: &Arc<Context>,
    source: &Arc<ImportedImage>,
    eyes: [Eye; 2],
    extent: [u32; 2],
) -> Result<Snapshot> {
    let view = |eye: Eye| SourceView {
        image: source,
        layout: ash::vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        layer: eye.layer,
        crop: eye.crop,
        flip: eye.flip,
        external: false,
    };
    let mut renderer = SnapshotRenderer::new(Arc::clone(context))?;
    // SAFETY: The fixture finished uploading the source and nothing writes it.
    let mut snapshot = unsafe { renderer.capture(eyes.map(view), extent)? };
    Ok(Snapshot {
        pixels: renderer.readback(&mut snapshot)?,
        extent,
    })
}

#[track_caller]
fn assert_rgba(actual: Rgba, expected: Rgba, what: &str) {
    assert!(
        actual
            .iter()
            .zip(expected)
            .all(|(&a, e)| a.abs_diff(e) <= 3),
        "{what}: {actual:?} vs {expected:?}"
    );
}

#[track_caller]
fn assert_corners(snapshot: &Snapshot, expected: [[Rgba; 4]; 2]) {
    let [width, height] = snapshot.extent;
    for (eye, corners) in expected.iter().enumerate() {
        let at = [
            (1, 1),
            (width - 2, 1),
            (1, height - 2),
            (width - 2, height - 2),
        ];
        for (corner, &(x, y)) in corners.iter().zip(&at) {
            assert_rgba(
                snapshot.pixel(eye, x, y),
                *corner,
                &format!("eye {eye} at {x},{y}"),
            );
        }
        assert_rgba(
            snapshot.pixel(eye, width / 2, height / 2),
            [LEFT, RIGHT][eye],
            &format!("eye {eye} centre"),
        );
    }
}

const UPRIGHT: [Rgba; 4] = [TOP_LEFT, TOP_RIGHT, BOTTOM_LEFT, BOTTOM_RIGHT];
const UPSIDE_DOWN: [Rgba; 4] = [BOTTOM_LEFT, BOTTOM_RIGHT, TOP_LEFT, TOP_RIGHT];

#[test]
#[ignore = "requires a Vulkan device"]
fn an_opengl_source_is_upright_only_with_its_reversed_bounds() -> Result<()> {
    let context = context()?;
    let (width, height) = (40, 30);
    let upright = uploaded_source(
        &context,
        width,
        height,
        &[
            picture(width, height, marked(0)),
            picture(width, height, marked(1)),
        ],
    )?;
    let eyes = [whole(1, 0), whole(1, 1)];
    // Control: an upright array source with identity bounds.
    let identity = eyes.map(|eye| eye.with_bounds([0.0, 0.0, 1.0, 1.0]).unwrap());
    assert_corners(
        &capture(&context, &upright, identity, [32, 24])?,
        [UPRIGHT; 2],
    );

    let stored = uploaded_source(
        &context,
        width,
        height,
        &[
            bottom_up(width, height, picture(width, height, marked(0))),
            bottom_up(width, height, picture(width, height, marked(1))),
        ],
    )?;
    // The whole image as stored: the inverted rim seen in Vader Immortal.
    // Rows are reversed within each eye; columns and eye order are kept.
    assert_corners(
        &capture(&context, &stored, eyes, [32, 24])?,
        [UPSIDE_DOWN; 2],
    );
    // The compositor's bounds for an OpenGL swapchain reverse the rows.
    let mapped = eyes.map(|eye| eye.with_bounds([0.0, 1.0, 1.0, 0.0]).unwrap());
    assert_corners(&capture(&context, &stored, mapped, [32, 24])?, [UPRIGHT; 2]);
    Ok(())
}

/// A source eye rendered with `game` horizontally, with vertical landmark
/// bands at view tangents `marks`.
fn landmarks(width: u32, height: u32, game: [f32; 2], marks: &[f32]) -> Vec<Rgba> {
    picture(width, height, |u, _| {
        let tangent = game[0] + u * (game[1] - game[0]);
        if marks.iter().any(|mark| (tangent - mark).abs() < 0.035) {
            LANDMARK
        } else {
            BLACK
        }
    })
}

/// The compositor's horizontal bounds when the game's field `game` differs
/// from the display eye's.
fn bounds(display: Projection, game: [f32; 2]) -> [f32; 4] {
    let span = game[1] - game[0];
    [
        (display.left as f32 - game[0]) / span,
        0.0,
        (display.right as f32 - game[0]) / span,
        1.0,
    ]
}

/// Measured landmark columns for each eye and mark.
fn measure(
    context: &Arc<Context>,
    source: &Arc<ImportedImage>,
    eyes: [Eye; 2],
    expected: &[[f32; 2]],
    width: u32,
) -> Result<Vec<[f32; 2]>> {
    let snapshot = capture(context, source, eyes, [width, 16])?;
    Ok(expected
        .iter()
        .map(|columns| {
            std::array::from_fn(|eye| {
                snapshot
                    .landmark_column(eye, 8, columns[eye], width as f32 * 0.06)
                    .unwrap_or(f32::NAN)
            })
        })
        .collect())
}

#[test]
#[ignore = "requires a Vulkan device"]
fn symmetric_game_fields_land_where_the_display_shows_them() -> Result<()> {
    const WIDTH: u32 = 128;
    let context = context()?;
    // Each eye's game field is symmetric about straight ahead and as wide
    // as the display eye's wider side, as Asgard's Wrath 2 renders.
    let game = DISPLAY.map(|display| {
        let half = display.left.abs().max(display.right.abs()) as f32;
        [-half, half]
    });
    // Landmarks either side of a 24° opening's edge (tan 12° ≈ 0.21) and
    // further out in the rim.
    let marks = [-0.9_f32, -0.25, 0.0, 0.25, 0.9];
    let source = |width: u32| {
        uploaded_source(
            &context,
            width,
            32,
            &game.map(|game| landmarks(width, 32, game, &marks)),
        )
    };
    // Where the live view shows each landmark, from the display projection.
    let expected: Vec<[f32; 2]> = marks
        .iter()
        .map(|&mark| {
            DISPLAY.map(|display| {
                (mark - display.left as f32) / (display.right - display.left) as f32 * WIDTH as f32
            })
        })
        .collect();
    let mapped: [Eye; 2] = std::array::from_fn(|eye| {
        whole(1, eye as u32)
            .with_bounds(bounds(DISPLAY[eye], game[eye]))
            .unwrap()
    });
    let fine = source(400)?;
    for (mark, (measured, expected)) in marks.iter().zip(
        measure(&context, &fine, mapped, &expected, WIDTH)?
            .iter()
            .zip(&expected),
    ) {
        for eye in 0..2 {
            assert!(
                (measured[eye] - expected[eye]).abs() <= 1.0,
                "eye {eye} landmark at tan {mark}: column {} vs {}",
                measured[eye],
                expected[eye]
            );
        }
    }
    // A resolution-only change of the same field keeps every landmark.
    let coarse = source(200)?;
    let half = measure(&context, &coarse, mapped, &expected, WIDTH)?;
    let full = measure(&context, &fine, mapped, &expected, WIDTH)?;
    for (a, b) in half.iter().zip(&full) {
        for eye in 0..2 {
            assert!(
                (a[eye] - b[eye]).abs() <= 1.0,
                "resolution changed {a:?} vs {b:?}"
            );
        }
    }
    // The whole image instead of the compositor's crop: the straight-ahead
    // landmark sits mid-image, so the two eyes' rims shift in opposite
    // directions, as measured on the Frame.
    let unmapped = capture(&context, &fine, [whole(1, 0), whole(1, 1)], [WIDTH, 16])?;
    let ahead = marks.iter().position(|&mark| mark == 0.0).unwrap();
    let shift = [0, 1].map(|eye| {
        let found = unmapped
            .landmark_column(eye, 8, WIDTH as f32 / 2.0, 3.0)
            .expect("straight-ahead landmark mid-image");
        found - expected[ahead][eye]
    });
    assert!(
        shift[0] < -5.0 && shift[1] > 5.0,
        "the unmapped rim should be displaced: {shift:?}"
    );
    Ok(())
}

#[test]
#[ignore = "requires a Vulkan device"]
fn a_narrower_game_field_fills_only_its_part_of_the_display() -> Result<()> {
    const WIDTH: u32 = 96;
    let context = context()?;
    let display = DISPLAY[0];
    let span = (display.right - display.left) as f32;
    // The game covers the middle two thirds of the display eye's width.
    let game = [
        display.left as f32 + span / 6.0,
        display.left as f32 + span * 5.0 / 6.0,
    ];
    let marks = [-0.5_f32, 0.0, 0.4];
    let image = landmarks(240, 32, game, &marks);
    let source = uploaded_source(&context, 240, 32, &[image.clone(), image])?;
    let eye = whole(1, 0).with_bounds(bounds(display, game)).unwrap();
    assert!((eye.field[0] - 1.0 / 6.0).abs() < 1e-4 && (eye.field[2] - 5.0 / 6.0).abs() < 1e-4);
    // The snapshot covers only that part, with the matching projection.
    let part = display.restrict(eye.field);
    let expected: Vec<[f32; 2]> = marks
        .iter()
        .map(|&mark| {
            let column = (mark - part.left as f32) / (part.right - part.left) as f32 * WIDTH as f32;
            [column; 2]
        })
        .collect();
    let eyes = [eye, Eye { layer: 1, ..eye }];
    for (measured, expected) in measure(&context, &source, eyes, &expected, WIDTH)?
        .iter()
        .zip(&expected)
    {
        assert!(
            (measured[0] - expected[0]).abs() <= 1.0,
            "column {} vs {}",
            measured[0],
            expected[0]
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires a Vulkan device"]
fn packed_eyes_stay_on_their_own_texels_and_flip_independently() -> Result<()> {
    let context = context()?;
    // Both eyes side by side in one image, each marked, the right eye
    // stored bottom row first: a reversal inside one packed region.
    let (half, height) = (24_u32, 20_u32);
    let left = picture(half, height, marked(0));
    let right = bottom_up(half, height, picture(half, height, marked(1)));
    let texels: Vec<Rgba> = left
        .chunks_exact(half as usize)
        .zip(right.chunks_exact(half as usize))
        .flat_map(|(a, b)| a.iter().chain(b).copied())
        .collect();
    let source = uploaded_source(&context, half * 2, height, &[texels])?;
    let eyes = [
        whole(1, 0).with_bounds([0.0, 0.0, 0.5, 1.0]).unwrap(),
        whole(1, 0).with_bounds([0.5, 1.0, 1.0, 0.0]).unwrap(),
    ];
    // Upsampled, so the copy samples beyond each half's last texel centre.
    let extent = [48, 40];
    let snapshot = capture(&context, &source, eyes, extent)?;
    assert_corners(&snapshot, [UPRIGHT; 2]);
    // The inner edges show only their own eye's colour.
    let (inner_left, inner_right) = (extent[0] - 1, 0);
    let row = extent[1] / 2;
    assert_rgba(
        snapshot.pixel(0, inner_left, row),
        LEFT,
        "left eye's inner edge",
    );
    assert_rgba(
        snapshot.pixel(1, inner_right, row),
        RIGHT,
        "right eye's inner edge",
    );
    Ok(())
}
