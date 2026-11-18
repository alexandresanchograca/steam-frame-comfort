//! Scene selection from Vulkan observer reports, without GPU calls.
use crate::stereo::{Layout, Packing};
pub use comfort_protocol::EyeBounds;
use comfort_protocol::{EyeUse, ImageAdd, MAX_IMAGES};
use std::collections::BTreeMap;

// VkFormat and VkImageLayout wire values, including the stock stream's P010.
const P010: u32 = 1_000_156_013;
const GENERAL: u32 = 1;
const SHADER_READ_ONLY: u32 = 5;

#[derive(Clone, Copy, Debug)]
pub struct Candidate {
    pub image: ImageAdd,
    pub layout: u32,
    pub acquired: bool,
    pub eye: EyeUse,
    /// The compositor's texture bounds for each labelled eye, when known.
    pub bounds: EyeBounds,
    pub revision: u64,
}

impl Candidate {
    fn stream(self) -> bool {
        self.acquired && self.image.exported == 0 && self.image.format == P010
    }

    fn dimensions(self) -> [u32; 2] {
        [self.image.width, self.image.height]
    }

    fn area(self) -> u64 {
        u64::from(self.image.width) * u64::from(self.image.height)
    }

    fn eye(self) -> Eye {
        Eye {
            id: self.image.id,
            layout: self.layout,
            crop: FULL,
            layer: 0,
            flip: [false; 2],
            field: FULL_FIELD,
        }
    }
}

pub struct Policy<'a> {
    pub complete: bool,
    pub epoch: u64,
    pub min_scene_size: u32,
    pub source_format: Option<u32>,
    pub excluded_ids: &'a [u64],
    pub excluded_sizes: &'a [[u32; 2]],
}

impl Default for Policy<'_> {
    fn default() -> Self {
        Self {
            complete: true,
            epoch: 0,
            min_scene_size: 512,
            source_format: None,
            excluded_ids: &[],
            excluded_sizes: &[],
        }
    }
}

const FULL: [f32; 4] = [0.0, 0.0, 1.0, 1.0];
/// The whole display eye, as `[x0, y0, x1, y1]` fractions of its field.
pub const FULL_FIELD: [f32; 4] = [0.0, 0.0, 1.0, 1.0];

/// One eye's picture in a source image and the part of the display eye it
/// covers. Rows and columns are normalised texture coordinates, top-left
/// first, and the display field runs left to right and top to bottom.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Eye {
    pub id: u64,
    pub layout: u32,
    /// Source rectangle `[x, y, width, height]`, positive size.
    pub crop: [f32; 4],
    pub layer: u32,
    /// Whether the rectangle's columns (x) or rows (y) run against the
    /// display's: an OpenGL image is stored bottom row first.
    pub flip: [bool; 2],
    /// The part of the display eye that the rectangle fills.
    pub field: [f32; 4],
}

impl Eye {
    /// Map this eye as the compositor does: display coordinate `d` in the
    /// eye's field shows source coordinate `bounds.xy + (bounds.zw - bounds.xy) * d`,
    /// as OpenVR `VRTextureBounds_t`. A reversed axis flips the picture.
    /// Bounds outside the texture leave part of the display without picture.
    pub fn with_bounds(self, bounds: [f32; 4]) -> Option<Self> {
        if !bounds.iter().all(|value| value.is_finite()) {
            return None;
        }
        let mut crop = [0.0; 4];
        let mut field = [0.0; 4];
        let mut flip = [false; 2];
        for axis in 0..2 {
            let start = f64::from(bounds[axis]);
            let span = f64::from(bounds[axis + 2]) - start;
            if span.abs() < 1e-4 {
                return None;
            }
            // Display coordinates where the source coordinate is 0 and 1.
            let [a, b] = [-start / span, (1.0 - start) / span];
            let low = a.min(b).max(0.0);
            let high = a.max(b).min(1.0);
            if high - low < 0.01 {
                return None;
            }
            let [s0, s1] = [start + span * low, start + span * high];
            crop[axis] = s0.min(s1).clamp(0.0, 1.0) as f32;
            crop[axis + 2] = ((s1 - s0).abs() as f32).min(1.0 - crop[axis]);
            field[axis] = low as f32;
            field[axis + 2] = high as f32;
            flip[axis] = span < 0.0;
        }
        Some(Self {
            crop,
            flip,
            field,
            ..self
        })
    }

    /// The source coordinate shown at display coordinate `display`, which
    /// must lie in this eye's field.
    pub fn source_at(&self, display: [f32; 2]) -> [f32; 2] {
        std::array::from_fn(|axis| {
            let [low, high] = [self.field[axis], self.field[axis + 2]];
            let along = (display[axis] - low) / (high - low);
            let along = if self.flip[axis] { 1.0 - along } else { along };
            self.crop[axis] + along * self.crop[axis + 2]
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reason {
    NoSource,
    Ambiguous,
    UnsupportedLayout,
    InvalidDescription,
    IncompleteReport,
}

impl Reason {
    /// The `gameSource` status name, shared with the C++ baseline.
    pub fn name(self) -> &'static str {
        match self {
            Self::NoSource => "no-stereo-source",
            Self::Ambiguous => "ambiguous-source",
            Self::UnsupportedLayout => "unsupported-stereo-layout",
            Self::InvalidDescription => "invalid-source-description",
            Self::IncompleteReport => "incomplete-stereo-report",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rule {
    Stream2x2,
    LocalSlots,
    LocalSideBySide,
    LocalAtlas,
    LocalArray,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Generation {
    pub epoch: u64,
    pub format: u32,
    pub extent: [u32; 2],
    pub layers: u32,
    pub images: Vec<(u64, u64)>, // ID and revision
}

pub struct Selection {
    pub generation: Generation,
    pub images: Vec<Eye>,
    pub stream: bool,
    labels: Vec<EyeUse>,
    bounds: Vec<EyeBounds>,
}

#[derive(Debug, PartialEq)]
pub struct Stereo {
    pub rule: Rule,
    pub slots: Vec<[Eye; 2]>,
    pub labelled: bool,
}

pub fn select(frame: &[Candidate], policy: Policy<'_>) -> Result<Selection, Reason> {
    if !policy.complete {
        return Err(Reason::IncompleteReport);
    }
    let mut candidates = Vec::new();
    for &candidate in frame {
        let d = candidate.image;
        if policy.excluded_ids.contains(&d.id)
            || policy.excluded_sizes.contains(&candidate.dimensions())
            || policy
                .source_format
                .is_some_and(|format| format != d.format)
        {
            continue;
        }
        if !candidate.stream()
            && (candidate.acquired
                || d.exported == 0
                || d.width < policy.min_scene_size
                || d.height < policy.min_scene_size)
        {
            continue;
        }
        if d.id == 0
            || d.width == 0
            || d.height == 0
            || d.array_layers == 0
            || !matches!(candidate.layout, GENERAL | SHADER_READ_ONLY)
        {
            return Err(Reason::InvalidDescription);
        }
        candidates.push(candidate);
    }
    if candidates.is_empty() {
        return Err(Reason::NoSource);
    }
    candidates.sort_unstable_by_key(|candidate| candidate.image.id);
    if candidates
        .windows(2)
        .any(|pair| pair[0].image.id == pair[1].image.id)
    {
        return Err(Reason::Ambiguous);
    }
    let stream = candidates.iter().copied().any(Candidate::stream);
    if stream {
        candidates.retain(|candidate| candidate.stream());
        let size = candidates[0].dimensions();
        if candidates
            .iter()
            .any(|candidate| candidate.image.array_layers != 1)
        {
            return Err(Reason::UnsupportedLayout);
        }
        if candidates
            .iter()
            .any(|candidate| candidate.dimensions() != size)
        {
            return Err(Reason::Ambiguous);
        }
    } else {
        // A compositor overlay can be larger than the scene's eye images.
        // A shape represented by one image cannot form a local stereo ring.
        let shape = |candidate: &Candidate| {
            let image = candidate.image;
            (image.format, image.width, image.height, image.array_layers)
        };
        let mut groups = BTreeMap::new();
        for candidate in &candidates {
            *groups.entry(shape(candidate)).or_insert(0) += 1;
        }
        candidates.retain(|candidate| groups[&shape(candidate)] >= 2);
        if candidates.is_empty() {
            return Err(Reason::NoSource);
        }
        let best = candidates
            .iter()
            .map(|candidate| candidate.area())
            .max()
            .unwrap_or_default();
        candidates.retain(|candidate| candidate.area() == best);
        let first = candidates[0];
        if candidates.iter().any(|candidate| {
            candidate.dimensions() != first.dimensions()
                || candidate.image.format != first.image.format
                || candidate.image.array_layers != first.image.array_layers
        }) {
            return Err(Reason::Ambiguous);
        }
        if !(1..=2).contains(&first.image.array_layers) || candidates.len() < 2 {
            return Err(Reason::UnsupportedLayout);
        }
    }
    if candidates.len() > MAX_IMAGES {
        return Err(Reason::Ambiguous);
    }
    let first = candidates[0];
    Ok(Selection {
        generation: Generation {
            epoch: policy.epoch,
            format: first.image.format,
            extent: first.dimensions(),
            layers: first.image.array_layers,
            images: candidates
                .iter()
                .map(|candidate| (candidate.image.id, candidate.revision))
                .collect(),
        },
        images: candidates.iter().copied().map(Candidate::eye).collect(),
        labels: candidates.iter().map(|candidate| candidate.eye).collect(),
        bounds: candidates
            .iter()
            .map(|candidate| candidate.bounds)
            .collect(),
        stream,
    })
}

impl Selection {
    /// Prefer direct eye labels when they form a complete and unambiguous set.
    pub fn from_labels(&self) -> Option<Stereo> {
        if self.stream {
            return None;
        }
        let mut left = BTreeMap::new();
        let mut right = BTreeMap::new();
        let mut layered = Vec::new();
        let layers = self.generation.layers;
        for ((&image, &label), bounds) in self.images.iter().zip(&self.labels).zip(&self.bounds) {
            // The layer the compositor sampled for this eye, mapped with its
            // texture bounds when they were read. Unusable bounds keep the
            // whole image, as before bounds were reported.
            let eye = |eye: usize| {
                let image = Eye {
                    layer: label.layers[eye],
                    ..image
                };
                bounds
                    .get(eye)
                    .and_then(|bounds| image.with_bounds(bounds))
                    .unwrap_or(image)
            };
            match label.eyes {
                1 if label.layers[0] < layers && left.insert(label.slot, eye(0)).is_none() => {}
                2 if label.layers[1] < layers && right.insert(label.slot, eye(1)).is_none() => {}
                3 if label.layers[0] != label.layers[1]
                    && label.layers.into_iter().all(|layer| layer < layers) =>
                {
                    layered.push([eye(0), eye(1)]);
                }
                _ => return None,
            }
        }
        let (rule, slots) = if left.is_empty() && right.is_empty() {
            if layered.is_empty() {
                return None;
            }
            (Rule::LocalArray, layered)
        } else {
            if !layered.is_empty() || left.len() != right.len() {
                return None;
            }
            let slots = left
                .into_iter()
                .map(|(slot, left)| Some([left, *right.get(&slot)?]))
                .collect::<Option<Vec<_>>>()?;
            (Rule::LocalSlots, slots)
        };
        Some(Stereo {
            rule,
            slots,
            labelled: true,
        })
    }

    pub fn packed_hypotheses(&self) -> &'static [Packing] {
        if self.stream {
            &[Packing::LeftQuadrants]
        } else if self.generation.layers == 2 {
            &[Packing::Array]
        } else {
            &[
                Packing::SideBySide,
                Packing::TopBottom,
                Packing::LeftQuadrants,
                Packing::RightQuadrants,
                Packing::TopQuadrants,
                Packing::BottomQuadrants,
                Packing::MainDiagonal,
                Packing::OtherDiagonal,
            ]
        }
    }

    pub fn from_proof(&self, layout: Layout, pairs: &[[usize; 2]]) -> Option<Stereo> {
        let (rule, slots) = if layout.packing == Packing::SeparateEyes {
            if self.stream
                || self.generation.layers != 1
                || pairs.len() < 2
                || pairs.len() * 2 != self.images.len()
            {
                return None;
            }
            let mut seen = [false; MAX_IMAGES];
            let mut slots = Vec::new();
            for &[left, right] in pairs {
                let eyes = [*self.images.get(left)?, *self.images.get(right)?];
                for index in [left, right] {
                    if std::mem::replace(&mut seen[index], true) {
                        return None;
                    }
                }
                slots.push(eyes);
            }
            (Rule::LocalSlots, slots)
        } else {
            if !self.packed_hypotheses().contains(&layout.packing) {
                return None;
            }
            let slots = self
                .images
                .iter()
                .map(|&image| {
                    let mut eyes = packed_eyes(image, layout.packing);
                    if layout.swapped {
                        eyes.swap(0, 1);
                    }
                    eyes
                })
                .collect();
            let rule = if self.stream {
                Rule::Stream2x2
            } else {
                match layout.packing {
                    Packing::Array => Rule::LocalArray,
                    Packing::SideBySide => Rule::LocalSideBySide,
                    _ => Rule::LocalAtlas,
                }
            };
            (rule, slots)
        };
        Some(Stereo {
            rule,
            slots,
            labelled: false,
        })
    }
}

pub fn packed_eyes(image: Eye, packing: Packing) -> [Eye; 2] {
    packing.eye_regions().map(|region| Eye {
        crop: region.crop,
        layer: region.layer,
        ..image
    })
}

#[derive(Default)]
pub struct SlotTracker {
    slots: BTreeMap<u64, (u64, u64)>, // signature and last observed change
    chosen: Option<(u64, u64, bool)>, // ID, changed time and stable
    pub needs_confirmation: bool,
}

impl SlotTracker {
    pub fn update(&mut self, signatures: &[(u64, u64)], now_ns: u64) -> Option<u64> {
        let previous_choice = self.chosen.map(|(id, _, _)| id);
        let observed: Vec<_> = signatures
            .iter()
            .map(|&(id, signature)| {
                let state = self.slots.entry(id).or_insert((signature, 0));
                let stable = state.0 == signature;
                if !stable {
                    *state = (signature, now_ns);
                }
                (id, state.1, stable)
            })
            .collect();
        self.chosen = observed.iter().copied().reduce(|best, candidate| {
            let rank = |(id, changed, stable)| (stable, changed, Some(id) == previous_choice);
            if rank(candidate) > rank(best) {
                candidate
            } else {
                best
            }
        });
        self.needs_confirmation = self.chosen.is_some_and(|(_, changed, _)| {
            observed
                .iter()
                .any(|&(_, time, stable)| !stable && time > changed)
        });
        self.chosen.map(|(id, _, _)| id)
    }

    pub fn capture_slot(&self, confirmed: bool) -> Option<u64> {
        let (id, changed, stable) = self.chosen?;
        (id != 0 && changed != 0 && stable && (!self.needs_confirmation || confirmed)).then_some(id)
    }

    pub fn changed_ns(&self) -> u64 {
        self.chosen.map_or(0, |(_, changed, _)| changed)
    }

    pub fn signature(&self, id: u64) -> Option<u64> {
        self.slots.get(&id).map(|&(signature, _)| signature)
    }
}

#[derive(Default)]
pub struct SlotSampling {
    sequence: u64,
    confirmation_ns: Option<u64>,
    confirming: bool,
}

impl SlotSampling {
    pub fn due(&mut self, sequence: u64, now: u64, reset: bool) -> bool {
        if reset || sequence != self.sequence {
            self.sequence = sequence;
            self.confirmation_ns = None;
            self.confirming = false;
            true
        } else if self.confirmation_ns.is_some_and(|deadline| now >= deadline) {
            self.confirmation_ns = None;
            self.confirming = true;
            true
        } else {
            false
        }
    }

    pub fn sampled(&mut self, needs_confirmation: bool, now: u64) {
        self.confirmation_ns =
            (needs_confirmation && !self.confirming).then(|| now.saturating_add(500_000));
    }

    pub fn wait_ns(&self, now: u64, maximum: u64) -> u64 {
        self.confirmation_ns.map_or(maximum, |deadline| {
            maximum.min(deadline.saturating_sub(now))
        })
    }
}

#[cfg(test)]
mod policy_tests;

#[cfg(test)]
mod tests {
    use super::*;

    fn candidate(id: u64, eye: u32, slot: u32) -> Candidate {
        Candidate {
            image: ImageAdd {
                id,
                width: 1024,
                height: 1024,
                array_layers: 1,
                format: 37,
                exported: 1,
                ..Default::default()
            },
            layout: SHADER_READ_ONLY,
            acquired: false,
            eye: EyeUse {
                eyes: eye,
                slot,
                layers: [0, 0],
            },
            bounds: EyeBounds::default(),
            revision: 1,
        }
    }

    #[test]
    fn eye_labels_require_one_partner_per_binding_regardless_of_report_order() {
        let mut frame = [
            candidate(3, 2, 2),
            candidate(4, 2, 1),
            candidate(1, 1, 1),
            candidate(2, 1, 2),
        ];
        let selected = select(&frame, Policy::default())
            .unwrap()
            .from_labels()
            .unwrap();
        assert_eq!(selected.rule, Rule::LocalSlots);
        assert_eq!(
            selected
                .slots
                .iter()
                .map(|eyes| eyes.map(|eye| eye.id))
                .collect::<Vec<_>>(),
            [[1, 4], [2, 3]]
        );
        frame[0].eye.slot = 1;
        assert!(
            select(&frame, Policy::default())
                .unwrap()
                .from_labels()
                .is_none()
        );
    }

    #[test]
    fn a_single_larger_overlay_cannot_displace_a_labelled_scene_ring() {
        let mut overlay = candidate(7, 3, 9);
        overlay.image.width = 1860;
        overlay.image.height = 2048;
        overlay.image.format = 50;
        let frame = [
            overlay,
            candidate(1, 1, 9),
            candidate(2, 1, 10),
            candidate(3, 2, 9),
            candidate(4, 2, 10),
        ];
        let scene = select(&frame, Policy::default())
            .unwrap()
            .from_labels()
            .unwrap();
        assert_eq!(
            scene
                .slots
                .iter()
                .map(|eyes| eyes.map(|eye| eye.id))
                .collect::<Vec<_>>(),
            [[1, 3], [2, 4]]
        );
        assert!(matches!(
            select(&[overlay], Policy::default()),
            Err(Reason::NoSource)
        ));
    }

    fn eye() -> Eye {
        candidate(1, 3, 9).eye()
    }

    #[track_caller]
    fn assert_close(actual: [f32; 4], expected: [f32; 4]) {
        for (a, e) in actual.iter().zip(expected) {
            assert!((a - e).abs() < 1e-5, "{actual:?} vs {expected:?}");
        }
    }

    #[test]
    fn identity_bounds_keep_the_whole_upright_image() {
        let mapped = eye().with_bounds([0.0, 0.0, 1.0, 1.0]).unwrap();
        assert_eq!(mapped, eye());
    }

    #[test]
    fn symmetric_game_fields_crop_each_eye_like_the_compositor() {
        // Asgard's Wrath 2, measured on the Frame: each eye renders a
        // horizontally symmetric field and the compositor shows only the
        // part inside the headset's asymmetric eye.
        let left = eye().with_bounds([0.0, 0.0, 0.867_11, 1.0]).unwrap();
        assert_close(left.crop, [0.0, 0.0, 0.867_11, 1.0]);
        assert_eq!(left.flip, [false; 2]);
        assert_eq!(left.field, FULL_FIELD);
        let right = eye().with_bounds([0.134_15, 0.0, 1.0, 1.0]).unwrap();
        assert_close(right.crop, [0.134_15, 0.0, 0.865_85, 1.0]);
        // The display's left edge shows the cropped source's left edge.
        assert_eq!(right.source_at([0.0, 0.5]), [0.134_15, 0.5]);
    }

    #[test]
    fn reversed_rows_flip_an_opengl_image_but_keep_its_columns() {
        // Vader Immortal renders through OpenGL ES, bottom row first.
        let mapped = eye().with_bounds([0.0, 1.0, 1.0, 0.0]).unwrap();
        assert_eq!(mapped.crop, FULL);
        assert_eq!(mapped.flip, [false, true]);
        assert_eq!(mapped.source_at([0.25, 0.0]), [0.25, 1.0]);
        assert_eq!(mapped.source_at([0.25, 1.0]), [0.25, 0.0]);
        // A 180° rotation reverses both axes.
        let rotated = eye().with_bounds([1.0, 1.0, 0.0, 0.0]).unwrap();
        assert_eq!(rotated.flip, [true, true]);
    }

    #[test]
    fn a_narrower_game_field_covers_only_part_of_the_display() {
        // Source coordinates -0.25..1.25 across the display: the image fills
        // the middle two thirds of the display eye's width.
        let mapped = eye().with_bounds([-0.25, 0.0, 1.25, 1.0]).unwrap();
        assert_close(mapped.crop, FULL);
        assert_close(mapped.field, [1.0 / 6.0, 0.0, 5.0 / 6.0, 1.0]);
        for x in [0.2_f32, 0.5, 0.8] {
            let expected = -0.25 + 1.5 * x;
            assert!((mapped.source_at([x, 0.5])[0] - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn bounds_without_a_usable_overlap_are_rejected() {
        assert_eq!(eye().with_bounds([1.5, 0.0, 2.5, 1.0]), None);
        assert_eq!(eye().with_bounds([0.5, 0.0, 0.5, 1.0]), None);
        assert_eq!(eye().with_bounds([f32::NAN, 0.0, 1.0, 1.0]), None);
    }

    #[test]
    fn labelled_eyes_take_their_layer_and_bounds() {
        let mut array = candidate(5, 3, 9);
        array.image.array_layers = 2;
        array.eye.layers = [0, 1];
        array.bounds.set(0, [0.0, 0.0, 0.867, 1.0]);
        array.bounds.set(1, [0.0, 1.0, 1.0, 0.0]);
        let mut sibling = array;
        sibling.image.id = 6;
        let stereo = select(&[array, sibling], Policy::default())
            .unwrap()
            .from_labels()
            .unwrap();
        assert_eq!(stereo.rule, Rule::LocalArray);
        let [left, right] = stereo.slots[0];
        assert_eq!((left.layer, right.layer), (0, 1));
        assert_close(left.crop, [0.0, 0.0, 0.867, 1.0]);
        assert_eq!(right.flip, [false, true]);

        // A single-eye label reports the layer it sampled.
        let mut frame = [candidate(1, 1, 1), candidate(2, 2, 1)];
        for candidate in &mut frame {
            candidate.image.array_layers = 2;
        }
        frame[1].eye.layers = [0, 1];
        let stereo = select(&frame, Policy::default())
            .unwrap()
            .from_labels()
            .unwrap();
        assert_eq!(stereo.slots[0].map(|eye| eye.layer), [0, 1]);
        frame[1].eye.layers = [0, 2];
        assert!(
            select(&frame, Policy::default())
                .unwrap()
                .from_labels()
                .is_none()
        );
    }

    #[test]
    fn ring_capture_waits_for_confirmation_when_a_newer_slot_is_changing() {
        let mut tracker = SlotTracker::default();
        tracker.update(&[(1, 1), (2, 2)], 1);
        tracker.update(&[(1, 3), (2, 2)], 2);
        tracker.update(&[(1, 3), (2, 4)], 3);
        assert!(tracker.needs_confirmation);
        assert_eq!(tracker.capture_slot(false), None);
        assert_eq!(tracker.capture_slot(true), Some(1));
        tracker.update(&[(1, 3), (2, 4)], 4);
        assert_eq!(tracker.capture_slot(false), Some(2));
    }
}
