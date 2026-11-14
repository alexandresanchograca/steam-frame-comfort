//! Horizontal detail correlation for identifying and ordering stereo images.
pub const THUMBNAIL_SIZE: usize = 64;
pub type LumaThumbnail = [u8; THUMBNAIL_SIZE * THUMBNAIL_SIZE];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StereoShift {
    pub fraction: f64,
    pub score: f64,
    pub alternative: f64,
    pub textured: bool,
    pub covered: bool,
}

impl Default for StereoShift {
    fn default() -> Self {
        Self {
            fraction: 0.0,
            score: -1.0,
            alternative: -1.0,
            textured: false,
            covered: false,
        }
    }
}

impl StereoShift {
    pub fn strong(self) -> bool {
        self.covered
            && (0.12..=0.30).contains(&self.fraction.abs())
            && self.score >= 0.35
            && self.score - self.alternative >= 0.15
    }
}

pub struct StereoFeatures {
    detail: [i16; THUMBNAIL_SIZE * THUMBNAIL_SIZE],
    valid: [bool; THUMBNAIL_SIZE * THUMBNAIL_SIZE],
}

/// Luma of scene content the thresholds below were set on. Dim scenes are
/// scaled up towards it before comparing.
const REFERENCE_LUMA: f32 = 96.0;
/// The most a dim thumbnail is scaled up.
const MAX_GAIN: f32 = 6.0;
/// Raw luma at or below this is black, such as a hidden-area mask.
const BLACK: u8 = 1;

impl StereoFeatures {
    pub fn new(pixels: &LumaThumbnail) -> Self {
        Self::with_gain(pixels, brightness_gain(pixels))
    }

    fn with_gain(pixels: &LumaThumbnail, gain: f32) -> Self {
        let mut features = Self {
            detail: [0; THUMBNAIL_SIZE * THUMBNAIL_SIZE],
            valid: [false; THUMBNAIL_SIZE * THUMBNAIL_SIZE],
        };
        // Thresholds on 8-bit luma alone reject a dim scene whose pictures
        // still match. Scale each thumbnail so its brighter content reaches
        // the reference level; a typical scene keeps a gain of 1.
        let level = |index: usize| (f32::from(pixels[index]) * gain).round().min(255.0) as i16;
        // Ignore the edges of hidden area masks, and cap detail so a HUD
        // or a frame code rendered in both eyes cannot dominate the score.
        let width = THUMBNAIL_SIZE;
        for y in 2..width - 2 {
            for x in 2..width - 2 {
                let at = y * width + x;
                features.valid[at] = [at, at - 2, at + 2, at - 2 * width, at + 2 * width]
                    .into_iter()
                    .all(|index| pixels[index] > BLACK && level(index) > 5);
                features.detail[at] = (level(at + 1) - level(at - 1)).clamp(-32, 32);
            }
        }
        features
    }
}

/// The gain that brings the 90th percentile of non-black interior luma to
/// the reference level, between 1 and the maximum.
fn brightness_gain(pixels: &LumaThumbnail) -> f32 {
    let width = THUMBNAIL_SIZE;
    let mut histogram = [0_u32; 256];
    let mut count = 0;
    for y in 2..width - 2 {
        for x in 2..width - 2 {
            let value = pixels[y * width + x];
            if value > BLACK {
                histogram[usize::from(value)] += 1;
                count += 1;
            }
        }
    }
    if count < 64 {
        return 1.0;
    }
    let mut seen = 0;
    let p90 = histogram
        .iter()
        .position(|&n| {
            seen += n;
            seen * 10 >= count * 9
        })
        .unwrap_or(255);
    (REFERENCE_LUMA / p90.max(1) as f32).clamp(1.0, MAX_GAIN)
}

fn correlation(
    first: &StereoFeatures,
    second: &StereoFeatures,
    shift: i32,
    x: std::ops::Range<i32>,
    y: std::ops::Range<i32>,
) -> f64 {
    let area = (x.end - x.start) * (y.end - y.start);
    let (mut a, mut b, mut aa, mut bb, mut ab) = (0.0, 0.0, 0.0, 0.0, 0.0);
    let mut count = 0;
    for y in y {
        for x in x.clone() {
            let at = y as usize * THUMBNAIL_SIZE + x as usize;
            let other = y as usize * THUMBNAIL_SIZE + (x - shift) as usize;
            if first.valid[at] && second.valid[other] {
                let left = f64::from(first.detail[at]);
                let right = f64::from(second.detail[other]);
                a += left;
                b += right;
                aa += left * left;
                bb += right * right;
                ab += left * right;
                count += 1;
            }
        }
    }
    if count < 64 || count * 3 < area {
        return -1.0;
    }
    let count = f64::from(count);
    let va = aa - a * a / count;
    let vb = bb - b * b / count;
    if va >= 4.0 * count && vb >= 4.0 * count {
        (ab - a * b / count) / (va * vb).sqrt()
    } else {
        -1.0
    }
}

pub fn stereo_correlation(first: &StereoFeatures, second: &StereoFeatures, shift: i32) -> f64 {
    let edge = THUMBNAIL_SIZE as i32 - 2;
    if !(-edge + 2..=edge - 2).contains(&shift) {
        return -1.0;
    }
    correlation(
        first,
        second,
        shift,
        2.max(2 + shift)..edge.min(edge + shift),
        2..edge,
    )
}

pub fn stereo_shift(first: &StereoFeatures, second: &StereoFeatures) -> StereoShift {
    const WIDTH: i32 = THUMBNAIL_SIZE as i32;
    const LIMIT: i32 = WIDTH / 3;
    const NEAREST: i32 = (WIDTH + 9) / 10;
    const TOP: i32 = 2;
    const BOTTOM: i32 = WIDTH - 2;
    const MIDDLE: i32 = WIDTH / 2;
    let mut scores = [0.0; (2 * LIMIT + 1) as usize];
    let mut result = StereoShift::default();
    let mut best = 0;
    for shift in -LIMIT..=LIMIT {
        let x = 2.max(2 + shift)..(WIDTH - 2).min(WIDTH - 2 + shift);
        let score = correlation(first, second, shift, x, TOP..BOTTOM);
        scores[(shift + LIMIT) as usize] = score;
        if shift.abs() >= NEAREST && score > result.score {
            result.score = score;
            best = shift;
        }
    }
    result.textured = result.score > -1.0;
    if !result.textured {
        return result;
    }
    for shift in -LIMIT..=LIMIT {
        if shift.abs() >= NEAREST && (shift - best).abs() > 3 {
            result.alternative = result.alternative.max(scores[(shift + LIMIT) as usize]);
        }
    }
    let mut peak = f64::from(best);
    if best > -LIMIT && best < LIMIT {
        let index = (best + LIMIT) as usize;
        let [a, b, c] = [scores[index - 1], scores[index], scores[index + 1]];
        if a > -1.0 && c > -1.0 && a + c < 2.0 * b {
            peak += 0.5 * (a - c) / (a + c - 2.0 * b);
        }
    }
    let x0 = 2.max(2 + best);
    let x1 = (WIDTH - 2).min(WIDTH - 2 + best);
    let center = (x0 + x1) / 2;
    let halves = [
        correlation(first, second, best, x0..x1, TOP..MIDDLE),
        correlation(first, second, best, x0..x1, MIDDLE..BOTTOM),
        correlation(first, second, best, x0..center, TOP..BOTTOM),
        correlation(first, second, best, center..x1, TOP..BOTTOM),
    ];
    result.covered = halves.iter().all(|&score| score > -1.0)
        && halves.iter().filter(|&&score| score >= 0.2).count() >= 3;
    result.fraction = peak / f64::from(WIDTH);
    result
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Packing {
    Array,
    SideBySide,
    TopBottom,
    LeftQuadrants,
    RightQuadrants,
    TopQuadrants,
    BottomQuadrants,
    MainDiagonal,
    OtherDiagonal,
    SeparateEyes,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Region {
    pub crop: [f32; 4],
    pub layer: u32,
}

impl Packing {
    pub const fn eye_regions(self) -> [Region; 2] {
        const FULL: [f32; 4] = [0.0, 0.0, 1.0, 1.0];
        const TL: [f32; 4] = [0.0, 0.0, 0.5, 0.5];
        const TR: [f32; 4] = [0.5, 0.0, 0.5, 0.5];
        const BL: [f32; 4] = [0.0, 0.5, 0.5, 0.5];
        const BR: [f32; 4] = [0.5, 0.5, 0.5, 0.5];
        let (left, right, layer) = match self {
            Self::Array => (FULL, FULL, 1),
            Self::SideBySide => ([0.0, 0.0, 0.5, 1.0], [0.5, 0.0, 0.5, 1.0], 0),
            Self::TopBottom => ([0.0, 0.0, 1.0, 0.5], [0.0, 0.5, 1.0, 0.5], 0),
            Self::LeftQuadrants => (TL, BL, 0),
            Self::RightQuadrants => (TR, BR, 0),
            Self::TopQuadrants => (TL, TR, 0),
            Self::BottomQuadrants => (BL, BR, 0),
            Self::MainDiagonal => (TL, BR, 0),
            Self::OtherDiagonal => (TR, BL, 0),
            Self::SeparateEyes => (FULL, FULL, 0),
        };
        [
            Region {
                crop: left,
                layer: 0,
            },
            Region { crop: right, layer },
        ]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Layout {
    pub packing: Packing,
    pub swapped: bool,
}

#[derive(Clone, Copy)]
pub struct LayoutSample {
    pub packing: Packing,
    pub shift: StereoShift,
    pub signatures: [u64; 2],
    pub new_frame: bool,
}

#[derive(Default, Debug, Clone, Copy)]
pub struct LayoutScore {
    pub shift: StereoShift,
    pub frames: u32,
    pub votes: u32,
}

#[derive(Default)]
pub struct LayoutVerifier {
    scores: [LayoutScore; Packing::SeparateEyes as usize + 1],
    candidate: Option<Layout>,
    seen: Vec<[u64; 2]>,
}

impl LayoutVerifier {
    const REQUIRED_FRAMES: usize = 4;

    pub fn observe(&mut self, samples: &[LayoutSample]) {
        let redundant_crop = |crop: &LayoutSample, parent: Packing, scale: f64| {
            samples.iter().any(|sample| {
                sample.packing == parent
                    && sample.shift.strong()
                    && (crop.shift.fraction - sample.shift.fraction * scale).abs() < 0.025
            })
        };
        let mut winner = None;
        let mut matches = 0;
        for sample in samples {
            let score = &mut self.scores[sample.packing as usize];
            score.shift = sample.shift;
            score.frames += 1;
            if !sample.shift.strong() {
                continue;
            }
            if matches!(
                sample.packing,
                Packing::TopQuadrants | Packing::BottomQuadrants
            ) && redundant_crop(sample, Packing::SideBySide, 1.0)
            {
                continue;
            }
            if matches!(
                sample.packing,
                Packing::LeftQuadrants | Packing::RightQuadrants
            ) && redundant_crop(sample, Packing::TopBottom, 2.0)
            {
                continue;
            }
            winner = Some(Layout {
                packing: sample.packing,
                swapped: sample.shift.fraction < 0.0,
            });
            matches += 1;
        }
        if matches > 1 {
            self.clear_votes();
            return;
        }
        let Some(winner) = winner else { return };
        if self.candidate != Some(winner) {
            self.clear_votes();
            self.candidate = Some(winner);
        }
        let Some(sample) = samples
            .iter()
            .find(|sample| sample.packing == winner.packing)
        else {
            return;
        };
        if !sample.new_frame
            || self
                .seen
                .iter()
                .any(|prior| prior[0] == sample.signatures[0] || prior[1] == sample.signatures[1])
        {
            return;
        }
        if self.seen.len() < Self::REQUIRED_FRAMES {
            self.seen.push(sample.signatures);
        }
        self.scores[winner.packing as usize].votes = self.seen.len() as u32;
    }

    pub fn proven(&self) -> Option<Layout> {
        self.candidate
            .filter(|_| self.seen.len() >= Self::REQUIRED_FRAMES)
    }

    pub fn score(&self, packing: Packing) -> LayoutScore {
        self.scores[packing as usize]
    }

    fn clear_votes(&mut self) {
        for score in &mut self.scores {
            score.votes = 0;
        }
        self.seen.clear();
        self.candidate = None;
    }
}

#[derive(Default, Clone)]
struct Partner {
    index: usize,
    cycles: u32,
}

/// Groups pairs of writes only when the surrounding quiet periods are both
/// at least 1.5 times the pair's maximum observed span.
#[derive(Default)]
pub struct PairTimingLearner {
    previous: Vec<u64>,
    group: Vec<usize>,
    partners: Vec<Partner>,
    last_ns: u64,
    quiet_since_ns: u64,
    quiet_before_ns: u64,
    first_possible_ns: u64,
    last_observed_ns: u64,
    pub conflicts: u32,
    pub looks: u32,
    pub gaps: u32,
    pub writes: u32,
    pub accepted: u32,
    pub ambiguous: u32,
    pub max_gap_ns: u64,
    pub last_span_ns: u64,
    pub last_quiet_ns: u64,
}

impl PairTimingLearner {
    pub const WINDOW_NS: u64 = 3_200_000;
    pub const CONFIRMATION_NS: u64 = Self::WINDOW_NS * 3 / 2;
    const REQUIRED_CYCLES: u32 = 3;

    pub fn begin_burst(&mut self) {
        self.previous.clear();
        self.group.clear();
    }

    pub fn observe(&mut self, now: u64, signatures: &[u64]) {
        self.looks += 1;
        let gap = now.saturating_sub(self.last_ns);
        if !self.previous.is_empty() && gap != 0 {
            self.max_gap_ns = self.max_gap_ns.max(gap);
            self.gaps += u32::from(gap > Self::WINDOW_NS);
        }
        if self.previous.len() != signatures.len() || gap == 0 || gap > Self::WINDOW_NS {
            self.begin_burst();
            self.previous.extend_from_slice(signatures);
            self.last_ns = now;
            self.quiet_since_ns = now;
            if self.partners.len() != signatures.len() {
                self.partners = vec![Partner::default(); signatures.len()];
                self.conflicts = 0;
            }
            return;
        }

        let changed: Vec<_> = self
            .previous
            .iter()
            .zip(signatures)
            .enumerate()
            .filter_map(|(index, (before, after))| (before != after).then_some(index))
            .collect();
        self.writes += changed.len() as u32;
        if changed.is_empty() {
            if !self.group.is_empty() {
                let quiet = now - self.last_observed_ns;
                let settled = if self.group.len() >= 2 {
                    apart(quiet, self.span_ns())
                } else {
                    now - self.first_possible_ns > Self::CONFIRMATION_NS
                };
                if settled {
                    self.finish(quiet);
                }
            }
        } else {
            if !self.group.is_empty()
                && (self.group.len() + changed.len() > 2
                    || now - self.first_possible_ns > Self::CONFIRMATION_NS)
            {
                self.finish(self.last_ns - self.last_observed_ns);
            }
            if self.group.is_empty() {
                self.first_possible_ns = self.last_ns;
                self.quiet_before_ns = self.last_ns - self.quiet_since_ns;
            }
            self.group.extend(changed);
            self.last_observed_ns = now;
        }
        self.previous.copy_from_slice(signatures);
        self.last_ns = now;
    }

    pub fn proven(&self) -> Vec<[usize; 2]> {
        if self.partners.len() < 4
            || !self.partners.len().is_multiple_of(2)
            || self
                .partners
                .iter()
                .any(|partner| partner.cycles < Self::REQUIRED_CYCLES)
        {
            return Vec::new();
        }
        self.partners
            .iter()
            .enumerate()
            .filter_map(|(index, partner)| {
                (index < partner.index).then_some([index, partner.index])
            })
            .collect()
    }

    fn span_ns(&self) -> u64 {
        self.last_observed_ns - self.first_possible_ns
    }

    fn finish(&mut self, quiet_after: u64) {
        let span = self.span_ns();
        let separate = apart(self.quiet_before_ns, span) && apart(quiet_after, span);
        match self.group.as_slice() {
            &[a, b] if a != b => {
                if separate && span <= Self::CONFIRMATION_NS {
                    if (self.partners[a].cycles != 0 && self.partners[a].index != b)
                        || (self.partners[b].cycles != 0 && self.partners[b].index != a)
                    {
                        self.partners.fill(Partner::default());
                        self.conflicts += 1;
                    }
                    let cycles = (self.partners[a.min(b)].cycles + 1).min(Self::REQUIRED_CYCLES);
                    self.partners[a] = Partner { index: b, cycles };
                    self.partners[b] = Partner { index: a, cycles };
                    self.accepted += 1;
                    self.last_span_ns = span;
                    self.last_quiet_ns = self.quiet_before_ns.min(quiet_after);
                } else {
                    self.ambiguous += 1;
                }
            }
            &[index] if separate => {
                if self.partners[index].cycles != 0 {
                    let partner = self.partners[index].index;
                    self.partners[partner] = Partner::default();
                }
                self.partners[index] = Partner::default();
                self.conflicts += 1;
            }
            [_] => self.ambiguous += 1,
            _ => self.conflicts += 1,
        }
        self.quiet_since_ns = self.last_observed_ns;
        self.group.clear();
    }
}

fn apart(quiet_ns: u64, span_ns: u64) -> bool {
    u128::from(quiet_ns) * 2 >= u128::from(span_ns) * 3
}

#[cfg(test)]
mod regression_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn black_thumbnails_keep_a_gain_of_one_and_dim_ones_get_the_maximum() {
        assert_eq!(brightness_gain(&[0; THUMBNAIL_SIZE * THUMBNAIL_SIZE]), 1.0);
        assert_eq!(brightness_gain(&[1; THUMBNAIL_SIZE * THUMBNAIL_SIZE]), 1.0);
        assert_eq!(
            brightness_gain(&[8; THUMBNAIL_SIZE * THUMBNAIL_SIZE]),
            MAX_GAIN
        );
    }

    #[test]
    fn delayed_eye_never_pairs_with_the_adjacent_frame() {
        for (delay, paired) in [(750_000_u64, true), (4_000_000, false)] {
            let mut learner = PairTimingLearner::default();
            let mut signatures = [0; 6];
            let mut frame = 0;
            let mut right_frame = 0;
            for now in (0..150_000_000_u64).step_by(125_000) {
                if now >= 2_000_000 + frame * 6_944_444 {
                    signatures[(frame % 3) as usize] += 1;
                    frame += 1;
                }
                if now >= 2_000_000 + right_frame * 6_944_444 + delay {
                    signatures[3 + (right_frame % 3) as usize] += 1;
                    right_frame += 1;
                }
                learner.observe(now, &signatures);
            }
            if paired {
                assert_eq!(learner.proven(), [[0, 3], [1, 4], [2, 5]]);
            } else {
                assert!(learner.proven().is_empty());
            }
        }
    }
}
