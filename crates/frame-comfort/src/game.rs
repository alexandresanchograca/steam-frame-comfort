use comfort_core::{
    geometry::{
        Pose, PoseHistory, Projection, estimated_pose_timestamp, matched_face_size,
        snapshot_eye_extent,
    },
    settings::Settings,
    source::{
        self, Candidate, Eye, EyeBounds, FULL_FIELD, Generation, Policy, Selection, SlotSampling,
        SlotTracker, Stereo, packed_eyes,
    },
    stereo::{
        LayoutSample, LayoutVerifier, LumaThumbnail, Packing, PairTimingLearner, StereoFeatures,
        THUMBNAIL_SIZE, stereo_correlation, stereo_shift,
    },
};
use comfort_runtime::{
    mailbox::monotonic_ns,
    source_client::SourceClient,
    vulkan::{Capture, Context, ImportedImage, Snapshot, SnapshotRenderer, SourceView},
};
use std::{
    collections::BTreeMap, error::Error, io, os::fd::AsFd, path::Path, sync::Arc, thread,
    time::Duration,
};

const FRESH_NS: u64 = 300_000_000;
type Result<T> = std::result::Result<T, Box<dyn Error>>;

struct Imported {
    revision: u64,
    image: Arc<ImportedImage>,
}

#[derive(Clone)]
struct Proof {
    layout: comfort_core::stereo::Layout,
    pairs: Vec<[usize; 2]>,
}

#[derive(Default)]
struct Evidence {
    generation: Option<Generation>,
    cached: Vec<(Generation, Proof)>,
    proof: Option<Proof>,
    layouts: LayoutVerifier,
    timing: PairTimingLearner,
    per_pair: Vec<LayoutVerifier>,
    pair_checks: Vec<PairCheck>,
    attempt_started_ns: Option<u64>,
    retry_ns: u64,
    signatures: BTreeMap<(u64, usize), [u64; 2]>,
    last_report: Option<u64>,
    next_look_ns: u64,
    packed_looks: usize,
    burst_until_ns: u64,
    separate_eyes: bool,
    // Why the last attempt ended without a proof, shown until the retry.
    unverified: Option<&'static str>,
}

#[derive(Default)]
struct PairCheck {
    signatures: Option<[u64; 2]>,
    conflicts: u8,
}

impl Evidence {
    fn select(&mut self, source: &Selection, generation: Generation) -> Option<Stereo> {
        if self.generation.as_ref() != Some(&generation) {
            let mut cached = std::mem::take(&mut self.cached);
            if let (Some(old), Some(proof)) = (self.generation.take(), self.proof.take()) {
                cached.retain(|(generation, _)| generation != &old);
                if cached.len() == 4 {
                    cached.remove(0);
                }
                cached.push((old, proof));
            }
            let proof = cached
                .iter()
                .find(|(known, _)| known == &generation)
                .map(|(_, proof)| proof.clone());
            *self = Self {
                generation: Some(generation),
                proof,
                cached,
                ..Self::default()
            };
        }
        source.from_labels().or_else(|| {
            let proof = self.proof.as_ref()?;
            source.from_proof(proof.layout, &proof.pairs)
        })
    }

    fn pause_burst(&mut self) {
        if self.burst_until_ns != 0 {
            self.burst_until_ns = 0;
            self.timing.begin_burst();
        }
    }

    /// End this attempt and retry later. `unproven` records that the pictures
    /// or timing did not verify, rather than a readback error.
    fn defer(&mut self, now: u64, unproven: bool) {
        self.unverified = unproven.then_some(if self.separate_eyes {
            "separate-eye-pairing-unverified"
        } else {
            "stereo-layout-unverified"
        });
        self.pause_burst();
        self.retry_ns = now.saturating_add(750_000_000);
        self.attempt_started_ns = None;
        self.packed_looks = 0;
    }

    fn observe(
        &mut self,
        source: &Selection,
        sequence: u64,
        at_rest: bool,
        gpu: &mut Readback<'_>,
    ) -> Result<()> {
        let now = monotonic_ns();
        if self.proof.is_some() || now < self.retry_ns {
            return Ok(());
        }
        let started = *self.attempt_started_ns.get_or_insert(now);
        let result = self.sample(source, sequence, at_rest, gpu, now);
        if result.is_err() {
            self.defer(now, false);
        } else if self.proof.is_none()
            && now.saturating_sub(started) >= 1_500_000_000
            && self.attempt_started_ns == Some(started)
        {
            self.defer(now, true);
        }
        result
    }

    /// The last attempt's failure while waiting to retry, as the C++ baseline
    /// reports it. A running attempt is reported as learning.
    fn unverified(&self, now: u64) -> Option<&'static str> {
        self.unverified
            .filter(|_| self.proof.is_none() && now < self.retry_ns)
    }

    fn sample(
        &mut self,
        source: &Selection,
        sequence: u64,
        at_rest: bool,
        gpu: &mut Readback<'_>,
        now: u64,
    ) -> Result<()> {
        let split = !source.stream
            && source.generation.layers == 1
            && source.images.len() >= 4
            && source.images.len().is_multiple_of(2);
        self.separate_eyes = split;
        if !at_rest {
            self.pause_burst();
        }
        if now < self.next_look_ns {
            return Ok(());
        }
        if at_rest && split && self.packed_looks >= 8 && self.timing.proven().is_empty() {
            if self.burst_until_ns == 0 {
                self.timing.begin_burst();
                self.burst_until_ns = now + 120_000_000;
            }
            let pairs = source.images.iter().map(|&eye| [eye; 2]);
            let signatures = gpu
                .fingerprints(pairs)?
                .into_iter()
                .map(|pair| pair[0])
                .collect::<Vec<_>>();
            self.timing.observe(monotonic_ns(), &signatures);
            let pairs = self.timing.proven();
            if !pairs.is_empty() {
                self.attempt_started_ns = Some(now);
                self.per_pair
                    .resize_with(pairs.len(), LayoutVerifier::default);
                self.pair_checks
                    .resize_with(pairs.len(), PairCheck::default);
            }
            if monotonic_ns() >= self.burst_until_ns || !pairs.is_empty() {
                self.pause_burst();
                self.packed_looks = 0;
            }
            self.next_look_ns = now.saturating_add(1_000_000);
            return Ok(());
        }
        if self.last_report == Some(sequence) {
            return Ok(());
        }
        self.last_report = Some(sequence);
        self.next_look_ns = now.saturating_add(if at_rest { 0 } else { 100_000_000 });

        let source_eye = source.images[self.packed_looks % source.images.len()];
        self.packed_looks += 1;
        let mut samples = Vec::new();
        let hypotheses = source.packed_hypotheses();
        let pairs = hypotheses
            .iter()
            .map(|&packing| packed_eyes(source_eye, packing));
        for (&packing, [a, b]) in hypotheses.iter().zip(gpu.thumbnails(pairs, source.stream)?) {
            let signatures = [hash(&a), hash(&b)];
            let old = self
                .signatures
                .insert((source_eye.id, packing as usize), signatures);
            let new_frame = source.stream
                || old.is_some_and(|old| old[0] != signatures[0] && old[1] != signatures[1]);
            samples.push(LayoutSample {
                packing,
                shift: stereo_shift(&StereoFeatures::new(&a), &StereoFeatures::new(&b)),
                signatures,
                new_frame,
            });
        }
        self.layouts.observe(&samples);
        if let Some(layout) = self.layouts.proven() {
            self.proof = Some(Proof {
                layout,
                pairs: Vec::new(),
            });
            return Ok(());
        }
        let pairs = self.timing.proven();
        if pairs.is_empty() {
            return Ok(());
        }
        let eyes = source.images.iter().map(|&eye| [eye; 2]);
        let thumbnails = gpu
            .thumbnails(eyes, false)?
            .into_iter()
            .map(|[eye, _]| (StereoFeatures::new(&eye), hash(&eye)))
            .collect::<Vec<_>>();
        let mut ordered = Vec::with_capacity(pairs.len());
        let mut samples = Vec::with_capacity(pairs.len());
        let mut verified = true;
        for (index, [a, b]) in pairs.into_iter().enumerate() {
            let shift = stereo_shift(&thumbnails[a].0, &thumbnails[b].0);
            let sample = LayoutSample {
                packing: Packing::SeparateEyes,
                shift,
                signatures: [thumbnails[a].1, thumbnails[b].1],
                new_frame: true,
            };
            samples.push(sample);
            self.per_pair[index].observe(&[sample]);
            let layout = self.per_pair[index].proven();
            verified &= layout.is_some();
            let swapped = layout.map_or(shift.strong() && shift.fraction < 0.0, |layout| {
                layout.swapped
            });
            ordered.push(if swapped { [b, a] } else { [a, b] });
        }

        for (i, &[left, right]) in ordered.iter().enumerate() {
            let sample = samples[i];
            let check = &mut self.pair_checks[i];
            if !sample.shift.strong()
                || check.signatures.is_some_and(|prior| {
                    prior[0] == sample.signatures[0] || prior[1] == sample.signatures[1]
                })
            {
                continue;
            }
            check.signatures = Some(sample.signatures);
            let disparity = (sample.shift.fraction.abs() * THUMBNAIL_SIZE as f64).round() as i32;
            let contradicted = ordered
                .iter()
                .enumerate()
                .any(|(j, &[other_left, other_right])| {
                    if i == j || !samples[j].shift.strong() {
                        return false;
                    }
                    let alternative = stereo_correlation(
                        &thumbnails[left].0,
                        &thumbnails[other_right].0,
                        disparity,
                    )
                    .max(stereo_correlation(
                        &thumbnails[other_left].0,
                        &thumbnails[right].0,
                        disparity,
                    ));
                    alternative > sample.shift.score + 0.05
                });
            if contradicted {
                check.conflicts += 1;
                if check.conflicts >= 2 {
                    self.timing = PairTimingLearner::default();
                    self.per_pair.clear();
                    self.pair_checks.clear();
                    self.defer(now, true);
                    return Ok(());
                }
            }
        }
        if verified {
            self.proof = Some(Proof {
                layout: comfort_core::stereo::Layout {
                    packing: Packing::SeparateEyes,
                    swapped: false,
                },
                pairs: ordered,
            });
        }
        Ok(())
    }
}

pub struct Game {
    context: Arc<Context>,
    source: SourceClient,
    source_format: Option<u32>,
    renderer: SnapshotRenderer,
    imports: BTreeMap<u64, Imported>,
    evidence: Evidence,
    selection: Option<Stereo>,
    // Why source selection last found no candidate stereo picture.
    reason: Option<source::Reason>,
    generation: Option<Generation>,
    pub generation_id: u64,
    epoch: u64,
    next_connect_ns: u64,
    last_selection_ns: u64,
    stream: bool,
    recent: BTreeMap<u64, (u64, Candidate)>,
    ring: SlotTracker,
    ring_primed: bool,
    sampling: SlotSampling,
    previous_pairs: BTreeMap<u64, [u64; 2]>,
    fingerprints: BTreeMap<u64, [u64; 2]>,
    snapshots: [Option<Snapshot>; 2],
    meta: [Option<Meta>; 2],
    pub captures: u64,
    audit: bool,
    pub unstable_captures: u64,
    pub(super) fingerprint_ms: super::metrics::Samples,
    pub(super) fingerprints_taken: u64,
    /// Apply the compositor's texture bounds to labelled eyes. Off restores
    /// the whole-image mapping for same-session comparisons.
    compositor_bounds: bool,
    // The last bounds reported for each image revision. A report from a
    // frame without a scene draw keeps the image's known mapping.
    known_bounds: BTreeMap<u64, (u64, EyeBounds)>,
}

#[derive(Clone, Copy)]
pub struct Meta {
    pub head: Pose,
    pub eyes: [(Pose, Projection); 2],
    pub face_size: u32,
}

impl Game {
    pub fn new(context: Arc<Context>, audit: bool, source_format: Option<u32>) -> Result<Self> {
        Self::with_source(context, audit, SourceClient::default(), source_format)
    }

    pub(super) fn with_source(
        context: Arc<Context>,
        audit: bool,
        source: SourceClient,
        source_format: Option<u32>,
    ) -> Result<Self> {
        Ok(Self {
            renderer: SnapshotRenderer::new(Arc::clone(&context))?,
            context,
            source,
            source_format: source_format.filter(|&format| format != 0),
            imports: BTreeMap::new(),
            evidence: Evidence::default(),
            selection: None,
            reason: None,
            generation: None,
            generation_id: 0,
            epoch: 0,
            next_connect_ns: 0,
            last_selection_ns: 0,
            stream: false,
            recent: BTreeMap::new(),
            ring: SlotTracker::default(),
            ring_primed: false,
            sampling: SlotSampling::default(),
            previous_pairs: BTreeMap::new(),
            fingerprints: BTreeMap::new(),
            snapshots: [None, None],
            meta: [None, None],
            captures: 0,
            audit,
            unstable_captures: 0,
            fingerprint_ms: super::metrics::Samples::default(),
            fingerprints_taken: 0,
            compositor_bounds: true,
            known_bounds: BTreeMap::new(),
        })
    }

    pub fn set_compositor_bounds(&mut self, enabled: bool) {
        self.compositor_bounds = enabled;
    }

    pub(super) fn source_client(&self) -> &SourceClient {
        &self.source
    }

    pub(super) fn same_device(&self) -> bool {
        self.source.hello.is_some_and(|hello| {
            hello.device_uuid == self.context.device_uuid()
                && hello.driver_uuid == self.context.driver_uuid()
        })
    }

    pub(super) fn compositor_sample(&self) -> Option<(u64, u64)> {
        let report = self.source.frame?;
        Some((self.epoch, report.monotonic_ns))
    }

    pub(super) fn allocated_bytes(&self) -> u64 {
        self.snapshots
            .iter()
            .flatten()
            .map(Snapshot::allocated_bytes)
            .sum()
    }

    pub(super) fn evidence_bursting(&self) -> bool {
        self.evidence.burst_until_ns != 0
    }

    pub(super) fn evidence_report(&self, now: u64) -> serde_json::Value {
        use serde_json::json;
        let evidence = &self.evidence;
        let packings = [
            Packing::Array,
            Packing::SideBySide,
            Packing::TopBottom,
            Packing::LeftQuadrants,
            Packing::RightQuadrants,
            Packing::TopQuadrants,
            Packing::BottomQuadrants,
            Packing::MainDiagonal,
            Packing::OtherDiagonal,
        ];
        let packings = packings
            .into_iter()
            .map(|packing| (packing, evidence.layouts.score(packing)))
            .chain(evidence.per_pair.iter().map(|pair| {
                let packing = Packing::SeparateEyes;
                (packing, pair.score(packing))
            }))
            .filter(|(_, score)| score.frames != 0)
            .map(|(packing, score)| {
                json!({
                    "packing": format!("{packing:?}"),
                    "frames": score.frames, "votes": score.votes,
                    "shift": score.shift.fraction, "score": score.shift.score,
                    "alternative": score.shift.alternative, "coverage": score.shift.covered
                })
            })
            .collect::<Vec<_>>();
        let pairs = self.selection.as_ref().map(|stereo| {
            stereo
                .slots
                .iter()
                .map(|pair| pair.map(|eye| eye.id))
                .collect::<Vec<_>>()
        });
        let timing = &evidence.timing;
        json!({
            "gameSource": self.source_status(now),
            "stream": self.stream, "rule": self.rule(),
            "eyeSource": self.eye_source(),
            "mapping": self.mapping_report(),
            "sourceFormat": self.generation.as_ref().map(|g| g.format),
            "extent": self.generation.as_ref().map(|g| g.extent),
            "images": self.generation.as_ref().map(|g| g.images.iter().map(|(id, _)| id).collect::<Vec<_>>()),
            "packings": packings, "pairing": pairs,
            "layout": evidence.proof.as_ref().map(|p| format!("{:?}", p.layout.packing)),
            "swapped": evidence.proof.as_ref().map(|p| p.layout.swapped),
            "timingConflicts": timing.conflicts, "timingLooks": timing.looks,
            "timingGaps": timing.gaps, "timingWrites": timing.writes,
            "maxTimingGapMs": timing.max_gap_ns as f64 / 1e6,
            "timingAccepted": timing.accepted, "timingAmbiguous": timing.ambiguous,
            "pairSpanMs": timing.last_span_ns as f64 / 1e6,
            "pairQuietMs": timing.last_quiet_ns as f64 / 1e6,
            "imagePairConflicts": evidence.pair_checks.iter().map(|check| u64::from(check.conflicts)).sum::<u64>(),
        })
    }

    /// Each selected slot's eyes: image, layer, source rectangle, flips and
    /// the part of the display eye the rectangle fills.
    pub(super) fn mapping_report(&self) -> serde_json::Value {
        use serde_json::json;
        let round = |values: [f32; 4]| values.map(|value| (value * 1e4).round() / 1e4);
        json!({
            "compositorBounds": self.compositor_bounds,
            "slots": self.selection.as_ref().map(|stereo| stereo.slots.iter().map(|eyes| {
                let mut index = 0;
                eyes.map(|eye| {
                    // The compositor's raw bounds for this eye, when reported.
                    let bounds = self
                        .known_bounds
                        .get(&eye.id)
                        .and_then(|(_, bounds)| bounds.get(index))
                        .map(round);
                    index += 1;
                    json!({"id": eye.id, "layer": eye.layer, "bounds": bounds,
                        "crop": round(eye.crop), "flip": eye.flip, "field": round(eye.field)})
                })
            }).collect::<Vec<_>>()),
        })
    }

    /// Read regions of a single image using its latest reported handoff.
    pub(super) fn probe_pixels(
        &mut self,
        id: u64,
        layer: u32,
        crops: [[f32; 4]; 2],
        size: u32,
    ) -> Result<Vec<u8>> {
        if monotonic_ns().saturating_sub(self.source.frame_received_ns) >= FRESH_NS {
            return Err(io::Error::other("no fresh source report").into());
        }
        let report = self
            .source
            .frame
            .ok_or_else(|| io::Error::other("no source report"))?;
        let area = report.count as usize;
        let handoff = report.ids[..area]
            .iter()
            .position(|&image| image == id)
            .map(|index| (report.layouts[index], true))
            .or_else(|| {
                report.touched_ids[..report.touched_count as usize]
                    .iter()
                    .position(|&image| image == id)
                    .map(|index| (report.touched_layouts[index], false))
            });
        let (layout, external) =
            handoff.ok_or_else(|| io::Error::other("image absent from source report"))?;
        self.import(id)?;
        let mut gpu = Readback {
            renderer: &mut self.renderer,
            imports: &self.imports,
        };
        gpu.pixels(
            [crops.map(|crop| Eye {
                id,
                layout,
                crop,
                layer,
                flip: [false; 2],
                field: FULL_FIELD,
            })],
            external,
            size,
        )
    }

    pub fn invalidate_tracking(&mut self) {
        self.meta = [None, None];
    }

    pub fn meta(&self, slot: u32) -> Option<Meta> {
        self.meta[slot as usize]
    }

    pub fn capture_meta(&self, slot: u32) -> Option<Capture<'_>> {
        let index = slot as usize;
        Some(Capture {
            snapshot: self.snapshots[index].as_ref()?,
            eyes: self.meta[index]?.eyes,
        })
    }

    pub fn ready(&self, now: u64) -> bool {
        self.selection.is_some()
            && now.saturating_sub(self.source.frame_received_ns) < FRESH_NS
            && (self.stream || self.ring_primed)
    }

    pub fn still_fresh(&self, now: u64) -> bool {
        self.selection.is_some() && now.saturating_sub(self.source.frame_received_ns) < FRESH_NS
    }

    pub fn learning_delay(&self) -> Duration {
        if self.evidence.burst_until_ns != 0 {
            Duration::from_nanos(self.evidence.next_look_ns.saturating_sub(monotonic_ns()))
        } else {
            Duration::from_millis(10)
        }
    }

    pub fn image_count(&self) -> usize {
        self.source.images.len()
    }

    pub fn rule(&self) -> Option<&'static str> {
        Some(match self.selection.as_ref()?.rule {
            source::Rule::Stream2x2 => "stream-2x2",
            source::Rule::LocalSlots => "local-slots",
            source::Rule::LocalSideBySide => "local-sbs",
            source::Rule::LocalAtlas => "local-atlas",
            source::Rule::LocalArray => "local-array",
        })
    }

    pub(super) fn eye_source(&self) -> Option<&'static str> {
        self.selection.as_ref().map(|stereo| {
            if stereo.labelled {
                "compositor"
            } else {
                "measured"
            }
        })
    }

    /// Why Game is or is not showing the game picture, as named on the panel.
    pub fn source_status(&self, now: u64) -> &'static str {
        source_status(Status {
            fresh: self.source.frame.is_some()
                && now.saturating_sub(self.source.frame_received_ns) < FRESH_NS,
            ready: self.ready(now),
            selected: self.selection.is_some(),
            reason: self.reason,
            unverified: self.evidence.unverified(now),
        })
    }

    fn clear_selection(&mut self) {
        self.selection = None;
        self.ring = SlotTracker::default();
        self.ring_primed = false;
        self.sampling = SlotSampling::default();
        self.previous_pairs.clear();
        self.fingerprints.clear();
    }

    pub(super) fn poll_source(&mut self, path: &Path, now: u64) -> io::Result<()> {
        let mut result = Ok(());
        if !self.source.connected() && now >= self.next_connect_ns {
            self.next_connect_ns = now.saturating_add(250_000_000);
            result = self.source.connect(path);
            if result.is_ok() {
                self.epoch += 1;
                self.reason = None;
                self.evidence = Evidence::default();
                self.recent.clear();
                self.known_bounds.clear();
                self.clear_selection();
            }
        }
        if self.source.connected() {
            result = self.source.poll(now);
            if result.is_err() {
                self.clear_selection();
            }
        }
        self.source.take_removals();
        self.imports.retain(|id, imported| {
            self.source
                .images
                .get(id)
                .is_some_and(|image| image.revision == imported.revision)
        });
        self.renderer.retain_sources(|image| {
            self.imports
                .get(&image.description.id)
                .is_some_and(|imported| std::ptr::eq(imported.image.as_ref(), image))
        });
        result
    }

    pub fn poll(
        &mut self,
        path: &Path,
        now: u64,
        dashboard: bool,
        at_rest: bool,
        excluded_sizes: &[[u32; 2]],
        observe_ring: bool,
    ) -> Result<()> {
        let _ = self.poll_source(path, now);
        let Some(frame) = self.source.frame else {
            return Ok(());
        };
        if !self.same_device() {
            self.clear_selection();
            return Err(io::Error::other("source UUIDs differ from the Vulkan renderer").into());
        }
        if dashboard || now.saturating_sub(self.source.frame_received_ns) >= FRESH_NS {
            self.evidence.pause_burst();
            self.clear_selection();
            return Ok(());
        }
        let acquired = frame.count as usize;
        let touched = frame.touched_count as usize;
        let mut candidates = (0..acquired)
            .map(|index| {
                (
                    frame.ids[index],
                    frame.layouts[index],
                    true,
                    frame.eyes[index],
                    frame.bounds[index],
                )
            })
            .chain(
                (0..touched)
                    .filter(|&index| !frame.ids[..acquired].contains(&frame.touched_ids[index]))
                    .map(|index| {
                        (
                            frame.touched_ids[index],
                            frame.touched_layouts[index],
                            false,
                            frame.touched_eyes[index],
                            frame.touched_bounds[index],
                        )
                    }),
            )
            .filter_map(|(id, layout, acquired, eye, bounds)| {
                let image = self.source.images.get(&id)?;
                Some(Candidate {
                    image: image.description,
                    layout,
                    acquired,
                    eye,
                    bounds,
                    revision: image.revision,
                })
            })
            .collect::<Vec<_>>();
        self.known_bounds.retain(|id, (revision, _)| {
            self.source
                .images
                .get(id)
                .is_some_and(|image| image.revision == *revision)
        });
        for candidate in &mut candidates {
            if !self.compositor_bounds {
                candidate.bounds = EyeBounds::default();
                continue;
            }
            let id = candidate.image.id;
            let known = self
                .known_bounds
                .entry(id)
                .or_insert((candidate.revision, EyeBounds::default()));
            // Fresh bounds replace the known ones eye by eye.
            let mut current = candidate.bounds;
            current.merge(&known.1);
            known.1 = current;
            candidate.bounds = current;
        }
        // Local render submissions may touch just the slot they updated.
        // Keep matching recent siblings in the ring as the baseline does.
        self.recent.retain(|id, (sequence, candidate)| {
            frame.sequence.saturating_sub(*sequence) <= 240
                && self
                    .source
                    .images
                    .get(id)
                    .is_some_and(|image| image.revision == candidate.revision)
        });
        for &candidate in &candidates {
            if !candidate.acquired && candidate.image.exported != 0 {
                self.recent
                    .insert(candidate.image.id, (frame.sequence, candidate));
            }
        }
        let reported = candidates.len();
        for index in 0..reported {
            let live = candidates[index];
            if live.acquired || live.image.exported == 0 {
                continue;
            }
            let shape = |candidate: Candidate| {
                let d = candidate.image;
                (d.format, d.width, d.height, d.array_layers)
            };
            for &(_, candidate) in self.recent.values() {
                if shape(candidate) == shape(live)
                    && !candidates
                        .iter()
                        .any(|current| current.image.id == candidate.image.id)
                {
                    candidates.push(candidate);
                }
            }
        }
        let source = match source::select(
            &candidates,
            Policy {
                epoch: self.epoch,
                complete: frame.truncated == 0 && frame.touched_truncated == 0,
                excluded_sizes,
                source_format: self.source_format,
                ..Policy::default()
            },
        ) {
            Ok(source) => source,
            // A report that touched only other images (overlays, or the slot
            // just written) changes nothing, the reason included.
            Err(source::Reason::NoSource)
                if !self.stream && now.saturating_sub(self.last_selection_ns) < FRESH_NS =>
            {
                return Ok(());
            }
            Err(reason) => {
                self.reason = Some(reason);
                self.clear_selection();
                return Ok(());
            }
        };
        self.reason = None;
        let mut generation = source.generation.clone();
        if source.stream {
            generation.images = self
                .source
                .images
                .iter()
                .filter_map(|(&id, image)| {
                    let d = image.description;
                    (d.exported == 0
                        && d.format == generation.format
                        && [d.width, d.height] == generation.extent
                        && d.array_layers == generation.layers)
                        .then_some((id, image.revision))
                })
                .collect();
        }
        if self.generation.as_ref() != Some(&generation) {
            self.generation = Some(generation.clone());
            self.generation_id += 1;
            self.clear_selection();
        }
        for eye in &source.images {
            self.import(eye.id)?;
        }
        self.stream = source.stream;
        let mut selection = self.evidence.select(&source, generation.clone());
        if selection.is_none() {
            let mut gpu = Readback {
                renderer: &mut self.renderer,
                imports: &self.imports,
            };
            self.evidence
                .observe(&source, frame.sequence, at_rest, &mut gpu)?;
            selection = self.evidence.select(&source, generation);
        }
        self.selection = selection;
        self.last_selection_ns = now;
        if !self.stream
            && (observe_ring || (at_rest && !self.ring_primed))
            && self.sampling.due(frame.sequence, now, false)
        {
            self.fingerprint_ring(now)?;
            self.sampling
                .sampled(self.ring.needs_confirmation, monotonic_ns());
        }
        Ok(())
    }

    fn import(&mut self, id: u64) -> Result<()> {
        if self.imports.contains_key(&id) {
            return Ok(());
        }
        if !self.same_device() {
            return Err(io::Error::other("source UUIDs differ from the Vulkan renderer").into());
        }
        let image = self
            .source
            .images
            .get(&id)
            .ok_or_else(|| io::Error::other("source image has been removed"))?;
        // SAFETY: The observer hello confirms the same device and driver.
        let imported = unsafe {
            ImportedImage::import(
                Arc::clone(&self.context),
                image.description,
                image.fd.as_fd(),
            )?
        };
        self.imports.insert(
            id,
            Imported {
                revision: image.revision,
                image: Arc::new(imported),
            },
        );
        Ok(())
    }

    fn fingerprint_ring(&mut self, now: u64) -> Result<()> {
        let Some(stereo) = &self.selection else {
            return Ok(());
        };
        let mut gpu = Readback {
            renderer: &mut self.renderer,
            imports: &self.imports,
        };
        let mut signatures = Vec::new();
        self.fingerprints.clear();
        let start = monotonic_ns();
        let readings = gpu.fingerprints(stereo.slots.iter().copied())?;
        self.fingerprint_ms.since(start);
        self.fingerprints_taken += 1;
        for (eyes, values) in stereo.slots.iter().zip(readings) {
            let id = eyes[0].id;
            if id != eyes[1].id {
                let previous = self.previous_pairs.entry(id).or_insert(values);
                if (values[0] != previous[0]) != (values[1] != previous[1]) {
                    continue;
                }
                *previous = values;
            }
            self.fingerprints.insert(id, values);
            signatures.push((
                id,
                values[0].wrapping_mul(1_099_511_628_211) ^ values[1].rotate_left(31),
            ));
        }
        self.ring.update(&signatures, now);
        self.ring_primed |= self.ring.capture_slot(true).is_some();
        Ok(())
    }

    pub fn is_stream(&self) -> bool {
        self.stream
    }

    pub fn report_pose_ns(&self, settings: &Settings) -> Option<u64> {
        let offset = if self.stream {
            settings.rest_frame_stream_pose_offset_ms
        } else {
            settings.rest_frame_pose_offset_ms
        };
        estimated_pose_timestamp(
            self.source.frame?.monotonic_ns,
            (f64::from(offset) * 1e6) as i64,
        )
    }

    pub fn capture(
        &mut self,
        slot: u32,
        settings: &Settings,
        history: &PoseHistory,
        epoch: u64,
        eyes: [(Pose, Projection); 2],
        now: u64,
    ) -> Result<bool> {
        if !self.still_fresh(now) {
            return Ok(false);
        }
        let Some(frame) = self.source.frame else {
            return Ok(false);
        };
        let confirmed = if !self.stream && self.ring.needs_confirmation {
            thread::sleep(Duration::from_micros(500));
            self.fingerprint_ring(monotonic_ns())?;
            true
        } else {
            false
        };
        let Some(stereo) = &self.selection else {
            return Ok(false);
        };
        let chosen = if self.stream {
            frame.ids[..frame.count as usize]
                .iter()
                .rev()
                .find_map(|id| stereo.slots.iter().find(|eyes| eyes[0].id == *id).copied())
        } else {
            self.ring
                .capture_slot(confirmed)
                .and_then(|id| stereo.slots.iter().find(|eyes| eyes[0].id == id).copied())
        };
        let Some(chosen) = chosen else {
            return Ok(false);
        };
        let Some(head) = self
            .report_pose_ns(settings)
            .and_then(|time| history.sample(time, epoch))
        else {
            return Ok(false);
        };
        // Each eye's snapshot covers only the part of the display eye its
        // source rectangle fills, and keeps that projection.
        let eyes: [(Pose, Projection); 2] = std::array::from_fn(|index| {
            (eyes[index].0, eyes[index].1.restrict(chosen[index].field))
        });
        let projection = eyes.map(|(_, projection)| projection);
        let extent = |eye: Eye| {
            let d = self.imports.get(&eye.id)?.image.description;
            Some([
                (d.width as f32 * eye.crop[2]).round().max(1.0) as u32,
                (d.height as f32 * eye.crop[3]).round().max(1.0) as u32,
            ])
        };
        // The eyes share one snapshot extent. Their rectangles may differ by
        // a few texels when the compositor crops each eye differently.
        let (Some(left), Some(right)) = (extent(chosen[0]), extent(chosen[1])) else {
            return Ok(false);
        };
        if (0..2).any(|axis| left[axis].max(right[axis]) > 2 * left[axis].min(right[axis])) {
            return Ok(false);
        }
        let source_extent = [left[0].max(right[0]), left[1].max(right[1])];
        let face_size = if settings.rest_frame_face_size != 0 {
            settings.rest_frame_face_size
        } else {
            matched_face_size(
                projection,
                source_extent,
                256,
                settings.rest_frame_max_face_size,
            )
            .unwrap_or(settings.rest_frame_max_face_size)
        };
        let extent = snapshot_eye_extent(projection, source_extent, face_size);
        let [left, right] = chosen.map(|eye| source_view(&self.imports, eye, self.stream));
        let slot = slot as usize;
        self.meta[slot] = None;
        // SAFETY: Stream buffers come from the current acquisition report.
        // Local rings use the baseline's stable-before/stable-after check.
        unsafe {
            self.renderer
                .capture_into([left?, right?], extent, &mut self.snapshots[slot])?
        };
        if !self.stream {
            if chosen[0].id != chosen[1].id {
                let deadline = self
                    .ring
                    .changed_ns()
                    .saturating_add(PairTimingLearner::CONFIRMATION_NS);
                thread::sleep(Duration::from_nanos(
                    deadline.saturating_sub(monotonic_ns()),
                ));
            }
            let mut gpu = Readback {
                renderer: &mut self.renderer,
                imports: &self.imports,
            };
            let start = monotonic_ns();
            let after = gpu.fingerprints([chosen])?[0];
            self.fingerprint_ms.since(start);
            self.fingerprints_taken += 1;
            if self.fingerprints.get(&chosen[0].id) != Some(&after) {
                self.unstable_captures += 1;
                return Ok(false);
            }
        }
        self.meta[slot] = Some(Meta {
            head,
            eyes: eyes.map(|(eye, projection)| (head.compose(eye), projection)),
            face_size,
        });
        if self.audit {
            let readback = self.renderer.readback(
                self.snapshots[slot]
                    .as_mut()
                    .expect("capture allocated snapshot"),
            );
            match readback {
                Ok(pixels) => {
                    println!("{}", super::audit::snapshot(&pixels, extent, source_extent))
                }
                Err(error) => println!(
                    "{}",
                    serde_json::json!({"snapshotAudit":false,"readback":error.to_string()})
                ),
            }
        }
        self.captures += 1;
        Ok(true)
    }
}

struct Status {
    fresh: bool,
    ready: bool,
    selected: bool,
    reason: Option<source::Reason>,
    unverified: Option<&'static str>,
}

fn source_status(status: Status) -> &'static str {
    if !status.fresh {
        "no-fresh-source"
    } else if status.ready {
        "verified"
    } else if status.selected {
        "waiting-for-complete-frame"
    } else if let Some(reason) = status.reason {
        reason.name()
    } else {
        status.unverified.unwrap_or("learning")
    }
}

fn source_view(
    imports: &BTreeMap<u64, Imported>,
    eye: Eye,
    external: bool,
) -> Result<SourceView<'_>> {
    let imported = imports
        .get(&eye.id)
        .ok_or_else(|| io::Error::other("source image is no longer imported"))?;
    Ok(SourceView {
        image: &imported.image,
        layout: ash::vk::ImageLayout::from_raw(eye.layout as i32),
        layer: eye.layer,
        crop: eye.crop,
        flip: eye.flip,
        external,
    })
}

struct Readback<'a> {
    renderer: &'a mut SnapshotRenderer,
    imports: &'a BTreeMap<u64, Imported>,
}

impl Readback<'_> {
    fn pixels(
        &mut self,
        pairs: impl IntoIterator<Item = [Eye; 2]>,
        external: bool,
        size: u32,
    ) -> Result<Vec<u8>> {
        let views = pairs
            .into_iter()
            .map(|pair| {
                let [left, right] = pair.map(|eye| source_view(self.imports, eye, external));
                Ok([left?, right?])
            })
            .collect::<Result<Vec<_>>>()?;
        // SAFETY: Same process/device and handoff rules as Game::capture.
        unsafe { self.renderer.measure(views, [size; 2]) }
    }

    fn fingerprints(&mut self, pairs: impl IntoIterator<Item = [Eye; 2]>) -> Result<Vec<[u64; 2]>> {
        let pixels = self.pixels(pairs, false, 16)?;
        Ok(pixels
            .chunks_exact(32 * 16 * 4)
            .map(|pair| {
                let mut signatures = [1_469_598_103_934_665_603; 2];
                for row in pair.chunks_exact(32 * 4) {
                    for (index, eye) in row.chunks_exact(16 * 4).enumerate() {
                        signatures[index] = append_hash(signatures[index], eye);
                    }
                }
                signatures
            })
            .collect())
    }

    fn thumbnails(
        &mut self,
        pairs: impl IntoIterator<Item = [Eye; 2]>,
        external: bool,
    ) -> Result<Vec<[LumaThumbnail; 2]>> {
        let pixels = self.pixels(pairs, external, 64)?;
        Ok(pixels
            .chunks_exact(128 * 64 * 4)
            .map(|pair| {
                let mut thumbnails = [[0; 64 * 64]; 2];
                for (y, row) in pair.chunks_exact(128 * 4).enumerate() {
                    for (eye, colors) in row.chunks_exact(64 * 4).enumerate() {
                        for (x, pixel) in colors.chunks_exact(4).enumerate() {
                            thumbnails[eye][y * 64 + x] = ((54 * u32::from(pixel[0])
                                + 183 * u32::from(pixel[1])
                                + 19 * u32::from(pixel[2])
                                + 128)
                                >> 8)
                                as u8;
                        }
                    }
                }
                thumbnails
            })
            .collect())
    }
}

fn hash(bytes: &[u8]) -> u64 {
    append_hash(1_469_598_103_934_665_603, bytes)
}

fn append_hash(start: u64, bytes: &[u8]) -> u64 {
    bytes.iter().fold(start, |hash, &byte| {
        (hash ^ u64::from(byte)).wrapping_mul(1_099_511_628_211)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_status_names_the_first_blocking_reason() {
        let base = Status {
            fresh: true,
            ready: false,
            selected: false,
            reason: None,
            unverified: None,
        };
        let status = |change: fn(&mut Status)| {
            let mut status = Status { ..base };
            change(&mut status);
            source_status(status)
        };
        assert_eq!(status(|_| {}), "learning");
        assert_eq!(
            status(|s| s.unverified = Some("separate-eye-pairing-unverified")),
            "separate-eye-pairing-unverified"
        );
        assert_eq!(
            status(|s| {
                s.reason = Some(source::Reason::Ambiguous);
                s.unverified = Some("stereo-layout-unverified");
            }),
            "ambiguous-source"
        );
        assert_eq!(status(|s| s.selected = true), "waiting-for-complete-frame");
        assert_eq!(
            status(|s| {
                s.selected = true;
                s.ready = true;
            }),
            "verified"
        );
        assert_eq!(
            status(|s| {
                s.fresh = false;
                s.selected = true;
                s.ready = true;
            }),
            "no-fresh-source"
        );
    }
}
