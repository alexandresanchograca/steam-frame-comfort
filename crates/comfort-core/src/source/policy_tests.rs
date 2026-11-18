//! Source selection and ring slot cases ported from the C++ baseline's
//! `source_policy_tests.cpp`: stream 2x2 frames, local side-by-side and array
//! rings, separate-eye rings, self-exclusion, UI filtering, newest-completed
//! slot tracking and the capture rule.
//!
//! The Rust selection has no `imported` flag, per-image identity check or
//! `excludedBySize` diagnostic, so those baseline checks have no counterpart.
//! A shape seen in a single image is dropped before the ambiguity checks here,
//! where the baseline reported it as unsupported or ambiguous.
use super::*;

// VkFormat wire values.
const RGBA: u32 = 37;
const SRGB: u32 = 43;
const BGRA: u32 = 50;
const CONFIRMATION_NS: u64 = 500_000;
/// Comfort's own outputs, excluded by size.
const OWN_SIZES: &[[u32; 2]] = &[[3072, 1536], [8, 8]];

/// An image the compositor imported: a decoder frame when it is P010,
/// otherwise a UI or overlay texture.
fn image(id: u64, format: u32, width: u32, height: u32, layers: u32) -> Candidate {
    Candidate {
        image: ImageAdd {
            id,
            format,
            width,
            height,
            array_layers: layers,
            ..Default::default()
        },
        layout: GENERAL,
        acquired: format == P010,
        eye: EyeUse::default(),
        bounds: EyeBounds::default(),
        revision: 1,
    }
}

/// A compositor-allocated (exported) local scene image.
fn scene(id: u64, format: u32, width: u32, height: u32, layers: u32) -> Candidate {
    let mut candidate = image(id, format, width, height, layers);
    candidate.image.exported = 1;
    candidate
}

fn policy() -> Policy<'static> {
    Policy {
        epoch: 7,
        excluded_sizes: OWN_SIZES,
        ..Policy::default()
    }
}

fn reason(frame: &[Candidate], policy: Policy<'_>) -> Option<Reason> {
    select(frame, policy).err()
}

fn proof(packing: Packing) -> Layout {
    Layout {
        packing,
        swapped: false,
    }
}

fn ids(stereo: &Stereo) -> Vec<[u64; 2]> {
    stereo
        .slots
        .iter()
        .map(|eyes| eyes.map(|eye| eye.id))
        .collect()
}

fn generation_ids(selection: &Selection) -> Vec<u64> {
    selection
        .generation
        .images
        .iter()
        .map(|&(id, _)| id)
        .collect()
}

/// Stream frame with SteamVR UI textures and our own overlay present.
fn stream() -> Vec<Candidate> {
    vec![
        image(1, BGRA, 1800, 120, 1),
        image(2, RGBA, 3072, 1536, 1),
        image(3, RGBA, 8, 8, 1),
        image(4, P010, 2304, 2304, 1),
        image(5, BGRA, 1920, 1080, 1),
    ]
}

/// Local buffer rings (3 slots per eye, created left first) among UI panels.
fn ring() -> Vec<Candidate> {
    let mut ring: Vec<_> = [112, 110, 115, 111, 113, 114]
        .into_iter()
        .map(|id| scene(id, SRGB, 1728, 1728, 1))
        .collect();
    ring.push(image(99, BGRA, 1800, 120, 1));
    ring
}

fn sbs_ring() -> Vec<Candidate> {
    (40..=42).map(|id| scene(id, SRGB, 3456, 1728, 1)).collect()
}

fn array_ring() -> Vec<Candidate> {
    (50..=52).map(|id| scene(id, SRGB, 1728, 1728, 2)).collect()
}

fn signatures(a: u64, b: u64, c: u64) -> [(u64, u64); 3] {
    [(1, a), (2, b), (3, c)]
}

/// Newest completed slot: the most recently changed slot that is stable now.
#[test]
fn newest_completed_slot() {
    let mut tracker = SlotTracker::default();
    assert_eq!(
        tracker.update(&signatures(10, 20, 30), 1),
        Some(1),
        "first poll: no history, first slot"
    );
    assert_eq!(
        tracker.update(&signatures(11, 20, 30), 2),
        Some(2),
        "slot 1 being written: take a stable slot"
    );
    // Round robin 1 -> 2 -> 3: at t3 slot 2 is being written, slot 1 (t2) is the newest complete.
    assert_eq!(
        tracker.update(&signatures(11, 21, 30), 3),
        Some(1),
        "must skip the slot being written and take the newest stable one"
    );
    assert_eq!(
        tracker.update(&signatures(11, 21, 31), 4),
        Some(2),
        "next poll: slot 3 in flight, slot 2 newest complete"
    );
    assert_eq!(
        tracker.update(&signatures(11, 21, 31), 5),
        Some(3),
        "no change: slot 3 is now complete and newest"
    );
    assert_eq!(
        tracker.update(&signatures(11, 21, 31), 6),
        Some(3),
        "static content keeps the choice"
    );
    assert_eq!(
        tracker.update(&signatures(12, 22, 32), 7),
        Some(3),
        "all changed at once: keep the previous choice among ties"
    );
}

/// Change times let the caller refuse to step backwards: a slot rewritten in
/// place makes an older slot the newest stable one for a moment.
#[test]
fn rewritten_slot_reports_its_older_and_newer_times() {
    let mut tracker = SlotTracker::default();
    tracker.update(&signatures(10, 20, 30), 1);
    tracker.update(&signatures(11, 20, 30), 2);
    tracker.update(&signatures(11, 21, 30), 3);
    assert!(
        tracker.update(&signatures(11, 21, 30), 4) == Some(2) && tracker.changed_ns() == 3,
        "slot 2 newest, changed at t3"
    );
    let shown = tracker.changed_ns();
    // Slot 2 is rewritten in place: slot 1 (t2) is now the newest stable one,
    // older than what is shown, so the caller must keep its picture.
    assert!(
        tracker.update(&signatures(11, 22, 30), 5) == Some(1) && tracker.changed_ns() < shown,
        "older slot reported with its older time"
    );
    // Once the rewrite is complete, slot 2 returns with a newer time: re-render it.
    assert!(
        tracker.update(&signatures(11, 22, 30), 6) == Some(2)
            && tracker.changed_ns() == 5
            && tracker.changed_ns() > shown,
        "rewritten slot must come back as newer content"
    );
    // The app restarts a tracker by replacing it.
    let mut tracker = SlotTracker::default();
    assert!(
        tracker.changed_ns() == 0
            && tracker.update(&signatures(1, 2, 3), 7) == Some(1)
            && tracker.changed_ns() == 0,
        "reset"
    );
}

/// Cadence regression: a game can rewrite a subset of its ring every 6.94 ms.
/// Once-per-report sampling rejects those slots on every frame because they
/// always just changed, leaving a stale baseline on screen. A bounded
/// confirmation within the SAME report must deliver each update.
#[test]
fn ring_reuse_keeps_the_source_cadence() {
    for changed_slots in 1..=3 {
        let mut tracker = SlotTracker::default();
        let mut sampling = SlotSampling::default();
        let mut signatures = signatures(10, 20, 30);
        let mut shown = 0;
        let mut updates = 0;
        assert!(
            sampling.due(1, 1_000_000, true),
            "initial ring must be sampled"
        );
        tracker.update(&signatures, 1_000_000);
        sampling.sampled(tracker.needs_confirmation, 1_000_000);
        for frame in 2..146_u64 {
            let now = frame * 6_944_444;
            for slot in &mut signatures[..changed_slots] {
                slot.1 += 1;
            }
            assert!(
                sampling.due(frame, now, false),
                "new source report must be sampled immediately"
            );
            tracker.update(&signatures, now);
            sampling.sampled(tracker.needs_confirmation, now);
            if tracker.changed_ns() > shown {
                shown = tracker.changed_ns();
                updates += 1;
            }
            if tracker.needs_confirmation {
                assert!(
                    !sampling.due(frame, now + 1, false),
                    "do not busy-poll a changing slot"
                );
                assert_eq!(
                    sampling.wait_ns(now, 4_000_000),
                    CONFIRMATION_NS,
                    "confirmation must wake before the next game frame"
                );
                let confirmed = now + CONFIRMATION_NS;
                assert!(
                    sampling.due(frame, confirmed, false),
                    "confirm without a new compositor report"
                );
                tracker.update(&signatures, confirmed);
                sampling.sampled(tracker.needs_confirmation, confirmed);
                if tracker.changed_ns() > shown {
                    shown = tracker.changed_ns();
                    updates += 1;
                }
            }
            assert!(
                !sampling.due(frame, now + 1_000_000, false),
                "static content must not keep polling the GPU"
            );
        }
        assert_eq!(
            updates, 144,
            "ring reuse must not reduce 144 source updates to a lower cadence \
             ({changed_slots} slots rewritten)"
        );
    }
}

#[test]
fn one_confirmation_per_report() {
    let mut sampling = SlotSampling::default();
    assert!(sampling.due(4, 100, true), "start confirmation fixture");
    sampling.sampled(true, 100);
    assert!(sampling.due(4, 500_100, false), "first confirmation is due");
    sampling.sampled(true, 500_100); // still changing: wait for the next report
    assert!(
        !sampling.due(4, 2_000_000, false),
        "at most one confirmation per report"
    );
    assert!(
        sampling.due(5, 2_000_001, false),
        "new report permits another observation"
    );
    sampling.sampled(true, 2_000_001);
    assert!(
        sampling.due(1, 2_000_002, true),
        "reconnect/ring switch resets pending confirmation"
    );
    assert!(
        !sampling.due(1, 9_000_000, false),
        "old ring deadline must not survive reset"
    );
}

/// Rest Frame capture from a busy local ring (SUPERHOT VR on the Frame: 2
/// captures against 345 invalid pairs). The game writes the next slot of a
/// three-slot ring every display frame and Rest Frame observes once per
/// display frame. Each write straddles the observation: at observation f,
/// slot f%3 has started and slot (f-1)%3 finished after the previous
/// observation, so both read as changed. The old rule (a choice with a change
/// time and no newer changing slot, without a confirmation observation)
/// therefore never captured.
#[test]
fn busy_ring_captures_after_a_confirmation() {
    let mut tracker = SlotTracker::default();
    let mut signatures = signatures(10, 20, 30);
    let slot_of = |frame: u64| frame % 3 + 1;
    // The first observation after activation only seeds the history.
    tracker.update(&signatures, 1_000_000);
    assert!(
        tracker.capture_slot(false).is_none() && tracker.capture_slot(true).is_none(),
        "seeded history has no change time"
    );
    let mut old_rule = 0;
    let mut captures = 0;
    for frame in 1..=120_u64 {
        let now = 1_000_000 + frame * 8_333_333;
        signatures[slot_of(frame - 1) as usize - 1].1 += 1; // the end of the previous frame's write
        signatures[slot_of(frame) as usize - 1].1 += 1; // the start of this frame's write
        tracker.update(&signatures, now);
        if tracker.chosen.is_some() && tracker.changed_ns() != 0 && !tracker.needs_confirmation {
            old_rule += 1;
        }
        assert!(
            tracker.capture_slot(false).is_none() || !tracker.needs_confirmation,
            "a newer slot that just changed may be complete: confirm before capturing"
        );
        // Prompt confirmation: this frame's slot is still being written.
        signatures[slot_of(frame) as usize - 1].1 += 1;
        tracker.update(&signatures, now + CONFIRMATION_NS);
        // The end of the previous write is already an observed change at frame 1.
        let slot = tracker.capture_slot(true);
        assert_eq!(
            slot,
            Some(slot_of(frame - 1)),
            "after confirmation: capture the newest completed slot"
        );
        let slot = slot.unwrap();
        assert_ne!(
            slot,
            slot_of(frame),
            "must never capture the slot being written"
        );
        assert_eq!(
            tracker.signature(slot),
            Some(signatures[slot as usize - 1].1),
            "signature of the captured slot"
        );
        captures += 1;
    }
    assert_eq!(old_rule, 0, "fixture must reproduce the old rule's failure");
    assert_eq!(
        captures, 120,
        "every confirmed observation must allow a capture"
    );
    assert!(
        tracker.signature(99).is_none(),
        "unknown slot has no signature"
    );
}

/// A lightly loaded game finishes each slot before the observation: the
/// confirmation finds it unchanged and it becomes the capture.
#[test]
fn quiet_ring_confirms_the_newest_slot() {
    let mut tracker = SlotTracker::default();
    let mut signatures = signatures(10, 20, 30);
    tracker.update(&signatures, 1_000_000);
    for frame in 1..=30_u64 {
        let now = 1_000_000 + frame * 8_333_333;
        let written = frame % 3 + 1;
        signatures[written as usize - 1].1 += 1;
        tracker.update(&signatures, now);
        assert!(
            tracker.capture_slot(false).is_none(),
            "a just-changed newest slot needs confirmation first"
        );
        tracker.update(&signatures, now + CONFIRMATION_NS);
        assert!(
            tracker.capture_slot(true) == Some(written) && !tracker.needs_confirmation,
            "confirmed newest slot"
        );
    }
}

/// Resuming observation with old history: every slot changed while nobody
/// looked, so the fallback choice is a slot that changed just now (it may be
/// mid-write). It must not be captured; the session also resets.
#[test]
fn resumed_observation_captures_nothing_until_a_slot_settles() {
    let stable = |tracker: &SlotTracker| tracker.chosen.is_some_and(|(_, _, stable)| stable);
    let mut tracker = SlotTracker::default();
    let mut signatures = signatures(10, 20, 30);
    tracker.update(&signatures, 1);
    signatures[0].1 += 1;
    tracker.update(&signatures, 2);
    tracker.update(&signatures, 3);
    assert_eq!(
        tracker.capture_slot(false),
        Some(1),
        "static content with a known change time"
    );
    for slot in &mut signatures {
        slot.1 += 5;
    }
    tracker.update(&signatures, 4_000_000_000);
    assert!(
        !stable(&tracker)
            && tracker.capture_slot(false).is_none()
            && tracker.capture_slot(true).is_none(),
        "all slots changed in the latest observation: nothing is known to be complete"
    );
    let tracker = SlotTracker::default();
    assert!(
        !stable(&tracker) && tracker.signature(1).is_none(),
        "reset clears stability and signatures"
    );
}

#[test]
fn stream_frame_among_ui_textures() {
    let stream = stream();
    let selection = select(&stream, policy()).unwrap();
    assert!(
        selection.stream && selection.from_labels().is_none(),
        "decoder handoff is a candidate, pending pixel evidence"
    );
    let mut resized = stream.clone();
    resized[3].image.width = 1920;
    resized[3].image.height = 2048;
    assert!(
        select(&resized, policy())
            .unwrap()
            .from_proof(proof(Packing::LeftQuadrants), &[])
            .is_some(),
        "P010 decoder candidates may have other resolutions"
    );
    resized[3].acquired = false;
    assert_eq!(
        reason(&resized, policy()),
        Some(Reason::NoSource),
        "P010 alone is not a decoder handoff"
    );
    let stereo = selection
        .from_proof(proof(Packing::LeftQuadrants), &[])
        .unwrap();
    assert!(
        stereo.rule == Rule::Stream2x2 && stereo.slots.len() == 1 && !stereo.labelled,
        "Rest Frame must identify the stock full-view stereo image"
    );
    let [left, right] = stereo.slots[0];
    assert!(
        left.id == 4
            && right.id == 4
            && left.crop == [0.0, 0.0, 0.5, 0.5]
            && right.crop == [0.0, 0.5, 0.5, 0.5],
        "Rest Frame eyes must use the same stream image's full left-column quadrants"
    );
    assert!(
        selection.generation.epoch == 7 && generation_ids(&selection) == [4],
        "Rest Frame stream identity must carry the connection epoch and selected images"
    );
    assert!(
        select(&stream, policy()).unwrap().generation == selection.generation,
        "static content must keep its source generation"
    );
    let reconnected = Policy {
        epoch: 8,
        ..policy()
    };
    assert!(
        select(&stream, reconnected).unwrap().generation != selection.generation,
        "reconnect must invalidate a capture even if the new connection repeats image IDs"
    );
    assert!(
        selection
            .from_proof(proof(Packing::SideBySide), &[])
            .is_none(),
        "a stream accepts only its own packing"
    );
}

#[test]
fn side_by_side_ring_and_its_generation() {
    let ring = sbs_ring();
    let selection = select(&ring, policy()).unwrap();
    assert!(
        selection.from_labels().is_none(),
        "candidates expose no capturable slot until a matching proof is applied"
    );
    let stereo = selection
        .from_proof(proof(Packing::SideBySide), &[])
        .unwrap();
    assert!(
        stereo.rule == Rule::LocalSideBySide && stereo.slots.len() == 3,
        "Rest Frame must enumerate complete SBS candidates"
    );
    for [left, right] in &stereo.slots {
        assert!(
            left.id == right.id
                && left.layer == 0
                && right.layer == 0
                && left.crop == [0.0, 0.0, 0.5, 1.0]
                && right.crop == [0.5, 0.0, 0.5, 1.0],
            "Rest Frame must never mix SBS slots or trim either eye"
        );
    }
    let previous = selection.generation;
    let changed = |edit: fn(&mut Candidate)| {
        let mut ring = sbs_ring();
        ring.iter_mut().for_each(edit);
        select(&ring, policy()).unwrap().generation
    };
    assert!(
        changed(|candidate| candidate.image.width += 1) != previous,
        "resolution changes with reused IDs must invalidate the source generation"
    );
    assert!(
        changed(|candidate| candidate.image.format = RGBA) != previous,
        "format changes with reused IDs must invalidate the source generation"
    );
    assert!(
        changed(|candidate| candidate.image.array_layers = 2) != previous,
        "layer changes with reused IDs must invalidate the source generation"
    );
    assert!(
        changed(|candidate| candidate.revision += 1) != previous,
        "a republished image with a reused ID must invalidate the source generation"
    );
    let mut replaced = sbs_ring();
    replaced.extend((60..=62).map(|id| scene(id, SRGB, 3456, 1728, 1)));
    let selection = select(&replaced, policy()).unwrap();
    assert!(
        generation_ids(&selection) == [40, 41, 42, 60, 61, 62] && selection.generation != previous,
        "changed membership must invalidate cached evidence"
    );
    replaced.remove(4); // removing id 61 must invalidate the identity
    let selection = select(&replaced, policy()).unwrap();
    assert!(
        generation_ids(&selection) == [40, 41, 42, 60, 62] && selection.generation != previous,
        "removing a slot must not preserve the old ring identity"
    );
}

#[test]
fn array_rings_need_two_layers_and_more_than_one_image() {
    let selection = select(&array_ring(), policy()).unwrap();
    let stereo = selection.from_proof(proof(Packing::Array), &[]).unwrap();
    assert!(
        stereo.rule == Rule::LocalArray && stereo.slots.len() == 3,
        "Rest Frame must enumerate validated two-layer array candidates"
    );
    for [left, right] in &stereo.slots {
        assert!(
            left.id == right.id
                && left.layer == 0
                && right.layer == 1
                && left.crop == [0.0, 0.0, 1.0, 1.0]
                && right.crop == [0.0, 0.0, 1.0, 1.0],
            "Rest Frame arrays must take both complete eye layers from one slot"
        );
    }
    assert!(
        selection
            .from_proof(proof(Packing::SideBySide), &[])
            .is_none(),
        "an array ring accepts only the array packing"
    );
    assert_eq!(
        reason(&[scene(50, SRGB, 2016, 2016, 2)], policy()),
        Some(Reason::NoSource),
        "a lone array is not a ring"
    );
    assert_eq!(
        reason(
            &[
                scene(60, SRGB, 1728, 1728, 4),
                scene(61, SRGB, 1728, 1728, 4)
            ],
            policy()
        ),
        Some(Reason::UnsupportedLayout),
        "four-layer arrays must not be guessed as stereo"
    );
    assert_eq!(
        reason(&[scene(30, SRGB, 2016, 2016, 1)], policy()),
        Some(Reason::NoSource),
        "a lone square image is not proven stereo"
    );
}

#[test]
fn separate_eye_rings_need_a_complete_proof() {
    let selection = select(&ring(), policy()).unwrap();
    assert!(
        selection.from_labels().is_none() && selection.images.len() == 6,
        "separate eye candidates must await layout and temporal evidence"
    );
    let split = [[5, 0], [3, 1], [4, 2]];
    let paired = selection
        .from_proof(proof(Packing::SeparateEyes), &split)
        .unwrap();
    assert!(
        paired.rule == Rule::LocalSlots && ids(&paired) == [[115, 110], [113, 111], [114, 112]],
        "use measured temporal partners and eye order, independent of image IDs"
    );
    assert!(
        selection
            .from_proof(proof(Packing::SeparateEyes), &[[5, 0], [3, 1], [4, 0]])
            .is_none(),
        "one image must not belong to multiple pairs"
    );
    assert!(
        selection
            .from_proof(proof(Packing::SeparateEyes), &[[5, 0], [3, 1], [4, 6]])
            .is_none(),
        "a pair must name images of this selection"
    );
    let atlas = selection
        .from_proof(
            Layout {
                packing: Packing::TopBottom,
                swapped: true,
            },
            &[],
        )
        .unwrap();
    assert!(
        atlas.rule == Rule::LocalAtlas
            && atlas.slots[0][0].crop == [0.0, 0.5, 1.0, 0.5]
            && atlas.slots[0][1].crop == [0.0, 0.0, 1.0, 0.5],
        "swapped vertical atlas is accepted with proof"
    );

    // Local pair of eye textures among UI panels.
    let pair = [
        image(10, BGRA, 900, 60, 1),
        scene(11, SRGB, 2016, 2016, 1),
        scene(12, SRGB, 2016, 2016, 1),
        image(13, BGRA, 1920, 1080, 1),
    ];
    assert!(
        select(&pair, policy())
            .unwrap()
            .from_proof(proof(Packing::SeparateEyes), &[[0, 1]])
            .is_none(),
        "one separately rewritten image per eye stays unsupported"
    );
    // A relaunched app: an old ring (110-115) lingers next to the new one (130-135).
    let mut rings = ring();
    rings.extend((130..=135).map(|id| scene(id, SRGB, 1728, 1728, 1)));
    let selection = select(&rings, policy()).unwrap();
    assert!(
        selection.images.len() == 12
            && selection
                .from_proof(proof(Packing::SeparateEyes), &split)
                .is_none(),
        "a proof must cover the complete current local candidate set"
    );
    let mut missing_eye = ring();
    missing_eye.remove(2);
    assert!(
        select(&missing_eye, policy())
            .unwrap()
            .from_proof(proof(Packing::SeparateEyes), &split)
            .is_none(),
        "missing separate-eye candidates remain unverified"
    );
}

#[test]
fn ambiguous_invalid_and_ui_sources_are_rejected() {
    let ambiguous = [
        scene(200, SRGB, 2000, 2000, 2),
        scene(201, SRGB, 2000, 2000, 2),
        scene(202, SRGB, 4000, 1000, 1),
        scene(203, SRGB, 4000, 1000, 1),
    ];
    assert_eq!(
        reason(&ambiguous, policy()),
        Some(Reason::Ambiguous),
        "equally large incompatible scene groups must not depend on candidate order"
    );
    let mut reversed = ambiguous;
    reversed.reverse();
    assert_eq!(
        reason(&reversed, policy()),
        Some(Reason::Ambiguous),
        "ambiguous source rejection must be independent of report order"
    );
    let sbs = scene(20, SRGB, 4032, 2016, 1);
    assert_eq!(
        reason(&[sbs, sbs], policy()),
        Some(Reason::Ambiguous),
        "duplicate image IDs cannot define two source slots"
    );
    let mut invalid = sbs;
    invalid.layout = 0; // VK_IMAGE_LAYOUT_UNDEFINED
    assert_eq!(
        reason(&[invalid], policy()),
        Some(Reason::InvalidDescription),
        "undefined source layout cannot be sampled for a snapshot"
    );
    let mut invalid = sbs;
    invalid.image.id = 0;
    assert_eq!(
        reason(&[invalid], policy()),
        Some(Reason::InvalidDescription),
        "an image without an identity must not be used"
    );
    assert_eq!(
        reason(&[image(220, P010, 2304, 2304, 2)], policy()),
        Some(Reason::UnsupportedLayout),
        "unknown layered stream layout must be rejected"
    );
    let panels = [
        image(45, BGRA, 2400, 1200, 1),
        image(46, BGRA, 1920, 1080, 1),
    ];
    assert_eq!(
        reason(&panels, policy()),
        Some(Reason::NoSource),
        "imported overlay panels must stay out of Rest Frame"
    );
    let ui = [scene(40, BGRA, 900, 60, 1), scene(41, BGRA, 450, 1200, 1)];
    assert_eq!(
        reason(&ui, policy()),
        Some(Reason::NoSource),
        "UI-sized textures are never a scene"
    );
}

#[test]
fn incomplete_reports_exclusions_and_format_filter() {
    let stream = stream();
    let incomplete = Policy {
        complete: false,
        ..policy()
    };
    assert_eq!(
        reason(&stream, incomplete),
        Some(Reason::IncompleteReport),
        "truncated observer reports must not be guessed as a complete stereo source"
    );
    let owned = Policy {
        excluded_ids: &[4],
        ..policy()
    };
    assert_eq!(
        reason(&stream, owned),
        Some(Reason::NoSource),
        "explicit owned-image IDs must be excluded"
    );
    let collision = Policy {
        excluded_sizes: &[[3072, 1536], [8, 8], [2304, 2304]],
        ..policy()
    };
    assert_eq!(
        reason(&stream, collision),
        Some(Reason::NoSource),
        "dimension collisions must be conservatively excluded"
    );
    // Without the exclusion the two own-size images would be the largest
    // local group and displace the array ring.
    let mut with_own = array_ring();
    with_own.push(scene(300, SRGB, 3072, 1536, 1));
    with_own.push(scene(301, SRGB, 3072, 1536, 1));
    let stereo = select(&with_own, policy())
        .unwrap()
        .from_proof(proof(Packing::Array), &[])
        .unwrap();
    assert!(
        stereo.rule == Rule::LocalArray && ids(&stereo) == [[50, 50], [51, 51], [52, 52]],
        "own-output compositor copies must not replace the actual stereo scene"
    );
    let unfiltered = Policy {
        excluded_sizes: &[],
        ..policy()
    };
    assert!(
        select(&with_own, unfiltered)
            .unwrap()
            .from_proof(proof(Packing::Array), &[])
            .is_none(),
        "fixture: without the size exclusion the copies win"
    );
    let filtered = || Policy {
        source_format: Some(BGRA),
        ..policy()
    };
    assert!(
        select(&stream, filtered()).is_err() && select(&array_ring(), filtered()).is_err(),
        "Rest Frame must honor its source format filter"
    );
}

/// Eye labels from the compositor's own draws verify a separate-eye ring
/// without measurement, pairing left slot k with right slot k whatever the
/// image IDs. Layout from the Frame trace: bindings 9, 10 and 11 in both eyes'
/// draws.
#[test]
fn compositor_labels_pair_rings_by_slot() {
    let labelled = |eyes_and_slots: &[(u32, u32)]| {
        let group: Vec<_> = eyes_and_slots
            .iter()
            .zip(200..)
            .map(|(&(eyes, slot), id)| {
                let mut candidate = scene(id, SRGB, 1440, 1440, 1);
                candidate.eye = EyeUse {
                    eyes,
                    slot,
                    layers: [0, 0],
                };
                candidate
            })
            .collect();
        select(&group, Policy::default()).unwrap().from_labels()
    };
    // Right eye's images created first, and slots out of order.
    let gorn = labelled(&[(2, 10), (2, 9), (2, 11), (1, 11), (1, 9), (1, 10)]).unwrap();
    assert!(
        gorn.labelled
            && gorn.rule == Rule::LocalSlots
            && ids(&gorn) == [[204, 201], [205, 200], [203, 202]],
        "labelled per-eye rings pair left and right images by slot"
    );
    assert!(
        labelled(&[(1, 9), (1, 10), (1, 11), (2, 9), (2, 10), (2, 12)]).is_none(),
        "an unmatched slot must not pair"
    );
    assert!(
        labelled(&[(1, 9), (1, 9), (2, 9), (2, 10)]).is_none(),
        "one slot used twice in an eye must not pair"
    );
    assert!(
        labelled(&[(1, 9), (1, 10), (2, 9), (0, 0)]).is_none(),
        "an unlabelled image keeps the measured path"
    );
    assert!(
        labelled(&[(1, 9), (2, 9), (3, 10), (3, 10)]).is_none(),
        "an image drawn into both eyes from one layer is not a pair"
    );
    assert!(
        labelled(&[(0, 0), (0, 0), (0, 0), (0, 0)]).is_none(),
        "without labels the selection is unchanged"
    );

    // Two-layer swapchain ring, each image drawn into both eyes from its own layers.
    let mut arrays: Vec<_> = (300..303)
        .map(|id| {
            let mut candidate = scene(id, SRGB, 2160, 2160, 2);
            candidate.eye = EyeUse {
                eyes: 3,
                slot: 9,
                layers: [1, 0], // this runtime puts the left eye in layer 1
            };
            candidate
        })
        .collect();
    let array = select(&arrays, Policy::default())
        .unwrap()
        .from_labels()
        .unwrap();
    assert!(
        array.labelled
            && array.rule == Rule::LocalArray
            && array.slots.len() == 3
            && array.slots[0].map(|eye| (eye.id, eye.layer)) == [(300, 1), (300, 0)],
        "labelled array rings take each eye's layer from its draw"
    );
    arrays[1].eye.layers = [0, 0];
    assert!(
        select(&arrays, Policy::default())
            .unwrap()
            .from_labels()
            .is_none(),
        "one layer in both eyes is not stereo"
    );
}
