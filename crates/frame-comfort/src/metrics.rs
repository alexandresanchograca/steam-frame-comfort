use comfort_runtime::mailbox::monotonic_ns;
use std::collections::VecDeque;

#[derive(Default)]
pub(super) struct Samples(VecDeque<f64>);

impl Samples {
    pub(super) fn add(&mut self, ms: f64) {
        if self.0.len() == 512 {
            self.0.pop_front();
        }
        self.0.push_back(ms);
    }

    pub(super) fn since(&mut self, ns: u64) {
        self.add(monotonic_ns().saturating_sub(ns) as f64 / 1e6);
    }

    pub(super) fn percentiles<const N: usize>(&self, positions: [f64; N]) -> [f64; N] {
        if self.0.is_empty() {
            return [-1.0; N];
        }
        let mut sorted = self.0.iter().copied().collect::<Vec<_>>();
        sorted.sort_unstable_by(f64::total_cmp);
        positions.map(|p| sorted[((p * sorted.len() as f64) as usize).min(sorted.len() - 1)])
    }
}

#[derive(Default)]
pub(super) struct Metrics {
    pub(super) work: Samples,
    pub(super) capture: Samples,
    pub(super) frame_wait: Samples,
    pub(super) submit: Samples,
    pub(super) gpu: Samples,
    pub(super) render_done: Samples,
    pub(super) after_vsync: Samples,
    pub(super) compositor_gap: Samples,
    pub(super) missed_frames: u64,
    pub(super) face_submissions: u64,
    last_compositor: Option<(u64, u64)>,
    last_vsync: Option<u64>,
}

impl Metrics {
    pub(super) fn compositor(&mut self, sample: Option<(u64, u64)>) {
        if let (Some((epoch, now)), Some((previous_epoch, previous))) =
            (sample, self.last_compositor)
            && epoch == previous_epoch
            && now > previous
        {
            self.compositor_gap.add((now - previous) as f64 / 1e6);
        }
        self.last_compositor = sample;
    }

    pub(super) fn submitted(&mut self, faces: u32, vsync: Option<(f32, u64)>) {
        self.face_submissions += u64::from(faces.count_ones());
        if let Some((seconds, counter)) = vsync {
            self.after_vsync.add(f64::from(seconds) * 1000.0);
            if let Some(last) = self.last_vsync {
                self.missed_frames += counter.saturating_sub(last).saturating_sub(1);
            }
            self.last_vsync = Some(counter);
        }
    }

    pub(super) fn hidden(&mut self) {
        self.last_vsync = None;
    }
}
