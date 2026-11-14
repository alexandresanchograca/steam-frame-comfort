//! Rest Frame capture scheduling. All times are monotonic milliseconds.
use crate::settings::{Periphery, Settings, Stick};

#[derive(Clone, Copy, Debug)]
pub struct Timing {
    pub stick: Stick,
    pub snapshots: bool,
    pub standby: bool,
    pub refresh_ms: f32,
    pub blend_ms: f32,
    pub stillness_deg_per_sec: f32,
    pub dwell_ms: f32,
    pub max_wait_ms: f32,
    pub early_refresh_deg: f32,
    pub capture_spacing_ms: f32,
    pub arming_timeout_ms: f32,
    pub retry_ms: f32,
    pub refresh_grace_ms: f32,
    pub shell_half_size: f32,
    pub pose_offset_ms: f32,
    pub deadzone: f32,
    pub release_ms: f32,
    pub fade_in_ms: f32,
    pub fade_out_ms: f32,
    pub strength: f32,
    pub display_period_ms: f32,
}

impl Default for Timing {
    fn default() -> Self {
        Self {
            stick: Stick::Left,
            snapshots: true,
            standby: false,
            refresh_ms: 1000.0,
            blend_ms: 200.0,
            stillness_deg_per_sec: 20.0,
            dwell_ms: 50.0,
            max_wait_ms: 200.0,
            early_refresh_deg: 30.0,
            capture_spacing_ms: 250.0,
            arming_timeout_ms: 300.0,
            retry_ms: 1000.0,
            refresh_grace_ms: 250.0,
            shell_half_size: 4.0,
            pose_offset_ms: 0.0,
            deadzone: 0.15,
            release_ms: 500.0,
            fade_in_ms: 120.0,
            fade_out_ms: 300.0,
            strength: 1.0,
            display_period_ms: 1000.0 / 90.0,
        }
    }
}

impl From<&Settings> for Timing {
    fn from(settings: &Settings) -> Self {
        Self {
            stick: settings.stick,
            snapshots: settings.periphery == Periphery::Game,
            refresh_ms: settings.rest_frame_interval_ms,
            blend_ms: settings.rest_frame_cross_fade_ms,
            stillness_deg_per_sec: settings.rest_frame_stillness_deg_per_sec,
            max_wait_ms: settings.rest_frame_max_wait_ms,
            early_refresh_deg: settings.rest_frame_early_refresh_deg,
            shell_half_size: settings.rest_frame_shell_half_size,
            pose_offset_ms: settings.rest_frame_pose_offset_ms,
            deadzone: settings.deadzone,
            release_ms: settings.fade_out_delay_ms,
            fade_in_ms: settings.fade_in_ms,
            fade_out_ms: settings.fade_out_ms,
            strength: settings.periphery_opacity,
            ..Self::default()
        }
    }
}

impl Timing {
    fn sane(&mut self) -> bool {
        for (value, low, high) in [
            (&mut self.refresh_ms, 500.0, 3000.0),
            (&mut self.blend_ms, 0.0, 500.0),
            (&mut self.stillness_deg_per_sec, 5.0, 90.0),
            (&mut self.dwell_ms, 0.0, 500.0),
            (&mut self.max_wait_ms, 0.0, 500.0),
            (&mut self.early_refresh_deg, 10.0, 90.0),
            (&mut self.capture_spacing_ms, 250.0, 2000.0),
            (&mut self.arming_timeout_ms, 50.0, 2000.0),
            (&mut self.retry_ms, 100.0, 10000.0),
            (&mut self.refresh_grace_ms, 0.0, 1000.0),
            (&mut self.shell_half_size, 2.0, 5.5),
            (&mut self.pose_offset_ms, -50.0, 250.0),
            (&mut self.deadzone, 0.0, 0.95),
            (&mut self.release_ms, 0.0, 3000.0),
            (&mut self.fade_in_ms, 0.0, 3000.0),
            (&mut self.fade_out_ms, 0.0, 3000.0),
            (&mut self.strength, 0.0, 1.0),
            (&mut self.display_period_ms, 1.0, 100.0),
        ] {
            if !value.is_finite() {
                return false;
            }
            *value = value.clamp(low, high);
        }
        self.blend_ms = self.blend_ms.min(self.refresh_ms * 0.5);
        true
    }

    fn standby_max_age(self) -> f64 {
        (f64::from(self.refresh_ms) - f64::from(self.capture_spacing_ms))
            .max(f64::from(self.refresh_ms) * 0.5)
    }

    fn standby_period(self) -> f64 {
        250.0_f64.max(self.standby_max_age() * 0.5)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Input {
    pub enabled: bool,
    pub dashboard: bool,
    pub input_available: bool,
    pub hand_magnitude: [f32; 2],
    pub source_fresh: bool,
    pub tracking_valid: bool,
    pub mapping_valid: bool,
    pub renderer_healthy: bool,
    pub source_generation: u64,
    pub tracking_epoch: u64,
    pub angular_speed_deg_per_sec: f32,
    pub image_angular_speed_deg_per_sec: f32,
    pub angle_since_capture_deg: f32,
    pub standby_angle_deg: f32,
}

impl Default for Input {
    fn default() -> Self {
        Self {
            enabled: true,
            dashboard: false,
            input_available: true,
            hand_magnitude: [0.0; 2],
            source_fresh: true,
            tracking_valid: true,
            mapping_valid: true,
            renderer_healthy: true,
            source_generation: 0,
            tracking_epoch: 0,
            angular_speed_deg_per_sec: 0.0,
            image_angular_speed_deg_per_sec: 0.0,
            angle_since_capture_deg: 0.0,
            standby_angle_deg: 0.0,
        }
    }
}

impl Input {
    fn magnitude(self, stick: Stick) -> Option<f32> {
        if !self.input_available {
            return None;
        }
        let [left, right] = self.hand_magnitude;
        match stick {
            Stick::Left if nonnegative(left) => Some(left),
            Stick::Right if nonnegative(right) => Some(right),
            Stick::Both if nonnegative(left) && nonnegative(right) => Some(left.max(right)),
            _ => None,
        }
    }
}

fn nonnegative(value: f32) -> bool {
    value.is_finite() && value >= 0.0
}

pub fn stick_force(magnitude: f32, deadzone: f32) -> f32 {
    if !magnitude.is_finite() || !deadzone.is_finite() {
        return 0.0;
    }
    let deadzone = deadzone.clamp(0.0, 0.95);
    ((magnitude - deadzone) / (1.0 - deadzone)).clamp(0.0, 1.0)
}

pub fn smooth_force(current: f32, target: f32, elapsed_ms: f64) -> f32 {
    if target > current {
        target.min(current + (elapsed_ms.max(0.0) / 200.0) as f32)
    } else {
        target.max(current - (elapsed_ms.max(0.0) / 300.0) as f32)
    }
}

pub fn opening_degrees(width: f32, height: f32, minimum_width: f32, force: f32) -> [f32; 2] {
    let width_now = width - force * (width - minimum_width);
    [width_now, height * (width_now / width)]
}

pub fn standby_stick_threshold(deadzone: f32) -> f32 {
    (if deadzone.is_finite() { deadzone } else { 0.0 }).clamp(0.05, 0.1)
}

fn fade_out(level: f32, full: f32, dt: f64, duration: f32) -> f32 {
    if !level.is_finite() || !full.is_finite() || full <= 0.0 || duration <= 0.0 {
        return 0.0;
    }
    let fraction = (level / full).clamp(0.0, 1.0);
    if fraction <= 0.0 {
        return 0.0;
    }
    const KNEE: f32 = 0.15;
    const K: f32 = 1.0 / (KNEE * (2.0 - KNEE));
    let faded = 1.0 - fraction;
    let progress = if faded < K * KNEE * KNEE {
        (faded / K).sqrt()
    } else {
        (faded / (K * KNEE) + KNEE) / 2.0
    } + (dt.max(0.0) / f64::from(duration)) as f32;
    if progress >= 1.0 {
        return 0.0;
    }
    let faded = if progress < KNEE {
        K * progress * progress
    } else {
        K * KNEE * (2.0 * progress - KNEE)
    };
    full * (1.0 - faded)
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum State {
    #[default]
    Idle,
    Arming,
    Entering,
    Active,
    Refreshing,
    Releasing,
    Suspended,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Reason {
    #[default]
    None,
    Disabled,
    Dashboard,
    InputUnavailable,
    SourceUnavailable,
    TrackingInvalid,
    MappingInvalid,
    RenderFailure,
    CaptureTimeout,
    CaptureInvalid,
    SnapshotExpired,
    ClockReset,
    SourceChanged,
    TrackingChanged,
    SettingsChanged,
    InvalidSettings,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CaptureReason {
    #[default]
    Initial,
    Periodic,
    HeadTurn,
    Standby,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureResult {
    Ready,
    InvalidPair,
    RenderFailure,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CaptureRequest {
    pub token: u64,
    pub requested_at_ms: u64,
    pub deadline_ms: u64,
    pub slot: u32,
    pub reason: CaptureReason,
    pub forced: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct Output {
    pub state: State,
    pub reason: Reason,
    pub opening_progress: f32,
    pub refresh_weight: f32,
    pub old_slot: u32,
    pub new_slot: u32,
    pub visible: bool,
    pub needs_both_snapshots: bool,
    pub capture_in_flight: bool,
    pub needs_reenable: bool,
    pub capture_request: Option<CaptureRequest>,
    pub refresh_due_ms: Option<u64>,
    pub last_capture_ms: Option<u64>,
    pub consecutive_render_failures: u32,
    pub standby_slot: Option<u32>,
    pub standby_at_ms: Option<u64>,
    pub activation: u64,
}

#[derive(Clone, Copy)]
struct Completion {
    capture_ms: u64,
    completed_ms: u64,
    result: CaptureResult,
}

impl Completion {
    fn timely(self, request: CaptureRequest, now: u64) -> bool {
        self.completed_ms >= request.requested_at_ms
            && self.completed_ms <= now
            && now <= request.deadline_ms
            && (self.result != CaptureResult::Ready
                || (self.capture_ms >= request.requested_at_ms
                    && self.capture_ms <= self.completed_ms))
    }
}

#[derive(Clone, Copy)]
struct Standby {
    slot: u32,
    capture_ms: u64,
}

pub struct Controller {
    state: State,
    reason: Reason,
    due_reason: CaptureReason,
    previous_ms: Option<u64>,
    last_movement_ms: Option<u64>,
    still_since_ms: Option<u64>,
    due_ms: Option<u64>,
    stick: Option<Stick>,
    mapping_settings: Option<(f32, f32, bool)>,
    pending: Option<CaptureRequest>,
    emitted: Option<CaptureRequest>,
    completion: Option<Completion>,
    standby: Option<Standby>,
    standby_due_ms: Option<u64>,
    standby_at_ms: Option<u64>,
    last_stick_ms: Option<u64>,
    standby_retry_ms: u64,
    activations: u64,
    arming_ms: u64,
    capture_ms: u64,
    next_token: u64,
    retry_ms: u64,
    capture_retry_ms: u64,
    generation: Option<(u64, u64)>,
    moving: bool,
    has_snapshot: bool,
    latched_failure: bool,
    opening_progress: f32,
    refresh_weight: f32,
    release_full: f32,
    old_slot: u32,
    new_slot: u32,
    failures: u32,
}

impl Default for Controller {
    fn default() -> Self {
        Self {
            state: State::default(),
            reason: Reason::default(),
            due_reason: CaptureReason::default(),
            previous_ms: None,
            last_movement_ms: None,
            still_since_ms: None,
            due_ms: None,
            stick: None,
            mapping_settings: None,
            pending: None,
            emitted: None,
            completion: None,
            standby: None,
            standby_due_ms: None,
            standby_at_ms: None,
            last_stick_ms: None,
            standby_retry_ms: 0,
            activations: 0,
            arming_ms: 0,
            capture_ms: 0,
            next_token: 0,
            retry_ms: 0,
            capture_retry_ms: 0,
            generation: None,
            moving: false,
            has_snapshot: false,
            latched_failure: false,
            opening_progress: 0.0,
            refresh_weight: 1.0,
            release_full: 0.0,
            old_slot: 0,
            new_slot: 0,
            failures: 0,
        }
    }
}

impl Controller {
    pub fn complete_capture(
        &mut self,
        token: u64,
        capture_ms: u64,
        completed_ms: u64,
        result: CaptureResult,
    ) -> bool {
        if self.pending.is_none_or(|request| request.token != token) || self.completion.is_some() {
            return false;
        }
        self.completion = Some(Completion {
            capture_ms,
            completed_ms,
            result,
        });
        true
    }

    pub fn reset(&mut self) {
        self.clear(State::Idle, Reason::None);
        self.previous_ms = None;
        self.generation = None;
        self.stick = None;
        self.mapping_settings = None;
        self.failures = 0;
        self.latched_failure = false;
        self.retry_ms = 0;
        self.drop_standby();
        self.last_stick_ms = None;
    }

    pub fn update(&mut self, mut settings: Timing, input: Input, now: u64) -> Output {
        self.emitted = None;
        if !settings.sane() {
            return self.pause(Reason::InvalidSettings, now, 1000.0);
        }
        let previous = self.previous_ms.replace(now).unwrap_or(now);
        if now < previous {
            self.generation = None;
            return self.pause(Reason::ClockReset, now, settings.retry_ms);
        }
        if !settings.standby {
            self.drop_standby();
        }
        if !input.enabled || settings.strength <= 0.0 {
            self.clear(State::Idle, Reason::Disabled);
            self.drop_standby();
            self.failures = 0;
            self.latched_failure = false;
            self.retry_ms = now;
            self.generation = None;
            self.stick = Some(settings.stick);
            return self.output();
        }

        let pause = if input.dashboard {
            Some(Reason::Dashboard)
        } else if settings.snapshots && !input.source_fresh {
            Some(Reason::SourceUnavailable)
        } else if !input.tracking_valid
            || !nonnegative(input.angular_speed_deg_per_sec)
            || !nonnegative(input.image_angular_speed_deg_per_sec)
        {
            Some(Reason::TrackingInvalid)
        } else if !input.mapping_valid || !nonnegative(input.angle_since_capture_deg) {
            Some(Reason::MappingInvalid)
        } else {
            None
        };
        if let Some(reason) = pause {
            return self.pause(reason, now, settings.retry_ms);
        }
        if !input.renderer_healthy {
            if self.state != State::Suspended || self.reason != Reason::RenderFailure {
                self.render_failure(now, settings);
            }
            return self.output();
        }
        let Some(magnitude) = input.magnitude(settings.stick) else {
            return self.pause(Reason::InputUnavailable, now, settings.retry_ms);
        };
        if self
            .stick
            .replace(settings.stick)
            .is_some_and(|stick| stick != settings.stick)
        {
            return self.pause(Reason::SettingsChanged, now, 0.0);
        }
        let mapping = (
            settings.shell_half_size,
            settings.pose_offset_ms,
            settings.snapshots,
        );
        if self
            .mapping_settings
            .replace(mapping)
            .is_some_and(|old| old != mapping)
        {
            self.drop_standby();
            if self.has_snapshot || self.pending.is_some() || self.state == State::Arming {
                return self.pause(Reason::SettingsChanged, now, 0.0);
            }
        }
        if let Some((source, tracking)) = self
            .generation
            .replace((input.source_generation, input.tracking_epoch))
        {
            if tracking != input.tracking_epoch {
                return self.pause(Reason::TrackingChanged, now, 0.0);
            }
            if settings.snapshots && source != input.source_generation {
                return self.pause(Reason::SourceChanged, now, 0.0);
            }
        }

        let threshold = if self.moving {
            (settings.deadzone - 0.03).max(0.0)
        } else {
            settings.deadzone
        };
        self.moving = magnitude > threshold;
        if self.moving {
            self.last_movement_ms = Some(now);
        }
        let target = self.moving
            || self
                .last_movement_ms
                .is_some_and(|last| elapsed(now, last) < f64::from(settings.release_ms));
        let still = input.angular_speed_deg_per_sec <= settings.stillness_deg_per_sec
            && input.image_angular_speed_deg_per_sec <= settings.stillness_deg_per_sec;
        if still {
            self.still_since_ms.get_or_insert(now);
        } else {
            self.still_since_ms = None;
        }

        if self
            .pending
            .is_some_and(|request| request.reason == CaptureReason::Standby)
            && let Some((request, done)) = self.take_completion()
        {
            if done.result == CaptureResult::RenderFailure {
                self.render_failure(now, settings);
                return self.output();
            }
            if done.result == CaptureResult::Ready && done.timely(request, now) {
                self.standby = Some(Standby {
                    slot: request.slot,
                    capture_ms: done.capture_ms,
                });
                self.standby_due_ms = None;
                self.failures = 0;
            } else {
                if self
                    .standby
                    .is_some_and(|standby| standby.slot == request.slot)
                {
                    self.standby = None;
                }
                self.standby_retry_ms = add(now, settings.display_period_ms);
            }
        }

        let last_stick = self.last_stick_ms;
        let any_stick = input
            .hand_magnitude
            .into_iter()
            .filter(|v| v.is_finite())
            .fold(0.0_f32, f32::max);
        if any_stick > standby_stick_threshold(settings.deadzone) {
            self.last_stick_ms = Some(now);
        }
        if let (Some(standby), Some(last_stick)) = (self.standby, last_stick)
            && last_stick >= standby.capture_ms
        {
            self.drop_standby();
        }

        let mut dt = elapsed(now, previous);
        if self.state == State::Arming && !self.moving {
            self.clear(State::Idle, Reason::None);
            return self.output();
        }
        if matches!(self.state, State::Idle | State::Suspended) {
            if !self.moving || self.latched_failure || now < self.retry_ms {
                self.request_standby(settings, input, now);
                return self.output();
            }
            if !settings.snapshots {
                self.clear(State::Entering, Reason::None);
                self.has_snapshot = true;
                self.capture_ms = now;
            } else if let Some(standby) = self.usable_standby(settings, input, now) {
                self.start_from_standby(standby);
            } else {
                self.arm(now);
                if still {
                    self.still_since_ms = Some(now);
                }
            }
            self.moving = true;
            self.last_movement_ms = Some(now);
            dt = 0.0;
        }

        let mut started_blend = false;
        if let Some((request, done)) = self.take_completion() {
            if !done.timely(request, now) {
                return self.pause(Reason::CaptureInvalid, now, settings.retry_ms);
            }
            match done.result {
                CaptureResult::RenderFailure => {
                    self.render_failure(now, settings);
                    return self.output();
                }
                CaptureResult::InvalidPair => {
                    self.capture_retry_ms = add(now, settings.display_period_ms);
                    self.reason = Reason::CaptureInvalid;
                }
                CaptureResult::Ready => {
                    let first = !self.has_snapshot;
                    self.old_slot = if first { request.slot } else { self.new_slot };
                    self.new_slot = request.slot;
                    self.capture_ms = done.capture_ms;
                    self.has_snapshot = true;
                    self.refresh_weight = if first || settings.blend_ms <= 0.0 {
                        1.0
                    } else {
                        0.0
                    };
                    self.state = if first {
                        State::Entering
                    } else {
                        self.shown_state()
                    };
                    self.reason = Reason::None;
                    self.due_ms = None;
                    self.failures = 0;
                    started_blend = !first;
                    if first {
                        dt = 0.0;
                        self.activations += 1;
                    }
                }
            }
        }
        if self.state == State::Arming
            && elapsed(now, self.arming_ms) >= f64::from(settings.arming_timeout_ms)
        {
            return self.pause(Reason::CaptureTimeout, now, settings.retry_ms);
        }
        let max_age =
            f64::from(settings.refresh_ms + settings.max_wait_ms + settings.refresh_grace_ms)
                + 2.0 * f64::from(settings.display_period_ms);
        if self.has_snapshot {
            let expired = settings.snapshots && elapsed(now, self.capture_ms) > max_age;
            if !target && self.state != State::Releasing {
                self.state = State::Releasing;
                self.release_full = self.opening_progress;
                self.pending = None;
                self.completion = None;
                self.due_ms = None;
                dt = 0.0;
            } else if target && self.state == State::Releasing {
                if expired {
                    if let Some(standby) = self.usable_standby(settings, input, now) {
                        self.start_from_standby(standby);
                    } else {
                        self.arm(now);
                    }
                } else {
                    self.state = self.shown_state();
                }
                dt = 0.0;
            }
            if settings.snapshots
                && self.has_snapshot
                && self.state != State::Releasing
                && elapsed(now, self.capture_ms) > max_age
            {
                return self.pause(Reason::SnapshotExpired, now, settings.retry_ms);
            }
            if self.state == State::Releasing {
                self.opening_progress = fade_out(
                    self.opening_progress,
                    self.release_full,
                    dt,
                    settings.fade_out_ms,
                );
                if self.opening_progress <= 0.0 {
                    self.clear(State::Idle, Reason::None);
                }
            } else if self.has_snapshot {
                self.opening_progress = advance(self.opening_progress, dt, settings.fade_in_ms);
                if self.refresh_weight < 1.0 {
                    self.refresh_weight = advance(
                        self.refresh_weight,
                        if started_blend { 0.0 } else { dt },
                        settings.blend_ms,
                    );
                }
                if self.opening_progress >= 1.0 {
                    self.state = self.shown_state();
                }
            }
        }
        self.request_refresh(settings, input, now, max_age);
        self.request_standby(settings, input, now);
        self.output()
    }

    fn take_completion(&mut self) -> Option<(CaptureRequest, Completion)> {
        let completion = self.completion.take()?;
        Some((self.pending.take()?, completion))
    }

    fn shown_state(&self) -> State {
        if self.opening_progress < 1.0 {
            State::Entering
        } else if self.refresh_weight < 1.0 {
            State::Refreshing
        } else {
            State::Active
        }
    }

    fn arm(&mut self, now: u64) {
        self.drop_standby();
        self.clear(State::Arming, Reason::None);
        self.arming_ms = now;
        self.due_ms = Some(now);
        self.due_reason = CaptureReason::Initial;
    }

    fn request_refresh(&mut self, settings: Timing, input: Input, now: u64, max_age: f64) {
        if settings.snapshots
            && self.has_snapshot
            && self.state != State::Releasing
            && self.pending.is_none()
            && self.refresh_weight >= 1.0
            && self.due_ms.is_none()
        {
            let deadline = add(self.capture_ms, settings.refresh_ms);
            if now >= deadline {
                self.due_ms = Some(deadline);
                self.due_reason = CaptureReason::Periodic;
            } else if input.angle_since_capture_deg >= settings.early_refresh_deg
                && elapsed(now, self.capture_ms) >= f64::from(settings.capture_spacing_ms)
            {
                self.due_ms = Some(now);
                self.due_reason = CaptureReason::HeadTurn;
            }
        }
        let Some(due) = self.due_ms else { return };
        if self.pending.is_some() || self.state == State::Releasing || now < self.capture_retry_ms {
            return;
        }
        let arming = self.state == State::Arming;
        let mut wait_cap = f64::from(settings.max_wait_ms);
        if arming {
            wait_cap = wait_cap.min(
                (f64::from(settings.arming_timeout_ms)
                    - 2.0 * f64::from(settings.display_period_ms))
                .max(0.0),
            );
        }
        let dwell = self.dwell(now, settings);
        if dwell || elapsed(now, due) >= wait_cap {
            let deadline = if arming {
                add(self.arming_ms, settings.arming_timeout_ms)
            } else {
                add(self.capture_ms, max_age)
            };
            let slot = if self.has_snapshot {
                1 - self.new_slot
            } else {
                0
            };
            self.emit(now, deadline, slot, self.due_reason, !dwell);
        }
    }

    fn usable_standby(&self, settings: Timing, input: Input, now: u64) -> Option<Standby> {
        self.standby.filter(|standby| {
            settings.standby
                && settings.snapshots
                && now >= standby.capture_ms
                && elapsed(now, standby.capture_ms) <= settings.standby_max_age()
                && input.standby_angle_deg.is_finite()
                && input.standby_angle_deg < settings.early_refresh_deg
        })
    }

    fn start_from_standby(&mut self, standby: Standby) {
        let still = self.still_since_ms;
        self.clear(State::Entering, Reason::None);
        self.still_since_ms = still;
        self.has_snapshot = true;
        self.old_slot = standby.slot;
        self.new_slot = standby.slot;
        self.capture_ms = standby.capture_ms;
        self.drop_standby();
        self.activations += 1;
    }

    fn drop_standby(&mut self) {
        self.standby = None;
        self.standby_due_ms = None;
        self.standby_at_ms = None;
        if self
            .pending
            .is_some_and(|request| request.reason == CaptureReason::Standby)
        {
            self.pending = None;
            self.completion = None;
            self.emitted = None;
        }
    }

    fn request_standby(&mut self, settings: Timing, input: Input, now: u64) {
        self.standby_at_ms = None;
        if !settings.standby
            || !settings.snapshots
            || self.moving
            || self.latched_failure
            || self.pending.is_some()
        {
            return;
        }
        let idle = matches!(self.state, State::Idle | State::Suspended);
        let releasing = self.state == State::Releasing && self.refresh_weight >= 1.0;
        if if idle {
            now < self.retry_ms
        } else {
            !releasing
        } {
            return;
        }
        let mut due = self.last_stick_ms.map_or(0, |last| add(last, 150.0));
        if let Some(standby) = self.standby {
            let turned = !input.standby_angle_deg.is_finite()
                || input.standby_angle_deg >= 0.5 * settings.early_refresh_deg;
            let interval = if turned {
                250.0
            } else {
                settings.standby_period()
            };
            due = due.max(add(standby.capture_ms, interval));
        }
        self.standby_at_ms = Some(due);
        if now < due {
            self.standby_due_ms = None;
            return;
        }
        let since = *self.standby_due_ms.get_or_insert(now);
        let dwell = self.dwell(now, settings);
        if now < self.standby_retry_ms
            || (!dwell && elapsed(now, since) < f64::from(settings.max_wait_ms))
        {
            return;
        }
        let slot = if idle {
            self.standby.map_or(0, |standby| 1 - standby.slot)
        } else {
            1 - self.new_slot
        };
        self.emit(now, add(now, 250.0), slot, CaptureReason::Standby, !dwell);
    }

    fn dwell(&self, now: u64, settings: Timing) -> bool {
        self.still_since_ms
            .is_some_and(|since| elapsed(now, since) >= f64::from(settings.dwell_ms))
    }

    fn emit(&mut self, now: u64, deadline_ms: u64, slot: u32, reason: CaptureReason, forced: bool) {
        self.next_token += 1;
        let request = CaptureRequest {
            token: self.next_token,
            requested_at_ms: now,
            deadline_ms,
            slot,
            reason,
            forced,
        };
        self.pending = Some(request);
        self.emitted = Some(request);
    }

    fn clear(&mut self, state: State, reason: Reason) {
        self.state = state;
        self.reason = reason;
        self.opening_progress = 0.0;
        self.refresh_weight = 1.0;
        self.has_snapshot = false;
        self.moving = false;
        self.last_movement_ms = None;
        self.still_since_ms = None;
        self.due_ms = None;
        self.pending = None;
        self.completion = None;
        self.emitted = None;
        self.capture_retry_ms = 0;
        self.old_slot = 0;
        self.new_slot = 0;
    }

    fn pause(&mut self, reason: Reason, now: u64, retry_ms: f32) -> Output {
        let changed = self.state != State::Suspended || self.reason != reason;
        self.clear(State::Suspended, reason);
        self.drop_standby();
        if changed {
            self.retry_ms = add(now, retry_ms);
        }
        self.output()
    }

    fn render_failure(&mut self, now: u64, settings: Timing) {
        self.failures += 1;
        self.latched_failure = self.failures >= 3;
        self.pause(Reason::RenderFailure, now, settings.retry_ms);
    }

    fn output(&self) -> Output {
        Output {
            state: self.state,
            reason: self.reason,
            opening_progress: self.opening_progress,
            refresh_weight: self.refresh_weight,
            old_slot: self.old_slot,
            new_slot: self.new_slot,
            visible: self.has_snapshot && self.opening_progress > 0.0,
            needs_both_snapshots: self.has_snapshot
                && self.old_slot != self.new_slot
                && self.refresh_weight < 1.0,
            capture_in_flight: self.pending.is_some(),
            needs_reenable: self.latched_failure,
            capture_request: self.emitted,
            refresh_due_ms: self.due_ms,
            last_capture_ms: self.has_snapshot.then_some(self.capture_ms),
            consecutive_render_failures: self.failures,
            standby_slot: self.standby.map(|standby| standby.slot),
            standby_at_ms: self.standby_at_ms,
            activation: self.activations,
        }
    }
}

fn elapsed(now: u64, then: u64) -> f64 {
    now.saturating_sub(then) as f64
}
fn add(now: u64, ms: impl Into<f64>) -> u64 {
    now.saturating_add(ms.into().max(0.0).ceil() as u64)
}

fn advance(value: f32, dt: f64, duration: f32) -> f32 {
    if duration <= 0.0 {
        1.0
    } else {
        (value + (dt / f64::from(duration)) as f32).clamp(0.0, 1.0)
    }
}

pub fn observe_ring(output: Output, settings: Timing, angle_deg: f32, now: u64) -> bool {
    if !settings.snapshots {
        return false;
    }
    if output
        .standby_at_ms
        .is_some_and(|due| now.saturating_add(40) >= due)
    {
        return true;
    }
    let pending = output.capture_request.is_some() || output.capture_in_flight;
    match output.state {
        State::Idle | State::Suspended | State::Releasing => pending,
        State::Arming => true,
        _ => {
            pending
                || output.refresh_due_ms.is_some()
                || output.last_capture_ms.is_none_or(|last| {
                    now as f64 + 100.0 >= last as f64 + f64::from(settings.refresh_ms)
                })
                || !angle_deg.is_finite()
                || angle_deg >= settings.early_refresh_deg - 10.0
        }
    }
}

#[cfg(test)]
mod controller_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standby_starts_on_the_push_with_the_original_snapshot_time() {
        let mut controller = Controller::default();
        let timing = Timing {
            standby: true,
            fade_in_ms: 0.0,
            ..Timing::default()
        };
        let mut input = Input::default();
        assert!(
            controller
                .update(timing, input, 0)
                .capture_request
                .is_none()
        );
        let request = controller
            .update(timing, input, 50)
            .capture_request
            .unwrap();
        assert_eq!(request.reason, CaptureReason::Standby);
        assert!(controller.complete_capture(request.token, 51, 53, CaptureResult::Ready));
        let ready = controller.update(timing, input, 53);
        assert!(!ready.visible);
        assert_eq!(ready.standby_slot, Some(request.slot));
        input.hand_magnitude[0] = 0.9;
        let shown = controller.update(timing, input, 60);
        assert!(shown.visible);
        assert_eq!(shown.last_capture_ms, Some(51));
        assert_eq!(shown.activation, 1);
        assert!(shown.capture_request.is_none());
    }

    #[test]
    fn tracking_loss_cancels_an_inflight_capture_and_ignores_its_completion() {
        let mut controller = Controller::default();
        let timing = Timing {
            dwell_ms: 0.0,
            ..Timing::default()
        };
        let mut input = Input {
            hand_magnitude: [0.8, 0.0],
            ..Input::default()
        };
        let request = controller.update(timing, input, 0).capture_request.unwrap();
        input.tracking_valid = false;
        let paused = controller.update(timing, input, 5);
        assert_eq!(paused.reason, Reason::TrackingInvalid);
        assert!(!paused.visible && !paused.capture_in_flight);
        assert!(!controller.complete_capture(request.token, 2, 4, CaptureResult::Ready));
    }

    #[test]
    fn grid_uses_release_delay_then_fades_without_capture() {
        let mut controller = Controller::default();
        let timing = Timing {
            snapshots: false,
            fade_in_ms: 0.0,
            release_ms: 20.0,
            fade_out_ms: 100.0,
            ..Timing::default()
        };
        let mut input = Input {
            hand_magnitude: [0.8, 0.0],
            ..Input::default()
        };
        let shown = controller.update(timing, input, 0);
        assert!(shown.visible && !shown.capture_in_flight);
        input.hand_magnitude[0] = 0.0;
        assert_eq!(controller.update(timing, input, 19).state, State::Active);
        assert_eq!(controller.update(timing, input, 20).state, State::Releasing);
        let fading = controller.update(timing, input, 70);
        assert!(fading.visible && fading.opening_progress < 1.0);
        let released = controller.update(timing, input, 120);
        assert!(!released.visible);
        assert_eq!(released.state, State::Idle);
    }
}
