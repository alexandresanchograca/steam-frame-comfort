//! Controller cases ported from the C++ baseline's `rest_frame_tests.cpp`.
//! Each test keeps the baseline's case name and assertion messages.
use super::*;

fn near(a: f32, b: f32) -> bool {
    (a - b).abs() < 0.0001
}

/// The fade-out curve at `progress` through the fade: a soft start below the
/// knee, then a steady rate.
fn shape(progress: f32) -> f32 {
    const KNEE: f32 = 0.15;
    let k = 1.0 / (KNEE * (2.0 - KNEE));
    if progress < KNEE {
        k * progress * progress
    } else {
        k * KNEE * (2.0 * progress - KNEE)
    }
}

struct Fixture {
    controller: Controller,
    timing: Timing,
    input: Input,
}

impl Fixture {
    /// These cases pin exact times with an 80 ms release delay, a 200 ms
    /// fade-out and no refresh grace. The shipped defaults have their own
    /// cases in `fade_delay_and_curve` and `refresh_grace`.
    fn new() -> Self {
        Self {
            controller: Controller::default(),
            timing: Timing {
                release_ms: 80.0,
                fade_out_ms: 200.0,
                refresh_grace_ms: 0.0,
                ..Timing::default()
            },
            input: Input::default(),
        }
    }

    fn shipped() -> Self {
        Self {
            timing: Timing::default(),
            ..Self::new()
        }
    }

    /// Idle with a still head from t = 0, keeping standby snapshots.
    fn standby() -> Self {
        let mut fixture = Self::new();
        fixture.timing.standby = true;
        fixture.timing.fade_in_ms = 0.0;
        fixture.at(0);
        fixture
    }

    fn at(&mut self, now: u64) -> Output {
        self.controller.update(self.timing, self.input, now)
    }

    fn complete(&mut self, token: u64, capture_ms: u64, completed_ms: u64) -> bool {
        self.complete_as(token, capture_ms, completed_ms, CaptureResult::Ready)
    }

    fn complete_as(
        &mut self,
        token: u64,
        capture_ms: u64,
        completed_ms: u64,
        result: CaptureResult,
    ) -> bool {
        self.controller
            .complete_capture(token, capture_ms, completed_ms, result)
    }

    fn arm(&mut self, now: u64) -> CaptureRequest {
        self.input.hand_magnitude[0] = 1.0;
        assert_eq!(self.at(now).state, State::Arming, "movement must arm");
        self.at(now + 50)
            .capture_request
            .expect("quiet entry needs first capture after dwell")
    }

    fn ready(&mut self, request: CaptureRequest, now: u64) {
        assert!(
            self.complete(request.token, request.requested_at_ms, now),
            "accept matching capture"
        );
        assert_eq!(
            self.at(now).state,
            State::Entering,
            "first GPU-complete capture enters"
        );
    }

    fn active(&mut self, now: u64) {
        let request = self.arm(now);
        self.ready(request, now + 60);
        let out = self.at(now + 180);
        assert!(
            out.state == State::Active && near(out.opening_progress, 1.0),
            "entry completes at fade duration"
        );
    }

    /// A still head: a standby capture is requested after the 50 ms dwell.
    fn standby_request(&mut self, now: u64) -> CaptureRequest {
        let request = self
            .at(now)
            .capture_request
            .expect("standby capture requested");
        assert_eq!(request.reason, CaptureReason::Standby);
        request
    }

    fn complete_standby(&mut self, request: CaptureRequest, now: u64, result: CaptureResult) {
        assert!(
            self.complete_as(request.token, request.requested_at_ms, now, result),
            "standby completion accepted"
        );
    }

    /// A standby snapshot taken at 50 ms and ready by 60 ms.
    fn with_standby() -> Self {
        let mut fixture = Self::standby();
        let request = fixture.standby_request(50);
        fixture.complete_standby(request, 55, CaptureResult::Ready);
        fixture.at(60);
        fixture
    }
}

fn output(state: State) -> Output {
    Output {
        state,
        reason: Reason::None,
        opening_progress: 0.0,
        refresh_weight: 1.0,
        old_slot: 0,
        new_slot: 0,
        visible: false,
        needs_both_snapshots: false,
        capture_in_flight: false,
        needs_reenable: false,
        capture_request: None,
        refresh_due_ms: None,
        last_capture_ms: None,
        consecutive_render_failures: 0,
        standby_slot: None,
        standby_at_ms: None,
        activation: 0,
    }
}

fn request() -> CaptureRequest {
    CaptureRequest {
        token: 0,
        requested_at_ms: 0,
        deadline_ms: 0,
        slot: 0,
        reason: CaptureReason::Initial,
        forced: false,
    }
}

#[test]
fn settings_and_selection() {
    let mut timing = Timing {
        refresh_ms: 500.0,
        blend_ms: 500.0,
        deadzone: 2.0,
        pose_offset_ms: -100.0,
        ..Timing::default()
    };
    assert!(
        timing.sane()
            && timing.blend_ms == 250.0
            && timing.deadzone == 0.95
            && timing.pose_offset_ms == -50.0,
        "clamp ranges and cross-field limits"
    );
    timing.refresh_ms = f32::NAN;
    assert!(!timing.sane(), "reject nonfinite settings");

    let mut input = Input {
        hand_magnitude: [0.4, 0.8],
        ..Input::default()
    };
    assert_eq!(input.magnitude(Stick::Left), Some(0.4), "left selection");
    assert_eq!(input.magnitude(Stick::Right), Some(0.8), "right selection");
    assert_eq!(
        input.magnitude(Stick::Both),
        Some(0.8),
        "both means either by maximum"
    );
    input.hand_magnitude[1] = f32::NAN;
    assert!(
        input.magnitude(Stick::Left).is_some(),
        "unselected invalid hand ignored"
    );
    assert!(
        input.magnitude(Stick::Both).is_none(),
        "invalid selected hand unavailable"
    );
    input.input_available = false;
    assert!(
        input.magnitude(Stick::Left).is_none(),
        "disconnected channel unavailable"
    );
}

#[test]
fn dynamic_fov() {
    assert!(
        stick_force(0.15, 0.15) == 0.0 && stick_force(0.08, 0.15) == 0.0,
        "no narrowing in the dead zone"
    );
    assert_eq!(
        stick_force(1.4, 0.15),
        1.0,
        "diagonal deflection saturates at full force"
    );
    let force = stick_force(0.8, 0.15);
    assert!(
        near(force, 0.65 / 0.85),
        "force is linear above the dead zone"
    );
    let [width, height] = opening_degrees(50.0, 50.0, 30.0, force);
    assert!(
        near(width, 50.0 - 20.0 * force) && near(height, width),
        "narrowing follows force and preserves a circle"
    );
    let [width, height] = opening_degrees(50.0, 40.0, 30.0, 1.0);
    assert!(
        near(width, 30.0) && near(height, 24.0),
        "height shrinks in proportion to width"
    );
    assert!(
        near(smooth_force(0.0, 1.0, 100.0), 0.5) && smooth_force(0.5, 1.0, 100.0) == 1.0,
        "200 ms to reach full force"
    );
    assert!(
        near(smooth_force(1.0, 0.0, 150.0), 0.5) && smooth_force(0.5, 0.0, 150.0) == 0.0,
        "300 ms to return to zero force"
    );
    assert!(
        smooth_force(0.0, 0.2, 100.0) == 0.2 && smooth_force(0.5, 0.49, 30.0) == 0.49,
        "a small force change never overshoots"
    );
}

#[test]
fn activation_and_release() {
    let mut f = Fixture::new();
    f.input.hand_magnitude[1] = 1.0;
    assert_eq!(
        f.at(0).state,
        State::Idle,
        "turning on unselected right stick cannot arm"
    );
    f.input.hand_magnitude[0] = 0.16;
    assert_eq!(f.at(1).state, State::Arming, "above activation dead zone");
    f.input.hand_magnitude[0] = 0.14;
    assert_eq!(
        f.at(20).state,
        State::Arming,
        "hysteresis retains held deflection"
    );
    f.input.hand_magnitude[0] = 0.11;
    assert_eq!(
        f.at(30).state,
        State::Idle,
        "release during arming cancels immediately"
    );
    f.active(100);
    // A held stick requires no advancing input timestamp or additional event.
    assert_eq!(
        f.at(300).state,
        State::Active,
        "held input stays active while channel alive"
    );
    f.input.hand_magnitude[0] = 0.0;
    assert_eq!(
        f.at(350).state,
        State::Active,
        "release hold preserves active frame"
    );
    assert_eq!(
        f.at(380).state,
        State::Releasing,
        "release delay starts expansion"
    );
    assert!(
        near(f.at(480).opening_progress, 1.0 - shape(0.5)),
        "halfway along the fade-out curve"
    );
    let out = f.at(580);
    assert!(
        out.state == State::Idle && !out.visible && near(out.opening_progress, 0.0),
        "release removes effect exactly"
    );
    assert!(
        !out.capture_in_flight && out.capture_request.is_none(),
        "idle issues no GPU capture"
    );
}

#[test]
fn capture_tokens_and_first_visibility() {
    let mut f = Fixture::new();
    let first = f.arm(0);
    assert!(
        f.at(55).capture_request.is_none() && f.at(56).capture_in_flight,
        "emit request once, retain pending state"
    );
    assert!(
        !f.complete(first.token + 1, 50, 60),
        "wrong completion token ignored"
    );
    assert!(!f.at(59).visible, "GPU-incomplete image is never visible");
    f.ready(first, 60);
    assert!(
        !f.complete(first.token, 50, 61),
        "duplicate completion ignored"
    );
    assert!(
        near(f.at(120).opening_progress, 0.5),
        "entry starts only when completed capture becomes available"
    );
    assert!(f.at(180).visible, "entry visible");
    f.input.dashboard = true;
    let out = f.at(181);
    assert!(
        !out.visible && out.state == State::Suspended && out.reason == Reason::Dashboard,
        "dashboard hides immediately"
    );
    f.input.dashboard = false;
    assert!(
        f.at(200).capture_request.is_none(),
        "recovery retry is bounded"
    );
    f.timing.max_wait_ms = 0.0;
    let out = f.at(1181);
    assert!(
        out.state == State::Arming && !out.visible,
        "recovery requires fresh first capture"
    );
    let recovery = out
        .capture_request
        .expect("recovery requires fresh first capture");
    assert!(
        recovery.token > first.token,
        "token not reused after suspension"
    );
    assert!(
        !f.complete(first.token, 50, 1182),
        "stale completion cannot restore old frame"
    );
}

#[test]
fn stillness_and_caps() {
    let mut f = Fixture::new();
    f.input.hand_magnitude[0] = 1.0;
    f.input.angular_speed_deg_per_sec = 30.0;
    f.at(0);
    assert!(
        f.at(199).capture_request.is_none(),
        "moving head waits before cap"
    );
    let forced = f
        .at(200)
        .capture_request
        .expect("moving head forced at wait cap");
    assert!(forced.forced, "moving head forced at wait cap");
    assert_eq!(forced.deadline_ms, 300, "entry deadline bounded");
    assert_eq!(
        f.at(300).reason,
        Reason::CaptureTimeout,
        "hanging capture times out without blocking"
    );

    let mut delayed = Fixture::new();
    delayed.input.hand_magnitude[0] = 1.0;
    delayed.input.image_angular_speed_deg_per_sec = 45.0;
    delayed.at(0);
    assert!(
        delayed.at(50).capture_request.is_none(),
        "current stillness does not imply delayed frame stillness"
    );
    delayed.input.image_angular_speed_deg_per_sec = 0.0;
    assert!(
        delayed.at(100).capture_request.is_none(),
        "stillness begins dwell"
    );
    delayed.input.angular_speed_deg_per_sec = 40.0;
    delayed.at(125);
    delayed.input.angular_speed_deg_per_sec = 0.0;
    delayed.at(140);
    assert!(
        delayed.at(189).capture_request.is_none(),
        "motion resets dwell"
    );
    let still = delayed
        .at(190)
        .capture_request
        .expect("sustained stillness preferred");
    assert!(!still.forced, "sustained stillness preferred");

    let mut long_wait = Fixture::new();
    long_wait.timing.max_wait_ms = 500.0;
    long_wait.input.hand_magnitude[0] = 1.0;
    long_wait.input.angular_speed_deg_per_sec = 100.0;
    long_wait.at(0);
    assert!(
        long_wait.at(278).capture_request.is_some(),
        "entry wait reserves completion tolerance before timeout"
    );
}

#[test]
fn periodic_refresh_and_blend() {
    let mut f = Fixture::new();
    f.active(0);
    assert!(
        f.at(1049).capture_request.is_none(),
        "periodic deadline is capture time plus interval"
    );
    let request = f
        .at(1050)
        .capture_request
        .expect("periodic capture selects spare snapshot");
    assert!(
        request.reason == CaptureReason::Periodic && request.slot == 1,
        "periodic capture selects spare snapshot"
    );
    assert!(f.complete(request.token, 1050, 1060), "complete refresh");
    let out = f.at(1060);
    assert!(
        out.state == State::Refreshing
            && out.old_slot == 0
            && out.new_slot == 1
            && out.refresh_weight == 0.0,
        "refresh retains old and new snapshots"
    );
    let out = f.at(1160);
    assert!(
        near(out.refresh_weight, 0.5)
            && near(out.opening_progress, 1.0)
            && out.needs_both_snapshots,
        "midpoint blends colour while opening stays constant"
    );
    assert!(out.capture_request.is_none(), "no overlapping refresh");
    let out = f.at(1260);
    assert!(
        out.state == State::Active && !out.needs_both_snapshots && out.refresh_weight == 1.0,
        "old snapshot can retire after fade"
    );
    assert!(
        f.at(2049).capture_request.is_none(),
        "next deadline excludes end-of-crossfade drift"
    );
    let next = f
        .at(2050)
        .capture_request
        .expect("alternate snapshot slot after fade complete");
    assert_eq!(next.slot, 0, "alternate snapshot slot after fade complete");
}

#[test]
fn refresh_during_long_entry() {
    let mut f = Fixture::new();
    f.timing.fade_in_ms = 3000.0;
    let first = f.arm(0);
    f.ready(first, 60);
    assert_eq!(f.at(500).state, State::Entering, "long entry continues");
    let out = f.at(1050);
    assert_eq!(
        out.state,
        State::Entering,
        "refresh can be due before entry finishes"
    );
    let refresh = out
        .capture_request
        .expect("refresh can be due before entry finishes");
    let before = out.opening_progress;
    f.complete(refresh.token, 1050, 1060);
    let out = f.at(1060);
    assert!(
        out.state == State::Entering && out.opening_progress > before,
        "refresh preserves and advances entry progress"
    );
    assert!(
        out.refresh_weight == 0.0 && out.needs_both_snapshots,
        "refresh blend starts independently from entry"
    );
    let out = f.at(1260);
    assert!(
        out.state == State::Entering && out.refresh_weight == 1.0 && out.visible,
        "entry survives original snapshot-age deadline with fresh capture"
    );
    assert!(
        out.opening_progress > 0.39 && out.opening_progress < 0.41,
        "entry timing does not reset at capture"
    );
}

#[test]
fn early_refresh_throttling() {
    let mut f = Fixture::new();
    f.active(0);
    f.input.angle_since_capture_deg = 60.0;
    assert!(
        f.at(299).capture_request.is_none(),
        "early refresh minimum spacing"
    );
    let request = f
        .at(300)
        .capture_request
        .expect("physical head angle can trigger early capture");
    assert_eq!(
        request.reason,
        CaptureReason::HeadTurn,
        "physical head angle can trigger early capture"
    );
    assert!(
        f.complete(request.token, 300, 310),
        "early refresh completion"
    );
    f.at(310);
    assert!(
        f.at(500).capture_request.is_none(),
        "no request while blending"
    );
    f.at(510);
    assert!(
        f.at(549).capture_request.is_none(),
        "throttle remains capture-relative after blend"
    );
    let again = f
        .at(550)
        .capture_request
        .expect("continued turning is throttled but may refresh again");
    assert_eq!(again.reason, CaptureReason::HeadTurn);
}

#[test]
fn release_mid_refresh_and_reentry() {
    let mut f = Fixture::new();
    f.active(0);
    let request = f.at(1050).capture_request.expect("refresh request");
    f.complete(request.token, 1050, 1060);
    f.at(1060);
    f.input.hand_magnitude[0] = 0.0;
    f.at(1110);
    let out = f.at(1140);
    assert_eq!(
        out.state,
        State::Releasing,
        "release during refresh expands opening"
    );
    let frozen = out.refresh_weight;
    let out = f.at(1240);
    assert!(
        near(out.refresh_weight, frozen) && out.needs_both_snapshots,
        "release freezes snapshot blend and retains both images"
    );
    let halfway = 1.0 - shape(0.5);
    assert!(
        near(out.opening_progress, halfway),
        "periphery fades during frozen blend"
    );
    f.input.hand_magnitude[0] = 1.0;
    let out = f.at(1250);
    assert!(
        near(out.opening_progress, halfway),
        "re-entry reverses without jump"
    );
    let out = f.at(1310);
    assert!(
        near(out.opening_progress, 1.0),
        "re-entry resumes from previous opening extent"
    );
    assert!(out.visible, "re-entry remains visible with valid snapshot");
}

#[test]
fn age_and_invalid_pair() {
    let mut f = Fixture::new();
    f.active(0);
    let request = f.at(1050).capture_request.expect("refresh request");
    f.complete_as(request.token, 1050, 1060, CaptureResult::InvalidPair);
    let out = f.at(1060);
    assert!(
        out.reason == Reason::CaptureInvalid && out.visible && out.capture_request.is_none(),
        "invalid pair cannot replace valid old snapshot"
    );
    assert!(
        f.at(1061).capture_request.is_none(),
        "invalid-pair retry bounded by display period"
    );
    assert!(
        f.at(1072).capture_request.is_some(),
        "invalid pair may retry within original deadline"
    );
    let out = f.at(1273);
    assert!(
        !out.visible && out.reason == Reason::SnapshotExpired,
        "cannot keep obsolete snapshot indefinitely"
    );
    assert!(
        !f.complete(request.token, 1050, 1273),
        "old tokens invalidated after age timeout"
    );

    let mut release = Fixture::new();
    release.active(0);
    release.at(1190);
    release.input.hand_magnitude[0] = 0.0;
    release.at(1269);
    assert_eq!(
        release.at(1273).state,
        State::Releasing,
        "age deadline must not replace requested clean release with cut"
    );
    assert_eq!(
        release.at(1473).state,
        State::Idle,
        "configured release can finish beyond snapshot refresh-age deadline"
    );
}

#[test]
fn generations_tracking_and_failures() {
    let mut f = Fixture::new();
    f.active(0);
    f.input.source_generation = 2;
    let out = f.at(190);
    assert!(
        out.reason == Reason::SourceChanged && !out.visible,
        "source generation change invalidates capture"
    );
    f.timing.max_wait_ms = 0.0;
    let request = f.at(191).capture_request.expect("new generation arms anew");
    assert_eq!(request.slot, 0, "new generation arms anew");
    f.input.tracking_epoch = 1;
    assert_eq!(
        f.at(192).reason,
        Reason::TrackingChanged,
        "recenter invalidates pending capture"
    );
    assert!(
        !f.complete(request.token, 191, 193),
        "pre-recenter GPU completion cannot become visible"
    );

    let mut errors = Fixture::new();
    errors.timing.max_wait_ms = 0.0;
    errors.input.hand_magnitude[0] = 1.0;
    let mut out = errors.at(0);
    for i in 0..3_u32 {
        let t = u64::from(i) * 1100;
        if i > 0 {
            out = errors.at(t);
        }
        let request = out.capture_request.expect("retry after a render failure");
        errors.complete_as(request.token, t, t + 1, CaptureResult::RenderFailure);
        out = errors.at(t + 1);
        assert!(
            !out.visible
                && out.reason == Reason::RenderFailure
                && out.consecutive_render_failures == i + 1,
            "render failure hides with bounded retries"
        );
    }
    assert!(
        out.needs_reenable,
        "persistent render failures require explicit re-enable"
    );
    assert!(
        errors.at(10000).capture_request.is_none(),
        "latched failure cannot retry forever"
    );
    errors.input.enabled = false;
    let out = errors.at(10001);
    assert!(
        !out.needs_reenable && out.state == State::Idle,
        "disable clears failure latch"
    );
    errors.input.enabled = true;
    assert!(
        errors.at(10002).capture_request.is_some(),
        "explicit enable permits retry"
    );
}

#[test]
fn geometry_settings_and_invalid_completions() {
    let mut f = Fixture::new();
    let request = f.arm(0);
    f.timing.pose_offset_ms = 30.0;
    let out = f.at(51);
    assert!(
        out.reason == Reason::SettingsChanged && !out.capture_in_flight,
        "calibration change cancels old capture"
    );
    assert!(
        !f.complete(request.token, 50, 52),
        "old calibration cannot restore pending capture"
    );

    let mut shell = Fixture::new();
    shell.active(0);
    shell.timing.shell_half_size = 5.0;
    assert_eq!(
        shell.at(190).reason,
        Reason::SettingsChanged,
        "shell resize invalidates active capture"
    );

    let mut invalid = Fixture::new();
    let bad = invalid.arm(0);
    invalid.complete(bad.token, 49, 60);
    let out = invalid.at(60);
    assert!(
        out.reason == Reason::CaptureInvalid && !out.visible,
        "copy timestamp preceding request rejected"
    );

    let mut future = Fixture::new();
    let late = future.arm(0);
    future.complete(late.token, 50, 500);
    assert_eq!(
        future.at(500).reason,
        Reason::CaptureInvalid,
        "late GPU completion cannot bypass acquisition deadline"
    );

    let mut delivery = Fixture::new();
    delivery.timing.max_wait_ms = 0.0;
    delivery.timing.dwell_ms = 0.0;
    delivery.timing.fade_in_ms = 0.0;
    delivery.input.hand_magnitude[0] = 1.0;
    let queued = delivery.at(0).capture_request.expect("immediate request");
    delivery.complete(queued.token, 0, 1);
    let out = delivery.at(600);
    assert!(
        out.reason == Reason::CaptureInvalid && !out.visible,
        "GPU completion on time cannot bypass late visible-delivery deadline"
    );

    let mut refresh_delivery = Fixture::new();
    refresh_delivery.active(0);
    let refresh = refresh_delivery
        .at(1050)
        .capture_request
        .expect("refresh request");
    refresh_delivery.complete(refresh.token, 1050, 1060);
    let out = refresh_delivery.at(1300);
    assert!(
        out.reason == Reason::CaptureInvalid && !out.visible,
        "late refresh delivery cannot reset old snapshot age"
    );

    let mut cancel = Fixture::new();
    let obsolete = cancel.arm(0);
    cancel.controller.reset();
    assert!(
        !cancel.complete(obsolete.token, 50, 60),
        "reset cancels asynchronous completion"
    );
    let replacement = cancel.arm(100);
    assert!(
        replacement.token > obsolete.token,
        "reset never recycles token identity"
    );
}

#[test]
fn zero_durations_pause_and_time() {
    let mut f = Fixture::new();
    f.timing.max_wait_ms = 0.0;
    f.timing.dwell_ms = 0.0;
    f.timing.fade_in_ms = 0.0;
    f.timing.fade_out_ms = 0.0;
    f.timing.release_ms = 0.0;
    f.timing.blend_ms = 0.0;
    f.input.hand_magnitude[0] = 1.0;
    let request = f.at(0).capture_request.expect("immediate request");
    f.complete(request.token, 0, 1);
    let out = f.at(1);
    assert!(
        out.state == State::Active && out.visible,
        "zero fade enters immediately after capture"
    );
    f.input.hand_magnitude[0] = 0.0;
    let out = f.at(2);
    assert!(
        out.state == State::Idle && !out.visible,
        "zero release/fade exits immediately"
    );

    let mut time = Fixture::new();
    time.active(100);
    let out = time.at(50);
    assert!(
        out.reason == Reason::ClockReset && !out.visible,
        "backwards clock invalidates all timing"
    );

    let mut invalid = Fixture::new();
    invalid.active(0);
    invalid.timing.strength = f32::INFINITY;
    let out = invalid.at(200);
    assert!(
        out.reason == Reason::InvalidSettings && !out.visible,
        "invalid settings never leak into visible state"
    );

    let mut stale = Fixture::new();
    stale.active(0);
    stale.input.input_available = false;
    assert!(
        !stale.at(190).visible,
        "stale channel bypasses release fade and hides immediately"
    );
}

/// The shipped fades: 120 ms in; after the stick is released the periphery
/// stays fully shown for 500 ms, then clears over 300 ms, slowly at first and
/// then at a steady rate.
#[test]
fn fade_delay_and_curve() {
    let defaults = Timing::default();
    assert!(
        defaults.fade_in_ms == 120.0
            && defaults.release_ms == 500.0
            && defaults.fade_out_ms == 300.0
            && defaults.refresh_grace_ms == 250.0,
        "default fade and grace timings"
    );
    assert!(
        near(fade_out(1.0, 1.0, 0.0, 300.0), 1.0) && fade_out(1.0, 1.0, 300.0, 300.0) == 0.0,
        "curve covers the whole fade"
    );
    assert!(
        1.0 - fade_out(1.0, 1.0, 0.075 * 300.0, 300.0) < 0.0375,
        "soft start: under half the steady rate at first"
    );
    let mut previous = 1.0;
    for n in 1..100 {
        let progress = n as f32 / 100.0;
        let level = fade_out(1.0, 1.0, f64::from(progress) * 300.0, 300.0);
        assert!(
            level < previous && near(level, 1.0 - shape(progress)),
            "curve falls steadily at {progress}"
        );
        // Continuing from a shown level lands where a single step would.
        assert!(
            near(
                fade_out(level, 1.0, 30.0, 300.0),
                fade_out(1.0, 1.0, f64::from(progress) * 300.0 + 30.0, 300.0)
            ),
            "curve inverts exactly at {progress}"
        );
        previous = level;
    }

    let mut f = Fixture::shipped();
    f.active(0); // held until t = 180, fully shown
    f.input.hand_magnitude[0] = 0.0;
    let out = f.at(400);
    assert!(
        out.state == State::Active && near(out.opening_progress, 1.0),
        "delay: still fully shown after release"
    );
    let out = f.at(679);
    assert!(
        out.state == State::Active && near(out.opening_progress, 1.0),
        "delay lasts 500 ms from the last movement"
    );
    assert_eq!(
        f.at(680).state,
        State::Releasing,
        "fade-out starts when the delay ends"
    );
    let out = f.at(725); // 45 ms = 15% of the fade
    assert!(
        near(out.opening_progress, 1.0 - shape(0.15)) && out.opening_progress > 0.9,
        "soft start of the fade-out"
    );
    let out = f.at(830); // halfway in time
    assert!(
        near(out.opening_progress, 1.0 - shape(0.5)),
        "steady part of the fade-out"
    );
    f.input.hand_magnitude[0] = 1.0; // pushed again mid-fade: no jump, then linear fade-in
    let level = out.opening_progress;
    assert!(
        near(f.at(831).opening_progress, level),
        "reversal starts from the shown level"
    );
    assert!(
        near(f.at(843).opening_progress, level + 0.1),
        "fade-in is linear over 120 ms"
    );
    f.input.hand_magnitude[0] = 0.0;
    f.at(900); // last movement was seen at t = 843
    assert!(
        f.at(1342).state != State::Releasing && f.at(1343).state == State::Releasing,
        "the delay restarts from the last movement"
    );
    assert_eq!(f.at(1643).state, State::Idle, "fade-out ends in normal VR");

    // A stick moved again during the delay keeps the periphery up without any fade.
    let mut hold = Fixture::shipped();
    hold.active(0);
    hold.input.hand_magnitude[0] = 0.0;
    hold.at(500);
    hold.input.hand_magnitude[0] = 1.0;
    let out = hold.at(600);
    assert!(
        out.state == State::Active && near(out.opening_progress, 1.0),
        "stop-start movement never flickers"
    );
}

/// A refresh that comes due while the head is moving is requested only after
/// the 200 ms stillness wait. It then needs room to retry: one rejected or
/// slow copy must not hide the periphery (8 dropouts in 3 minutes on the Frame).
#[test]
fn refresh_grace() {
    let mut f = Fixture::shipped();
    f.active(0); // snapshot taken at t = 50
    f.input.angular_speed_deg_per_sec = 40.0; // head turning: never still
    assert!(
        f.at(1249).capture_request.is_none(),
        "moving head waits for the stillness cap"
    );
    let request = f
        .at(1250)
        .capture_request
        .expect("forced refresh after the wait");
    assert!(request.forced, "forced refresh after the wait");
    assert!(
        request.deadline_ms >= 1500,
        "the request can be retried within the grace period"
    );
    f.complete_as(request.token, 1250, 1262, CaptureResult::InvalidPair);
    let out = f.at(1262);
    assert!(
        out.visible && out.state != State::Suspended && out.reason == Reason::CaptureInvalid,
        "a rejected copy keeps the snapshot that is shown"
    );
    let out = f.at(1274);
    assert!(out.visible, "retried a display period later");
    let request = out.capture_request.expect("retried a display period later");
    f.complete(request.token, 1274, 1290);
    let out = f.at(1290);
    assert!(
        out.visible && out.state == State::Refreshing && out.reason == Reason::None,
        "the retry replaces the snapshot without a dropout"
    );

    // The grace is bounded: with no good copy at all the snapshot still expires.
    let mut stuck = Fixture::shipped();
    stuck.active(0);
    stuck.input.angular_speed_deg_per_sec = 40.0;
    assert!(stuck.at(1250).capture_request.is_some(), "request");
    assert!(stuck.at(1522).visible, "shown through the grace period");
    let out = stuck.at(1523);
    assert!(
        !out.visible && out.reason == Reason::SnapshotExpired,
        "an old snapshot is not kept indefinitely"
    );

    // Without grace the same rejected copy hid the periphery.
    let mut old = Fixture::shipped();
    old.timing.refresh_grace_ms = 0.0;
    old.active(0);
    old.input.angular_speed_deg_per_sec = 40.0;
    let request = old.at(1250).capture_request.expect("request");
    old.complete_as(request.token, 1250, 1262, CaptureResult::InvalidPair);
    old.at(1262);
    assert!(
        !old.at(1274).visible,
        "fixture reproduces the dropout without grace"
    );
}

/// Plain and Grid: no snapshot, so a press opens at once and never asks for a
/// capture, whatever the game source does; tracking resets and release still
/// apply.
#[test]
fn without_snapshots() {
    let mut f = Fixture::new();
    f.timing.snapshots = false;
    f.input.source_fresh = false; // no game picture at all
    f.input.hand_magnitude[0] = 1.0;
    let out = f.at(0);
    assert!(
        out.state == State::Entering && out.capture_request.is_none() && !out.visible,
        "press enters without a capture"
    );
    let out = f.at(60);
    assert!(
        out.visible && near(out.opening_progress, 0.5) && out.capture_request.is_none(),
        "opening closes in without a snapshot"
    );
    let out = f.at(200);
    assert!(
        out.state == State::Active && near(out.opening_progress, 1.0),
        "settles like the game periphery"
    );
    f.input.source_generation = 99;
    for t in (300..=5000).step_by(100) {
        let out = f.at(t);
        assert!(
            out.state == State::Active
                && out.visible
                && out.capture_request.is_none()
                && out.refresh_due_ms.is_none(),
            "no refresh, expiry or source dependence at {t}"
        );
    }
    f.input.tracking_epoch = 1;
    assert_eq!(
        f.at(5100).reason,
        Reason::TrackingChanged,
        "a tracking reset still stops it"
    );
    assert_eq!(
        f.at(5200).state,
        State::Entering,
        "re-enters at once while held"
    );
    f.at(5300);
    f.input.hand_magnitude[0] = 0.0;
    assert_eq!(
        f.at(5400).state,
        State::Releasing,
        "release after the hold time"
    );
    assert_eq!(
        f.at(5700).state,
        State::Idle,
        "release fades back to normal VR"
    );
    f.input.hand_magnitude[0] = 1.0;
    f.at(5800);
    assert_eq!(f.at(6000).state, State::Active, "active again");
    f.input.source_fresh = true; // (without a source Game would suspend as source-unavailable)
    f.timing.snapshots = true; // switching to Game while shown restarts cleanly
    let out = f.at(6100);
    assert!(
        out.state == State::Suspended && out.reason == Reason::SettingsChanged && !out.visible,
        "periphery switch while shown"
    );
}

/// Game mode fingerprints a local ring only shortly before a capture.
#[test]
fn ring_observation_window() {
    let timing = Timing::default();
    assert!(
        timing.snapshots && timing.refresh_ms == 1000.0 && timing.early_refresh_deg == 30.0,
        "fixture timings"
    );
    for state in [State::Idle, State::Suspended, State::Releasing] {
        let mut out = output(state);
        out.refresh_due_ms = Some(0);
        assert!(
            !observe_ring(out, timing, 90.0, 0),
            "no observation while idle, suspended or releasing"
        );
    }
    let arming = output(State::Arming);
    assert!(
        observe_ring(arming, timing, 0.0, 0),
        "arming observes from its first tick"
    );
    let plain = Timing {
        snapshots: false,
        ..timing
    };
    assert!(
        !observe_ring(arming, plain, 0.0, 0),
        "Plain and Grid never observe a ring"
    );

    let mut out = output(State::Active);
    out.last_capture_ms = Some(10000);
    assert!(
        !observe_ring(out, timing, 0.0, 10500),
        "no observation mid-interval"
    );
    assert!(
        !observe_ring(out, timing, 0.0, 10899),
        "no observation before the lead"
    );
    assert!(
        observe_ring(out, timing, 0.0, 10900),
        "observation starts 100 ms before the periodic refresh"
    );
    assert!(
        !observe_ring(out, timing, 19.9, 10500),
        "small head turns do not observe"
    );
    assert!(
        observe_ring(out, timing, 20.0, 10500),
        "a turn within 10 degrees of an early refresh observes"
    );
    assert!(
        observe_ring(out, timing, f32::NAN, 10500),
        "unknown turn angle observes"
    );
    for state in [State::Entering, State::Refreshing] {
        out.state = state;
        assert!(
            !observe_ring(out, timing, 0.0, 10500),
            "entering and refreshing wait for the lead too"
        );
    }
    out.state = State::Active;
    let mut due = out;
    due.refresh_due_ms = Some(11000);
    assert!(
        observe_ring(due, timing, 0.0, 10500),
        "a due refresh observes until it is captured"
    );
    let mut flight = out;
    flight.capture_in_flight = true;
    assert!(
        observe_ring(flight, timing, 0.0, 10500),
        "a capture in flight observes"
    );
    let mut requested = out;
    requested.capture_request = Some(request());
    assert!(
        observe_ring(requested, timing, 0.0, 10500),
        "a capture request observes"
    );
    let mut unknown = out;
    unknown.last_capture_ms = None;
    assert!(
        observe_ring(unknown, timing, 0.0, 10500),
        "no capture time observes"
    );

    // The controller reports the shown snapshot's capture time.
    let mut f = Fixture::new();
    let first = f.arm(0);
    assert!(
        f.at(55).last_capture_ms.is_none(),
        "no capture time before the first capture"
    );
    f.ready(first, 60);
    assert_eq!(
        f.at(70).last_capture_ms,
        Some(first.requested_at_ms),
        "capture time of the shown snapshot"
    );

    // Standby captures are observed shortly before they are due, and while
    // requested or in flight.
    let mut idle = output(State::Idle);
    idle.standby_at_ms = Some(1000);
    assert!(
        !observe_ring(idle, timing, 0.0, 959),
        "no standby observation before its lead"
    );
    assert!(
        observe_ring(idle, timing, 0.0, 960),
        "standby observed 40 ms before it is due"
    );
    assert!(
        !observe_ring(idle, plain, 0.0, 960),
        "Plain and Grid keep no standby"
    );
    idle.standby_at_ms = None;
    idle.capture_in_flight = true;
    assert!(
        observe_ring(idle, timing, 0.0, 0),
        "a standby capture in flight observes"
    );
    idle.capture_in_flight = false;
    idle.capture_request = Some(request());
    idle.state = State::Releasing;
    assert!(
        observe_ring(idle, timing, 0.0, 0),
        "a standby request while releasing observes"
    );
}

/// Standby: with the sticks at rest Game mode keeps a recent snapshot, so a
/// push shows the periphery at once. Fixture: 1000 ms refresh, 250 ms capture
/// spacing, so a standby may be 750 ms old and is replaced every 375 ms.
#[test]
fn standby_snapshots() {
    let defaults = Timing::default();
    assert!(
        defaults.standby_max_age() == 750.0 && defaults.standby_period() == 375.0,
        "standby age and period from the refresh interval and spacing"
    );
    let fast = Timing {
        refresh_ms: 500.0,
        capture_spacing_ms: 400.0,
        ..defaults
    };
    assert!(
        fast.standby_max_age() == 250.0 && fast.standby_period() == 250.0,
        "short intervals keep half an interval"
    );
    assert!(
        standby_stick_threshold(0.15) == 0.1 && standby_stick_threshold(0.0) == 0.05,
        "stick threshold between 0.05 and 0.1"
    );

    let mut f = Fixture::standby();
    let out = f.at(10);
    assert!(
        out.state == State::Idle && out.capture_request.is_none() && out.standby_at_ms == Some(0),
        "standby due at once, waiting for the still dwell"
    );
    assert!(
        f.at(49).capture_request.is_none(),
        "dwell before a standby capture"
    );
    let first = f.standby_request(50);
    assert!(
        first.slot == 0 && !first.forced && first.deadline_ms == 300,
        "standby into slot 0 after a still dwell"
    );
    assert!(
        f.at(52).capture_in_flight && f.at(53).capture_request.is_none(),
        "one request, then in flight"
    );
    f.complete_standby(first, 55, CaptureResult::Ready);
    let out = f.at(60);
    assert!(
        out.state == State::Idle && !out.visible && out.standby_slot == Some(0),
        "standby ready, nothing shown"
    );
    assert_eq!(
        out.standby_at_ms,
        Some(425),
        "next standby due after 375 ms"
    );

    // The push shows it in the same update, as a first snapshot.
    f.input.hand_magnitude[0] = 1.0;
    let out = f.at(100);
    assert!(
        out.state == State::Active
            && out.visible
            && out.new_slot == 0
            && out.old_slot == 0
            && out.capture_request.is_none(),
        "push shows the standby snapshot at once"
    );
    assert!(
        out.last_capture_ms == Some(50) && out.activation == 1 && out.standby_slot.is_none(),
        "standby becomes the shown first snapshot"
    );
    assert!(
        f.at(1049).capture_request.is_none(),
        "first refresh keeps the snapshot's own interval"
    );
    let refresh = f
        .at(1050)
        .capture_request
        .expect("periodic refresh into the other slot");
    assert!(
        refresh.reason == CaptureReason::Periodic && refresh.slot == 1,
        "periodic refresh into the other slot"
    );

    // Without standby the same push still arms and captures.
    let mut off = Fixture::new();
    off.timing.fade_in_ms = 0.0;
    assert!(
        off.at(50).capture_request.is_none() && off.at(100).standby_at_ms.is_none(),
        "no standby unless enabled"
    );
    off.input.hand_magnitude[0] = 1.0;
    assert_eq!(
        off.at(150).state,
        State::Arming,
        "push arms without standby"
    );

    // Replaced every period into the other slot; the old one stays usable meanwhile.
    let mut cycle = Fixture::with_standby();
    assert!(
        cycle.at(424).capture_request.is_none(),
        "standby kept for its period"
    );
    let request = cycle.standby_request(425);
    assert_eq!(request.slot, 1, "replacement goes to the other slot");
    assert_eq!(
        cycle.at(426).standby_slot,
        Some(0),
        "previous standby kept while its replacement is copied"
    );
    cycle.complete_standby(request, 430, CaptureResult::Ready);
    assert_eq!(
        cycle.at(435).standby_slot,
        Some(1),
        "replacement becomes the standby"
    );
    assert_eq!(cycle.standby_request(800).slot, 0, "slots alternate");

    // Too old: the push arms as before.
    let mut old = Fixture::with_standby();
    old.input.hand_magnitude[0] = 1.0;
    let out = old.at(801);
    assert!(
        out.state == State::Arming && !out.visible,
        "a standby older than 750 ms is not used"
    );

    // A failed copy retries a display period later; a render failure suspends.
    let mut retry = Fixture::standby();
    let request = retry.standby_request(50);
    retry.complete_standby(request, 55, CaptureResult::InvalidPair);
    let out = retry.at(60);
    assert!(
        out.standby_slot.is_none() && out.capture_request.is_none() && out.reason == Reason::None,
        "a rejected standby copy is quiet"
    );
    assert!(
        retry.at(65).capture_request.is_none() && retry.at(72).capture_request.is_some(),
        "retried at display cadence"
    );
    let mut broken = Fixture::standby();
    let request = broken.standby_request(50);
    broken.complete_standby(request, 55, CaptureResult::RenderFailure);
    let out = broken.at(60);
    assert!(
        out.reason == Reason::RenderFailure && out.consecutive_render_failures == 1,
        "a standby render failure counts as one"
    );
    assert!(
        broken.at(1060).capture_request.is_none(),
        "retry time reached; dwell starts"
    );
    let request = broken.standby_request(1110);
    broken.complete_standby(request, 1115, CaptureResult::Ready);
    let out = broken.at(1120);
    assert!(
        out.standby_slot.is_some() && out.consecutive_render_failures == 0,
        "a good standby copy resets the failure count"
    );

    // Late delivery cannot become a standby.
    let mut late = Fixture::standby();
    let request = late.standby_request(50);
    late.complete_standby(request, 55, CaptureResult::Ready);
    assert!(
        late.at(301).standby_slot.is_none(),
        "standby completion delivered after its deadline is dropped"
    );
}

/// Any stick may move the game world, a head turn changes what a snapshot
/// covers, and a suspension starts over: each ends or replaces the standby.
#[test]
fn standby_invalidation() {
    let mut stick = Fixture::with_standby();
    stick.input.hand_magnitude[1] = 0.5; // the other stick (trigger: left), e.g. a turn
    assert_eq!(stick.at(100).state, State::Idle, "other stick cannot push");
    stick.input.hand_magnitude[1] = 0.0;
    let out = stick.at(110);
    assert!(
        out.standby_slot.is_none() && out.standby_at_ms == Some(250),
        "other stick ends the standby; new one 150 ms after it settles"
    );
    assert!(
        stick.at(249).capture_request.is_none() && stick.at(250).capture_request.is_some(),
        "replacement after the sticks settle"
    );

    let mut small = Fixture::with_standby();
    small.input.hand_magnitude[0] = 0.12; // below the dead zone, but a game may move
    small.at(100);
    small.input.hand_magnitude[0] = 1.0;
    assert_eq!(
        small.at(110).state,
        State::Arming,
        "a small deflection before the push ends the standby"
    );

    let mut turn = Fixture::with_standby();
    turn.input.standby_angle_deg = 16.0;
    let out = turn.at(299);
    assert!(
        out.standby_at_ms == Some(300) && out.capture_request.is_none(),
        "a head turn of half the early-refresh angle brings the next standby forward"
    );
    assert!(
        turn.at(300).capture_request.is_some(),
        "replaced 250 ms after its capture"
    );

    let mut far = Fixture::with_standby();
    far.input.standby_angle_deg = 30.0;
    far.input.hand_magnitude[0] = 1.0;
    assert_eq!(
        far.at(100).state,
        State::Arming,
        "a standby the head has turned 30 degrees from is not used"
    );

    let mut unknown = Fixture::with_standby();
    unknown.input.standby_angle_deg = f32::INFINITY;
    unknown.input.hand_magnitude[0] = 1.0;
    assert_eq!(
        unknown.at(100).state,
        State::Arming,
        "unknown turn angle is not used"
    );

    let mut dashboard = Fixture::with_standby();
    dashboard.input.dashboard = true;
    let out = dashboard.at(100);
    assert!(
        out.state == State::Suspended && out.standby_slot.is_none() && out.standby_at_ms.is_none(),
        "dashboard ends the standby"
    );
    dashboard.input.dashboard = false;
    assert!(
        dashboard.at(200).capture_request.is_none() && dashboard.at(1099).capture_request.is_none(),
        "no standby before the retry time"
    );
    assert!(
        dashboard.at(1100).capture_request.is_some(),
        "standby again once the retry time has passed"
    );

    let mut geometry = Fixture::with_standby();
    geometry.timing.shell_half_size = 5.0;
    assert!(
        geometry.at(70).standby_slot.is_none(),
        "a shell or calibration change ends the standby"
    );

    let mut plain = Fixture::standby();
    plain.timing.snapshots = false;
    for t in (0..=1000).step_by(50) {
        assert!(
            plain.at(t).capture_request.is_none(),
            "Plain and Grid never capture"
        );
    }

    let mut source = Fixture::with_standby();
    source.input.source_generation = 7;
    assert!(
        source.at(70).standby_slot.is_none(),
        "a new game source ends the standby"
    );
}

/// Releasing: once the sticks settle the other slot takes standby snapshots,
/// ready for the next push after the fade-out.
#[test]
fn standby_after_release() {
    let mut f = Fixture::standby();
    f.timing.fade_in_ms = 120.0;
    f.active(0); // first capture at 50, shown from 60, active at 180
    f.input.hand_magnitude[0] = 0.0;
    assert_eq!(f.at(200).state, State::Active, "release delay");
    let out = f.at(260);
    assert!(
        out.state == State::Releasing
            && out.capture_request.is_none()
            && out.standby_at_ms == Some(330),
        "standby due 150 ms after the stick settled"
    );
    let request = f.standby_request(330);
    assert!(
        request.slot == 1 && out.new_slot == 0,
        "releasing standby uses the slot not shown"
    );
    f.complete_standby(request, 335, CaptureResult::Ready);
    let out = f.at(340);
    assert!(
        out.state == State::Releasing
            && out.standby_slot == Some(1)
            && out.new_slot == 0
            && out.visible,
        "fade-out continues with the shown snapshot"
    );
    let out = f.at(460);
    assert!(
        out.state == State::Idle && out.standby_slot == Some(1),
        "standby survives the end of the fade-out"
    );
    f.input.hand_magnitude[0] = 1.0;
    let out = f.at(500);
    assert!(
        out.state == State::Entering
            && out.new_slot == 1
            && out.last_capture_ms == Some(330)
            && out.activation == 2,
        "next push starts from the standby"
    );

    // Pushed again during a long fade-out once the shown snapshot is too old.
    let mut again = Fixture::standby();
    again.timing.fade_in_ms = 120.0;
    again.timing.fade_out_ms = 3000.0;
    again.active(0);
    again.input.hand_magnitude[0] = 0.0;
    again.at(260);
    let request = again.standby_request(330);
    again.complete_standby(request, 335, CaptureResult::Ready);
    again.at(340);
    let request = again.standby_request(705);
    again.complete_standby(request, 710, CaptureResult::Ready);
    again.at(715);
    let request = again.standby_request(1080);
    assert_eq!(
        request.slot, 1,
        "replacements while releasing stay in the free slot"
    );
    again.complete_standby(request, 1085, CaptureResult::Ready);
    again.at(1090);
    again.input.hand_magnitude[0] = 1.0;
    let out = again.at(1300); // shown snapshot is 1250 ms old (limit 1222)
    assert!(
        out.state == State::Entering
            && !out.visible
            && out.new_slot == 1
            && out.last_capture_ms == Some(1080),
        "a push with an expired shown snapshot restarts from the standby"
    );
    assert!(again.at(1360).visible, "and fades in");
}
