use super::{
    Options,
    game::Game,
    metrics::{Metrics, Samples},
};
use comfort_core::{
    geometry::{Aperture, Face, Pose, PoseHistory, Projection, Shell, Vec3},
    head_overlays,
    panel::{self, Panel, System},
    release::{self, Version},
    settings::{OpeningShape, Periphery, SaveScope, Settings},
    timing::{self, Controller, Input, Output, State, Timing},
};
use comfort_runtime::{
    mailbox::{Mailbox, Options as MailboxOptions, monotonic_ns},
    openvr::{OpenVr, Overlay, OverlayEvent, SystemEvent},
    vulkan::{Context, ContextOptions, Gap, OverlayTexture, ShellParams, ShellRenderer},
};
use serde_json::json;
use std::{
    error::Error,
    fs,
    io::{self, Write},
    path::{Path, PathBuf},
    process::Command,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const DASHBOARD: &std::ffi::CStr = c"org.vrwindow.comfort.controls";
const HIDDEN_RELEASE_NS: u64 = 100_000_000;
/// Consecutive failed panel texture uploads before raw uploads take over.
const TEXTURE_ATTEMPTS: u8 = 3;
/// How often the shown panel rechecks the observer and the update state.
const SYSTEM_CHECK: Duration = Duration::from_secs(2);
/// How long a started restart or update shows as busy before the update
/// state (or the restarted app) takes over.
const TASK_BUSY: Duration = Duration::from_secs(15);

struct Dashboard<'a> {
    vr: &'a OpenVr,
    panel: Panel,
    overlay: Overlay<'a>,
    _thumbnail: Overlay<'a>,
    // Owned by `run` so the images outlive OpenVR shutdown. Without it the
    // panel falls back to raw uploads, which flicker on every change.
    texture: Option<&'a mut OverlayTexture>,
    texture_failures: u8,
    context: Arc<Context>,
    cursor: [f32; 2],
    source: &'static str,
    drawn_source: &'static str,
    system: System,
    drawn_system: System,
    system_checked: Option<Instant>,
    task: Option<(System, Instant)>,
    /// Where the installer put the app, `frame-comfortctl` and the update state.
    install_dir: Option<PathBuf>,
    request: Option<PathBuf>,
    settings_path: Option<PathBuf>,
}

impl<'a> Dashboard<'a> {
    fn new(
        vr: &'a OpenVr,
        settings: &Settings,
        options: &Options,
        texture: Option<&'a mut OverlayTexture>,
        context: Arc<Context>,
    ) -> Result<Self> {
        let [overlay, thumbnail] = vr.dashboard(
            DASHBOARD,
            c"Frame Comfort",
            panel::WIDTH as u32,
            panel::HEIGHT as u32,
        )?;
        let mut image = Panel::thumbnail();
        thumbnail.upload_rgba(&mut image.pixels, image.width as u32, image.height as u32)?;
        let request = options
            .socket
            .as_ref()
            .and_then(|path| path.parent().map(Path::to_owned))
            .or_else(|| std::env::var_os("XDG_RUNTIME_DIR").map(PathBuf::from))
            .map(|path| path.join("vrw-comfort-controls.request"));
        let mut dashboard = Self {
            vr,
            panel: Panel::default(),
            overlay,
            _thumbnail: thumbnail,
            texture,
            texture_failures: 0,
            context,
            cursor: [0.0; 2],
            source: "no-fresh-source",
            drawn_source: "",
            system: System::Ready,
            drawn_system: System::Ready,
            system_checked: None,
            task: None,
            install_dir: std::env::current_exe()
                .ok()
                .and_then(|exe| exe.parent().map(Path::to_owned)),
            request,
            settings_path: options.dashboard_settings(),
        };
        dashboard.system = dashboard.system_state();
        dashboard.redraw(settings)?;
        Ok(dashboard)
    }

    fn redraw(&mut self, settings: &Settings) -> Result<()> {
        self.drawn_source = self.source;
        self.drawn_system = self.system.clone();
        let mut image = self.panel.render(settings, self.source, &self.system);
        if let Some(texture) = &mut self.texture {
            let (overlay, context) = (&self.overlay, &self.context);
            let uploaded = texture.upload(image.pixels.as_flattened(), |image, extent| {
                // SAFETY: The image belongs to this context, is in transfer
                // source layout and outlives OpenVR shutdown. The overlay
                // loop is the only user of the shared queue.
                unsafe { overlay.submit_image(context, image, extent) }
            });
            match uploaded {
                Ok(()) => {
                    self.texture_failures = 0;
                    return Ok(());
                }
                // A rejected submission can be transient, so this redraw
                // uses a raw upload and the next one tries the texture again.
                Err(error) => {
                    self.texture_failures += 1;
                    if self.texture_failures >= TEXTURE_ATTEMPTS {
                        eprintln!("Dashboard panel texture failed, using raw uploads: {error}");
                        self.texture = None;
                    } else {
                        eprintln!("Dashboard panel texture upload failed: {error}");
                    }
                }
            }
        }
        self.overlay
            .upload_rgba(&mut image.pixels, image.width as u32, image.height as u32)?;
        Ok(())
    }

    /// Show the latest Game source status on the Mode panel and the
    /// installation state in its footer. The source status only changes in
    /// the game, so the panel is redrawn once the dashboard shows. A failed
    /// redraw waits for the next status change.
    fn refresh_status(&mut self, settings: &Settings, visible: bool) -> Result<()> {
        if !visible {
            return Ok(());
        }
        if self
            .system_checked
            .is_none_or(|checked| checked.elapsed() >= SYSTEM_CHECK)
        {
            self.system_checked = Some(Instant::now());
            self.system = self.system_state();
        }
        let source_changed =
            settings.periphery == Periphery::Game && self.drawn_source != self.source;
        if source_changed || self.drawn_system != self.system {
            self.redraw(settings)?;
        }
        Ok(())
    }

    /// A task just started shows as busy; then a running update, the
    /// observer, then an available update, in that order.
    fn system_state(&self) -> System {
        if let Some((busy, since)) = &self.task
            && since.elapsed() < TASK_BUSY
        {
            return busy.clone();
        }
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_secs());
        let update = self
            .install_dir
            .as_ref()
            .and_then(|dir| fs::read(dir.join(release::UPDATE_FILE)).ok())
            .and_then(|bytes| serde_json::from_slice(&bytes).ok())
            .map(|state| release::update_notice(&state, Version::current(), now))
            .unwrap_or_default();
        if matches!(update, System::Busy(_)) {
            return update;
        }
        match observer_state() {
            System::Ready => update,
            observer => observer,
        }
    }

    /// Runs `frame-comfortctl` outside this service: restarting SteamVR stops
    /// Comfort and an update restarts it, and either would end the task.
    fn start_task(&mut self, args: &[&str], busy: &'static str) {
        let Some(ctl) = self
            .install_dir
            .as_ref()
            .map(|dir| dir.join("frame-comfortctl"))
            .filter(|ctl| ctl.is_file())
        else {
            eprintln!("frame-comfortctl is not installed next to the app");
            return;
        };
        match Command::new("systemd-run")
            .args(["--user", "--collect", "--quiet", "--"])
            .arg(&ctl)
            .args(args)
            .status()
        {
            Ok(status) if status.success() => {
                self.task = Some((System::Busy(busy), Instant::now()));
                self.system = self.system_state();
            }
            result => eprintln!("Could not start frame-comfortctl {args:?}: {result:?}"),
        }
    }

    fn poll(&mut self, settings: &mut Settings) -> Result<bool> {
        if let Some(path) = &self.request
            && fs::remove_file(path).is_ok()
        {
            self.vr.show_dashboard(DASHBOARD)?;
        }
        let mut redraw = false;
        while let Some(event) = self.overlay.poll_dashboard_event() {
            match event {
                OverlayEvent::Move(point) => self.cursor = point,
                OverlayEvent::Leave => self.cursor = [0.0; 2],
                OverlayEvent::Press(point) => {
                    let [x, y] = if point == [0.0; 2] {
                        self.cursor
                    } else {
                        point
                    };
                    let system = self.system.clone();
                    let click = self
                        .panel
                        .click([x, panel::HEIGHT as f32 - y], settings, &system);
                    redraw |= click.changed;
                    match click.action {
                        panel::Action::Quit => return Ok(false),
                        panel::Action::RestartSteamVr => {
                            self.start_task(
                                &["enable-source", "--yes", "--start"],
                                "RESTARTING STEAMVR...",
                            );
                            redraw = true;
                        }
                        panel::Action::Update => {
                            self.start_task(&["update"], "UPDATING FRAME COMFORT...");
                            redraw = true;
                        }
                        panel::Action::SaveSettings => {
                            if let Some(path) = &self.settings_path
                                && let Err(error) =
                                    write_json(path, &settings.to_json(SaveScope::Dashboard))
                            {
                                eprintln!("Could not save dashboard settings: {error}");
                                self.panel.save_failed();
                            }
                            break;
                        }
                        panel::Action::None => {}
                    }
                }
                OverlayEvent::Other => {}
            }
        }
        if redraw {
            self.redraw(settings)?;
        }
        Ok(true)
    }
}

/// Whether SteamVR's compositor runs the installed observer. Without a
/// compositor (SteamVR stopped, or a test fixture) there is nothing to report.
fn observer_state() -> System {
    let compositor = fs::read_dir("/proc")
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| entry.file_name().to_str()?.parse::<u32>().ok())
        .find(|pid| {
            fs::read_to_string(format!("/proc/{pid}/comm"))
                .is_ok_and(|name| name.trim() == "vrcompositor")
        });
    compositor
        .and_then(|pid| fs::read_to_string(format!("/proc/{pid}/maps")).ok())
        .map_or(System::Ready, |maps| observer_in_maps(&maps))
}

/// A replaced library shows as `(deleted)` in the mappings of a process that
/// loaded it before the update.
fn observer_in_maps(maps: &str) -> System {
    match maps
        .lines()
        .find(|line| line.contains("libvrw-comfort-source.so"))
    {
        None => System::ObserverMissing,
        Some(line) if line.ends_with("(deleted)") => System::ObserverOutdated,
        Some(_) => System::Ready,
    }
}

/// The most recent failure and when it happened, in app seconds, so an old
/// failure does not read as current in the status line.
#[derive(Default)]
struct LastError {
    message: String,
    at: Option<f64>,
}

impl LastError {
    fn set(&mut self, failure: impl std::fmt::Display, start: Instant) {
        self.message = failure.to_string();
        self.at = Some(start.elapsed().as_secs_f64());
    }
}

fn write_json(path: &Path, value: &serde_json::Value) -> Result<()> {
    let parent = path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let mut file = tempfile::NamedTempFile::new_in(parent)?;
    serde_json::to_writer_pretty(&mut file, value)?;
    writeln!(file)?;
    file.persist(path).map_err(|error| error.error)?;
    Ok(())
}

struct HeadOverlay<'a> {
    overlay: Overlay<'a>,
    last_style: Option<[f32; 3]>,
    shown: bool,
    nose: bool,
}

impl<'a> HeadOverlay<'a> {
    fn new(vr: &'a OpenVr, nose: bool) -> Result<Self> {
        let key = if nose {
            c"org.vrwindow.comfort.rust.nose"
        } else {
            c"org.vrwindow.comfort.rust.crosshair"
        };
        let name = if nose {
            c"Frame Comfort Nose"
        } else {
            c"Frame Comfort Crosshair"
        };
        let overlay = vr.create_overlay(key, name)?;
        // Both sit above the ring's faces (sort order 103).
        if nose {
            overlay.stereo(104)?;
        } else {
            overlay.set_sort_order(10000)?;
        }
        Ok(Self {
            overlay,
            last_style: None,
            shown: false,
            nose,
        })
    }

    fn update(&mut self, settings: &Settings, enabled: bool) -> Result<()> {
        let wanted = enabled
            && if self.nose {
                settings.nose
            } else {
                settings.crosshair
            };
        if wanted {
            let style = if self.nose {
                [
                    settings.nose_size,
                    settings.nose_opacity,
                    head_overlays::NOSE_DISTANCE_M,
                ]
            } else {
                [
                    settings.crosshair_size_deg,
                    settings.crosshair_opacity,
                    settings.crosshair_distance_m,
                ]
            };
            if self.last_style != Some(style) {
                let (rgba, width, height, degrees) = if self.nose {
                    let n = head_overlays::NOSE_EYE_SIZE as u32;
                    (
                        head_overlays::nose_pixels(style[0], style[1]),
                        n * 2,
                        n,
                        head_overlays::NOSE_TEXTURE_DEG,
                    )
                } else {
                    let n = head_overlays::CROSSHAIR_TEXTURE_SIZE as u32;
                    (
                        head_overlays::crosshair_pixels(style[0], style[1]),
                        n,
                        n,
                        head_overlays::CROSSHAIR_TEXTURE_DEG,
                    )
                };
                let mut pixels: Vec<_> = rgba
                    .chunks_exact(4)
                    .map(|bytes| [bytes[0], bytes[1], bytes[2], bytes[3]])
                    .collect();
                self.overlay.upload_rgba(&mut pixels, width, height)?;
                self.overlay
                    .set_width(head_overlays::overlay_width_m(degrees, style[2]))?;
                self.overlay.set_transform(
                    Pose {
                        position: Vec3(0.0, 0.0, -f64::from(style[2])),
                        ..Pose::default()
                    }
                    .matrix(),
                    true,
                )?;
                self.last_style = Some(style);
            }
        }
        if wanted != self.shown {
            self.overlay.visible(wanted)?;
            self.shown = wanted;
        }
        Ok(())
    }
}

struct Presentation<'a> {
    context: Arc<Context>,
    overlays: Vec<Overlay<'a>>,
    renderer: Option<ShellRenderer>,
    size: u32,
    shown: u32,
    hidden_at_ns: u64,
    frames: u64,
    leased: [Option<u64>; 3],
    next: usize,
}

impl<'a> Presentation<'a> {
    fn new(vr: &'a OpenVr, context: Arc<Context>) -> Result<Self> {
        let mut overlays = Vec::new();
        for face in 0..6 {
            let key =
                std::ffi::CString::new(format!("org.vrwindow.comfort.rust.restframe.{face}.face"))?;
            let overlay = vr.create_overlay(&key, c"Frame Comfort Rest Frame")?;
            overlay.stereo(103)?;
            overlays.push(overlay);
        }
        Ok(Self {
            context,
            overlays,
            renderer: None,
            size: 0,
            shown: 0,
            hidden_at_ns: 0,
            frames: 0,
            leased: [None; 3],
            next: 0,
        })
    }

    fn hide(&mut self, now: u64) {
        if self.shown != 0 {
            for overlay in &self.overlays {
                let _ = overlay.visible(false);
                let _ = overlay.clear();
            }
            self.shown = 0;
            self.hidden_at_ns = now;
        }
        if now.saturating_sub(self.hidden_at_ns) >= HIDDEN_RELEASE_NS {
            if let Some(renderer) = &mut self.renderer {
                for set in 0..3 {
                    // SAFETY: The overlays were cleared and the same hidden
                    // grace interval as the baseline has passed.
                    let _ = unsafe { renderer.retire(set) };
                }
            }
            self.leased = [None; 3];
        }
    }

    fn place(&self, shell: Shell) -> Result<()> {
        for (face, overlay) in Face::ALL.into_iter().zip(&self.overlays) {
            overlay.set_width((shell.half_size * 2.0) as f32)?;
            overlay.set_transform(
                shell
                    .face_matrix(face)
                    .ok_or_else(|| io::Error::other("invalid overlay placement"))?,
                false,
            )?;
        }
        Ok(())
    }

    fn prepare(&mut self, size: u32) -> Result<()> {
        if self.renderer.is_none() || (size != self.size && self.leased.iter().all(Option::is_none))
        {
            let mut renderer = ShellRenderer::new(Arc::clone(&self.context), size)?;
            renderer.enable_timing();
            self.renderer = Some(renderer);
            self.size = size;
        }
        Ok(())
    }

    fn show(&mut self, params: ShellParams<'_>, size: u32) -> Result<Option<u64>> {
        self.prepare(size)?;
        let renderer = self.renderer.as_mut().expect("renderer initialized");
        let set = self.next;
        if let Some(frame) = self.leased[set] {
            if self.frames.saturating_sub(frame) < 2 {
                return Ok(None);
            }
            // SAFETY: As in the C++ baseline, the overlay has been replaced
            // twice before reusing a submitted set. Verify on the headset.
            unsafe { renderer.retire(set) }?;
            self.leased[set] = None;
        }
        renderer.queue_render(params, set)?;
        self.frames += 1;
        self.leased[set] = Some(self.frames);
        let old = self.shown;
        // Track possibly visible overlays if a submit or visibility call fails.
        self.shown |= params.faces;
        let start = monotonic_ns();
        for ((index, face), overlay) in Face::ALL.into_iter().enumerate().zip(&self.overlays) {
            let bit = 1 << index;
            if let Some(image) = renderer.output_image(set, face) {
                let extent = ash::vk::Extent2D {
                    width: self.size * 2,
                    height: self.size,
                };
                // SAFETY: OpenVR uses the supplied render queue, where
                // rendering was submitted first. Calls stay serialized and
                // the output set is retained until retirement.
                unsafe { overlay.submit_image(&self.context, image, extent)? };
                if old & bit == 0 {
                    overlay.visible(true)?;
                }
            } else if old & bit != 0 {
                overlay.visible(false)?;
                overlay.clear()?;
            }
        }
        self.shown = params.faces;
        self.next = (set + 1) % 3;
        Ok(Some(start))
    }
}

impl Drop for Presentation<'_> {
    fn drop(&mut self) {
        self.hide(monotonic_ns());
        self.overlays.clear();
        if self.leased.iter().any(Option::is_some) {
            thread::sleep(Duration::from_nanos(HIDDEN_RELEASE_NS));
        }
        let _ = self.context.wait_idle();
    }
}

pub(super) fn run(options: Options, mut settings: Settings) -> Result<()> {
    let stop = Arc::new(AtomicBool::new(false));
    for signal in [signal_hook::consts::SIGINT, signal_hook::consts::SIGTERM] {
        signal_hook::flag::register(signal, Arc::clone(&stop))?;
    }
    // Keep the rendering device alive through OpenVR shutdown. SteamVR
    // holds Vulkan objects on this device after the last overlay is destroyed.
    let render_device;
    // The dashboard panel's images are freed after OpenVR shutdown too, as
    // in the C++ baseline.
    let mut panel_texture: Option<OverlayTexture>;
    // SAFETY: The CLI uses the installed SteamVR client or an explicitly
    // selected fixture built against the same pinned ABI.
    let vr = unsafe { OpenVr::init(&options.openvr_library)? };
    let names = vr.instance_extensions()?;
    let extensions: Vec<_> = names.iter().map(|name| name.as_c_str()).collect();
    let context = Context::new(ContextOptions {
        instance_extensions: &extensions,
        require_dma_buf: true,
        high_priority: !options.no_high_priority_queue,
        ..Default::default()
    })
    .or_else(|_| {
        Context::new(ContextOptions {
            instance_extensions: &extensions,
            high_priority: !options.no_high_priority_queue,
            ..Default::default()
        })
    })?;
    let (_, physical, _, _) = context.raw_handles();
    // SAFETY: context keeps the physical device's instance alive.
    let device_extensions = unsafe { vr.device_extensions(physical)? };
    render_device = if device_extensions.is_empty() {
        context
    } else {
        let names: Vec<_> = device_extensions
            .iter()
            .map(|name| name.as_c_str())
            .collect();
        Context::new(ContextOptions {
            instance_extensions: &extensions,
            device_extensions: &names,
            device_uuid: Some(context.device_uuid()),
            require_dma_buf: context.supports_dma_buf(),
            high_priority: !options.no_high_priority_queue,
            ..Default::default()
        })?
    };
    let mut game = Game::new(
        Arc::clone(&render_device),
        options.check_frame_code,
        options.source_format,
    )?;
    let mut view = Presentation::new(&vr, Arc::clone(&render_device))?;
    let socket = options.source_socket();
    panel_texture = OverlayTexture::new(
        Arc::clone(&render_device),
        panel::WIDTH as u32,
        panel::HEIGHT as u32,
    )
    .inspect_err(|error| eprintln!("Dashboard panel texture: {error} (using raw uploads)"))
    .ok();
    let mut dashboard = Dashboard::new(
        &vr,
        &settings,
        &options,
        panel_texture.as_mut(),
        Arc::clone(&render_device),
    )?;
    let mut nose = HeadOverlay::new(&vr, true)?;
    let mut crosshair = HeadOverlay::new(&vr, false)?;
    let mailbox = Mailbox::start(MailboxOptions {
        port: options.mailbox_port,
        ..Default::default()
    })?;
    let start = Instant::now();
    let mut eyes = vr.eye_geometry()?;
    let (mut period, mut photons) = vr.display_timing();
    let mut controller = Controller::default();
    let mut tracking_epoch = 0;
    let mut anchor = None;
    let mut force = 0.0;
    let mut previous_ns = monotonic_ns();
    let mut last_status = 0;
    let mut active_frames = 0;
    let mut render_failures = 0;
    let mut error = LastError::default();
    let mut status_write_failed = false;
    let mut render_healthy = true;
    let mut history = PoseHistory::default();
    let mut last_output = None;
    let mut previous_mode = settings.periphery;
    let mut game_activation = None;
    let mut activation = 0;
    let mut starts = 0_u64;
    let mut standby_starts = 0_u64;
    let mut starting = None;
    let mut start_samples = Samples::default();
    let mut metrics = Metrics::default();
    let mut last_settings = settings.clone();

    while !stop.load(Ordering::Relaxed)
        && options
            .seconds
            .is_none_or(|limit| limit.is_zero() || start.elapsed() < limit)
    {
        let loop_started = monotonic_ns();
        if let Some(renderer) = &mut view.renderer {
            match renderer.completions() {
                Ok(samples) => {
                    for sample in samples {
                        if let Some(ms) = sample.gpu_ms {
                            metrics.gpu.add(ms);
                        }
                        metrics.render_done.add(sample.elapsed_ms);
                    }
                }
                Err(failure) => error.set(failure, start),
            }
        }
        while let Some(event) = vr.poll_system_event() {
            match event {
                SystemEvent::Quit => return Ok(()),
                SystemEvent::TrackingChanged => {
                    tracking_epoch += 1;
                    history.reset(tracking_epoch);
                    game.invalidate_tracking();
                    eyes = vr.eye_geometry()?;
                    (period, photons) = vr.display_timing();
                    anchor = None;
                }
                SystemEvent::Other => {}
            }
        }
        if !dashboard.poll(&mut settings)? {
            break;
        }
        let now = monotonic_ns();
        let dashboard_visible = vr.dashboard_visible()?;
        let dashboard_open = dashboard_visible && !options.ignore_dashboard;
        let tracking = vr.tracking(0.0);
        if let Some(tracking) = &tracking {
            history.add(now, tracking.head, tracking_epoch);
        } else {
            history.invalidate();
        }
        let snapshot = mailbox.snapshot();
        let hands = snapshot.read(
            settings.stick,
            settings.deadzone,
            Duration::from_millis(1500),
            now,
        );
        let magnitude = selected_magnitude(settings.stick, hands.unwrap_or_default());
        let at_rest = last_output
            .is_none_or(|out: Output| matches!(out.state, State::Idle | State::Suspended));
        if settings.periphery != previous_mode {
            game_activation = None;
            previous_mode = settings.periphery;
        }
        if last_settings != settings {
            last_settings = settings.clone();
            render_healthy = true;
        }
        let mut config = Timing::from(&settings);
        config.display_period_ms = period;
        config.standby = !options.no_standby_snapshot;
        if settings.refresh_blink_ms > 0.0 {
            config.blend_ms = settings.refresh_blink_ms;
            config.capture_spacing_ms = config.capture_spacing_ms.max(400.0);
        }
        if settings.periphery == Periphery::Game && settings.enabled {
            let excluded = [
                [view.size * 2, view.size],
                [panel::WIDTH as u32, panel::HEIGHT as u32],
                [head_overlays::CROSSHAIR_TEXTURE_SIZE as u32; 2],
                [
                    head_overlays::NOSE_EYE_SIZE as u32 * 2,
                    head_overlays::NOSE_EYE_SIZE as u32,
                ],
            ];
            let observe = last_output.is_some_and(|out| {
                let angle = game
                    .meta(out.new_slot)
                    .and_then(|meta| {
                        let head = tracking.as_ref()?.head;
                        Some(head.rotation.angle_to(meta.head.rotation)?.to_degrees() as f32)
                    })
                    .unwrap_or(0.0);
                timing::observe_ring(out, config, angle, now / 1_000_000)
            });
            if let Err(failure) =
                game.poll(&socket, now, dashboard_open, at_rest, &excluded, observe)
            {
                error.set(failure, start);
            }
            metrics.compositor(game.compositor_sample());
        } else {
            metrics.compositor(None);
        }
        let source_status = match game.source_status(now) {
            // This movement started before verification and stays black.
            "verified" if game_activation.is_some_and(|p| p != Periphery::Game) => {
                "verified-next-movement"
            }
            status => status,
        };
        if !dashboard_open {
            dashboard.source = source_status;
        }
        // A status-only redraw is cosmetic and never stops Comfort.
        if let Err(failure) = dashboard.refresh_status(&settings, dashboard_visible) {
            error.set(failure, start);
        }
        let effective = effective_periphery(&settings, game_activation, game.ready(now));
        config.snapshots = effective == Periphery::Game;
        if config.snapshots && game.is_stream() {
            config.pose_offset_ms = settings.rest_frame_stream_pose_offset_ms;
        }
        let head_speed = tracking
            .as_ref()
            .map_or(0.0, |tracking| tracking.speed_deg_per_sec);
        let image_speed = game
            .report_pose_ns(&settings)
            .and_then(|time| history.angular_speed(time.min(now), 16_000_000, tracking_epoch))
            .map_or(head_speed, |speed| speed as f32);
        let angle = |slot: Option<u32>| {
            let meta = game.meta(slot?)?;
            let head = tracking.as_ref()?.head;
            Some(head.rotation.angle_to(meta.head.rotation)?.to_degrees() as f32)
        };
        let standby_angle =
            angle(last_output.and_then(|out| out.standby_slot)).unwrap_or(f32::INFINITY);
        let recent_angle = angle(last_output.map(|out| out.new_slot)).unwrap_or(0.0);
        let input = Input {
            enabled: settings.enabled,
            dashboard: dashboard_open,
            tracking_epoch,
            input_available: hands.is_some(),
            hand_magnitude: hands.unwrap_or_default(),
            source_fresh: game.still_fresh(now),
            source_generation: game.generation_id,
            tracking_valid: tracking.is_some(),
            renderer_healthy: render_healthy,
            angular_speed_deg_per_sec: head_speed,
            image_angular_speed_deg_per_sec: image_speed,
            angle_since_capture_deg: recent_angle,
            standby_angle_deg: standby_angle,
            mapping_valid: anchor.is_none_or(|shell: Shell| {
                tracking.as_ref().is_some_and(|tracking| {
                    let Vec3(x, y, z) = shell.pose.position - tracking.head.position;
                    x.abs().max(y.abs()).max(z.abs()) <= f64::from(settings.rest_frame_move_limit_m)
                })
            }),
        };
        let out = controller.update(config, input, now / 1_000_000);
        let inactive = matches!(out.state, State::Idle | State::Suspended);
        if settings.periphery == Periphery::Game {
            if inactive {
                if out.state == State::Idle || magnitude <= (settings.deadzone - 0.03).max(0.0) {
                    game_activation = None;
                }
            } else if game_activation.is_none() {
                game_activation = Some(effective);
            }
        }
        let new_activation = out.activation != activation;
        let before = last_output.map_or(State::Idle, |out| out.state);
        if inactive {
            starting = None;
        } else if at_rest
            || (before == State::Releasing && (out.state == State::Arming || new_activation))
        {
            let updated = snapshot
                .hands
                .iter()
                .map(|hand| hand.updated_ns)
                .max()
                .unwrap_or(now);
            starting = Some((
                updated.clamp(previous_ns.min(now), now),
                config.snapshots && out.state != State::Arming,
            ));
        }
        if config.snapshots
            && new_activation
            && let Some(meta) = game.meta(out.new_slot)
        {
            let shell = Shell {
                pose: Pose {
                    position: meta.head.position,
                    ..Pose::default()
                },
                half_size: f64::from(settings.rest_frame_shell_half_size),
            };
            view.place(shell)?;
            anchor = Some(shell);
        }
        activation = out.activation;
        last_output = Some(out);
        render_healthy = true;
        let target = timing::stick_force(magnitude, config.deadzone);
        force = if inactive || !settings.dynamic_fov {
            0.0
        } else {
            timing::smooth_force(force, target, now.saturating_sub(previous_ns) as f64 / 1e6)
        };
        previous_ns = now;
        if out.visible
            && let Some(tracking) = tracking
        {
            if anchor.is_none() {
                let shell = Shell {
                    pose: Pose {
                        position: tracking.head.position,
                        ..Pose::default()
                    },
                    half_size: f64::from(settings.rest_frame_shell_half_size),
                };
                view.place(shell)?;
                anchor = Some(shell);
            }
            let shell = anchor.expect("initialized above");
            let predicted = (period / 1000.0 - vr.since_vsync()
                + photons
                + settings.rest_frame_mask_prediction_ms / 1000.0)
                .clamp(0.0, 0.1);
            if let Some(display) = vr.tracking(predicted) {
                let mut params = render_params(
                    &settings,
                    &options,
                    shell,
                    eyes,
                    display.head,
                    out,
                    force,
                    view.shown,
                    effective,
                );
                let size = if config.snapshots {
                    let old_slot = if out.needs_both_snapshots {
                        out.old_slot
                    } else {
                        out.new_slot
                    };
                    let captures = game
                        .capture_meta(old_slot)
                        .zip(game.capture_meta(out.new_slot));
                    // A slot without a snapshot cannot be drawn. Hide the
                    // ring but keep pacing the loop and serving the capture
                    // request below, which is what replaces the slot.
                    captures.map(|(old, new)| {
                        params.captures = Some([old, new]);
                        params.blink = out.needs_both_snapshots && settings.refresh_blink_ms > 0.0;
                        game.meta(out.new_slot)
                            .expect("capture metadata exists")
                            .face_size
                    })
                } else if settings.rest_frame_face_size != 0 {
                    Some(settings.rest_frame_face_size)
                } else if effective == Periphery::Plain {
                    Some(512)
                } else {
                    Some(settings.rest_frame_max_face_size)
                };
                let result = size.map(|size| view.show(params, size)).transpose();
                match result {
                    Ok(None) => view.hide(now),
                    Ok(Some(None)) => {}
                    Ok(Some(Some(submitting))) => {
                        metrics.submit.since(submitting);
                        metrics.submitted(params.faces, vr.vsync());
                        active_frames += 1;
                        if view.shown != 0
                            && let Some((stick_ns, standby)) = starting.take()
                        {
                            starts += 1;
                            standby_starts += u64::from(standby);
                            start_samples.since(stick_ns);
                        }
                    }
                    Err(failure) => {
                        error.set(failure, start);
                        render_healthy = false;
                        render_failures += 1;
                        view.hide(now);
                    }
                }
            } else {
                view.hide(now);
            }
        } else {
            view.hide(now);
            if matches!(out.state, State::Idle | State::Suspended) {
                anchor = None;
            }
        }
        if view.shown == 0 {
            metrics.hidden();
        }
        if let Some(request) = out.capture_request {
            use comfort_core::timing::{CaptureReason, CaptureResult};
            let start_ns = monotonic_ns();
            let capture = game.capture(
                request.slot,
                &settings,
                &history,
                tracking_epoch,
                eyes,
                start_ns,
            );
            metrics.capture.since(start_ns);
            let result = match capture {
                Ok(true) => {
                    if request.reason == CaptureReason::Standby
                        && view.shown == 0
                        && let Some(meta) = game.meta(request.slot)
                    {
                        let prepared = view.prepare(meta.face_size).and_then(|()| {
                            view.renderer
                                .as_mut()
                                .expect("renderer prepared")
                                .prepare_outputs()
                        });
                        if let Err(failure) = prepared {
                            error.set(failure, start);
                        }
                    }
                    CaptureResult::Ready
                }
                Ok(false) => CaptureResult::InvalidPair,
                Err(failure) => {
                    error.set(failure, start);
                    CaptureResult::RenderFailure
                }
            };
            controller.complete_capture(
                request.token,
                start_ns / 1_000_000,
                monotonic_ns() / 1_000_000,
                result,
            );
        }
        nose.update(
            &settings,
            settings.enabled && !dashboard_open && input.tracking_valid,
        )?;
        crosshair.update(
            &settings,
            settings.enabled && !dashboard_open && input.tracking_valid,
        )?;
        let paced = out.visible || matches!(out.state, State::Arming | State::Entering);
        if paced {
            metrics.work.since(loop_started);
        }
        if now.saturating_sub(last_status) >= 250_000_000 {
            let [start_p50, start_p95, start_max] = start_samples.percentiles([0.5, 0.95, 1.0]);
            let [work_p50, work_p95] = metrics.work.percentiles([0.5, 0.95]);
            let [gap_p50, gap_p95, gap_max] = metrics.compositor_gap.percentiles([0.5, 0.95, 1.0]);
            let [wait_p50] = metrics.frame_wait.percentiles([0.5]);
            let [gpu_p50, gpu_p95, gpu_max] = metrics.gpu.percentiles([0.5, 0.95, 1.0]);
            let [done_p50, done_p95, done_max] = metrics.render_done.percentiles([0.5, 0.95, 1.0]);
            let [submit_p50, submit_p95] = metrics.submit.percentiles([0.5, 0.95]);
            let [vsync_p50, vsync_p95, vsync_max] =
                metrics.after_vsync.percentiles([0.5, 0.95, 1.0]);
            let [fingerprint_p50, fingerprint_p95] = game.fingerprint_ms.percentiles([0.5, 0.95]);
            let [capture_p95] = metrics.capture.percentiles([0.95]);
            let bytes = game.allocated_bytes()
                + view
                    .renderer
                    .as_ref()
                    .map_or(0, ShellRenderer::allocated_bytes);
            let rest_frame = json!({"visible":view.shown != 0,"state":format!("{:?}",out.state).to_lowercase(),
                    "reason":format!("{:?}",out.reason),
                    "rule":game.rule(),"eyeSource":game.eye_source(),
                    "captures":game.captures,"unstableCaptures":game.unstable_captures,
                    "starts":starts,"standbyStarts":standby_starts,"startMsP50":start_p50,
                    "startMsP95":start_p95,"startMsMax":start_max,"standbyReady":out.standby_slot.is_some(),
                    "gpuMsP50":gpu_p50,"gpuMsP95":gpu_p95,"gpuMsMax":gpu_max,
                    "renderDoneMsP50":done_p50,"renderDoneMsP95":done_p95,"renderDoneMsMax":done_max,
                    "captureMsP95":capture_p95,"frameWaitMsP50":wait_p50,
                    "fingerprints":game.fingerprints_taken,"fingerprintMsP50":fingerprint_p50,"fingerprintMsP95":fingerprint_p95,
                    "submitMsP50":submit_p50,"submitMsP95":submit_p95,
                    "afterVsyncMsP50":vsync_p50,"afterVsyncMsP95":vsync_p95,"afterVsyncMsMax":vsync_max,
                    "missedFrames":metrics.missed_frames,"facesPerFrame":metrics.face_submissions as f64 / active_frames.max(1) as f64,
                    "allocatedBytes":bytes,"faceSize":view.size,"highPriorityQueue":u8::from(render_device.high_priority())});
            let status = json!({"t":start.elapsed().as_secs_f64(), "activeFrames":active_frames,
                "dashboard":dashboard_open,"tracking":input.tracking_valid,"input":input.input_available,
                "rendered":view.frames,"renderFailures":render_failures,
                "error":error.message,"errorT":error.at,"enabled":settings.enabled,
                "periphery":settings.periphery.name(),"effectivePeriphery":effective.name(),
                "gridOverRing":settings.grid_over_ring,
                "workMsP50":work_p50,"workMsP95":work_p95,
                "compositorGapMsP50":gap_p50,"compositorGapMsP95":gap_p95,"compositorGapMsMax":gap_max,
                "restFrame":rest_frame,
                "gameSource":source_status,"images":game.image_count()});
            // The status file is a diagnostic. Failing to write it is reported
            // once per run of failures and never stops Comfort.
            if let Some(path) = &options.status {
                match write_json(path, &status) {
                    Ok(()) => status_write_failed = false,
                    Err(failure) => {
                        if !status_write_failed {
                            eprintln!("Could not write status {}: {failure}", path.display());
                        }
                        status_write_failed = true;
                    }
                }
            }
            println!("{status}");
            last_status = now;
        }
        if paced {
            let wait_started = monotonic_ns();
            let wait = vr.wait_frame((period * 2.0) as u32);
            metrics.frame_wait.since(wait_started);
            if wait.is_err() {
                thread::sleep(Duration::from_secs_f32(period / 1000.0));
            }
        } else {
            let delay = if settings.periphery == Periphery::Game && settings.enabled {
                game.learning_delay()
            } else {
                Duration::from_millis(10)
            };
            mailbox.wait_for_stick(delay, settings.deadzone);
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn render_params(
    settings: &Settings,
    options: &Options,
    shell: Shell,
    eyes: [(Pose, Projection); 2],
    head: Pose,
    out: Output,
    force: f32,
    shown: u32,
    effective: Periphery,
) -> ShellParams<'static> {
    let display_eyes = eyes.map(|(eye, _)| head.head_aligned_eye(eye.position));
    let view_eyes = eyes.map(|(eye, projection)| (head.compose(eye), projection));
    let mut faces = if options.rest_frame_all_faces {
        0x3f
    } else {
        shell.faces_in_view(view_eyes, 30.0) | (shown & shell.faces_in_view(view_eyes, 45.0))
    };
    if faces == 0 {
        faces = 0x3f;
    }
    let [width, height] = timing::opening_degrees(
        settings.opening_width_deg,
        settings.opening_height_deg,
        settings.opening_min_width_deg,
        if settings.dynamic_fov { force } else { 0.0 },
    );
    let (gap, grid_over_ring) = periphery_look(settings, effective);
    ShellParams {
        shell,
        display_eyes,
        faces,
        aperture: Aperture {
            rectangle: settings.shape == OpeningShape::Rectangle,
            half_width_tan: (f64::from(width).to_radians() * 0.5).tan(),
            half_height_tan: (f64::from(height).to_radians() * 0.5).tan(),
            feather: f64::from(settings.opening_fade),
            strength: f64::from(settings.periphery_opacity),
            entry: f64::from(out.opening_progress),
        },
        gap,
        refresh_weight: out.refresh_weight,
        grid_over_ring,
        dim: settings.periphery_dim,
        ..Default::default()
    }
}

/// What fills the periphery for this movement.
///
/// Game shows the snapshot only once the game's picture is verified. A
/// movement that starts earlier stays black like Plain until release, so grid
/// lines appear only with Grid over ring on, as in every other Game moment.
fn effective_periphery(
    settings: &Settings,
    activation: Option<Periphery>,
    game_ready: bool,
) -> Periphery {
    if settings.periphery != Periphery::Game {
        return settings.periphery;
    }
    activation.unwrap_or(if game_ready {
        Periphery::Game
    } else {
        Periphery::Plain
    })
}

/// The gap fill and whether grid lines are layered over the periphery.
///
/// Only the Grid periphery fills with the grid look. Plain and Game keep a
/// black gap fill and take their lines from Grid over ring, which the shader
/// draws over the picture and holds steady through the refresh blink. Marking
/// those as Grid would leave the lines in the blink dip and the gaps only.
fn periphery_look(settings: &Settings, effective: Periphery) -> (Gap, bool) {
    let grid = effective == Periphery::Grid;
    let gap = Gap {
        grid,
        background: if grid { settings.brightness } else { 0.0 },
        lines: settings.grid_brightness,
        thickness_cm: settings.grid_thickness_cm,
        spacing: settings.grid_spacing,
    };
    (gap, settings.grid_over_ring && !grid)
}

fn selected_magnitude(stick: comfort_core::settings::Stick, [left, right]: [f32; 2]) -> f32 {
    use comfort_core::settings::Stick;
    match stick {
        Stick::Left => left,
        Stick::Right => right,
        Stick::Both => left.max(right),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observer_state_comes_from_the_compositor_mappings() {
        let line = "ffff91f00000-ffff91f81000 r-xp 00000000 08:08 742424 /home/u/.local/share/vrw-comfort/libvrw-comfort-source.so";
        assert_eq!(observer_in_maps(line), System::Ready);
        assert_eq!(
            observer_in_maps(&format!("{line} (deleted)")),
            System::ObserverOutdated
        );
        assert_eq!(
            observer_in_maps("ffff0000-ffff1000 r-xp 0 08:08 1 /usr/lib/libvulkan.so.1"),
            System::ObserverMissing
        );
    }

    #[test]
    fn grid_over_ring_layers_lines_over_plain_and_game() {
        let mut settings = Settings {
            grid_over_ring: true,
            ..Settings::default()
        };
        for effective in [Periphery::Plain, Periphery::Game] {
            let (gap, over_ring) = periphery_look(&settings, effective);
            assert!(over_ring, "{effective:?}");
            assert!(!gap.grid, "{effective:?}");
            assert_eq!(gap.background, 0.0);
            assert_eq!(gap.lines, settings.grid_brightness);
        }
        // Grid mode draws its own lines.
        let (gap, over_ring) = periphery_look(&settings, Periphery::Grid);
        assert!(gap.grid && !over_ring);
        assert_eq!(gap.background, settings.brightness);

        settings.grid_over_ring = false;
        for effective in [Periphery::Plain, Periphery::Game] {
            let (gap, over_ring) = periphery_look(&settings, effective);
            assert!(!gap.grid && !over_ring, "{effective:?}");
        }
    }

    #[test]
    fn unverified_game_movement_follows_the_grid_over_ring_toggle() {
        let mut settings = Settings::default();
        assert_eq!(settings.periphery, Periphery::Game);
        assert_eq!(effective_periphery(&settings, None, true), Periphery::Game);
        // Not verified yet: black, latched for the whole movement.
        let fallback = effective_periphery(&settings, None, false);
        assert_eq!(fallback, Periphery::Plain);
        let latched = Some(fallback);
        assert_eq!(
            effective_periphery(&settings, latched, true),
            Periphery::Plain
        );
        for toggle in [false, true] {
            settings.grid_over_ring = toggle;
            let (gap, over_ring) = periphery_look(&settings, fallback);
            assert!(!gap.grid, "the fallback never uses the Grid mode's look");
            assert_eq!(gap.background, 0.0);
            assert_eq!(over_ring, toggle, "lines only with Grid over ring on");
        }
        for mode in [Periphery::Plain, Periphery::Grid] {
            settings.periphery = mode;
            assert_eq!(effective_periphery(&settings, None, false), mode);
        }
    }
}
