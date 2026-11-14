use clap::{Parser, ValueEnum};
use comfort_core::settings::{OpeningShape, Periphery, SaveScope, Settings, Stick};
use std::{
    error::Error,
    fs, io,
    path::{Path, PathBuf},
    time::Duration,
};

#[cfg(target_os = "linux")]
mod audit;
#[cfg(target_os = "linux")]
mod game;
#[cfg(target_os = "linux")]
mod metrics;
#[cfg(target_os = "linux")]
mod probe;
#[cfg(target_os = "linux")]
mod session;

#[derive(Clone, Copy, ValueEnum)]
enum Mode {
    Plain,
    Grid,
    Game,
}
#[derive(Clone, Copy, ValueEnum)]
enum Hand {
    Left,
    Right,
    Both,
}
#[derive(Clone, Copy, ValueEnum)]
enum Shape {
    Circle,
    Rectangle,
}

#[derive(Parser)]
#[command(
    about = "Frame Comfort: a still periphery around a live stereo opening while you move, on Steam Frame"
)]
struct Options {
    #[arg(long)]
    settings: Option<PathBuf>,
    #[arg(long)]
    status: Option<PathBuf>,
    #[arg(long)]
    socket: Option<PathBuf>,
    /// Restrict Game source selection and evidence to this numeric VkFormat.
    #[arg(long)]
    source_format: Option<u32>,
    #[arg(long, value_parser = duration)]
    seconds: Option<Duration>,
    #[arg(long, default_value_t = 27062, value_parser = clap::value_parser!(u16).range(1..))]
    mailbox_port: u16,
    #[arg(long)]
    print_default_settings: bool,
    #[arg(long, default_value = "libopenvr_api.so")]
    openvr_library: PathBuf,
    #[arg(long, value_enum)]
    periphery: Option<Mode>,
    #[arg(long, value_enum, aliases = ["hands", "rest-frame-trigger"])]
    stick: Option<Hand>,
    #[arg(long, value_enum)]
    shape: Option<Shape>,
    #[arg(long)]
    opening_width_deg: Option<f32>,
    #[arg(long)]
    opening_height_deg: Option<f32>,
    #[arg(long)]
    opening_min_width_deg: Option<f32>,
    #[arg(long, alias = "feather")]
    opening_fade: Option<f32>,
    #[arg(long, alias = "opacity")]
    periphery_opacity: Option<f32>,
    #[arg(long)]
    periphery_dim: Option<f32>,
    #[arg(long)]
    deadzone: Option<f32>,
    #[arg(long)]
    fade_in_ms: Option<f32>,
    #[arg(long, alias = "release-ms")]
    fade_out_delay_ms: Option<f32>,
    #[arg(long)]
    fade_out_ms: Option<f32>,
    #[arg(long)]
    refresh_blink_ms: Option<f32>,
    #[arg(long, alias = "refresh-ms")]
    rest_frame_interval_ms: Option<f32>,
    #[arg(long)]
    rest_frame_cross_fade_ms: Option<f32>,
    #[arg(long)]
    rest_frame_stillness: Option<f32>,
    #[arg(long)]
    rest_frame_max_wait_ms: Option<f32>,
    #[arg(long)]
    rest_frame_early_deg: Option<f32>,
    #[arg(long)]
    rest_frame_shell: Option<f32>,
    #[arg(long)]
    rest_frame_pose_offset_ms: Option<f32>,
    #[arg(long)]
    rest_frame_stream_pose_offset_ms: Option<f32>,
    #[arg(long)]
    rest_frame_mask_prediction_ms: Option<f32>,
    #[arg(long)]
    rest_frame_move_limit: Option<f32>,
    #[arg(long)]
    rest_frame_face_size: Option<u32>,
    #[arg(long)]
    rest_frame_max_face_size: Option<u32>,
    #[arg(long)]
    surround: Option<f32>,
    #[arg(long)]
    dynamic_fov: bool,
    #[arg(long)]
    no_dynamic_fov: bool,
    #[arg(long)]
    grid_over_ring: bool,
    #[arg(long)]
    crosshair: bool,
    #[arg(long)]
    nose: bool,
    #[arg(long)]
    ignore_dashboard: bool,
    #[arg(long)]
    rest_frame_all_faces: bool,
    #[arg(long)]
    no_high_priority_queue: bool,
    #[arg(long)]
    no_standby_snapshot: bool,
    /// Verify the synthetic frame code and eye colors in each accepted snapshot.
    #[arg(long)]
    check_frame_code: bool,
    /// Report input without opening an overlay.
    #[arg(long, group = "probe")]
    probe_input: bool,
    /// List observer images and sample importable ones.
    #[arg(long, group = "probe")]
    probe_source: bool,
    /// Measure the streamed stereo atlas.
    #[arg(long, group = "probe")]
    probe_layout: bool,
    /// Print 32x32 luma grids for both full-view stream quadrants.
    #[arg(long, group = "probe")]
    probe_grid: bool,
    /// Show the live stereo verifier's scores and timing.
    #[arg(long, group = "probe")]
    probe_evidence: bool,
    /// Report raw OpenVR projections and eye-to-head transforms.
    #[arg(long)]
    probe_projection: bool,
    /// Sample unpredicted standing-space HMD poses.
    #[arg(long)]
    probe_tracking: bool,
    /// Save a strip of each streamed frame with its report time, and log HMD
    /// poses, to measure the stream's render-to-report delay. Needs
    /// --probe-dump FILE for the strips.
    #[arg(long, group = "probe", requires = "probe_dump")]
    probe_stream_motion: bool,
    /// Save private diagnostic images when probing layout or evidence.
    #[arg(
        long,
        requires = "probe",
        conflicts_with = "probe_input",
        conflicts_with = "probe_source"
    )]
    probe_dump: Option<PathBuf>,
}

fn duration(value: &str) -> Result<Duration, String> {
    let value = value.parse().map_err(|_| "expected seconds")?;
    Duration::try_from_secs_f64(value).map_err(|_| "expected finite nonnegative seconds".into())
}

fn load(path: &Path, settings: &mut Settings) -> Result<(), Box<dyn Error>> {
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(error.into()),
    };
    settings
        .apply_json(&serde_json::from_slice(&bytes)?)
        .map_err(io::Error::other)?;
    Ok(())
}

impl Options {
    #[cfg(target_os = "linux")]
    fn source_socket(&self) -> PathBuf {
        self.socket.clone().unwrap_or_else(|| {
            std::env::var_os("XDG_RUNTIME_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("/tmp"))
                .join("vrw-comfort-source.sock")
        })
    }

    fn dashboard_settings(&self) -> Option<PathBuf> {
        self.settings
            .as_ref()
            .map(|path| path.with_extension("dashboard.json"))
    }

    fn apply(&self, settings: &mut Settings) {
        macro_rules! apply {
            ($($arg:ident => $field:ident),* $(,)?) => {
                $(if let Some(value) = self.$arg { settings.$field = value; })*
            };
        }
        apply!(
            opening_width_deg => opening_width_deg,
            opening_height_deg => opening_height_deg,
            opening_min_width_deg => opening_min_width_deg,
            opening_fade => opening_fade,
            periphery_opacity => periphery_opacity,
            periphery_dim => periphery_dim,
            deadzone => deadzone,
            fade_in_ms => fade_in_ms,
            fade_out_delay_ms => fade_out_delay_ms,
            fade_out_ms => fade_out_ms,
            refresh_blink_ms => refresh_blink_ms,
            rest_frame_interval_ms => rest_frame_interval_ms,
            rest_frame_cross_fade_ms => rest_frame_cross_fade_ms,
            rest_frame_stillness => rest_frame_stillness_deg_per_sec,
            rest_frame_max_wait_ms => rest_frame_max_wait_ms,
            rest_frame_early_deg => rest_frame_early_refresh_deg,
            rest_frame_shell => rest_frame_shell_half_size,
            rest_frame_pose_offset_ms => rest_frame_pose_offset_ms,
            rest_frame_stream_pose_offset_ms => rest_frame_stream_pose_offset_ms,
            rest_frame_mask_prediction_ms => rest_frame_mask_prediction_ms,
            rest_frame_move_limit => rest_frame_move_limit_m,
            rest_frame_face_size => rest_frame_face_size,
            rest_frame_max_face_size => rest_frame_max_face_size,
            surround => brightness,
        );
        if let Some(mode) = self.periphery {
            settings.periphery = match mode {
                Mode::Plain => Periphery::Plain,
                Mode::Grid => Periphery::Grid,
                Mode::Game => Periphery::Game,
            };
        }
        if let Some(stick) = self.stick {
            settings.stick = match stick {
                Hand::Left => Stick::Left,
                Hand::Right => Stick::Right,
                Hand::Both => Stick::Both,
            };
        }
        if let Some(shape) = self.shape {
            settings.shape = match shape {
                Shape::Circle => OpeningShape::Circle,
                Shape::Rectangle => OpeningShape::Rectangle,
            };
        }
        if self.dynamic_fov || self.no_dynamic_fov {
            settings.dynamic_fov = !self.no_dynamic_fov;
        }
        settings.grid_over_ring |= self.grid_over_ring;
        settings.crosshair |= self.crosshair;
        settings.nose |= self.nose;
        settings.clamp();
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let options = Options::parse();
    if options.print_default_settings {
        println!("{:#}", Settings::default().to_json(SaveScope::All));
        return Ok(());
    }
    let mut settings = Settings::default();
    if let Some(path) = &options.settings {
        load(path, &mut settings)?;
    }
    if let Some(path) = options.dashboard_settings()
        && let Err(error) = load(&path, &mut settings)
    {
        eprintln!(
            "Ignoring invalid dashboard settings {}: {error}",
            path.display()
        );
    }
    options.apply(&mut settings);
    #[cfg(target_os = "linux")]
    return if options.probe_input
        || options.probe_source
        || options.probe_layout
        || options.probe_grid
        || options.probe_evidence
        || options.probe_projection
        || options.probe_tracking
        || options.probe_stream_motion
    {
        probe::run(&options, &settings)
    } else {
        session::run(options, settings)
    };
    #[cfg(not(target_os = "linux"))]
    Err("Frame Comfort requires Linux and SteamVR".into())
}
