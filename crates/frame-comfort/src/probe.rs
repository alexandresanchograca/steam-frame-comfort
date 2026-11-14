use super::{Options, game::Game};
use ash::vk;
use comfort_core::{
    geometry::Projection,
    panel,
    settings::{Settings, Stick},
    stereo::{Packing, StereoFeatures, stereo_shift},
};
use comfort_runtime::{
    mailbox::{Mailbox, Options as MailboxOptions, monotonic_ns},
    openvr::OpenVr,
    source_client::SourceClient,
    vulkan::{Context, ContextOptions},
};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    error::Error,
    ffi::OsString,
    fs::OpenOptions,
    io::{self, Write},
    os::unix::fs::OpenOptionsExt,
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

type Result<T> = std::result::Result<T, Box<dyn Error>>;
const INTERVAL: Duration = Duration::from_millis(10);
const FULL: [f32; 4] = [0.0, 0.0, 1.0, 1.0];

pub(super) fn run(options: &Options, settings: &Settings) -> Result<()> {
    let stop = Arc::new(AtomicBool::new(false));
    for signal in [signal_hook::consts::SIGINT, signal_hook::consts::SIGTERM] {
        signal_hook::flag::register(signal, Arc::clone(&stop))?;
    }
    let seconds = options
        .seconds
        .filter(|limit| !limit.is_zero())
        .unwrap_or(Duration::from_secs(5));
    if options.probe_stream_motion {
        return stream_motion(options, seconds, &stop);
    }
    if options.probe_projection || options.probe_tracking {
        geometry(options, seconds, &stop)?;
    }
    if options.probe_input {
        input(options, settings, seconds, &stop)
    } else if options.probe_source
        || options.probe_layout
        || options.probe_grid
        || options.probe_evidence
    {
        source(options, seconds, &stop)
    } else {
        Ok(())
    }
}

fn geometry(options: &Options, seconds: Duration, stop: &AtomicBool) -> Result<()> {
    // SAFETY: Uses the installed OpenVR client or a pinned ABI fixture.
    let vr = unsafe { OpenVr::init_background(&options.openvr_library)? };
    if options.probe_projection {
        let mut eyes = Vec::new();
        for index in 0..2 {
            let eye = vr.raw_eye_geometry(index)?;
            let [left, right, top, bottom] = eye.tangents.map(f64::from);
            let projection = Projection::from_openvr_raw(left, right, top, bottom);
            let row = vr.projection_matrix(index)?[1].map(f64::from);
            eyes.push(json!({
                "eye": (["left", "right"][index as usize]),
                "left": left, "right": right, "top": top, "bottom": bottom,
                "u0": -left / (right - left), "v0": -projection.top / (projection.bottom - projection.top),
                "eyeX": eye.eye_to_head[0][3], "eyeToHead": eye.eye_to_head,
                "matrixRow1": row, "matrixUpTan": (1.0 + row[2]) / row[1], "matrixDownTan": (1.0 - row[2]) / row[1],
                "upTan": -projection.top, "downTan": projection.bottom,
            }));
        }
        emit(&json!({"recommendedEyeSize": vr.recommended_size()?, "projection": eyes}))?;
    }
    if options.probe_tracking {
        let start = Instant::now();
        while running(stop, start, seconds.min(Duration::from_secs(120))) {
            let before = monotonic_ns();
            let pose = vr.raw_tracking(0.0)?;
            let duration = monotonic_ns().saturating_sub(before);
            emit(&json!({
                "trackingPose": true, "monotonicNs": before + duration / 2, "queryDurationNs": duration,
                "origin": "standing", "predictionSeconds": 0,
                "valid": pose.bPoseIsValid, "connected": pose.bDeviceIsConnected, "trackingResult": pose.eTrackingResult,
                "headToStanding": pose.mDeviceToAbsoluteTracking.m,
                "angularVelocityRadPerSecond": pose.vAngularVelocity.v,
                "velocityMetresPerSecond": pose.vVelocity.v,
            }))?;
            thread::sleep(INTERVAL);
        }
    }
    Ok(())
}

fn input(
    options: &Options,
    settings: &Settings,
    seconds: Duration,
    stop: &AtomicBool,
) -> Result<()> {
    let mailbox = Mailbox::start(MailboxOptions {
        port: options.mailbox_port,
        ..Default::default()
    })?;
    let start = Instant::now();
    let mut next_log = Duration::ZERO;
    let mut samples = 0;
    let mut deflected = 0;
    let mut max = 0.0_f32;
    while running(stop, start, seconds) {
        let state = mailbox.snapshot();
        let hands = state.read(
            settings.stick,
            settings.deadzone,
            Duration::from_millis(1500),
            monotonic_ns(),
        );
        let magnitude = hands.map_or(0.0, |[left, right]| match settings.stick {
            Stick::Left => left,
            Stick::Right => right,
            Stick::Both => left.max(right),
        });
        samples += 1;
        deflected += u64::from(magnitude > settings.deadzone);
        max = max.max(magnitude);
        if start.elapsed() >= next_log {
            let [left, right] = state.hands.map(|hand| [hand.x, hand.y]);
            emit(&json!({
                "connected": state.connected, "available": hands.is_some(), "magnitude": magnitude,
                "left": left, "right": right, "messages": state.messages, "error": state.error,
            }))?;
            next_log = start.elapsed() + Duration::from_millis(250);
        }
        thread::sleep(INTERVAL);
    }
    emit(
        &json!({"summary": true, "maxMagnitude": max, "deflectedSamples": deflected, "samples": samples}),
    )
}

/// Connect to the observer and open a Vulkan device on its GPU.
fn open_source(
    options: &Options,
    seconds: Duration,
    stop: &AtomicBool,
    start: Instant,
) -> Result<Game> {
    let path = options.source_socket();
    let mut client = SourceClient::default();
    let context = loop {
        if !running(stop, start, seconds) {
            return Err(
                io::Error::other(format!("No observer report from {}", path.display())).into(),
            );
        }
        if !client.connected() && client.connect(&path).is_err() {
            thread::sleep(Duration::from_millis(100));
            continue;
        }
        if client.poll(monotonic_ns()).is_err() {
            thread::sleep(Duration::from_millis(100));
            continue;
        }
        if let (Some(hello), Some(_)) = (client.hello, client.frame) {
            let context_options = ContextOptions {
                device_uuid: Some(hello.device_uuid),
                high_priority: !options.no_high_priority_queue,
                ..Default::default()
            };
            let context = Context::new(ContextOptions {
                require_dma_buf: true,
                ..context_options
            })
            .or_else(|_| Context::new(context_options))?;
            if context.driver_uuid() != hello.driver_uuid {
                return Err(io::Error::other("Vulkan driver differs from observer").into());
            }
            break context;
        }
        thread::sleep(INTERVAL);
    };
    Game::with_source(context, false, client, options.source_format)
}

fn source(options: &Options, seconds: Duration, stop: &AtomicBool) -> Result<()> {
    let start = Instant::now();
    let path = options.source_socket();
    let mut game = open_source(options, seconds, stop, start)?;
    let mut next_log = Duration::ZERO;
    let mut next_dump = Duration::ZERO;
    let mut dump_sets = 0;
    let mut trace = StreamTrace::default();
    let mut frames = 0;
    let mut previous = None;
    let mut last_error = String::new();
    while running(stop, start, seconds) {
        let now = monotonic_ns();
        if options.probe_evidence {
            if let Err(error) = game.poll(
                &path,
                now,
                false,
                true,
                &[[panel::WIDTH as u32, panel::HEIGHT as u32]],
                false,
            ) {
                last_error = error.to_string();
                eprintln!("Evidence: {error}");
            }
            if start.elapsed() >= next_log {
                emit(&game.evidence_report(now))?;
                next_log = start.elapsed() + Duration::from_secs(1);
            }
            if dump_sets < 4
                && !game.evidence_bursting()
                && start.elapsed() >= next_dump
                && let Some(prefix) = &options.probe_dump
            {
                if dump_evidence(&mut game, prefix, dump_sets)? {
                    dump_sets += 1;
                }
                next_dump = start.elapsed() + Duration::from_millis(1500);
            }
            thread::sleep(game.learning_delay().max(Duration::from_micros(100)));
            continue;
        }
        if let Err(error) = game.poll_source(&path, now) {
            last_error = error.to_string();
        }
        if let Some(frame) = game.source_client().frame
            && previous != Some(frame.sequence)
        {
            previous = Some(frame.sequence);
            frames += 1;
            if options.probe_source && start.elapsed() >= next_log {
                let reports = (0..frame.count as usize)
                    .map(|i| (frame.ids[i], frame.layouts[i], false))
                    .chain(
                        (0..frame.touched_count as usize)
                            .map(|i| (frame.touched_ids[i], frame.touched_layouts[i], true)),
                    );
                let images = reports
                    .map(|(id, layout, touched)| image_report(&mut game, id, layout, touched))
                    .collect::<Vec<_>>();
                emit(
                    &json!({"frame": frame.sequence, "count": frame.count, "touchedCount": frame.touched_count,
                    "images": images, "sourceFrames": frames}),
                )?;
                next_log = start.elapsed() + Duration::from_millis(500);
            } else if !options.probe_source && trace.measured < 400 {
                let ids = &frame.ids[..frame.count as usize];
                let p010: Vec<_> = ids
                    .iter()
                    .copied()
                    .filter(|id| {
                        game.source_client()
                            .images
                            .get(id)
                            .is_some_and(|image| image.description.format == P010)
                    })
                    .collect();
                if trace.sequence.len() < 48 {
                    trace.sequence.push(if p010.is_empty() {
                        "-".into()
                    } else {
                        p010.iter()
                            .map(u64::to_string)
                            .collect::<Vec<_>>()
                            .join("+")
                    });
                }
                for id in p010 {
                    match game.probe_pixels(id, 0, [FULL; 2], 128) {
                        Ok(pixels) => trace.observe(id, &pixels, options)?,
                        Err(error) => last_error = error.to_string(),
                    }
                    if trace.measured >= 400 {
                        break;
                    }
                }
            }
        } else if !game.source_client().connected() {
            previous = None;
        }
        thread::sleep(INTERVAL);
    }
    if options.probe_evidence {
        return emit(&game.evidence_report(monotonic_ns()));
    }
    let client = game.source_client();
    if !options.probe_source {
        return trace.finish(client, &last_error);
    }
    let mut shapes = BTreeMap::<String, Vec<u64>>::new();
    for (&id, image) in &client.images {
        let d = image.description;
        shapes
            .entry(format!(
                "{}:{}x{}:L{}:h{}:t{}:e{}",
                d.format, d.width, d.height, d.array_layers, d.handle_type, d.tiling, d.exported
            ))
            .or_default()
            .push(id);
    }
    emit(&json!({"publishedShapes": shapes}))?;
    emit(
        &json!({"summary": true, "connected": u8::from(client.connected()), "frames": frames,
        "images": client.images.len(), "hello": u8::from(client.hello.is_some()),
        "sameDevice": u8::from(game.same_device()), "lastError": last_error}),
    )
}

fn image_report(game: &mut Game, id: u64, layout: u32, touched: bool) -> Value {
    let mut value = json!({"id": id, "layout": layout, "touched": u8::from(touched)});
    let Some(d) = game
        .source_client()
        .images
        .get(&id)
        .map(|image| image.description)
    else {
        return value;
    };
    value = json!({
        "id": id, "layout": layout, "touched": u8::from(touched), "format": d.format,
        "size": [d.width, d.height], "layers": d.array_layers, "handle": d.handle_type,
        "modifier": d.modifier, "inode": d.inode, "usage": d.usage, "flags": d.flags,
        "tiling": d.tiling, "mips": d.mip_levels, "memType": d.memory_type_index,
        "alloc": d.allocation_size, "exported": d.exported,
    });
    match game.probe_pixels(id, 0, [FULL; 2], 32) {
        Ok(pixels) => {
            for (name, x, y, w, h) in [
                ("mean", 0, 0, 32, 32),
                ("top", 0, 0, 32, 8),
                ("left", 0, 0, 8, 32),
                ("right", 24, 0, 8, 32),
                ("bottom", 0, 24, 32, 8),
                ("leftHalf", 0, 0, 16, 32),
                ("rightHalf", 16, 0, 16, 32),
                ("topHalf", 0, 0, 32, 16),
                ("bottomHalf", 0, 16, 32, 16),
            ] {
                value[name] = json!(mean_rgb(&pixels, 32, x, y, w, h));
            }
        }
        Err(error) => value["renderError"] = json!(error.to_string()),
    }
    value
}

fn mean_rgb(
    pixels: &[u8],
    side: usize,
    x: usize,
    y: usize,
    width: usize,
    height: usize,
) -> [f64; 3] {
    let mut sum = [0u32; 3];
    for row in pixels.chunks_exact(side * 8).skip(y).take(height) {
        for pixel in row.chunks_exact(4).skip(x).take(width) {
            for (channel, &value) in sum.iter_mut().zip(pixel) {
                *channel += u32::from(value);
            }
        }
    }
    sum.map(|v| f64::from(v) / (width * height) as f64)
}

const P010: u32 = vk::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16.as_raw() as u32;

#[derive(Default)]
struct StreamTrace {
    sequence: Vec<String>,
    shifts: Vec<f64>,
    per_image: BTreeMap<u64, [f64; 4]>,
    measured: u32,
    mean: f64,
    deviation: f64,
    differences: [f64; 3],
}

impl StreamTrace {
    fn observe(&mut self, id: u64, pixels: &[u8], options: &Options) -> Result<()> {
        let picture = unpack(pixels, 128);
        let luma = |x: usize, y: usize| f64::from(picture[y * 128 + x]);
        let count = picture.len() as f64;
        let mean = picture.iter().map(|&v| f64::from(v)).sum::<f64>() / count;
        let deviation = (picture
            .iter()
            .map(|&v| (f64::from(v) - mean).powi(2))
            .sum::<f64>()
            / count)
            .sqrt();
        if deviation < 2.0 {
            return Ok(());
        }
        if self.measured == 0 {
            if options.probe_grid {
                let grid = |offset: usize| {
                    (0..1024)
                        .map(|i| {
                            let (x, y) = (i % 32 * 2, offset + i / 32 * 2);
                            (luma(x, y) + luma(x + 1, y) + luma(x, y + 1) + luma(x + 1, y + 1))
                                / 4.0
                        })
                        .collect::<Vec<_>>()
                };
                emit(
                    &json!({"source": "frame-stream", "id": id, "quadrants": {"top": grid(0), "bottom": grid(64)}}),
                )?;
            }
            if let Some(path) = &options.probe_dump {
                let rgb: Vec<_> = pixels
                    .chunks_exact(128 * 8)
                    .flat_map(|row| {
                        row[..128 * 4]
                            .chunks_exact(4)
                            .flat_map(|pixel| pixel[..3].iter().copied())
                    })
                    .collect();
                write_image(path, "P6", 128, &rgb)?;
            }
        }
        let thumb = |y| std::array::from_fn(|i| picture[(y + i / 64) * 128 + i % 64]);
        let shift = stereo_shift(
            &StereoFeatures::new(&thumb(0)),
            &StereoFeatures::new(&thumb(64)),
        );
        if shift.textured && shift.score > 0.5 {
            self.shifts.push(shift.fraction);
        }
        for y in 0..64 {
            for x in 0..64 {
                self.differences[0] += ((luma(x, y) - luma(x + 64, y)).abs()
                    + (luma(x, y + 64) - luma(x + 64, y + 64)).abs())
                    / 8192.0;
                self.differences[1] += ((luma(x, y) - luma(x, y + 64)).abs()
                    + (luma(x + 64, y) - luma(x + 64, y + 64)).abs())
                    / 8192.0;
                self.differences[2] += (luma(x, y) - luma(127 - x, 127 - y)).abs() / 4096.0;
            }
        }
        let totals = self.per_image.entry(id).or_default();
        for (total, sample) in totals.iter_mut().zip(mean_rgb(pixels, 128, 0, 0, 128, 128)) {
            *total += sample;
        }
        totals[3] += 1.0;
        self.measured += 1;
        self.mean += mean;
        self.deviation += deviation;
        Ok(())
    }

    fn finish(mut self, client: &SourceClient, error: &str) -> Result<()> {
        self.shifts.sort_by(f64::total_cmp);
        let n = self.shifts.len();
        if n > 0 {
            emit(
                &json!({"rowShift": {"frames": n, "medianFraction": self.shifts[n/2],
                "p10": self.shifts[n/10], "p90": self.shifts[n*9/10],
                "topRight": self.shifts.iter().filter(|&&v| v > 0.0).count(),
                "topLeft": self.shifts.iter().filter(|&&v| v < 0.0).count() }}),
            )?;
        }
        let images = self
            .per_image
            .iter()
            .map(|(id, [r, g, b, n])| (id.to_string(), [r / n, g / n, b / n, *n]))
            .collect::<BTreeMap<_, _>>();
        let samples = f64::from(self.measured.max(1));
        emit(
            &json!({"perImageMeanRgb": images, "perFrameStreamImages": self.sequence,
            "publishedStreamImages": client.images.values().filter(|image| image.description.format == P010).count(),
            "acquiredStreamImages": images.len(), "measured": self.measured, "meanLuma": self.mean / samples,
            "lumaDeviation": self.deviation / samples, "leftRightDiff": self.differences[0] / samples,
            "topBottomDiff": self.differences[1] / samples, "unrelatedDiff": self.differences[2] / samples, "lastError": error}),
        )
    }
}

/// Strip width and row count saved per eye by `--probe-stream-motion`.
const MOTION_WIDTH: usize = 256;
const MOTION_ROWS: usize = 32;

/// Record, for every report that acquires a streamed P010 frame, a strip
/// across the middle of each eye's full view with the report time, and log
/// unpredicted HMD poses on the same clock. Matching the strips' sideways
/// motion with the head's yaw gives the delay between the head orientation
/// a frame shows and its report, which `restFrameStreamPoseOffsetMs` sets.
///
/// The file holds the header `FCSTRIP1`, then little-endian u32 width and
/// row count, then per frame: u64 report time (monotonic ns), u64 report
/// sequence, u64 image ID and both eyes' luma strips, left eye first. Each
/// strip covers the middle half of the eye's width and middle quarter of
/// its height. Poses are JSON lines on stdout.
fn stream_motion(options: &Options, seconds: Duration, stop: &AtomicBool) -> Result<()> {
    let output = options
        .probe_dump
        .as_ref()
        .ok_or_else(|| io::Error::other("--probe-stream-motion needs --probe-dump FILE"))?;
    // SAFETY: Uses the installed OpenVR client or a pinned ABI fixture.
    let vr = unsafe { OpenVr::init_background(&options.openvr_library)? };
    let start = Instant::now();
    let seconds = seconds.min(Duration::from_secs(120));
    let mut game = open_source(options, seconds, stop, start)?;
    // Raw left, right, top, bottom tangents, for converting strip pixels to
    // angles in the analysis.
    let eyes = [vr.raw_eye_geometry(0)?, vr.raw_eye_geometry(1)?].map(|eye| eye.tangents);
    emit(&json!({"eyeTangents": eyes}))?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(output)?;
    file.write_all(b"FCSTRIP1")?;
    file.write_all(&(MOTION_WIDTH as u32).to_le_bytes())?;
    file.write_all(&(MOTION_ROWS as u32).to_le_bytes())?;
    let crops = Packing::LeftQuadrants.eye_regions().map(|region| {
        let [x, y, w, h] = region.crop;
        [x + 0.25 * w, y + 0.375 * h, 0.5 * w, 0.25 * h]
    });
    let path = options.source_socket();
    let (mut previous, mut frames, mut poses) = (None, 0_u64, 0_u64);
    let mut last_pose_ns = 0;
    let mut last_error = String::new();
    while running(stop, start, seconds) {
        let before = monotonic_ns();
        if before.saturating_sub(last_pose_ns) >= 4_000_000 {
            let pose = vr.raw_tracking(0.0)?;
            let after = monotonic_ns();
            if pose.bPoseIsValid {
                last_pose_ns = before;
                poses += 1;
                emit(&json!({"pose": before + (after - before) / 2,
                    "m": pose.mDeviceToAbsoluteTracking.m}))?;
            }
        }
        if let Err(error) = game.poll_source(&path, monotonic_ns()) {
            last_error = error.to_string();
        }
        if let Some(frame) = game.source_client().frame
            && previous != Some(frame.sequence)
        {
            previous = Some(frame.sequence);
            let stream = frame.ids[..frame.count as usize]
                .iter()
                .copied()
                .find(|id| {
                    game.source_client().images.get(id).is_some_and(|image| {
                        let d = image.description;
                        d.format == P010 && d.width == d.height
                    })
                });
            if let Some(id) = stream {
                match game.probe_pixels(id, 0, crops, MOTION_WIDTH as u32) {
                    Ok(pixels) => {
                        file.write_all(&frame.monotonic_ns.to_le_bytes())?;
                        file.write_all(&frame.sequence.to_le_bytes())?;
                        file.write_all(&id.to_le_bytes())?;
                        file.write_all(&motion_strips(&pixels))?;
                        frames += 1;
                    }
                    Err(error) => last_error = error.to_string(),
                }
            }
        }
        thread::sleep(Duration::from_millis(1));
    }
    file.flush()?;
    emit(
        &json!({"summary": true, "streamFrames": frames, "poses": poses,
        "seconds": start.elapsed().as_secs_f64(), "lastError": last_error}),
    )
}

/// Both eyes' strips from a side-by-side RGBA readback, as luma rows
/// averaged down to the saved row count.
fn motion_strips(pixels: &[u8]) -> Vec<u8> {
    let side = MOTION_WIDTH;
    let group = side / MOTION_ROWS;
    let mut strips = vec![0_u8; 2 * MOTION_ROWS * side];
    for eye in 0..2 {
        for row in 0..MOTION_ROWS {
            for column in 0..side {
                let sum: u32 = (0..group)
                    .map(|offset| {
                        let at = ((row * group + offset) * 2 * side + eye * side + column) * 4;
                        let rgba = &pixels[at..at + 4];
                        (54 * u32::from(rgba[0])
                            + 183 * u32::from(rgba[1])
                            + 19 * u32::from(rgba[2])
                            + 128)
                            >> 8
                    })
                    .sum();
                strips[(eye * MOTION_ROWS + row) * side + column] = (sum / group as u32) as u8;
            }
        }
    }
    strips
}

fn unpack(pixels: &[u8], side: usize) -> Vec<u8> {
    pixels
        .chunks_exact(side * 8)
        .flat_map(|row| {
            row[..side * 4].chunks_exact(4).map(|rgba| {
                ((54 * u32::from(rgba[0])
                    + 183 * u32::from(rgba[1])
                    + 19 * u32::from(rgba[2])
                    + 128)
                    >> 8) as u8
            })
        })
        .collect()
}

fn dump_evidence(game: &mut Game, prefix: &Path, set: u32) -> Result<bool> {
    let client = game.source_client();
    let Some(frame) = client.frame else {
        return Ok(false);
    };
    let ids: BTreeSet<_> = frame.ids[..frame.count as usize]
        .iter()
        .chain(&frame.touched_ids[..frame.touched_count as usize])
        .copied()
        .collect();
    let images: Vec<_> = ids
        .into_iter()
        .filter_map(|id| {
            client
                .images
                .get(&id)
                .map(|i| (id, i.description.array_layers.min(2)))
        })
        .collect();
    let mut wrote = false;
    for (id, layers) in images {
        for layer in 0..layers {
            let Ok(thumb) = game.probe_pixels(id, layer, [FULL; 2], 64) else {
                continue;
            };
            let suffix = if layers > 1 {
                format!("-{set}-id{id}-layer{layer}")
            } else {
                format!("-{set}-id{id}")
            };
            let path = |end| {
                let mut name = OsString::from(prefix);
                name.push(&suffix);
                name.push(end);
                PathBuf::from(name)
            };
            write_image(&path("-thumb.pgm"), "P5", 64, &unpack(&thumb, 64))?;
            wrote = true;
            if let Ok(full) = game.probe_pixels(id, layer, [FULL; 2], 512) {
                write_image(&path(".pgm"), "P5", 512, &unpack(&full, 512))?;
            }
        }
    }
    Ok(wrote)
}

fn write_image(path: &Path, magic: &str, side: usize, pixels: &[u8]) -> io::Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)?;
    write!(file, "{magic}\n{side} {side}\n255\n")?;
    file.write_all(pixels)
}

fn emit(value: &Value) -> Result<()> {
    let mut stdout = io::stdout().lock();
    serde_json::to_writer(&mut stdout, value)?;
    writeln!(stdout)?;
    stdout.flush()?;
    Ok(())
}

fn running(stop: &AtomicBool, start: Instant, duration: Duration) -> bool {
    !stop.load(Ordering::Relaxed) && start.elapsed() < duration
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn motion_strips_keep_each_eye_and_average_rows() {
        let side = MOTION_WIDTH;
        let mut pixels = vec![0_u8; side * 2 * side * 4];
        for row in 0..side {
            for column in 0..2 * side {
                let at = (row * 2 * side + column) * 4;
                let value = if column < side {
                    (column % 256) as u8
                } else {
                    200
                };
                let value = if column < side && row % 2 == 1 {
                    value / 2
                } else {
                    value
                };
                pixels[at..at + 4].copy_from_slice(&[value, value, value, 255]);
            }
        }
        let strips = motion_strips(&pixels);
        assert_eq!(strips.len(), 2 * MOTION_ROWS * side);
        // Left eye: rows alternate full and half brightness, so averaging 8
        // rows gives three quarters of the column's value.
        assert_eq!(strips[100], (100 * 3 / 4) as u8);
        assert_eq!(strips[MOTION_ROWS * side + 7], 200);
    }
}
