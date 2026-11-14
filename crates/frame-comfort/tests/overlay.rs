#![cfg(target_os = "linux")]

use serde_json::{Value, json};
use std::{
    fs, io,
    net::TcpListener,
    path::PathBuf,
    process::{Child, Command},
    thread,
    time::Duration,
};

#[test]
#[ignore = "requires the Rust OpenVR fixture and a Vulkan device"]
fn plain_overlay_shows_on_stick_push_and_hides_on_release() -> Result<(), Box<dyn std::error::Error>>
{
    run_overlay("plain", None, false, false)
}

#[test]
#[ignore = "requires the Rust OpenVR fixture"]
fn geometry_probe_reports_projection_and_unpredicted_tracking()
-> Result<(), Box<dyn std::error::Error>> {
    let library = std::env::var_os("FRAME_COMFORT_FAKE_OPENVR").ok_or("missing OpenVR fixture")?;
    let temp = tempfile::tempdir()?;
    let calls = temp.path().join("calls.jsonl");
    let output = Command::new(env!("CARGO_BIN_EXE_frame-comfort"))
        .args([
            "--probe-projection",
            "--probe-tracking",
            "--seconds",
            "0.05",
            "--openvr-library",
        ])
        .arg(library)
        .env("FAKE_OPENVR_LOG", &calls)
        .output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let lines = std::str::from_utf8(&output.stdout)?
        .lines()
        .map(serde_json::from_str::<Value>)
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(lines[0]["recommendedEyeSize"], json!([1728, 1728]));
    for eye in lines[0]["projection"].as_array().unwrap() {
        for axis in ["Up", "Down"] {
            let raw = eye[format!("{}Tan", axis.to_lowercase())].as_f64().unwrap();
            let matrix = eye[format!("matrix{axis}Tan")].as_f64().unwrap();
            assert!((raw - matrix).abs() < 1e-5, "{eye}");
        }
    }
    assert!(lines[1..].iter().any(|line| line["trackingPose"] == true
        && line["predictionSeconds"] == 0
        && line["monotonicNs"].as_u64().is_some()));
    let calls = fs::read_to_string(calls)?;
    let calls = calls
        .lines()
        .map(serde_json::from_str::<Value>)
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(calls.first().unwrap()["type"], 3);
    assert!(
        calls
            .iter()
            .all(|call| matches!(call["call"].as_str(), Some("VR_Init" | "VR_Shutdown")))
    );
    Ok(())
}

#[test]
#[ignore = "requires the Rust cadence and OpenVR fixtures, source layer and Vulkan"]
fn game_overlay_uses_a_standby_snapshot() -> Result<(), Box<dyn std::error::Error>> {
    run_overlay(
        "game",
        Some(("rotate", "0", Some("local-array"))),
        false,
        false,
    )
}

#[test]
#[ignore = "requires the Rust cadence and OpenVR fixtures, source layer and Vulkan"]
fn delayed_separate_eye_source_falls_back_to_black() -> Result<(), Box<dyn std::error::Error>> {
    run_overlay("game", Some(("per-eye", "4000", None)), false, false)
}

#[test]
#[ignore = "requires the Rust cadence and OpenVR fixtures, source layer and Vulkan"]
fn game_overlay_pairs_separate_eye_rings() -> Result<(), Box<dyn std::error::Error>> {
    let run = || {
        run_overlay(
            "game",
            Some(("per-eye", "750", Some("local-slots"))),
            false,
            false,
        )
    };
    // Pairing separate eyes from write timing occasionally does not finish
    // within one run under a software Vulkan driver. Only that is retried:
    // any other failure, such as a bad audit, fails the test at once.
    match run() {
        Err(error) if error.is::<NeverVerified>() => {
            eprintln!("{error}; retrying once.");
            run()
        }
        result => result,
    }
}

#[test]
#[ignore = "requires the Rust cadence and OpenVR fixtures, source layer and Vulkan"]
fn game_overlay_layers_grid_lines_over_the_snapshot() -> Result<(), Box<dyn std::error::Error>> {
    run_overlay(
        "game",
        Some(("rotate", "0", Some("local-array"))),
        false,
        true,
    )
}

#[test]
#[ignore = "requires the Rust OpenVR fixture and a Vulkan device"]
fn dashboard_tabs_and_head_overlays_follow_openvr_events() -> Result<(), Box<dyn std::error::Error>>
{
    run_overlay("grid", None, true, false)
}

#[test]
#[ignore = "requires the Rust cadence fixture, source layer and Vulkan"]
fn source_socket_is_guarded_and_removed_on_teardown() -> Result<(), Box<dyn std::error::Error>> {
    use std::{os::unix::fs::PermissionsExt, time::Instant};

    let binary = std::env::var_os("FRAME_COMFORT_CADENCE").ok_or("set FRAME_COMFORT_CADENCE")?;
    let manifest =
        std::env::var_os("FRAME_COMFORT_LAYER_DIR").ok_or("set FRAME_COMFORT_LAYER_DIR")?;
    let temp = tempfile::tempdir()?;
    let socket = temp.path().join("source.sock");
    for (guard, enabled) in [("vrcompositor", false), ("", true)] {
        let output = fs::File::create(temp.path().join("source.stdout"))?;
        let errors = fs::File::create(temp.path().join("source.stderr"))?;
        let mut process = FixtureProcess(
            Command::new(&binary)
                .args(["1", "rotate"])
                .env("VK_ADD_LAYER_PATH", &manifest)
                .env(
                    "VK_INSTANCE_LAYERS",
                    "VK_LAYER_VRW_comfort_source:VK_LAYER_KHRONOS_validation",
                )
                .env("VRW_COMFORT_SOURCE_COMM", guard)
                .env("VRW_COMFORT_SOURCE_SOCKET", &socket)
                .stdout(output)
                .stderr(errors)
                .spawn()?,
        );
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut saw_socket = false;
        let status = loop {
            if let Ok(metadata) = socket.symlink_metadata() {
                let mode = metadata.permissions().mode() & 0o777;
                saw_socket |= mode == 0o600;
            }
            if let Some(status) = process.0.try_wait()? {
                break status;
            }
            assert!(Instant::now() < deadline, "cadence fixture did not exit");
            thread::sleep(Duration::from_millis(5));
        };
        let stdout = fs::read_to_string(temp.path().join("source.stdout"))?;
        let stderr = fs::read_to_string(temp.path().join("source.stderr"))?;
        assert!(status.success(), "{stdout}\n{stderr}");
        validation_clean(&stdout);
        validation_clean(&stderr);
        assert_eq!(saw_socket, enabled);
        assert!(!socket.exists());
    }
    Ok(())
}

/// A Game run that never showed the frozen game picture, because the source
/// verified too late or not at all. Returned before any check that depends
/// on the snapshot.
#[derive(Debug)]
struct NeverVerified;

impl std::fmt::Display for NeverVerified {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("the frozen game picture was never shown")
    }
}

impl std::error::Error for NeverVerified {}

struct FixtureProcess(Child);

impl Drop for FixtureProcess {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn run_overlay(
    mode: &str,
    cadence: Option<(&str, &str, Option<&str>)>,
    dashboard: bool,
    grid_over_ring: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let library = PathBuf::from(
        std::env::var_os("FRAME_COMFORT_FAKE_OPENVR")
            .ok_or("set FRAME_COMFORT_FAKE_OPENVR to libframe_comfort_fake_openvr.so")?,
    );
    let temp = tempfile::tempdir()?;
    let log = temp.path().join("openvr.jsonl");
    let socket = temp.path().join("source.sock");
    let status_path = temp.path().join("status.json");
    let game = cadence.is_some();
    let rule = cadence.and_then(|(_, _, rule)| rule);
    // Steps are 25 ms plus a little work each. Pairing separate eyes learns
    // from write timing and was seen to take 6 to 10 s under a software
    // driver, so that run waits up to 14 s for a standby snapshot before
    // pushing, instead of 6 s. The app runs about 3 s longer than the steps,
    // so it does not exit while the mailbox is still sending.
    let separate_eyes = matches!(cadence, Some(("per-eye", _, Some(_))));
    let (producer_seconds, game_seconds, game_steps, last_push) = if separate_eyes {
        ("28", "22", 740, 560)
    } else {
        ("20", "14", 420, 240)
    };
    let face_size = if game { "160" } else { "64" };
    let _producer = cadence
        .map(|(cadence, gap, _)| {
            let binary = std::env::var_os("FRAME_COMFORT_CADENCE").ok_or_else(|| {
                io::Error::other("set FRAME_COMFORT_CADENCE to frame-comfort-cadence")
            })?;
            let manifest = std::env::var_os("FRAME_COMFORT_LAYER_DIR").ok_or_else(|| {
                io::Error::other("set FRAME_COMFORT_LAYER_DIR to the layer manifest directory")
            })?;
            let layers = match std::env::var("VK_INSTANCE_LAYERS") {
                Ok(layers) if !layers.is_empty() => format!("VK_LAYER_VRW_comfort_source:{layers}"),
                _ => "VK_LAYER_VRW_comfort_source".into(),
            };
            Command::new(binary)
                .args([producer_seconds, cadence, gap])
                .env("VK_ADD_LAYER_PATH", manifest)
                .env("VK_INSTANCE_LAYERS", layers)
                .env("VRW_COMFORT_SOURCE_COMM", "")
                .env("VRW_COMFORT_SOURCE_SOCKET", &socket)
                .stdout(fs::File::create(temp.path().join("producer.stdout"))?)
                .stderr(fs::File::create(temp.path().join("producer.stderr"))?)
                .spawn()
                .map(FixtureProcess)
        })
        .transpose()?;
    let listener = TcpListener::bind(("127.0.0.1", 0))?;
    let port = listener.local_addr()?.port();
    let stdout = fs::File::create(temp.path().join("app.stdout"))?;
    let stderr = fs::File::create(temp.path().join("app.stderr"))?;
    let mut app = Command::new(env!("CARGO_BIN_EXE_frame-comfort"));
    if dashboard {
        app.args(["--nose", "--crosshair"])
            .env("FAKE_OPENVR_PANEL_DUMP", temp.path().join("panel"));
    }
    if grid_over_ring {
        app.arg("--grid-over-ring");
    }
    let mut child = FixtureProcess(
        app.args([
            "--periphery",
            mode,
            "--seconds",
            if game {
                game_seconds
            } else if dashboard {
                "7"
            } else {
                "3"
            },
            "--rest-frame-face-size",
            face_size,
            "--fade-in-ms",
            "0",
            "--fade-out-delay-ms",
            "0",
            "--fade-out-ms",
            "0",
            "--check-frame-code",
            "--rest-frame-interval-ms",
            "600",
            "--mailbox-port",
            &port.to_string(),
            "--openvr-library",
        ])
        .arg(library)
        .arg("--socket")
        .arg(&socket)
        .arg("--status")
        .arg(&status_path)
        .env("FAKE_OPENVR_LOG", &log)
        .env("FAKE_OPENVR_DUMP", temp.path().join("front"))
        .stdout(stdout)
        .stderr(stderr)
        .spawn()?,
    );
    listener.set_nonblocking(true)?;
    let start = std::time::Instant::now();
    let connection = loop {
        match listener.accept() {
            Ok((stream, _)) => break Ok(stream),
            Err(error)
                if error.kind() == io::ErrorKind::WouldBlock
                    && start.elapsed() < Duration::from_secs(5) =>
            {
                if let Some(status) = child.0.try_wait()? {
                    break Err(format!(
                        "Frame Comfort exited before mailbox connection: {status}"
                    ));
                }
                thread::sleep(Duration::from_millis(10));
            }
            Err(error) => break Err(error.to_string()),
        }
    };
    let stream = match connection {
        Ok(stream) => stream,
        Err(error) => {
            let _ = child.0.kill();
            child.0.wait()?;
            let stderr = fs::read_to_string(temp.path().join("app.stderr"))?;
            return Err(format!("{error}: {stderr}").into());
        }
    };
    stream.set_read_timeout(Some(Duration::from_secs(2)))?;
    stream.set_write_timeout(Some(Duration::from_secs(2)))?;
    let mut mailbox = tungstenite::accept(stream)?;
    for _ in 0..3 {
        assert!(mailbox.read()?.is_text());
    }
    let hold = if game { 80 } else { 40 };
    let mut push = if !game {
        Some(10)
    } else if rule.is_none() {
        Some(100)
    } else {
        None
    };
    for step in 0..if game {
        game_steps
    } else if dashboard {
        240
    } else {
        90
    } {
        if push.is_none() {
            let standby_ready = fs::read(&status_path)
                .ok()
                .and_then(|status| serde_json::from_slice::<Value>(&status).ok())
                .is_some_and(|status| status["restFrame"]["standbyReady"] == true);
            if standby_ready || step == last_push {
                push = Some(step);
            }
        }
        let deflection = if push.is_some_and(|start| (start..start + hold).contains(&step)) {
            0.8
        } else {
            0.0
        };
        let event = json!({"type":"update_component_states","device":"/user/hand/left",
            "components":{"/input/thumbstick/y":deflection}});
        if let Err(error) = mailbox.send(tungstenite::Message::text(event.to_string())) {
            eprintln!("Mailbox fixture stopped at step {step}: {error}");
            break;
        }
        thread::sleep(Duration::from_millis(25));
    }
    let status = child.0.wait()?;
    let stdout = fs::read_to_string(temp.path().join("app.stdout"))?;
    let stderr = fs::read_to_string(temp.path().join("app.stderr"))?;
    assert!(status.success(), "{stderr}\n{stdout}");
    validation_clean(&stdout);
    validation_clean(&stderr);
    if game {
        for name in ["producer.stdout", "producer.stderr"] {
            validation_clean(&fs::read_to_string(temp.path().join(name))?);
        }
        // A movement that starts before verification stays on the black
        // fallback until release, even if the source verifies meanwhile.
        if rule.is_some()
            && !stdout
                .lines()
                .filter_map(|line| serde_json::from_str::<Value>(line).ok())
                .any(|state| {
                    state["effectivePeriphery"] == "game" && state["restFrame"]["visible"] == true
                })
        {
            eprintln!("{stdout}");
            return Err(NeverVerified.into());
        }
    }
    let entries: Vec<Value> = fs::read_to_string(log)?
        .lines()
        .map(serde_json::from_str)
        .collect::<serde_json::Result<_>>()?;
    let mut show = None;
    let mut hide = None;
    for (index, entry) in entries.iter().enumerate() {
        if !entry["key"]
            .as_str()
            .is_some_and(|key| key.contains(".restframe."))
        {
            continue;
        }
        match entry["call"].as_str() {
            Some("ShowOverlay") => {
                show.get_or_insert(index);
            }
            Some("HideOverlay") if show.is_some() => {
                hide.get_or_insert(index);
            }
            _ => {}
        }
    }
    assert!(
        show.is_some() && hide > show,
        "expected the shell to show and then hide: {}",
        stdout
    );
    let side = face_size.parse::<usize>()?;
    let width = 2 * side;
    assert!(
        entries
            .iter()
            .any(|entry| entry["call"] == "SetOverlayTexture"
                && entry["w"] == width
                && entry["h"] == side)
    );
    let face = fs::read(temp.path().join(format!("front-30-{width}x{side}.rgba")))
        .map_err(|error| format!("Missing fixture image: {error}\n{stdout}"))?;
    for eye in 0..2 {
        let center = (side / 2 * width + side / 2 + eye * side) * 4;
        assert_eq!(face[center + 3], 0);
        if rule.is_some() {
            let pixel = (side / 4 * width + side / 4 + eye * side) * 4;
            assert!(
                face[pixel + eye * 2] > face[pixel + (1 - eye) * 2],
                "{stdout}"
            );
        }
        if rule.is_some() {
            // The default grid has a room-fixed line a quarter of the way
            // across the 8 m face. Every channel of the fixture's picture
            // differs, so its darkest channel stays low except where the grey
            // line is layered over it. The line must be there in every frame,
            // not just in the refresh blink.
            let darkest = |x: usize| {
                let pixel = ((side * 3 / 8) * width + x + eye * side) * 4;
                *face[pixel..pixel + 3].iter().min().expect("three channels")
            };
            let (line, beside) = (
                darkest(side / 4),
                darkest(side / 4 + 4).max(darkest(side / 4 - 5)),
            );
            assert!(beside <= 48, "eye {eye}: picture {beside}\n{stdout}");
            assert_eq!(
                line >= 80,
                grid_over_ring,
                "eye {eye}: line {line}\n{stdout}"
            );
        }
        if game && rule.is_none() {
            // An unverified Game movement is black like Plain, not the Grid
            // mode's shade. Sample between the default grid lines.
            let pixel = ((side / 8) * width + side / 8 + eye * side) * 4;
            assert_eq!(face[pixel + 3], 255, "eye {eye}\n{stdout}");
            assert!(
                face[pixel..pixel + 3].iter().all(|&value| value <= 8),
                "eye {eye}: {:?}\n{stdout}",
                &face[pixel..pixel + 4]
            );
        }
    }
    if dashboard {
        let panels = ["opening", "timing", "extras"]
            .map(|tab| fs::read(temp.path().join(format!("panel-{tab}-1024x640.rgba"))));
        let [opening, timing, extras] = panels;
        let (opening, timing, extras) = (opening?, timing?, extras?);
        assert_eq!(opening.len(), 1024 * 640 * 4);
        assert_ne!(opening, timing);
        assert_ne!(timing, extras);
        // The header bar's surface colour, top-left up: no flip or swizzle.
        let at = (20 * 1024 + 20) * 4;
        assert_eq!(opening[at..at + 4], [29, 41, 56, 255]);
        // The panel alternates GPU images. Raw uploads flickered on changes.
        let controls = "org.vrwindow.comfort.controls";
        assert!(
            entries
                .iter()
                .any(|e| e["call"] == "SetOverlayTexture" && e["key"] == controls)
        );
        assert!(
            !entries
                .iter()
                .any(|e| e["call"] == "SetOverlayRaw" && e["key"] == controls),
            "the panel used a raw upload"
        );

        for (overlay, width, height) in [("nose", 640, 320), ("crosshair", 256, 256)] {
            let calls = entries
                .iter()
                .filter(|event| {
                    event["key"]
                        .as_str()
                        .is_some_and(|key| key.contains(overlay))
                })
                .collect::<Vec<_>>();
            assert!(calls.iter().any(|event| event["call"] == "SetOverlayRaw"
                && event["w"] == width
                && event["h"] == height));
            assert!(calls.iter().any(|event| event["call"]
                == "SetOverlayTransformTrackedDeviceRelative"
                && event["device"] == 0));
            let visibility = calls
                .iter()
                .filter_map(|event| match event["call"].as_str() {
                    Some("ShowOverlay") => Some(true),
                    Some("HideOverlay") => Some(false),
                    _ => None,
                })
                .collect::<Vec<_>>();
            assert!(
                visibility.starts_with(&[true, false, true]),
                "{overlay}: {visibility:?}"
            );
        }
    }
    if game {
        let states: Vec<Value> = stdout
            .lines()
            .map(serde_json::from_str)
            .collect::<serde_json::Result<_>>()?;
        let (audits, statuses): (Vec<_>, Vec<_>) = states
            .iter()
            .partition(|state| state.get("snapshotAudit").is_some());
        if let Some(rule) = rule {
            assert!(
                statuses
                    .iter()
                    .any(|state| state["gameSource"] == "verified"
                        && state["restFrame"]["rule"] == rule
                        && state["restFrame"]["standbyReady"] == true),
                "{stdout}"
            );
            if rule == "local-array" {
                assert!(
                    statuses
                        .iter()
                        .any(|state| state["restFrame"]["standbyStarts"] == 1),
                    "{stdout}"
                );
            }
            assert!(
                audits.len() >= 2 && audits.iter().all(|audit| audit["snapshotAudit"] == true),
                "{stdout}"
            );
            let last_status = states
                .iter()
                .rposition(|state| state.get("restFrame").is_some())
                .unwrap();
            let captures = states[..last_status]
                .iter()
                .filter(|state| state.get("snapshotAudit").is_some())
                .count();
            assert_eq!(states[last_status]["restFrame"]["captures"], captures);
        } else {
            assert!(
                statuses.iter().any(|state| state["images"] == 6),
                "{stdout}"
            );
            assert!(
                audits.is_empty()
                    && statuses
                        .iter()
                        .all(|state| state["gameSource"] != "verified"),
                "{stdout}"
            );
            // A 4 ms gap can pass the timing check, so once the images arrive
            // the app keeps re-checking the pairs or reports the failed
            // attempt. It never reports a selection reason or readiness.
            let learning = statuses
                .iter()
                .skip_while(|state| state["gameSource"] != "learning")
                .collect::<Vec<_>>();
            assert!(
                !learning.is_empty()
                    && learning.iter().all(|state| matches!(
                        state["gameSource"].as_str(),
                        Some("learning" | "separate-eye-pairing-unverified")
                    )),
                "{stdout}"
            );
        }
    }
    Ok(())
}

fn validation_clean(output: &str) {
    let messages: Vec<_> = output
        .lines()
        .filter(|line| line.contains("Validation Error") || line.contains("Validation Warning"))
        .collect();
    assert!(messages.is_empty(), "{}", messages.join("\n"));
}
