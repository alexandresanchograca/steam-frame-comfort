//! Build the fixtures and run the Vulkan end-to-end tests in one command.
use serde_json::json;
use std::{
    env,
    ffi::OsString,
    fs, io,
    path::{Path, PathBuf},
    process::Command,
};

const FEATURES: &str = "frame-comfort-fixtures/vulkan-fixtures";

fn run_step(command: &mut Command) -> io::Result<()> {
    eprintln!("+ {command:?}");
    let status = command.status()?;
    if status.success() {
        Ok(())
    } else {
        Err(io::Error::other(format!("step failed ({status})")))
    }
}

/// Runs the shell, imported source, and fake-runtime overlay tests.
/// Extra arguments are passed to each suite as test name filters.
pub fn e2e(filters: &[String]) -> io::Result<()> {
    if !cfg!(target_os = "linux") {
        return Err(io::Error::other(
            "the end-to-end tests need Linux with a Vulkan device",
        ));
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()?;
    let cargo = env::var_os("CARGO").unwrap_or_else(|| OsString::from("cargo"));
    // Cargo resolves a relative CARGO_TARGET_DIR against the directory it
    // was started in, which `cargo run` keeps as this process's.
    let target = match env::var_os("CARGO_TARGET_DIR").map(PathBuf::from) {
        Some(path) if path.is_relative() => env::current_dir()?.join(path),
        Some(path) => path,
        None => root.join("target"),
    };
    let debug = target.join("debug");
    run_step(
        Command::new(&cargo)
            .current_dir(&root)
            .env("CARGO_TARGET_DIR", &target)
            .args(["build", "--locked", "-p", "frame-comfort"])
            .args(["-p", "comfort-source-layer", "-p", "frame-comfort-fixtures"])
            .args(["--features", FEATURES]),
    )?;

    // The observer layer, found through VK_ADD_LAYER_PATH by the producer.
    let layers = target.join("test-layer");
    fs::create_dir_all(&layers)?;
    let manifest = json!({"file_format_version": "1.2.0", "layer": {
        "name": "VK_LAYER_VRW_comfort_source", "type": "GLOBAL",
        "library_path": debug.join("libvrw_comfort_source.so"), "api_version": "1.3.0",
        "implementation_version": "1", "description": "Frame Comfort test observer",
        "functions": {"vkNegotiateLoaderLayerInterfaceVersion": "vkNegotiateLoaderLayerInterfaceVersion"}
    }});
    fs::write(
        layers.join("VK_LAYER_VRW_comfort_source.json"),
        serde_json::to_string_pretty(&manifest)? + "\n",
    )?;

    let validation = env::var_os("VK_INSTANCE_LAYERS")
        .unwrap_or_else(|| OsString::from("VK_LAYER_KHRONOS_validation"));
    let test = |package: &str, target_args: &[&str], sync_validation: bool| {
        let mut command = Command::new(&cargo);
        command
            .current_dir(&root)
            .env("CARGO_TARGET_DIR", &target)
            .env(
                "FRAME_COMFORT_FAKE_OPENVR",
                debug.join("libframe_comfort_fake_openvr.so"),
            )
            .env("FRAME_COMFORT_CADENCE", debug.join("frame-comfort-cadence"))
            .env("FRAME_COMFORT_LAYER_DIR", &layers)
            .env("VK_INSTANCE_LAYERS", &validation)
            .args(["test", "--locked", "-p", package])
            .args(target_args)
            .args(["--", "--ignored", "--nocapture", "--test-threads=1"])
            .args(filters);
        if sync_validation && env::var_os("VK_KHRONOS_VALIDATION_VALIDATE_SYNC").is_none() {
            command.env("VK_KHRONOS_VALIDATION_VALIDATE_SYNC", "true");
        }
        if env::var_os("LP_NUM_THREADS").is_none() {
            // Keeps Mesa lavapipe from starving the timing fixtures.
            command.env("LP_NUM_THREADS", "2");
        }
        // Validation messages from Vulkan can be written directly to the
        // process streams while the test harness still returns success.
        eprintln!("+ {command:?}");
        let output = command.output()?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        print!("{stdout}");
        eprint!("{stderr}");
        if !output.status.success()
            || [stdout.as_ref(), stderr.as_ref()]
                .into_iter()
                .any(|output| {
                    output.contains("Validation Error") || output.contains("Validation Warning")
                })
        {
            return Err(io::Error::other(format!(
                "{package} Vulkan tests failed ({})",
                output.status,
            )));
        }
        Ok(())
    };
    test(
        "frame-comfort-fixtures",
        &[
            "--features",
            "vulkan-fixtures",
            "--test",
            "shell",
            "--test",
            "imported_source",
            "--test",
            "mapping",
        ],
        true,
    )?;
    test(
        "comfort-runtime",
        &["--features", "scene-fixtures", "--lib"],
        true,
    )?;
    // Extra synchronization instrumentation can delay the software Vulkan
    // readbacks beyond the real separate-eye pairing window.
    test("frame-comfort", &["--test", "overlay"], false)
}
