#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use comfort_runtime::vulkan::scene::{Layout, XrPatternOptions, run_openxr};
    use std::{env, io, path::PathBuf, time::Duration};

    const USAGE: &str = "frame-comfort-local-xr-pattern [--layout per-eye|array|sbs] [--seconds 1-120] [--runtime MANIFEST] [--frame-code] [--trace-truth]";
    let mut options = XrPatternOptions {
        manifest: PathBuf::new(),
        layout: Layout::PerEye,
        seconds: Duration::from_secs(30),
        frame_code: false,
        trace_truth: false,
    };
    let mut args = env::args_os().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || args.next().ok_or_else(|| io::Error::other(USAGE));
        match arg.to_str() {
            Some("--layout") => {
                options.layout = match value()?.to_str() {
                    Some("per-eye") => Layout::PerEye,
                    Some("array") => Layout::Array,
                    Some("sbs") => Layout::SideBySide,
                    _ => return Err(io::Error::other(USAGE).into()),
                };
            }
            Some("--seconds") => {
                let seconds = value()?
                    .to_str()
                    .and_then(|s| s.parse::<f64>().ok())
                    .filter(|s| s.is_finite())
                    .ok_or_else(|| io::Error::other(USAGE))?;
                options.seconds = Duration::from_secs_f64(seconds.clamp(1.0, 120.0));
            }
            Some("--runtime") => options.manifest = value()?.into(),
            Some("--frame-code") => options.frame_code = true,
            Some("--trace-truth") => options.trace_truth = true,
            Some("--help" | "-h") => {
                println!("{USAGE}");
                return Ok(());
            }
            _ => return Err(io::Error::other(USAGE).into()),
        }
    }
    if options.manifest.as_os_str().is_empty() {
        options.manifest = env::var_os("XR_RUNTIME_JSON")
            .map(PathBuf::from)
            .or_else(|| {
                env::var_os("XDG_CONFIG_HOME")
                    .map(PathBuf::from)
                    .map(|root| root.join("openxr/1/active_runtime.json"))
            })
            .or_else(|| {
                env::var_os("HOME")
                    .map(PathBuf::from)
                    .map(|home| home.join(".config/openxr/1/active_runtime.json"))
            })
            .ok_or_else(|| io::Error::other("no OpenXR runtime manifest, use --runtime"))?;
    }
    run_openxr(options)
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("The OpenXR scene fixture requires Linux.");
    std::process::exit(1);
}
