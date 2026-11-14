#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use comfort_runtime::vulkan::scene::{EyePatternOptions, run_openvr};
    use std::{env, io, path::PathBuf, time::Duration};

    let mut options = EyePatternOptions {
        library: PathBuf::from("libopenvr_api.so"),
        seconds: Duration::from_secs(20),
        bar: false,
        markers: false,
        frame_code: false,
        trace_truth: false,
    };
    const USAGE: &str = "frame-comfort-local-eye-pattern [SECONDS] [--bar] [--markers] [--frame-code] [--trace-truth] [--openvr-library PATH]";
    let mut seconds = None;
    let mut args = env::args_os().skip(1);
    while let Some(arg) = args.next() {
        match arg.to_str() {
            Some("--bar") => options.bar = true,
            Some("--markers") => options.markers = true,
            Some("--frame-code") => options.frame_code = true,
            Some("--trace-truth") => options.trace_truth = true,
            Some("--openvr-library") => {
                options.library = args.next().ok_or_else(|| io::Error::other(USAGE))?.into();
            }
            Some("--help" | "-h") => {
                println!("{USAGE}");
                return Ok(());
            }
            Some(value) if seconds.is_none() => {
                let value = value.parse::<f64>().map_err(|_| io::Error::other(USAGE))?;
                if !value.is_finite() {
                    return Err(io::Error::other(USAGE).into());
                }
                seconds = Some(Duration::from_secs_f64(value.clamp(1.0, 900.0)));
            }
            _ => return Err(io::Error::other(USAGE).into()),
        }
    }
    if let Some(seconds) = seconds {
        options.seconds = seconds;
    }
    run_openvr(options)
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("The OpenVR scene fixture requires Linux.");
    std::process::exit(1);
}
