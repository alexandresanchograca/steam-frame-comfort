#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use comfort_runtime::vulkan::fixture::{Cadence, run_cadence};
    use std::{env, io, time::Duration};

    let args = env::args().skip(1).collect::<Vec<_>>();
    let seconds: f64 = args.first().map_or(Ok(12.0), |value| value.parse())?;
    let gap: u64 = args.get(2).map_or(Ok(750), |value| value.parse())?;
    let usage = || {
        io::Error::other(
            "usage: frame-comfort-cadence [1–60 seconds] [reuse|rotate|burst|per-eye] [0–5000 µs gap]",
        )
    };
    let mode = match args.get(1).map(String::as_str).unwrap_or("reuse") {
        "reuse" => Cadence::Reuse,
        "rotate" => Cadence::Rotate,
        "burst" => Cadence::Burst,
        "per-eye" => Cadence::SeparateEyes(Duration::from_micros(gap)),
        _ => return Err(usage().into()),
    };
    if !(1.0..=60.0).contains(&seconds) || gap > 5000 || args.len() > 3 {
        return Err(usage().into());
    }
    let frames = run_cadence(Duration::from_secs_f64(seconds), mode)?;
    println!("cadence frames={frames}");
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("The Vulkan cadence fixture requires Linux.");
    std::process::exit(1);
}
