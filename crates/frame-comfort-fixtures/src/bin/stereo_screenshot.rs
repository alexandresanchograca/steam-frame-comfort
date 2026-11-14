#[cfg(target_os = "linux")]
mod linux {
    use comfort_runtime::openvr::OpenVr;
    use rustix::{
        event::{PollFd, PollFlags, Timespec, poll},
        process::{Pid, PidfdFlags, Signal, pidfd_open, pidfd_send_signal},
    };
    use std::{
        env,
        error::Error,
        ffi::{CString, OsString},
        fs, io,
        os::unix::ffi::OsStrExt,
        path::{Path, PathBuf},
        thread,
        time::{Duration, Instant},
    };

    type Result<T> = std::result::Result<T, Box<dyn Error>>;
    const USAGE: &str = "frame-comfort-stereo-screenshot [--openvr-library PATH] PREVIEW_PREFIX STEREO_PREFIX\nframe-comfort-stereo-screenshot [--openvr-library PATH] --pair COMFORT_PID OUTPUT_PREFIX";

    fn request(library: &Path, preview: &Path, stereo: &Path) -> Result<()> {
        let preview = CString::new(preview.as_os_str().as_bytes())?;
        let stereo = CString::new(stereo.as_os_str().as_bytes())?;
        // SAFETY: The operator selects the installed OpenVR library or a
        // fixture implementing the pinned ABI.
        let vr = unsafe { OpenVr::init(library)? };
        let handle = vr.request_stereo_screenshot(&preview, &stereo)?;
        println!("Stereo screenshot result=0 handle={handle}");
        Ok(())
    }

    /// SteamVR resolves a relative path against its own working directory,
    /// so a relative screenshot prefix silently lands elsewhere.
    fn absolute(prefix: &Path) -> Result<PathBuf> {
        let directory = prefix
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        Ok(directory.canonicalize()?.join(
            prefix
                .file_name()
                .ok_or_else(|| io::Error::other("missing output prefix"))?,
        ))
    }

    fn suffix(prefix: &Path, suffix: &str) -> PathBuf {
        let mut path = prefix.as_os_str().to_owned();
        path.push(suffix);
        path.into()
    }

    fn capture(library: &Path, preview: &Path, stereo: &Path) -> Result<()> {
        request(library, preview, stereo)?;
        let png = suffix(stereo, ".png");
        let deadline = Instant::now() + Duration::from_secs(4);
        while Instant::now() < deadline {
            if fs::metadata(&png).is_ok_and(|file| file.is_file() && file.len() != 0) {
                return Ok(());
            }
            thread::sleep(Duration::from_millis(100));
        }
        Err(io::Error::other(format!("SteamVR did not write {}", png.display())).into())
    }

    fn pair(library: &Path, pid: &OsString, prefix: &Path) -> Result<()> {
        let pid = pid
            .to_str()
            .and_then(|pid| pid.parse().ok())
            .and_then(Pid::from_raw)
            .ok_or_else(|| io::Error::other("expected a positive Comfort PID"))?;
        // A pidfd ties the signal and exit check to this exact process,
        // even if its numeric PID is reused while capturing the first image.
        let process = pidfd_open(pid, PidfdFlags::empty())?;
        let executable = fs::read_link(format!("/proc/{}/exe", pid.as_raw_nonzero()))?;
        if !executable
            .file_name()
            .is_some_and(|name| name == "vrw-comfort" || name == "frame-comfort")
        {
            return Err(
                io::Error::other("PID does not identify a Frame Comfort executable").into(),
            );
        }

        let prefix = absolute(prefix)?;
        let [game, normal] = ["-G", "-N"].map(|suffix_text| suffix(&prefix, suffix_text));
        let [game_preview, normal_preview] = [&game, &normal].map(|path| suffix(path, "-preview"));
        for path in [&game, &normal, &game_preview, &normal_preview] {
            let png = suffix(path, ".png");
            match fs::symlink_metadata(&png) {
                Ok(_) => {
                    return Err(io::Error::other(format!(
                        "Output already exists: {}",
                        png.display()
                    ))
                    .into());
                }
                Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                Err(error) => return Err(error.into()),
            }
        }
        let start = Instant::now();
        capture(library, &game_preview, &game)?;
        pidfd_send_signal(&process, Signal::TERM)?;
        if poll(
            &mut [PollFd::new(&process, PollFlags::IN)],
            Some(&Timespec {
                tv_sec: 1,
                tv_nsec: 500_000_000,
            }),
        )? == 0
        {
            return Err(io::Error::other("Comfort is still running after SIGTERM").into());
        }

        thread::sleep(Duration::from_millis(150));
        let normal_requested = start.elapsed().as_millis();
        capture(library, &normal_preview, &normal)?;
        println!("Normal VR requested {normal_requested} ms after Game");
        println!(
            "{}\n{}",
            suffix(&game, ".png").display(),
            suffix(&normal, ".png").display()
        );
        Ok(())
    }

    pub fn run() -> Result<()> {
        let mut args = env::args_os().skip(1).peekable();
        let library = if args.peek().is_some_and(|arg| arg == "--openvr-library") {
            args.next();
            PathBuf::from(args.next().ok_or_else(|| io::Error::other(USAGE))?)
        } else {
            PathBuf::from("libopenvr_api.so")
        };
        let arguments: Vec<_> = args.collect();
        match arguments.as_slice() {
            [flag] if flag == "--help" || flag == "-h" => {
                println!("{USAGE}");
                Ok(())
            }
            [flag, pid, prefix] if flag == "--pair" => pair(&library, pid, Path::new(prefix)),
            // Wait for the file, so a dropped request is an error.
            [preview, stereo] => capture(
                &library,
                &absolute(Path::new(preview))?,
                &absolute(Path::new(stereo))?,
            ),
            _ => Err(io::Error::other(USAGE).into()),
        }
    }
}

#[cfg(target_os = "linux")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    linux::run()
}

#[cfg(not(target_os = "linux"))]
fn main() {
    eprintln!("The OpenVR screenshot fixture requires Linux.");
    std::process::exit(1);
}
