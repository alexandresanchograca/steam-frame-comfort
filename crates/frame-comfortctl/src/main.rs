#![forbid(unsafe_code)]

mod install;
mod launch;
mod monitor;
mod update;

use clap::{Parser, Subcommand};
use serde_json::{Value, json};
use std::{
    env, fs, io,
    path::{Path, PathBuf},
    process::{Command, ExitCode, Output},
    thread,
    time::{Duration, Instant},
};

const SERVICE: &str = "vrw-comfort.service";
const STEAMVR: &str = "steamvr.service";
const LAYER: &str = "VK_LAYER_VRW_comfort_source";

#[derive(Parser)]
#[command(about = "Install and manage Frame Comfort as the logged-in Frame user")]
struct Cli {
    #[command(subcommand)]
    command: Action,
}

#[derive(Subcommand)]
enum Action {
    Install {
        #[arg(long)]
        build: PathBuf,
    },
    /// Set up or update Frame Comfort from this folder if needed, then start
    /// it and open its panel (what the `frame-comfort` Steam shortcut runs).
    Launch,
    EnableSource {
        #[arg(long)]
        yes: bool,
        #[arg(long, default_value = "30", value_parser = seconds)]
        timeout: Duration,
        /// Start Comfort and open its panel afterwards, even if the restart failed.
        #[arg(long)]
        start: bool,
    },
    /// Look for a newer release and record the result for the panel.
    CheckUpdate,
    /// Download, verify and install the latest release.
    Update,
    DisableSource {
        #[arg(long)]
        restart: bool,
        #[arg(long)]
        yes: bool,
        #[arg(long, default_value = "30", value_parser = seconds)]
        timeout: Duration,
    },
    Start,
    Stop,
    EnableAutostart,
    DisableAutostart,
    Status,
    Monitor {
        #[arg(long, default_value = "600", value_parser = seconds)]
        seconds: Duration,
        #[arg(long, default_value = "1.0", value_parser = seconds)]
        interval: Duration,
        #[arg(long)]
        out: Option<PathBuf>,
        #[arg(long)]
        verbose: bool,
    },
    Uninstall {
        #[arg(long)]
        yes: bool,
    },
}

fn seconds(value: &str) -> Result<Duration, String> {
    let parsed: f64 = value.parse().map_err(|_| "expected seconds")?;
    let duration =
        Duration::try_from_secs_f64(parsed).map_err(|_| "expected finite positive seconds")?;
    if duration.is_zero() {
        return Err("expected positive seconds".into());
    }
    Ok(duration)
}

struct Paths {
    root: PathBuf,
    settings: PathBuf,
    unit: PathBuf,
    dropin: PathBuf,
    desktop: PathBuf,
    socket: PathBuf,
    status: PathBuf,
    request: PathBuf,
    log: PathBuf,
}

impl Paths {
    fn from_environment() -> io::Result<Self> {
        let home =
            env::home_dir().ok_or_else(|| io::Error::other("cannot determine home directory"))?;
        let data = home.join(".local/share");
        let desktop_dir = env::var_os("XDG_DATA_HOME")
            .map(PathBuf::from)
            .filter(|path| path.is_absolute())
            .unwrap_or_else(|| data.clone());
        let runtime = env::var_os("XDG_RUNTIME_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(format!("/run/user/{}", rustix::process::getuid().as_raw()))
            });
        let unit_dir = home.join(".config/systemd/user");
        let state = env::var_os("XDG_STATE_HOME")
            .map(PathBuf::from)
            .filter(|path| path.is_absolute())
            .unwrap_or_else(|| home.join(".local/state"));
        Ok(Self {
            root: data.join("vrw-comfort"),
            settings: home.join(".config/vrw-comfort/settings.json"),
            unit: unit_dir.join(SERVICE),
            dropin: unit_dir.join("steamvr.service.d/99-vrw-comfort-source.conf"),
            desktop: desktop_dir.join("applications/vrw-comfort.desktop"),
            socket: runtime.join("vrw-comfort-source.sock"),
            status: runtime.join("vrw-comfort-status.json"),
            request: runtime.join("vrw-comfort-controls.request"),
            log: state.join("vrw-comfort/frame-comfortctl.log"),
        })
    }

    fn app(&self) -> PathBuf {
        self.root.join("vrw-comfort")
    }
    fn layer(&self) -> PathBuf {
        self.root.join("libvrw-comfort-source.so")
    }
    fn manifest(&self) -> PathBuf {
        self.root.join(format!("{LAYER}.json"))
    }
    fn source_env(&self) -> PathBuf {
        self.root.join("source.env")
    }
    fn launcher(&self) -> PathBuf {
        self.root.join("vrw-comfort-launch")
    }
    fn ctl(&self) -> PathBuf {
        self.root.join("frame-comfortctl")
    }
    fn package_manifest(&self) -> PathBuf {
        self.root.join("package.json")
    }
}

fn systemctl(args: &[&str]) -> io::Result<Output> {
    Command::new("systemctl").arg("--user").args(args).output()
}

fn require_success(output: Output, action: &str) -> io::Result<()> {
    if output.status.success() {
        return Ok(());
    }
    Err(io::Error::other(format!(
        "{action}: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    )))
}

fn pids(name: &str) -> Vec<u32> {
    Command::new("pgrep")
        .args(["-x", name])
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| {
            String::from_utf8_lossy(&out.stdout)
                .split_whitespace()
                .filter_map(|pid| pid.parse().ok())
                .collect()
        })
        .unwrap_or_default()
}

fn proc_text(pid: u32, file: &str) -> Option<String> {
    fs::read(format!("/proc/{pid}/{file}"))
        .ok()
        .map(|bytes| String::from_utf8_lossy(&bytes).into_owned())
}

fn mapped(pid: u32, path: &Path) -> bool {
    proc_text(pid, "maps").is_some_and(|maps| maps.contains(path.to_string_lossy().as_ref()))
}

fn proc_ids() -> impl Iterator<Item = u32> {
    fs::read_dir("/proc")
        .into_iter()
        .flatten()
        .filter_map(|entry| entry.ok()?.file_name().to_str()?.parse().ok())
}

fn app_status(path: &Path) -> Option<Value> {
    serde_json::from_slice(&fs::read(path).ok()?).ok()
}

fn wait_for<T>(timeout: Duration, mut check: impl FnMut() -> Option<T>) -> Option<T> {
    let start = Instant::now();
    loop {
        if let Some(result) = check() {
            return Some(result);
        }
        let remaining = timeout.saturating_sub(start.elapsed());
        if remaining.is_zero() {
            return None;
        }
        thread::sleep(remaining.min(Duration::from_millis(500)));
    }
}

fn restart_steamvr(timeout: Duration) -> io::Result<Vec<u32>> {
    require_success(systemctl(&["daemon-reload"])?, "daemon-reload failed")?;
    require_success(systemctl(&["restart", STEAMVR])?, "SteamVR restart failed")?;
    Ok(wait_for(timeout, || {
        let pids = pids("vrcompositor");
        (!pids.is_empty()).then_some(pids)
    })
    .unwrap_or_default())
}

fn enable_source(paths: &Paths, yes: bool, timeout: Duration) -> io::Result<Value> {
    use std::os::unix::fs::PermissionsExt;
    for path in [paths.layer(), paths.manifest(), paths.source_env()] {
        if !path.is_file() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("not installed: {}", path.display()),
            ));
        }
    }
    if fs::metadata(paths.source_env())?.permissions().mode() & 0o077 != 0 {
        return Err(io::Error::other("environment file is not private"));
    }
    if !yes {
        return Err(io::Error::other(
            "refusing to restart SteamVR without --yes (this disconnects an active stream)",
        ));
    }
    write_dropin(paths)?;
    let mut compositors = restart_steamvr(timeout)?;
    let loaded = !compositors.is_empty()
        && wait_for(timeout, || {
            let current = pids("vrcompositor");
            if !current.is_empty() {
                compositors = current;
            }
            compositors
                .iter()
                .any(|&pid| mapped(pid, &paths.layer()))
                .then_some(())
        })
        .is_some();
    if !loaded {
        return Err(io::Error::other(
            "SteamVR restarted without loading the Frame Comfort observer",
        ));
    }
    let others: Vec<_> = proc_ids()
        .filter(|pid| !compositors.contains(pid) && mapped(*pid, &paths.layer()))
        .filter_map(|pid| proc_text(pid, "comm").map(|name| name.trim().to_owned()))
        .collect();
    Ok(
        json!({"sourceEnabled": true, "compositorPids": compositors, "layerMappedInCompositor": loaded,
        "otherProcessesMappingLayer": others, "socket": paths.socket.exists()}),
    )
}

/// Makes SteamVR load the observer from its next start.
fn write_dropin(paths: &Paths) -> io::Result<()> {
    install::write_private(
        &paths.dropin,
        b"[Service]\nEnvironmentFile=%h/.local/share/vrw-comfort/source.env\n",
        0o644,
    )
}

fn disable_source(paths: &Paths, restart: bool, yes: bool, timeout: Duration) -> io::Result<Value> {
    if restart && !yes {
        return Err(io::Error::other(
            "refusing to restart SteamVR without --yes",
        ));
    }
    let removed = remove_file_if_present(&paths.dropin)?;
    require_success(systemctl(&["daemon-reload"])?, "daemon-reload failed")?;
    let restarted = restart && !restart_steamvr(timeout)?.is_empty();
    Ok(
        json!({"dropinRemoved": removed, "steamvrRestarted": restarted,
        "layerStillMapped": pids("vrcompositor").iter().any(|&pid| mapped(pid, &paths.layer()))}),
    )
}

fn remove_file_if_present(path: &Path) -> io::Result<bool> {
    match fs::remove_file(path) {
        Ok(()) => Ok(true),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error),
    }
}

fn uninstall(paths: &Paths, yes: bool) -> io::Result<Value> {
    if paths.dropin.exists() && !yes {
        return Err(io::Error::other(
            "the source drop-in is still enabled; run disable-source first or pass --yes",
        ));
    }
    let _ = systemctl(&["disable", "--now", SERVICE])?;
    for path in [&paths.dropin, &paths.unit, &paths.desktop] {
        remove_file_if_present(path)?;
    }
    if paths.root.exists() {
        fs::remove_dir_all(&paths.root)?;
    }
    require_success(systemctl(&["daemon-reload"])?, "daemon-reload failed")?;
    Ok(json!({"uninstalled": true, "settingsKept": paths.settings}))
}

fn status(paths: &Paths) -> io::Result<Value> {
    let compositors = pids("vrcompositor");
    let layer_mapped = compositors.iter().any(|&pid| mapped(pid, &paths.layer()));
    let active = systemctl(&["is-active", SERVICE])?;
    let enabled = systemctl(&["is-enabled", SERVICE])?;
    Ok(
        json!({"installed": paths.app().is_file() && paths.layer().is_file(), "sourceDropin": paths.dropin.exists(),
        "compositorPids": compositors, "layerMapped": layer_mapped, "socket": paths.socket.exists(),
        "service": String::from_utf8_lossy(&active.stdout).trim(), "autostart": String::from_utf8_lossy(&enabled.stdout).trim(),
        "app": app_status(&paths.status)}),
    )
}

fn run(action: Action) -> io::Result<(Value, bool)> {
    let paths = Paths::from_environment()?;
    let output = match action {
        Action::Install { build } => install::install(&paths, &build)?,
        Action::Launch => launch::launch(&paths)?,
        Action::EnableSource {
            yes,
            timeout,
            start,
        } => {
            let result = enable_source(&paths, yes, timeout);
            if start {
                launch::start_and_show(&paths, false)?;
            }
            result?
        }
        Action::CheckUpdate => update::check(&paths)?,
        Action::Update => update::update(&paths)?,
        Action::DisableSource {
            restart,
            yes,
            timeout,
        } => disable_source(&paths, restart, yes, timeout)?,
        Action::Status => status(&paths)?,
        Action::Uninstall { yes } => uninstall(&paths, yes)?,
        Action::Monitor {
            seconds,
            interval,
            out,
            verbose,
        } => monitor::monitor(&paths, seconds, interval, out, verbose)?,
        action => {
            let action = match action {
                Action::Start => "start",
                Action::Stop => "stop",
                Action::EnableAutostart => "enable",
                Action::DisableAutostart => "disable",
                _ => unreachable!(),
            };
            let result = systemctl(&[action, SERVICE])?;
            let ok = result.status.success();
            return Ok((
                json!({"action": action, "ok": ok,
                "stderr": String::from_utf8_lossy(&result.stderr).trim()}),
                ok,
            ));
        }
    };
    Ok((output, true))
}

/// Commands started from Steam or the panel have no terminal, so their
/// results also go to a log.
fn log_result(command: &str, result: &Value) {
    let Ok(paths) = Paths::from_environment() else {
        return;
    };
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs());
    let line = json!({"t": now, "command": command, "result": result}).to_string() + "\n";
    let written = paths
        .log
        .parent()
        .map_or(Ok(()), fs::create_dir_all)
        .and_then(|()| {
            use std::io::Write;
            fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&paths.log)?
                .write_all(line.as_bytes())
        });
    if let Err(error) = written {
        eprintln!("Could not write {}: {error}", paths.log.display());
    }
}

fn main() -> ExitCode {
    let action = Cli::parse().command;
    let logged = match &action {
        Action::Launch => Some("launch"),
        Action::CheckUpdate => Some("check-update"),
        Action::Update => Some("update"),
        Action::EnableSource { start: true, .. } => Some("enable-source --start"),
        _ => None,
    };
    let (output, ok, failed) = match run(action) {
        Ok((output, ok)) => (output, ok, false),
        Err(error) => (json!({"error": error.to_string()}), false, true),
    };
    if let Some(command) = logged {
        log_result(command, &output);
    }
    if failed {
        println!("{output}");
    } else {
        println!("{output:#}");
    }
    if ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
