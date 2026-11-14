//! `launch`: what the Steam shortcut runs. It installs or upgrades from the
//! folder it was started from, asks SteamVR to load the observer at its next
//! start, starts Comfort and opens its panel.
use super::{Paths, SERVICE, install, require_success, systemctl, write_dropin};
use comfort_core::release::Version;
use serde_json::{Value, json};
use std::{
    env, fs, io,
    os::unix::process::CommandExt,
    path::Path,
    process::{Command, Stdio},
};

#[derive(Debug, PartialEq, Eq)]
enum Plan {
    /// A newer version is installed (an in-app update): let it launch.
    Delegate,
    /// Install or upgrade from the folder this copy runs from.
    Install,
    Run,
}

/// `package` is the manifest of the release folder this copy runs from, if it
/// runs from one; `installed` is the installed manifest.
fn plan(own: Version, installed: Option<&Value>, package: Option<&Value>) -> Plan {
    let installed_version = installed
        .and_then(|manifest| manifest["version"].as_str())
        .and_then(Version::parse);
    if package.is_some() && installed_version.is_some_and(|version| version > own) {
        return Plan::Delegate;
    }
    match (package, installed) {
        (Some(_), None) => Plan::Install,
        (Some(package), Some(installed))
            if installed_version.is_none_or(|version| version < own)
                || package["sha256"] != installed["sha256"] =>
        {
            Plan::Install
        }
        _ => Plan::Run,
    }
}

fn read_json(path: &Path) -> Option<Value> {
    serde_json::from_slice(&fs::read(path).ok()?).ok()
}

pub(super) fn launch(paths: &Paths) -> io::Result<Value> {
    let exe = env::current_exe()?.canonicalize()?;
    let here = exe
        .parent()
        .ok_or_else(|| io::Error::other("cannot locate the launcher's folder"))?;
    let in_install = paths.root.canonicalize().is_ok_and(|root| root == here);
    let package = (!in_install && here.join("comfort/vrw-comfort").is_file())
        .then(|| read_json(&here.join("package.json")))
        .flatten();
    let installed = read_json(&paths.package_manifest());
    let mut installed_now = false;
    match plan(Version::current(), installed.as_ref(), package.as_ref()) {
        Plan::Delegate if paths.ctl().is_file() => {
            return Err(Command::new(paths.ctl()).arg("launch").exec());
        }
        Plan::Delegate | Plan::Run => {}
        Plan::Install => {
            install::install(paths, here)?;
            installed_now = true;
        }
    }
    if !paths.app().is_file() {
        return Err(io::Error::other(
            "Frame Comfort is not installed: start frame-comfort from the unpacked release folder",
        ));
    }
    // The observer loads when SteamVR next starts; the panel offers a
    // restart until then.
    let source_requested = !paths.dropin.exists();
    if source_requested {
        write_dropin(paths)?;
        require_success(systemctl(&["daemon-reload"])?, "daemon-reload failed")?;
    }
    if !systemctl(&["is-active", "--quiet", super::STEAMVR])?
        .status
        .success()
    {
        return Err(io::Error::other(
            "SteamVR is not running: start SteamVR, then Frame Comfort",
        ));
    }
    // A running Comfort keeps its old binary until it restarts.
    start_and_show(paths, installed_now)?;
    // Look for a newer release in the background so the panel can offer it.
    let checking = Command::new("systemd-run")
        .args(["--user", "--collect", "--quiet", "--"])
        .arg(paths.ctl())
        .arg("check-update")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|status| status.success());
    Ok(json!({"launched": true, "installed": installed_now,
        "sourceRequested": source_requested, "checkingForUpdate": checking}))
}

/// Starts (or restarts) the Comfort service and asks it to open its
/// dashboard panel.
pub(super) fn start_and_show(paths: &Paths, restart: bool) -> io::Result<()> {
    let action = if restart { "restart" } else { "start" };
    require_success(systemctl(&[action, SERVICE])?, "could not start Comfort")?;
    fs::write(&paths.request, b"")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn launches_install_upgrade_or_hand_over_by_version() {
        let v = |text: &str, hash: &str| json!({"version": text, "sha256": {"app": hash}});
        let own = Version(0, 2, 0);
        let package = v("0.2.0", "b");
        // First launch from the download.
        assert_eq!(plan(own, None, Some(&package)), Plan::Install);
        // A newer download, or the same version rebuilt.
        assert_eq!(
            plan(own, Some(&v("0.1.0", "a")), Some(&package)),
            Plan::Install
        );
        assert_eq!(
            plan(own, Some(&v("0.2.0", "a")), Some(&package)),
            Plan::Install
        );
        // Already installed from this folder.
        assert_eq!(plan(own, Some(&v("0.2.0", "b")), Some(&package)), Plan::Run);
        // The panel updated past this download: never downgrade.
        assert_eq!(
            plan(own, Some(&v("0.3.0", "c")), Some(&package)),
            Plan::Delegate
        );
        // The installed copy itself, from the desktop entry or a delegation.
        assert_eq!(plan(own, Some(&v("0.3.0", "c")), None), Plan::Run);
    }
}
