use super::{LAYER, Paths, require_success, systemctl};
use comfort_core::settings::is_version_two;
use rustix::fs::{FlockOperation, flock};
use serde_json::{Value, json};
use std::{
    fs::{self, File, Permissions},
    io::{self, Write},
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    process::Command,
};

fn parent(path: &Path) -> io::Result<&Path> {
    path.parent()
        .ok_or_else(|| io::Error::other("installation path has no parent"))
}

fn check_destination(path: &Path) -> io::Result<()> {
    match path.symlink_metadata() {
        Ok(meta) if meta.is_file() => Ok(()),
        Ok(_) => Err(io::Error::other(format!(
            "refusing to replace non-regular file: {}",
            path.display()
        ))),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error),
    }
}

pub(super) fn write_private(path: &Path, bytes: &[u8], mode: u32) -> io::Result<()> {
    check_destination(path)?;
    // Leave an identical file alone: replacing the observer would make
    // SteamVR's loaded copy look outdated and the panel ask for a restart.
    if fs::symlink_metadata(path).is_ok_and(|meta| meta.permissions().mode() & 0o7777 == mode)
        && fs::read(path).is_ok_and(|current| current == bytes)
    {
        return Ok(());
    }
    let dir = parent(path)?;
    fs::create_dir_all(dir)?;
    let mut file = tempfile::NamedTempFile::new_in(dir)?;
    file.write_all(bytes)?;
    file.as_file()
        .set_permissions(Permissions::from_mode(mode))?;
    file.persist(path).map_err(|error| error.error)?;
    Ok(())
}

fn unused_backup(path: &Path) -> io::Result<PathBuf> {
    let mut base = path.as_os_str().to_owned();
    base.push(".before-rest-frame");
    for index in 0.. {
        let mut candidate = base.clone();
        if index != 0 {
            candidate.push(format!(".{index}"));
        }
        let candidate = PathBuf::from(candidate);
        match candidate.symlink_metadata() {
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(candidate),
            Err(error) => return Err(error),
            Ok(_) => {}
        }
    }
    unreachable!()
}

fn migrate_settings(settings: &Path, defaults: &[u8]) -> io::Result<()> {
    let dir = parent(settings)?;
    fs::create_dir_all(dir)?;
    let directory = File::open(dir)?;
    flock(&directory, FlockOperation::LockExclusive)?;
    if super::app_status(settings)
        .as_ref()
        .is_some_and(is_version_two)
    {
        return Ok(());
    }
    let mut backups = Vec::new();
    for path in [
        settings.with_extension("dashboard.json"),
        settings.to_owned(),
    ] {
        match path.symlink_metadata() {
            Ok(meta) if meta.is_file() || meta.is_symlink() => {
                backups.push((unused_backup(&path)?, path))
            }
            Ok(_) => {
                return Err(io::Error::other(format!(
                    "not a settings file: {}",
                    path.display()
                )));
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
    }
    let mut staged = tempfile::NamedTempFile::new_in(dir)?;
    staged.write_all(defaults)?;
    staged
        .as_file()
        .set_permissions(Permissions::from_mode(0o600))?;
    let mut moved = Vec::new();
    let migration = (|| -> io::Result<()> {
        for (backup, path) in &backups {
            fs::rename(path, backup)?;
            moved.push((backup, path));
        }
        staged
            .persist_noclobber(settings)
            .map_err(|error| error.error)?;
        Ok(())
    })();
    if let Err(error) = migration {
        for (backup, path) in moved.into_iter().rev() {
            if matches!(path.symlink_metadata(), Err(error) if error.kind() == io::ErrorKind::NotFound)
                && let Err(rollback) = fs::rename(backup, path)
            {
                eprintln!(
                    "Could not restore {}. Backup kept at {}: {rollback}",
                    path.display(),
                    backup.display()
                );
            }
        }
        return Err(error);
    }
    Ok(())
}

fn path_text(path: &Path) -> io::Result<&str> {
    path.to_str()
        .filter(|text| !text.contains(['\n', '\r']))
        .ok_or_else(|| {
            io::Error::other(format!("unsupported installation path: {}", path.display()))
        })
}

pub(super) fn unit_arg(path: &Path) -> io::Result<String> {
    let value = path_text(path)?
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('%', "%%")
        .replace('$', "$$");
    Ok(format!("\"{value}\""))
}

fn env_value(path: &Path) -> io::Result<String> {
    let value = path_text(path)?
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('$', "\\$")
        .replace('`', "\\`");
    Ok(format!("\"{value}\""))
}

fn desktop_exec(path: &Path) -> io::Result<String> {
    let path = path_text(path)?;
    let mut escaped = String::new();
    for ch in path.chars() {
        match ch {
            '\\' => escaped.push_str(r"\\\\"),
            '"' | '$' | '`' => {
                escaped.push_str(r"\\");
                escaped.push(ch);
            }
            '%' => escaped.push_str("%%"),
            _ => escaped.push(ch),
        }
    }
    Ok(format!("\"{escaped}\""))
}

/// Runs `frame-comfortctl launch` from the launcher's own folder.
pub(super) const LAUNCHER: &[u8] = include_bytes!("../frame-comfort.sh");

/// Installs from a release folder: `comfort/` with the app and observer,
/// and, in a package, `frame-comfortctl` and `package.json`.
pub(super) fn install(paths: &Paths, build: &Path) -> io::Result<Value> {
    let package = build.canonicalize()?;
    let build = package.join("comfort");
    let app = build.join("vrw-comfort");
    let layer = build.join("libvrw-comfort-source.so");
    // The panel runs restarts and updates through the installed copy.
    let ctl = Some(package.join("frame-comfortctl"))
        .filter(|path| path.is_file())
        .map_or_else(std::env::current_exe, Ok)?;
    let manifest_bytes = match fs::read(package.join("package.json")) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            (serde_json::to_string_pretty(&json!({"version": comfort_core::release::CURRENT}))?
                + "\n")
                .into_bytes()
        }
        Err(error) => return Err(error),
    };
    for source in [&app, &layer] {
        if !source.is_file() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("missing build output: {}", source.display()),
            ));
        }
    }
    let output = Command::new(&app)
        .arg("--print-default-settings")
        .output()?;
    if !output.status.success() {
        return Err(io::Error::other(format!(
            "could not read Comfort settings defaults: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    let defaults: Value = serde_json::from_slice(&output.stdout)?;
    if !is_version_two(&defaults) {
        return Err(io::Error::other(
            "Comfort did not return version 2 settings defaults",
        ));
    }

    let manifest = json!({"file_format_version": "1.2.0", "layer": {
        "name": LAYER, "type": "GLOBAL", "library_path": paths.layer(), "api_version": "1.3.0",
        "implementation_version": "1", "description": "VR Window Frame Comfort scene-source observer (vrcompositor only)",
        "functions": {"vkNegotiateLoaderLayerInterfaceVersion": "vkNegotiateLoaderLayerInterfaceVersion"}
    }});
    let env = format!(
        "VK_ADD_LAYER_PATH={}\nVK_INSTANCE_LAYERS={LAYER}\nVRW_COMFORT_SOURCE_COMM=vrcompositor\nVRW_COMFORT_SOURCE_SOCKET={}\n",
        env_value(&paths.root)?,
        env_value(&paths.socket)?
    );
    let unit = format!(
        "[Unit]\nDescription=VR Window Frame Comfort\nAfter=steamvr.service\nPartOf=steamvr.service\n\n[Service]\nExecStart={} --settings {} --status {} --socket {}\nRestart=on-failure\nRestartSec=3\n\n[Install]\nWantedBy=steamvr.service\n",
        unit_arg(&paths.app())?,
        unit_arg(&paths.settings)?,
        unit_arg(&paths.status)?,
        unit_arg(&paths.socket)?
    );
    let desktop = format!(
        "[Desktop Entry]\nType=Application\nName=Frame Comfort\nComment=Start the Steam Frame Comfort overlay\nExec={}\nTerminal=false\nCategories=Utility;\n",
        desktop_exec(&paths.launcher())?
    );

    match paths.root.symlink_metadata() {
        Ok(meta) if !meta.is_dir() => {
            return Err(io::Error::other("installation path is not a directory"));
        }
        Err(error) if error.kind() != io::ErrorKind::NotFound => return Err(error),
        _ => {}
    }
    fs::create_dir_all(&paths.root)?;
    let directory = File::open(&paths.root)?;
    flock(&directory, FlockOperation::LockExclusive)?;
    directory.set_permissions(Permissions::from_mode(0o700))?;
    for target in [
        paths.app(),
        paths.layer(),
        paths.manifest(),
        paths.source_env(),
        paths.unit.clone(),
        paths.launcher(),
        paths.desktop.clone(),
        paths.ctl(),
        paths.package_manifest(),
    ] {
        check_destination(&target)?;
    }
    migrate_settings(&paths.settings, &output.stdout)?;
    write_private(&paths.app(), &fs::read(app)?, 0o755)?;
    write_private(&paths.layer(), &fs::read(layer)?, 0o755)?;
    let manifest = serde_json::to_string_pretty(&manifest)? + "\n";
    write_private(&paths.manifest(), manifest.as_bytes(), 0o644)?;
    write_private(&paths.source_env(), env.as_bytes(), 0o600)?;
    write_private(&paths.unit, unit.as_bytes(), 0o644)?;
    write_private(&paths.launcher(), LAUNCHER, 0o755)?;
    write_private(&paths.desktop, desktop.as_bytes(), 0o644)?;
    if ctl.canonicalize()? != paths.ctl().canonicalize().unwrap_or_default() {
        write_private(&paths.ctl(), &fs::read(&ctl)?, 0o755)?;
    }
    // Last, so an interrupted install is retried on the next launch.
    write_private(&paths.package_manifest(), &manifest_bytes, 0o644)?;
    require_success(systemctl(&["daemon-reload"])?, "daemon-reload failed")?;

    Ok(
        json!({"installed": paths.root, "unit": paths.unit, "settings": paths.settings,
        "launcher": paths.launcher(), "desktopEntry": paths.desktop, "sourceEnabled": paths.dropin.exists()}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn migrates_settings_once_without_overwriting_existing_backups() -> io::Result<()> {
        let root = tempfile::tempdir()?;
        let settings = root.path().join("settings.json");
        let dashboard = root.path().join("settings.dashboard.json");
        let defaults = br#"{"settingsVersion":2,"shape":"circle"}"#;
        fs::write(&settings, "{}")?;
        fs::write(&dashboard, "{\"shape\":\"rectangle\"}")?;
        fs::write(
            root.path().join("settings.json.before-rest-frame"),
            "existing",
        )?;
        migrate_settings(&settings, defaults)?;
        assert_eq!(fs::read(&settings)?, defaults);
        assert_eq!(
            fs::read_to_string(root.path().join("settings.json.before-rest-frame.1"))?,
            "{}"
        );
        assert_eq!(
            fs::read_to_string(
                root.path()
                    .join("settings.dashboard.json.before-rest-frame")
            )?,
            "{\"shape\":\"rectangle\"}"
        );
        fs::write(
            &dashboard,
            "{\"settingsVersion\":2,\"shape\":\"rectangle\"}",
        )?;
        migrate_settings(&settings, b"new defaults")?;
        assert_eq!(fs::read(&settings)?, defaults);
        assert!(dashboard.exists());
        Ok(())
    }
}
