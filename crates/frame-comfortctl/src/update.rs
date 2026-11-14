//! `check-update` and `update`: find the latest GitHub release, and download,
//! verify and install it. Releases are signed with the key in `release.json`;
//! an archive whose signature does not verify is never unpacked.
use super::{Paths, SERVICE, install, launch::start_and_show, systemctl};
use comfort_core::release::{UPDATE_FILE, Version};
use ed25519_dalek::{Signature, VerifyingKey};
use serde_json::{Value, json};
use std::{
    fs, io,
    path::Path,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

/// Where releases come from and the key they are signed with.
const RELEASE: &str = include_str!("../../../release.json");

struct Source {
    repository: String,
    key: VerifyingKey,
}

/// `None` when this build has no release repository or key configured.
fn source() -> io::Result<Option<Source>> {
    let config: Value = serde_json::from_str(RELEASE)?;
    let (Some(repository), Some(key)) =
        (config["repository"].as_str(), config["publicKey"].as_str())
    else {
        return Err(io::Error::other(
            "release.json needs repository and publicKey",
        ));
    };
    if repository.is_empty() || key.is_empty() {
        return Ok(None);
    }
    let key = hex::<32>(key)
        .and_then(|bytes| VerifyingKey::from_bytes(&bytes).ok())
        .ok_or_else(|| io::Error::other("release.json has an invalid publicKey"))?;
    Ok(Some(Source {
        repository: repository.into(),
        key,
    }))
}

fn hex<const N: usize>(text: &str) -> Option<[u8; N]> {
    let text = text.trim();
    if text.len() != N * 2 || !text.is_ascii() {
        return None;
    }
    let mut bytes = [0; N];
    for (byte, pair) in bytes.iter_mut().zip(text.as_bytes().chunks(2)) {
        *byte = u8::from_str_radix(std::str::from_utf8(pair).ok()?, 16).ok()?;
    }
    Some(bytes)
}

/// Checks `archive` against a hex signature file's contents.
fn verify(key: &VerifyingKey, archive: &[u8], signature: &str) -> io::Result<()> {
    let signature = hex::<64>(signature)
        .map(|bytes| Signature::from_bytes(&bytes))
        .ok_or_else(|| io::Error::other("the release signature is malformed"))?;
    key.verify_strict(archive, &signature)
        .map_err(|_| io::Error::other("the release signature does not match"))
}

fn archive_name(version: Version) -> String {
    format!("frame-comfort-{version}-linux-aarch64.tar.gz")
}

struct Latest {
    version: Version,
    archive: String,
    signature: String,
}

fn curl(args: &[&str]) -> io::Result<Vec<u8>> {
    let output = Command::new("curl")
        .args(["--fail", "--silent", "--show-error", "--location"])
        .args(args)
        .output()?;
    if !output.status.success() {
        return Err(io::Error::other(format!(
            "download failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    Ok(output.stdout)
}

fn latest(source: &Source) -> io::Result<Latest> {
    let url = format!(
        "https://api.github.com/repos/{}/releases/latest",
        source.repository
    );
    let release: Value = serde_json::from_slice(&curl(&[
        "--max-time",
        "20",
        "--header",
        "Accept: application/vnd.github+json",
        &url,
    ])?)?;
    parse_latest(&release)
}

fn parse_latest(release: &Value) -> io::Result<Latest> {
    let version = release["tag_name"]
        .as_str()
        .and_then(Version::parse)
        .ok_or_else(|| io::Error::other("the latest release has no version tag"))?;
    let name = archive_name(version);
    let asset = |wanted: &str| {
        release["assets"]
            .as_array()?
            .iter()
            .find(|asset| asset["name"] == wanted)?["browser_download_url"]
            .as_str()
            .map(str::to_owned)
    };
    match (asset(&name), asset(&format!("{name}.sig"))) {
        (Some(archive), Some(signature)) => Ok(Latest {
            version,
            archive,
            signature,
        }),
        _ => Err(io::Error::other(format!(
            "release {version} lacks {name} or its signature"
        ))),
    }
}

fn installed_version(paths: &Paths) -> Version {
    fs::read(paths.package_manifest())
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
        .and_then(|manifest| manifest["version"].as_str().and_then(Version::parse))
        .unwrap_or_else(Version::current)
}

fn write_state(paths: &Paths, mut state: Value) -> io::Result<Value> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs());
    state["at"] = json!(now);
    let text = serde_json::to_string_pretty(&state)? + "\n";
    install::write_private(&paths.root.join(UPDATE_FILE), text.as_bytes(), 0o644)?;
    Ok(state)
}

fn availability(installed: Version, latest: Version) -> &'static str {
    if latest > installed {
        "available"
    } else {
        "current"
    }
}

pub(super) fn check(paths: &Paths) -> io::Result<Value> {
    let Some(source) = source()? else {
        return write_state(paths, json!({"state": "disabled"}));
    };
    let installed = installed_version(paths);
    match latest(&source) {
        Ok(latest) => write_state(
            paths,
            json!({"state": availability(installed, latest.version),
                "installed": installed.to_string(), "latest": latest.version.to_string()}),
        ),
        Err(error) => write_state(
            paths,
            json!({"state": "error", "installed": installed.to_string(), "error": error.to_string()}),
        ),
    }
}

pub(super) fn update(paths: &Paths) -> io::Result<Value> {
    let source = source()?
        .ok_or_else(|| io::Error::other("this build has no release repository configured"))?;
    let installed = installed_version(paths);
    write_state(
        paths,
        json!({"state": "updating", "installed": installed.to_string()}),
    )?;
    let mut latest_version = None;
    let result = (|| {
        let latest = latest(&source)?;
        latest_version = Some(latest.version);
        if latest.version <= installed {
            return Ok(false);
        }
        write_state(
            paths,
            json!({"state": "updating", "installed": installed.to_string(),
                "latest": latest.version.to_string()}),
        )?;
        let work = tempfile::tempdir_in(&paths.root)?;
        let archive = curl(&["--max-time", "300", &latest.archive])?;
        let signature = String::from_utf8(curl(&["--max-time", "60", &latest.signature])?)
            .map_err(|_| io::Error::other("the release signature is not text"))?;
        verify(&source.key, &archive, &signature)?;
        let unpacked = unpack(&archive, work.path())?;
        let folder = unpacked.join(format!("frame-comfort-{}", latest.version));
        let manifest: Value = serde_json::from_slice(&fs::read(folder.join("package.json"))?)?;
        if manifest["version"] != latest.version.to_string() {
            return Err(io::Error::other(
                "the release archive has the wrong version",
            ));
        }
        install::install(paths, &folder)?;
        Ok(true)
    })();
    let latest = latest_version.map(|version| version.to_string());
    match result {
        Ok(updated) => {
            let state = write_state(
                paths,
                json!({"state": "current",
                    "installed": if updated { latest.clone().unwrap_or_default() } else { installed.to_string() },
                    "latest": latest, "updated": updated}),
            )?;
            if updated
                && systemctl(&["is-active", "--quiet", SERVICE])?
                    .status
                    .success()
            {
                start_and_show(paths, true)?;
            }
            Ok(state)
        }
        Err(error) => {
            write_state(
                paths,
                json!({"state": "failed", "installed": installed.to_string(),
                    "latest": latest, "error": error.to_string()}),
            )?;
            Err(error)
        }
    }
}

/// Unpacks a verified archive into `work/unpacked`.
fn unpack(archive: &[u8], work: &Path) -> io::Result<std::path::PathBuf> {
    let file = work.join("release.tar.gz");
    fs::write(&file, archive)?;
    let unpacked = work.join("unpacked");
    fs::create_dir(&unpacked)?;
    let output = Command::new("tar")
        .arg("-xzf")
        .arg(&file)
        .arg("-C")
        .arg(&unpacked)
        .arg("--no-same-owner")
        .output()?;
    if !output.status.success() {
        return Err(io::Error::other(format!(
            "could not unpack the release: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    Ok(unpacked)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};

    #[test]
    fn only_correctly_signed_archives_verify() {
        let key = SigningKey::from_bytes(&[7; 32]);
        let archive = b"release archive";
        let signature: String = key
            .sign(archive)
            .to_bytes()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        let public = key.verifying_key();
        verify(&public, archive, &format!("{signature}\n")).unwrap();
        assert!(verify(&public, b"tampered archive", &signature).is_err());
        assert!(verify(&public, archive, &signature[2..]).is_err());
        let other = SigningKey::from_bytes(&[8; 32]).verifying_key();
        assert!(verify(&other, archive, &signature).is_err());
    }

    #[test]
    fn reads_the_release_assets() {
        let release = json!({"tag_name": "v0.2.0", "assets": [
            {"name": "frame-comfort-0.2.0-linux-aarch64.tar.gz", "browser_download_url": "https://a"},
            {"name": "frame-comfort-0.2.0-linux-aarch64.tar.gz.sig", "browser_download_url": "https://s"},
        ]});
        let latest = parse_latest(&release).unwrap();
        assert_eq!(latest.version, Version(0, 2, 0));
        assert_eq!(
            (latest.archive.as_str(), latest.signature.as_str()),
            ("https://a", "https://s")
        );
        let unsigned = json!({"tag_name": "v0.2.0", "assets": [release["assets"][0]]});
        assert!(parse_latest(&unsigned).is_err());
        assert_eq!(
            availability(Version(0, 1, 0), Version(0, 2, 0)),
            "available"
        );
        assert_eq!(availability(Version(0, 2, 0), Version(0, 2, 0)), "current");
    }

    #[test]
    fn the_shipped_release_configuration_parses() {
        source().unwrap();
    }
}
