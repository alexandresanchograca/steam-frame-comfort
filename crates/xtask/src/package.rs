use ed25519_dalek::{Signer, SigningKey};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions, Permissions},
    io::{self, Read, Write},
    os::unix::fs::{OpenOptionsExt, PermissionsExt},
    path::{Path, PathBuf},
    process::Command,
};

const VERSION: &str = env!("CARGO_PKG_VERSION");
const ARTIFACTS: [(&str, &str); 3] = [
    ("frame-comfort", "comfort/vrw-comfort"),
    (
        "libvrw_comfort_source.so",
        "comfort/libvrw-comfort-source.so",
    ),
    ("frame-comfortctl", "frame-comfortctl"),
];
/// The file users add to Steam; it runs `frame-comfortctl launch`.
const LAUNCHER: &[u8] = include_bytes!("../../frame-comfortctl/frame-comfort.sh");

fn sha256(path: &Path) -> io::Result<String> {
    let mut hash = Sha256::new();
    io::copy(&mut File::open(path)?, &mut hash)?;
    Ok(format!("{:x}", hash.finalize()))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

pub fn package(artifacts: &Path, output: &Path) -> io::Result<()> {
    let parent = output
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    if output.symlink_metadata().is_ok() {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            format!("package already exists: {}", output.display()),
        ));
    }
    fs::create_dir_all(parent)?;
    let staging = tempfile::tempdir_in(parent)?;
    fs::create_dir(staging.path().join("comfort"))?;

    let mut hashes = serde_json::Map::new();
    for (source, name) in ARTIFACTS {
        let target = staging.path().join(name);
        fs::copy(artifacts.join(source), &target)?;
        hashes.insert(name.into(), json!(sha256(&target)?));
    }
    let launcher = staging.path().join("frame-comfort");
    fs::write(&launcher, LAUNCHER)?;
    fs::set_permissions(&launcher, Permissions::from_mode(0o755))?;
    hashes.insert("frame-comfort".into(), json!(sha256(&launcher)?));
    fs::write(
        staging.path().join("LICENSE"),
        include_bytes!("../../../LICENSE"),
    )?;
    fs::write(
        staging.path().join("OPENVR-LICENSE.txt"),
        include_bytes!("../../../OPENVR-LICENSE.txt"),
    )?;
    let mut manifest = File::create(staging.path().join("package.json"))?;
    serde_json::to_writer_pretty(
        &mut manifest,
        &json!({"version": VERSION, "sha256": hashes}),
    )?;
    writeln!(manifest)?;
    // The temporary directory is private; the package is for other users too.
    fs::set_permissions(staging.path(), Permissions::from_mode(0o755))?;
    fs::rename(staging.path(), output)
}

fn read_key(path: &Path) -> io::Result<SigningKey> {
    let text = fs::read_to_string(path)?;
    let text = text.trim();
    let mut bytes = [0u8; 32];
    if text.len() != 64 {
        return Err(io::Error::other("the signing key file needs 64 hex digits"));
    }
    for (byte, pair) in bytes.iter_mut().zip(text.as_bytes().chunks(2)) {
        *byte = std::str::from_utf8(pair)
            .ok()
            .and_then(|pair| u8::from_str_radix(pair, 16).ok())
            .ok_or_else(|| io::Error::other("the signing key file is not hex"))?;
    }
    Ok(SigningKey::from_bytes(&bytes))
}

/// Writes a new private signing key (never overwriting one) and prints the
/// public key for `release.json`.
pub fn keygen(path: &Path) -> io::Result<()> {
    let mut seed = [0u8; 32];
    File::open("/dev/urandom")?.read_exact(&mut seed)?;
    let key = SigningKey::from_bytes(&seed);
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(path)?;
    writeln!(file, "{}", hex(&seed))?;
    println!("{}", hex(key.verifying_key().as_bytes()));
    Ok(())
}

/// Packages the build, archives it and signs the archive, ready to attach to
/// a GitHub release tagged `vVERSION`.
pub fn release(artifacts: &Path, output: &Path, key: &Path) -> io::Result<()> {
    let key = read_key(key)?;
    let configured = serde_json::from_str::<serde_json::Value>(include_str!(
        "../../../release.json"
    ))?["publicKey"]
        .as_str()
        .map(str::to_owned)
        .unwrap_or_default();
    if configured != hex(key.verifying_key().as_bytes()) {
        return Err(io::Error::other(
            "the signing key does not match release.json's publicKey, so installed copies would reject this release",
        ));
    }
    let folder = format!("frame-comfort-{VERSION}");
    fs::create_dir_all(output)?;
    package(artifacts, &output.join(&folder))?;
    let name = format!("{folder}-linux-aarch64.tar.gz");
    let archive: PathBuf = output.join(&name);
    let status = Command::new("tar")
        .args(["--owner=0", "--group=0", "--numeric-owner", "-czf"])
        .arg(&archive)
        .arg("-C")
        .arg(output)
        .arg(&folder)
        .status()?;
    if !status.success() {
        return Err(io::Error::other("tar failed"));
    }
    fs::write(
        output.join(format!("{name}.sha256")),
        format!("{}  {name}\n", sha256(&archive)?),
    )?;
    let signature = key.sign(&fs::read(&archive)?);
    fs::write(
        output.join(format!("{name}.sig")),
        hex(&signature.to_bytes()) + "\n",
    )?;
    println!("{}", archive.display());
    Ok(())
}
