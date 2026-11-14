#[cfg(all(target_os = "linux", feature = "steamvr"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use comfort_runtime::openvr::{OpenVr, vulkan_device_names};
    use std::path::PathBuf;

    let path = std::env::args_os().nth(1).map(PathBuf::from);
    let devices = vulkan_device_names()?;
    println!("Vulkan devices: {}", devices.join(", "));
    if let Some(path) = path {
        // SAFETY: The diagnostic operator supplies the installed OpenVR client.
        let vr = unsafe { OpenVr::init(&path) }?;
        println!("Recommended size: {:?}", vr.recommended_size()?);
        let _overlay = vr.create_overlay(
            c"org.vrwindow.comfort.rust-probe",
            c"Frame Comfort Rust probe",
        )?;
        println!("Created a hidden overlay.");
    }
    Ok(())
}

#[cfg(not(all(target_os = "linux", feature = "steamvr")))]
fn main() {
    eprintln!("The SteamVR probe requires Linux and the steamvr feature.");
    std::process::exit(1);
}
