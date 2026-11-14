fn main() {
    #[cfg(feature = "steamvr")]
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        let out = std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap());
        let shaders = std::path::Path::new("../../shaders");
        println!("cargo:rerun-if-env-changed=GLSLC");
        let glslc = std::env::var_os("GLSLC").unwrap_or_else(|| "glslc".into());
        for shader in ["overlay.vert", "overlay.frag", "restframe.frag"] {
            let source = shaders.join(shader);
            println!("cargo:rerun-if-changed={}", source.display());
            let status = std::process::Command::new(&glslc)
                .args(["--target-env=vulkan1.1", "-O"])
                .arg(&source)
                .arg("-o")
                .arg(out.join(format!("{shader}.spv")))
                .status()
                .expect("run glslc");
            assert!(status.success(), "glslc failed for {}", source.display());
        }
        let header = "sdk/openvr_capi.h";
        println!("cargo:rerun-if-changed={header}");
        bindgen::Builder::default()
            .header(header)
            .allowlist_type("VR_IVR(System|Overlay|Compositor|Screenshots)_FnTable|EVRApplicationType|VRVulkanTextureData_t|EVRMouseButton")
            .allowlist_var(
                "(IVR(System|Overlay|Compositor|Screenshots)_Version|k_unMaxTrackedDeviceCount)",
            )
            .generate_comments(false)
            .clang_arg(format!("--target={}", std::env::var("TARGET").unwrap()))
            .generate()
            .expect("generate pinned OpenVR bindings")
            .write_to_file(
                std::path::PathBuf::from(std::env::var_os("OUT_DIR").unwrap()).join("openvr.rs"),
            )
            .expect("write OpenVR bindings");
    }
}
