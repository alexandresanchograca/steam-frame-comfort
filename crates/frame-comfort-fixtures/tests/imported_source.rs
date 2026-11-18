//! Export, reimport and observe a real Vulkan image on a private source socket.
#![cfg(all(target_os = "linux", feature = "vulkan-fixtures"))]

use ash::vk;
use comfort_core::geometry::{Aperture, Face, Pose, Projection};
use comfort_runtime::{
    mailbox::monotonic_ns,
    source_client::SourceClient,
    vulkan::{
        Capture, Context, ContextOptions, ImportedImage, ShellParams, ShellRenderer,
        SnapshotRenderer, SourceView,
        fixture::{keep_driver_loaded, linear_p010_source},
    },
};
use std::{
    fs::{self, File},
    os::{fd::AsFd, unix::fs::PermissionsExt},
    process::{Child, Command, Stdio},
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

struct Producer(Child);

impl Drop for Producer {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

#[test]
#[ignore = "requires Vulkan external memory and the observer layer"]
fn imported_opaque_fd_reaches_the_stereo_shell() -> Result<()> {
    assert!(
        check_source("--opaque-fd", vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD)?,
        "the Vulkan test driver must support an opaque FD fixture"
    );
    Ok(())
}

#[test]
#[ignore = "requires Vulkan DMA-BUF with a linear DRM modifier and the observer layer"]
fn imported_dma_buf_reaches_the_stereo_shell() -> Result<()> {
    if !check_source("--dma-buf", vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)? {
        eprintln!(
            "SKIP: the selected Vulkan device cannot import and export DRM-linear RGBA DMA-BUFs"
        );
    }
    Ok(())
}

/// Ported from the C++ baseline's renderer GPU test: a 10-bit P010 source
/// with explicit BT.709 narrow-range metadata goes through the YCbCr sampler
/// conversion into the snapshot.
#[test]
#[ignore = "requires a Vulkan device that samples linear P010"]
fn p010_source_converts_through_the_ycbcr_sampler() -> Result<()> {
    // Black (Y 64) on the left, white (Y 940) on the right.
    p010_snapshot_bands(
        vk::SamplerYcbcrModelConversion::YCBCR_709,
        vk::SamplerYcbcrRange::ITU_NARROW,
        [64, 940],
        [0..=7, 248..=255],
    )
}

/// Stream images arrive with an identity model, which the compositor
/// converts as full range (measured against its own output on November 14).
/// Y 64 must come out near 16, where limited range would make it black,
/// whatever range the metadata names.
#[test]
#[ignore = "requires a Vulkan device that samples linear P010"]
fn identity_p010_converts_as_full_range() -> Result<()> {
    p010_snapshot_bands(
        vk::SamplerYcbcrModelConversion::YCBCR_IDENTITY,
        vk::SamplerYcbcrRange::ITU_NARROW,
        [64, 1023],
        [12..=20, 248..=255],
    )
}

/// Capture a P010 source split into two luma bands with neutral chroma, and
/// check each band's grey level in both eyes of the snapshot.
fn p010_snapshot_bands(
    model: vk::SamplerYcbcrModelConversion,
    range: vk::SamplerYcbcrRange,
    luma_bands: [u16; 2],
    expected: [std::ops::RangeInclusive<u8>; 2],
) -> Result<()> {
    let context = Context::new(ContextOptions::default())?;
    keep_driver_loaded(&context);
    let (width, height) = (64_u32, 64_u32);
    let (w, h) = (width as usize, height as usize);
    let luma: Vec<u16> = (0..w * h)
        .map(|index| luma_bands[usize::from(index % w >= w / 2)])
        .collect();
    let chroma = vec![[512_u16, 512]; (w / 2) * (h / 2)];
    let Some(source) = linear_p010_source(&context, width, height, &luma, &chroma, model, range)?
    else {
        eprintln!("SKIP: the selected Vulkan device cannot sample linear P010 through YCbCr");
        return Ok(());
    };
    let view = || SourceView {
        image: &source,
        layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        layer: 0,
        crop: [0.0, 0.0, 1.0, 1.0],
        flip: [false; 2],
        external: false,
    };
    let mut renderer = SnapshotRenderer::new(Arc::clone(&context))?;
    // SAFETY: The fixture finished writing the source before returning, and
    // nothing else uses it.
    let mut snapshot = unsafe { renderer.capture([view(), view()], [width, height])? };
    let pixels = renderer.readback(&mut snapshot)?;
    for eye in 0..2 {
        for (band, column) in [w / 5, w * 4 / 5].into_iter().enumerate() {
            let at = ((h / 2) * w * 2 + eye * w + column) * 4;
            let rgb = &pixels[at..at + 3];
            assert!(
                rgb.iter().all(|value| expected[band].contains(value)),
                "eye {eye} Y {}: {rgb:?} outside {:?}",
                luma_bands[band],
                expected[band]
            );
        }
    }
    eprintln!("P010 {model:?} conversion ok");
    Ok(())
}

fn check_source(flag: &str, handle: vk::ExternalMemoryHandleTypeFlags) -> Result<bool> {
    let layer_dir = std::env::var_os("FRAME_COMFORT_LAYER_DIR")
        .ok_or("run with cargo xtask e2e to supply the observer manifest")?;
    let layers = match std::env::var("VK_INSTANCE_LAYERS") {
        Ok(layers) if !layers.is_empty() => format!("VK_LAYER_VRW_comfort_source:{layers}"),
        _ => "VK_LAYER_VRW_comfort_source".into(),
    };
    let temp = tempfile::tempdir()?;
    let socket = temp.path().join("source.sock");
    let stdout = temp.path().join("producer.stdout");
    let stderr = temp.path().join("producer.stderr");
    let binary = std::env::var_os("FRAME_COMFORT_SOURCE_PRODUCER")
        .unwrap_or_else(|| env!("CARGO_BIN_EXE_frame-comfort-source-producer").into());
    let mut producer = Producer(
        Command::new(binary)
            .arg(flag)
            .env("VK_ADD_LAYER_PATH", layer_dir)
            .env("VK_INSTANCE_LAYERS", layers)
            .env("VRW_COMFORT_SOURCE_COMM", "")
            .env("VRW_COMFORT_SOURCE_SOCKET", &socket)
            .stdin(Stdio::piped())
            .stdout(File::create(&stdout)?)
            .stderr(File::create(&stderr)?)
            .spawn()?,
    );

    let deadline = Instant::now() + Duration::from_secs(10);
    while !fs::read_to_string(&stdout)?.contains("source ready") || !socket.exists() {
        if let Some(status) = producer.0.try_wait()? {
            let output = fs::read_to_string(&stdout)?;
            let errors = fs::read_to_string(&stderr)?;
            assert_validation_clean(&output, &errors);
            if status.code() == Some(77) {
                return Ok(false);
            }
            return Err(format!("producer exited with {status}\n{output}\n{errors}").into());
        }
        if Instant::now() >= deadline {
            return Err(format!(
                "producer did not publish its external handoff\n{}\n{}",
                fs::read_to_string(&stdout)?,
                fs::read_to_string(&stderr)?,
            )
            .into());
        }
        thread::sleep(Duration::from_millis(10));
    }

    let mode = socket.symlink_metadata()?.permissions().mode() & 0o777;
    assert_eq!(mode, 0o600);
    let mut client = SourceClient::default();
    client.connect(&socket)?;
    let deadline = Instant::now() + Duration::from_secs(5);
    let (hello, frame) = loop {
        client.poll(monotonic_ns())?;
        if let (Some(hello), Some(frame)) = (client.hello, client.frame) {
            break (hello, frame);
        }
        if Instant::now() >= deadline {
            return Err("observer did not report a completed external acquire".into());
        }
        thread::sleep(Duration::from_millis(5));
    };
    assert_eq!(hello.pid, producer.0.id());
    assert_eq!(frame.count, 1);
    assert_eq!(frame.truncated, 0);
    assert_eq!(frame.layouts[0], vk::ImageLayout::GENERAL.as_raw() as u32);
    let source = client
        .images
        .get(&frame.ids[0])
        .ok_or("acquired image was not published")?;
    let d = source.description;
    assert_eq!(d.exported, 0, "test the import hook's passed FD");
    assert_eq!(d.handle_type, handle.as_raw());
    assert_eq!([d.width, d.height, d.array_layers], [256, 256, 1]);
    let dma = handle == vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT;
    if dma {
        assert_eq!(
            d.tiling,
            vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT.as_raw() as u32
        );
        assert_eq!(d.modifier, 0);
        assert_eq!(d.plane_count, 1);
        assert!(d.planes[0].row_pitch >= 256 * 4);
    }
    let context = Context::new(ContextOptions {
        device_uuid: Some(hello.device_uuid),
        require_dma_buf: dma,
        ..Default::default()
    })?;
    keep_driver_loaded(&context);
    assert_eq!(context.driver_uuid(), hello.driver_uuid);
    // SAFETY: The fixture exported this image on this physical device and
    // remains alive and idle with the image released to EXTERNAL.
    let image =
        Arc::new(unsafe { ImportedImage::import(Arc::clone(&context), d, source.fd.as_fd())? });
    let view = || SourceView {
        image: &image,
        layout: vk::ImageLayout::GENERAL,
        layer: 0,
        crop: [0.0, 0.0, 1.0, 1.0],
        flip: [false; 2],
        external: true,
    };
    let mut renderer = SnapshotRenderer::new(Arc::clone(&context))?;
    // SAFETY: The producer waits for this test to finish before destroying
    // the shared image. capture waits for its work and releases ownership.
    let mut snapshot = unsafe { renderer.capture([view(), view()], [128, 96])? };
    let pixels = renderer.readback(&mut snapshot)?;
    assert_eq!(pixels.len(), 256 * 96 * 4);
    for eye in 0..2 {
        for (x, y, expected) in [
            (16, 48, [220, 200, 0, 255]),
            (64, 12, [0, 200, 0, 255]),
            (64, 48, [0, 0, 200, 255]),
        ] {
            let offset = (y * 256 + eye * 128 + x) * 4;
            assert_eq!(&pixels[offset..offset + 4], &expected);
        }
    }

    let capture = Capture {
        snapshot: &snapshot,
        eyes: [(Pose::default(), Projection::default()); 2],
    };
    let mut shell = ShellRenderer::new(context, 64)?;
    shell.render(
        ShellParams {
            captures: Some([capture; 2]),
            aperture: Aperture {
                half_width_tan: 0.25,
                half_height_tan: 0.25,
                ..Aperture::default()
            },
            faces: 1 << Face::Front as u32,
            ..Default::default()
        },
        0,
    )?;
    let face = shell.read_face(0, Face::Front)?;
    for eye in 0..2 {
        let centre = (32 * 128 + eye * 64 + 32) * 4;
        let edge = (16 * 128 + eye * 64 + 8) * 4;
        assert_eq!(face[centre + 3], 0);
        assert_eq!(&face[edge..edge + 4], &[220, 200, 0, 255]);
    }
    drop(shell);
    drop(renderer);
    drop(image);
    drop(client);

    drop(producer.0.stdin.take());
    let deadline = Instant::now() + Duration::from_secs(5);
    let status = loop {
        if let Some(status) = producer.0.try_wait()? {
            break status;
        }
        if Instant::now() >= deadline {
            return Err("producer did not shut down after its stdin closed".into());
        }
        thread::sleep(Duration::from_millis(5));
    };
    let output = fs::read_to_string(stdout)?;
    let errors = fs::read_to_string(stderr)?;
    assert!(status.success(), "{output}\n{errors}");
    assert_validation_clean(&output, &errors);
    assert!(!socket.exists());
    Ok(true)
}

fn assert_validation_clean(stdout: &str, stderr: &str) {
    for output in [stdout, stderr] {
        assert!(
            !output.contains("Validation Error") && !output.contains("Validation Warning"),
            "{output}"
        );
    }
}
