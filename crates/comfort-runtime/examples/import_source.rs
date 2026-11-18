//! Import the first image supplied by the C++ source fixture.
#[cfg(all(target_os = "linux", feature = "steamvr"))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use comfort_core::geometry::{Face, Pose, Projection};
    use comfort_runtime::{
        source_client::SourceClient,
        vulkan::{
            Capture, Context, ContextOptions, ImportedImage, ShellParams, ShellRenderer,
            SnapshotRenderer, SourceView,
        },
    };
    use std::{
        os::fd::AsFd,
        path::PathBuf,
        time::{Duration, Instant},
    };

    let socket = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: import_source SOCKET")?;
    let mut client = SourceClient::default();
    client.connect(&socket)?;
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(5) {
        client.poll(start.elapsed().as_nanos() as u64)?;
        if let (Some(hello), Some(frame)) = (client.hello, client.frame)
            && let Some((&id, &layout)) = frame
                .ids
                .iter()
                .zip(&frame.layouts)
                .take(frame.count as usize)
                .next()
            && let Some(source) = client.images.get(&id)
        {
            let context = Context::new(ContextOptions {
                device_uuid: Some(hello.device_uuid),
                require_dma_buf: source.description.handle_type
                    == ash::vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT.as_raw(),
                validation: std::env::var_os("FRAME_COMFORT_VALIDATION").is_some(),
                ..Default::default()
            })?;
            if context.driver_uuid() != hello.driver_uuid {
                return Err("Vulkan driver differs from producer".into());
            }
            println!(
                "Importing {}x{} observer image on {}",
                source.description.width,
                source.description.height,
                context.name()
            );
            // SAFETY: The C++ Vulkan fixture created the image on the same
            // device and supplied a live external-memory descriptor.
            let imported = std::sync::Arc::new(unsafe {
                ImportedImage::import(
                    std::sync::Arc::clone(&context),
                    source.description,
                    source.fd.as_fd(),
                )?
            });
            let mut renderer = SnapshotRenderer::new(std::sync::Arc::clone(&context))?;
            let view = || SourceView {
                image: &imported,
                layout: ash::vk::ImageLayout::from_raw(layout as i32),
                layer: 0,
                crop: [0.0, 0.0, 1.0, 1.0],
                flip: [false; 2],
                external: true,
            };
            // SAFETY: The fixture reports a completed release to the external
            // queue family and keeps the pixel contents unchanged.
            let mut snapshot = unsafe { renderer.capture([view(), view()], [256, 256])? };
            let pixels = renderer.readback(&mut snapshot)?;
            let at = (32 * 512 + 32) * 4;
            let pixel = &pixels[at..at + 4];
            if pixel != [220, 200, 0, 255] {
                return Err(format!("unexpected snapshot color: {pixel:?}").into());
            }
            assert_eq!(pixel, &pixels[at + 256 * 4..at + 256 * 4 + 4]);
            println!("Captured matching left and right fixture pixels: {pixel:?}");
            let mut shell = ShellRenderer::new(context, 64)?;
            let capture = Capture {
                snapshot: &snapshot,
                eyes: [(Pose::default(), Projection::default()); 2],
            };
            let mut params = ShellParams {
                captures: Some([capture; 2]),
                faces: 1 << Face::Front as u32,
                ..Default::default()
            };
            params.aperture.half_width_tan = 0.25;
            params.aperture.half_height_tan = 0.25;
            shell.render(params, 0)?;
            assert!(shell.render(params, 0).is_err());
            assert!(shell.output_image(0, Face::Back).is_none());
            let pixels = shell.read_face(0, Face::Front)?;
            for eye in 0..2 {
                let center = (32 * 128 + 32 + eye * 64) * 4;
                let edge = (8 * 128 + 8 + eye * 64) * 4;
                assert_eq!(pixels[center + 3], 0);
                assert_eq!(&pixels[edge..edge + 4], &[220, 200, 0, 255]);
            }
            // SAFETY: This diagnostic has no external consumers.
            unsafe { shell.retire(0) }?;
            shell.render(params, 0)?;
            println!("Rendered the source into a stereo shell face with a transparent center.");
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    Err("no importable image received from the observer".into())
}

#[cfg(not(all(target_os = "linux", feature = "steamvr")))]
fn main() {
    eprintln!("The source import diagnostic requires Linux and the steamvr feature.");
    std::process::exit(1);
}
