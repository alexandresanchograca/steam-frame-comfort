use super::*;
use crate::vulkan::{ContextOptions, fixture::keep_driver_loaded};

#[test]
#[ignore = "requires Vulkan"]
fn scene_layouts_preserve_frame_codes_and_stereo_colors() -> Result<()> {
    let context = Context::new(ContextOptions::default())?;
    keep_driver_loaded(&context);
    let extent = vk::Extent2D {
        width: 64,
        height: 64,
    };
    let format = vk::Format::R8G8B8A8_UNORM;
    let eye_size = (extent.width * extent.height * 4) as usize;
    let frame = 0xa1b2_c3d4_u32;
    let mut painter = ScenePainter::new(&context, extent, format, true)?;
    let mut readback = Buffer::new(&context, 2 * eye_size, vk::BufferUsageFlags::TRANSFER_DST)?;
    let mut commands = Commands::new(&context)?;

    // Match the three destination arrangements in the live scene. Decode
    // pixels from actual Vulkan images to catch offset and layer mistakes.
    for (layers, wide) in [(1, false), (2, false), (1, true)] {
        let target_extent = vk::Extent2D {
            width: extent.width * if wide { 2 } else { 1 },
            height: extent.height,
        };
        let images = (0..if layers == 1 && !wide { 2 } else { 1 })
            .map(|_| SceneImage::new(&context, target_extent, layers, format))
            .collect::<Result<Vec<_>>>()?;
        let eyes = [
            EyeTarget {
                image: images[0].handle,
                x: 0,
                layer: 0,
            },
            EyeTarget {
                image: images.last().expect("one or two images").handle,
                x: if wide { extent.width } else { 0 },
                layer: layers - 1,
            },
        ];
        let release = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
        for (code, old) in [
            (frame, vk::ImageLayout::UNDEFINED),
            (frame.wrapping_add(1), release),
        ] {
            // SAFETY: The images are locally owned, and all earlier
            // submissions finished before the rewrite.
            unsafe {
                painter.paint(eyes, code, true, Some([[4, 29]; 2]), old, release)?;
            }
            commands.begin()?;
            for (eye, target) in eyes.into_iter().enumerate() {
                let region = vk::BufferImageCopy::default()
                    .buffer_offset((eye * eye_size) as u64)
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(COLOR)
                            .base_array_layer(target.layer)
                            .layer_count(1),
                    )
                    .image_offset(vk::Offset3D {
                        x: target.x as i32,
                        y: 0,
                        z: 0,
                    })
                    .image_extent(vk::Extent3D {
                        width: extent.width,
                        height: extent.height,
                        depth: 1,
                    });
                // SAFETY: All source subresources were transitioned to
                // TRANSFER_SRC_OPTIMAL and both target ranges are disjoint.
                unsafe {
                    context.device.cmd_copy_image_to_buffer(
                        commands.handle,
                        target.image,
                        release,
                        readback.handle,
                        &[region],
                    );
                }
            }
            commands.submit()?;
            commands.wait()?;
            let pixels = readback.read()?;
            for eye in 0..2 {
                let texel = |x, y| {
                    let at = eye * eye_size + ((y * extent.width + x) * 4) as usize;
                    &pixels[at..at + 4]
                };
                let cell = 4;
                let mut decoded = 0_u64;
                for bit in 0..48 {
                    let p = texel(
                        16 + bit % 8 * cell + cell / 2,
                        16 + bit / 8 * cell + cell / 2,
                    );
                    assert!(p == [8, 8, 8, 255] || p == [240, 240, 240, 255]);
                    decoded |= u64::from(p[0] == 240) << bit;
                }
                assert_eq!(decoded, u64::from(code) << 16 | 0xa55a);
                let [r, _, b, _] = <[u8; 4]>::try_from(texel(0, 54)).unwrap();
                assert!(if eye == 0 { r > b } else { b > r }, "swapped eye colors");
                assert_eq!(texel(7, 2), &[32, 208, 32, 255]);
                let bar_x = (u64::from(code) * 6 % 49) as u32;
                if bar_x >= 12 {
                    assert_eq!(texel(bar_x, 2), &[240, 240, 240, 255]);
                }
                let tag = (24 + code.wrapping_mul(73) % 208) as u8;
                assert_eq!(
                    texel(if eye == 0 { 42 } else { 32 }, 56),
                    &[tag, tag, tag, 255]
                );
            }
        }
    }
    Ok(())
}
