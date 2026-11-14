//! Vulkan transfers shared by the manual OpenVR and OpenXR scene diagnostics.

#[cfg(feature = "scene-fixtures")]
mod openvr;
#[cfg(feature = "scene-fixtures")]
mod openxr;

#[cfg(feature = "scene-fixtures")]
pub use openvr::{EyePatternOptions, run_openvr};
#[cfg(feature = "scene-fixtures")]
pub use openxr::{Layout, XrPatternOptions, run_openxr};

#[cfg(test)]
mod tests;

use super::{
    Context, Result,
    gpu::{Buffer, COLOR, Commands, color_range},
    pattern::{CODE_CELLS, ORIGIN, code_cell_size, stereo_pixel, write_code},
};
use ash::vk;
use serde_json::json;
use std::{io, sync::Arc};

#[derive(Clone, Copy)]
struct EyeTarget {
    image: vk::Image,
    x: u32,
    layer: u32,
}

struct SceneImage {
    context: Arc<Context>,
    handle: vk::Image,
    memory: vk::DeviceMemory,
}

impl SceneImage {
    fn new(
        context: &Arc<Context>,
        extent: vk::Extent2D,
        layers: u32,
        format: vk::Format,
    ) -> Result<Self> {
        let mut image = Self {
            context: Arc::clone(context),
            handle: vk::Image::null(),
            memory: vk::DeviceMemory::null(),
        };
        let device = &context.device;
        // SAFETY: The fixture owns these resources. Their handles start null
        // for cleanup if creation, allocation or binding fails.
        unsafe {
            image.handle = device.create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(format)
                    .extent(vk::Extent3D {
                        width: extent.width,
                        height: extent.height,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(layers)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .usage(
                        vk::ImageUsageFlags::TRANSFER_DST
                            | vk::ImageUsageFlags::TRANSFER_SRC
                            | vk::ImageUsageFlags::SAMPLED,
                    ),
                None,
            )?;
            let req = device.get_image_memory_requirements(image.handle);
            let index = context
                .memory_type(req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
                .ok_or_else(|| io::Error::other("no local scene image memory"))?;
            image.memory = device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(index),
                None,
            )?;
            device.bind_image_memory(image.handle, image.memory, 0)?;
        }
        Ok(image)
    }
}

impl Drop for SceneImage {
    fn drop(&mut self) {
        // SAFETY: Consumers finish using the image before teardown. A null
        // handle is permitted for partial allocations.
        unsafe {
            self.context.device.destroy_image(self.handle, None);
            self.context.device.free_memory(self.memory, None);
        }
    }
}

struct ScenePainter {
    commands: Commands,
    context: Arc<Context>,
    background: Buffer,
    decorations: Buffer,
    code: Option<(Buffer, Vec<u8>)>,
    cell: u32,
    extent: vk::Extent2D,
    bar_width: u32,
    marker_offset: u64,
}

const MARKER_WIDTH: u32 = 8;

impl ScenePainter {
    fn new(
        context: &Arc<Context>,
        extent: vk::Extent2D,
        format: vk::Format,
        frame_code: bool,
    ) -> Result<Self> {
        let vk::Extent2D { width, height } = extent;
        let cell = code_cell_size(width, height)
            .ok_or_else(|| io::Error::other("scene eye image is too small"))?;
        let eye_bytes = usize::try_from(u64::from(width) * u64::from(height))
            .ok()
            .and_then(|pixels| pixels.checked_mul(4))
            .and_then(|size| size.checked_mul(2))
            .ok_or_else(|| io::Error::other("scene eye image is too large"))?
            / 2;
        let mut pixels = vec![0; 2 * eye_bytes];
        let bgra = matches!(
            format,
            vk::Format::B8G8R8A8_SRGB | vk::Format::B8G8R8A8_UNORM
        );
        for (eye, image) in pixels.chunks_exact_mut(eye_bytes).enumerate() {
            for (index, texel) in image.chunks_exact_mut(4).enumerate() {
                let mut rgba = if frame_code {
                    stereo_pixel(
                        (index % width as usize) as u32,
                        (index / width as usize) as u32,
                        width,
                        height,
                        eye,
                    )
                } else if eye == 0 {
                    [204, 8, 5, 255]
                } else {
                    [5, 8, 204, 255]
                };
                if bgra {
                    rgba.swap(0, 2);
                }
                texel.copy_from_slice(&rgba);
            }
        }
        let usage = vk::BufferUsageFlags::TRANSFER_SRC;
        let mut background = Buffer::new(context, pixels.len(), usage)?;
        background.write(&pixels)?;

        let bar_width = 16.max(width / 24);
        let bar_bytes = bar_width as usize * height as usize * 4;
        let marker_bytes = MARKER_WIDTH as usize * width.max(height) as usize * 4;
        let mut pixels = vec![0; bar_bytes + marker_bytes];
        for texel in pixels[..bar_bytes].chunks_exact_mut(4) {
            texel.copy_from_slice(&[240, 240, 240, 255]);
        }
        for texel in pixels[bar_bytes..].chunks_exact_mut(4) {
            texel.copy_from_slice(&[32, 208, 32, 255]);
        }
        let mut decorations = Buffer::new(context, pixels.len(), usage)?;
        decorations.write(&pixels)?;

        let code = if frame_code {
            let size = CODE_CELLS * (cell as usize).pow(2) * 4
                + (width / 8) as usize * (height / 8) as usize * 4;
            println!(
                "{}",
                json!({"frameCode":{"enabled":true,"bits":32,"syncBits":16,"syncHex":"a55a",
                    "originPerEye":[ORIGIN,ORIGIN],"cellPixels":cell,"grid":[8,6],
                    "order":"row-major-lsb-first-sync-then-frame","zeroRGB":8,"oneRGB":240}})
            );
            Some((Buffer::new(context, size, usage)?, vec![0; size]))
        } else {
            None
        };
        Ok(Self {
            commands: Commands::new(context)?,
            context: Arc::clone(context),
            background,
            decorations,
            code,
            cell,
            extent,
            bar_width,
            marker_offset: bar_bytes as u64,
        })
    }

    /// # Safety
    /// Both destinations must be writable on this queue in the supplied layout.
    /// Regions must fit within the image's subresources and must not overlap.
    unsafe fn paint(
        &mut self,
        eyes: [EyeTarget; 2],
        frame: u32,
        bar: bool,
        markers: Option<[[u32; 2]; 2]>,
        old: vk::ImageLayout,
        release: vk::ImageLayout,
    ) -> Result<()> {
        if let Some((buffer, pixels)) = &mut self.code {
            write_code(pixels, self.cell as usize, frame);
            buffer.write(pixels)?;
        }
        self.commands.begin()?;
        let device = &self.context.device;
        let cmd = self.commands.handle;
        let transfer = vk::PipelineStageFlags::TRANSFER;
        let writing = vk::AccessFlags::TRANSFER_WRITE;
        let transfer_dst = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
        let (source_stage, source_access) = if old == vk::ImageLayout::UNDEFINED {
            (
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::AccessFlags::empty(),
            )
        } else {
            (transfer, vk::AccessFlags::TRANSFER_READ)
        };
        let ranges = eyes.map(|eye| {
            vk::ImageMemoryBarrier::default()
                .image(eye.image)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .subresource_range(color_range(1).base_array_layer(eye.layer))
        });
        let count = if eyes[0].image == eyes[1].image && eyes[0].layer == eyes[1].layer {
            1
        } else {
            2
        };
        let enter = ranges.map(|range| {
            range
                .old_layout(old)
                .new_layout(transfer_dst)
                .src_access_mask(source_access)
                .dst_access_mask(writing)
        });
        // SAFETY: The caller controls the images and queue for this recording.
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                source_stage,
                transfer,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &enter[..count],
            )
        };
        let vk::Extent2D { width, height } = self.extent;
        let bar_x = (u64::from(frame) * 6 % u64::from(width - self.bar_width + 1)) as u32;
        let order_writes = || {
            let barrier = vk::MemoryBarrier::default()
                .src_access_mask(writing)
                .dst_access_mask(writing);
            // SAFETY: The recorded transfers may overlap and need a write dependency.
            unsafe {
                device.cmd_pipeline_barrier(
                    cmd,
                    transfer,
                    transfer,
                    vk::DependencyFlags::empty(),
                    &[barrier],
                    &[],
                    &[],
                )
            };
        };
        for (eye, target) in eyes.into_iter().enumerate() {
            let region = |buffer_offset, [x, y]: [u32; 2], [width, height]: [u32; 2]| {
                vk::BufferImageCopy::default()
                    .buffer_offset(buffer_offset)
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(COLOR)
                            .base_array_layer(target.layer)
                            .layer_count(1),
                    )
                    .image_offset(vk::Offset3D {
                        x: (target.x + x) as i32,
                        y: y as i32,
                        z: 0,
                    })
                    .image_extent(vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    })
            };
            let copy = |buffer: &Buffer, regions: &[vk::BufferImageCopy]| {
                // SAFETY: All destination ranges belong to this eye, and the buffers
                // contain exactly the texels copied by these regions.
                unsafe {
                    device.cmd_copy_buffer_to_image(
                        cmd,
                        buffer.handle,
                        target.image,
                        transfer_dst,
                        regions,
                    )
                };
            };
            let byte_offset = eye as u64 * u64::from(width) * u64::from(height) * 4;
            copy(
                &self.background,
                &[region(byte_offset, [0, 0], [width, height])],
            );
            if bar {
                order_writes();
                copy(
                    &self.decorations,
                    &[region(0, [bar_x, 0], [self.bar_width, height])],
                );
            }
            if let Some(markers) = markers {
                let [x, y] = markers[eye];
                order_writes();
                copy(
                    &self.decorations,
                    &[region(self.marker_offset, [x, 0], [MARKER_WIDTH, height])],
                );
                order_writes();
                copy(
                    &self.decorations,
                    &[region(self.marker_offset, [0, y], [width, MARKER_WIDTH])],
                );
            }
            if let Some((buffer, _)) = &self.code {
                let cell = self.cell;
                let cell_bytes = u64::from(cell).pow(2) * 4;
                let cells: [vk::BufferImageCopy; CODE_CELLS] = std::array::from_fn(|index| {
                    let index = index as u32;
                    region(
                        u64::from(index) * cell_bytes,
                        [ORIGIN + (index % 8) * cell, ORIGIN + (index / 8) * cell],
                        [cell; 2],
                    )
                });
                order_writes();
                copy(buffer, &cells);
                order_writes();
                let tag_x = width / 2 + if eye == 0 { width * 5 / 32 } else { 0 };
                copy(
                    buffer,
                    &[region(
                        CODE_CELLS as u64 * cell_bytes,
                        [tag_x, height * 7 / 8],
                        [width / 8, height / 8],
                    )],
                );
            }
        }
        let (destination_stage, destination_access) =
            if release == vk::ImageLayout::TRANSFER_SRC_OPTIMAL {
                (transfer, vk::AccessFlags::TRANSFER_READ)
            } else {
                (
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                    vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                )
            };
        let leave = ranges.map(|range| {
            range
                .old_layout(transfer_dst)
                .new_layout(release)
                .src_access_mask(writing)
                .dst_access_mask(destination_access)
        });
        // SAFETY: The same subresources are returned in the runtime's release layout.
        unsafe {
            device.cmd_pipeline_barrier(
                cmd,
                transfer,
                destination_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &leave[..count],
            )
        };
        self.commands.submit()?;
        self.commands.wait()
    }
}
