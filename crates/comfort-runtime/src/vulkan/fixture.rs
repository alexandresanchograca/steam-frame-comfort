use super::{
    Context, ContextOptions, ImportedImage, Result, Snapshot,
    gpu::{Buffer, COLOR, Commands, RGBA, Target, create_render_pass},
    pattern::{CODE_CELLS, stereo_pixel, write_code},
};
use ash::vk;
use comfort_protocol::{ImageAdd, Ycbcr};
use std::{
    io,
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

const SIDE: u32 = 512;
const EYE_BYTES: usize = SIDE as usize * SIDE as usize * 4;
const CODE_CELL: u32 = 16;
const CODE_BYTES: usize = CODE_CELL as usize * CODE_CELL as usize * 4;
const TAG: u32 = SIDE / 8;
const TAG_BYTES: usize = TAG as usize * TAG as usize * 4;
const CODE_COUNT: usize = CODE_CELLS;
const STAGING_BYTES: usize = CODE_BYTES * CODE_COUNT + TAG_BYTES;
const SLOTS: usize = 3;
const RIGHT_SLOTS: [usize; SLOTS] = [4, 5, 3];

mod imported_source;
pub use imported_source::publish_imported_source;

#[derive(Clone, Copy)]
pub enum Cadence {
    Reuse,
    Rotate,
    Burst,
    SeparateEyes(Duration),
}

struct Export {
    context: Arc<Context>,
    image: vk::Image,
    memory: vk::DeviceMemory,
}

impl Export {
    fn new(context: &Arc<Context>, layers: u32) -> Result<Self> {
        let mut image = Self {
            context: Arc::clone(context),
            image: vk::Image::null(),
            memory: vk::DeviceMemory::null(),
        };
        let device = &context.device;
        let kind = vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD;
        let mut external = vk::ExternalMemoryImageCreateInfo::default().handle_types(kind);
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(RGBA)
            .extent(vk::Extent3D {
                width: SIDE,
                height: SIDE,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(layers)
            .samples(vk::SampleCountFlags::TYPE_1)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
            .push_next(&mut external);
        // SAFETY: The image is created and exclusively owned by this fixture.
        unsafe {
            image.image = device.create_image(&info, None)?;
            let requirements = device.get_image_memory_requirements(image.image);
            let index = context
                .memory_type(
                    requirements.memory_type_bits,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                )
                .ok_or_else(|| io::Error::other("no device memory for the cadence image"))?;
            let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(image.image);
            let mut export = vk::ExportMemoryAllocateInfo::default().handle_types(kind);
            image.memory = device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(index)
                    .push_next(&mut dedicated)
                    .push_next(&mut export),
                None,
            )?;
            device.bind_image_memory(image.image, image.memory, 0)?;
        }
        Ok(image)
    }
}

impl Drop for Export {
    fn drop(&mut self) {
        // SAFETY: Producer::write waits for all fixture work before returning.
        unsafe {
            self.context.device.destroy_image(self.image, None);
            self.context.device.free_memory(self.memory, None);
        }
    }
}

struct Producer {
    commands: Commands,
    staging: Buffer,
    initial: Buffer,
    images: Vec<Export>,
    context: Arc<Context>,
    split: bool,
    bytes: Vec<u8>,
}

impl Producer {
    fn new(split: bool) -> Result<Self> {
        let context = Context::new(ContextOptions {
            device_extensions: &[ash::khr::external_memory_fd::NAME],
            high_priority: true,
            ..Default::default()
        })?;
        let count = if split { 2 * SLOTS } else { SLOTS };
        let layers = if split { 1 } else { 2 };
        let images = (0..count)
            .map(|_| Export::new(&context, layers))
            .collect::<Result<Vec<_>>>()?;
        let mut initial = Buffer::new(&context, 2 * EYE_BYTES, vk::BufferUsageFlags::TRANSFER_SRC)?;
        initial.write(&pattern())?;
        Ok(Self {
            commands: Commands::new(&context)?,
            staging: Buffer::new(&context, STAGING_BYTES, vk::BufferUsageFlags::TRANSFER_SRC)?,
            initial,
            images,
            context,
            split,
            bytes: vec![0; STAGING_BYTES],
        })
    }

    fn write(
        &mut self,
        frame: u32,
        eyes: std::ops::Range<usize>,
        slot: Option<usize>,
    ) -> Result<()> {
        write_code(&mut self.bytes, CODE_CELL as usize, frame);
        self.staging.write(&self.bytes)?;
        self.commands.begin()?;
        for eye in eyes {
            for candidate in 0..SLOTS {
                if slot.is_none_or(|slot| slot == candidate) {
                    self.record(eye, candidate, slot.is_none());
                }
            }
        }
        self.commands.submit()?;
        self.commands.wait()
    }

    fn record(&self, eye: usize, slot: usize, initial: bool) {
        let image = self.images[if self.split && eye == 1 {
            RIGHT_SLOTS[slot]
        } else {
            slot
        }]
        .image;
        let layer = if self.split { 0 } else { eye as u32 };
        let device = &self.context.device;
        let command = self.commands.handle;
        let layout = vk::ImageLayout::GENERAL;
        let barrier = vk::ImageMemoryBarrier::default()
            .image(image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(COLOR)
                    .level_count(1)
                    .base_array_layer(layer)
                    .layer_count(1),
            )
            .old_layout(if initial {
                vk::ImageLayout::UNDEFINED
            } else {
                layout
            })
            .new_layout(layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .src_access_mask(if initial {
                vk::AccessFlags::empty()
            } else {
                vk::AccessFlags::TRANSFER_WRITE
            })
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
        let region = |offset: usize, x: u32, y: u32, side: u32| {
            vk::BufferImageCopy::default()
                .buffer_offset(offset as u64)
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(COLOR)
                        .base_array_layer(layer)
                        .layer_count(1),
                )
                .image_offset(vk::Offset3D {
                    x: x as i32,
                    y: y as i32,
                    z: 0,
                })
                .image_extent(vk::Extent3D {
                    width: side,
                    height: side,
                    depth: 1,
                })
        };
        // SAFETY: All buffers and images outlive the submission. Each write
        // targets just one layer. The producer orders its writes before
        // publishing the image's shader-readable layout to the observer.
        unsafe {
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
            if initial {
                device.cmd_copy_buffer_to_image(
                    command,
                    self.initial.handle,
                    image,
                    layout,
                    &[region(eye * EYE_BYTES, 0, 0, SIDE)],
                );
                let writes = vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE);
                device.cmd_pipeline_barrier(
                    command,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[writes],
                    &[],
                    &[],
                );
            }
            let regions = std::array::from_fn::<_, { CODE_COUNT + 1 }, _>(|bit| {
                if bit == CODE_COUNT {
                    let x = SIDE / 2 + if eye == 0 { SIDE * 5 / 32 } else { 0 };
                    region(CODE_COUNT * CODE_BYTES, x, SIDE * 7 / 8, TAG)
                } else {
                    let x = 16 + (bit as u32 % 8) * CODE_CELL;
                    let y = 16 + (bit as u32 / 8) * CODE_CELL;
                    region(bit * CODE_BYTES, x, y, CODE_CELL)
                }
            });
            device.cmd_copy_buffer_to_image(command, self.staging.handle, image, layout, &regions);
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier
                    .old_layout(layout)
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)],
            );
        }
    }
}

/// A packed stereo snapshot with each eye cleared to one colour (gamma
/// encoded, like captured snapshots), for testing the shell shader.
/// Keep this Vulkan device, and so the driver, loaded until the process
/// exits. libtest runs each test on its own thread. On the Frame's Turnip
/// driver, a thread that exits after the process has destroyed its last
/// Vulkan instance crashes in glibc's per-thread cleanup: the loader has
/// unloaded the driver whose cleanup function glibc then calls. Seen on
/// November 14 with every implicit layer disabled. The app is unaffected,
/// because it releases Vulkan on the main thread, whose per-thread cleanup
/// does not run at process exit.
pub fn keep_driver_loaded(context: &Arc<Context>) {
    std::mem::forget(Arc::clone(context));
}

/// The stream decoder's 10-bit two-plane format.
pub const P010: vk::Format = vk::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16;

/// A host-filled linear P010 source with the given YCbCr metadata, as the
/// observer reports it, in shader-read layout. `luma`
/// holds one 10-bit sample per pixel and `chroma` one Cb, Cr pair per 2×2
/// block, both row-major. Returns None when the device cannot sample linear
/// P010 through a YCbCr conversion.
///
/// Linear tiling keeps the fixture independent of the driver's upload path
/// to tiled multi-planar images, which was unreliable on Turnip. Decoder
/// output is tiled, so this checks the conversion, not the tiling.
pub fn linear_p010_source(
    context: &Arc<Context>,
    width: u32,
    height: u32,
    luma: &[u16],
    chroma: &[[u16; 2]],
    model: vk::SamplerYcbcrModelConversion,
    range: vk::SamplerYcbcrRange,
) -> Result<Option<Arc<ImportedImage>>> {
    let (w, h) = (width as usize, height as usize);
    if width < 2 || height < 2 || !width.is_multiple_of(2) || !height.is_multiple_of(2) {
        return Err(io::Error::other("P010 needs an even extent").into());
    }
    if luma.len() != w * h || chroma.len() != (w / 2) * (h / 2) {
        return Err(io::Error::other("P010 planes differ from the extent").into());
    }
    let instance = &context.instance.handle;
    // SAFETY: The physical device belongs to this instance.
    let features =
        unsafe { instance.get_physical_device_format_properties(context.physical, P010) }
            .linear_tiling_features;
    let chroma_offset = if features.contains(vk::FormatFeatureFlags::COSITED_CHROMA_SAMPLES) {
        vk::ChromaLocation::COSITED_EVEN
    } else if features.contains(vk::FormatFeatureFlags::MIDPOINT_CHROMA_SAMPLES) {
        vk::ChromaLocation::MIDPOINT
    } else {
        return Ok(None);
    };
    if !features.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE) {
        return Ok(None);
    }
    let device = &context.device;
    let info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(P010)
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::LINEAR)
        .usage(vk::ImageUsageFlags::SAMPLED)
        .initial_layout(vk::ImageLayout::PREINITIALIZED);
    // SAFETY: The image and memory are owned here until wrapped below, and
    // freed on every early return.
    unsafe {
        let Ok(image) = device.create_image(&info, None) else {
            return Ok(None);
        };
        let requirements = device.get_image_memory_requirements(image);
        let Some(memory_type) = context.memory_type(
            requirements.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ) else {
            device.destroy_image(image, None);
            return Ok(None);
        };
        let memory = match device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(memory_type),
            None,
        ) {
            Ok(memory) => memory,
            Err(error) => {
                device.destroy_image(image, None);
                return Err(error.into());
            }
        };
        let source = Arc::new(ImportedImage::owned(
            Arc::clone(context),
            image,
            memory,
            ImageAdd {
                format: P010.as_raw() as u32,
                width,
                height,
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1.as_raw(),
                tiling: vk::ImageTiling::LINEAR.as_raw() as u32,
                usage: vk::ImageUsageFlags::SAMPLED.as_raw(),
                ycbcr: Ycbcr {
                    valid: 1,
                    model: model.as_raw() as u32,
                    range: range.as_raw() as u32,
                    chroma_filter: vk::Filter::NEAREST.as_raw() as u32,
                    x_chroma_offset: chroma_offset.as_raw() as u32,
                    y_chroma_offset: chroma_offset.as_raw() as u32,
                    ..Ycbcr::default()
                },
                ..ImageAdd::default()
            },
        ));
        device.bind_image_memory(image, memory, 0)?;
        let mapped = device
            .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())?
            .cast::<u8>();
        // P010 keeps each 10-bit sample in the high bits of 16.
        let planes = [
            (
                vk::ImageAspectFlags::PLANE_0,
                h,
                luma.iter().map(|&y| y << 6).collect::<Vec<_>>(),
            ),
            (
                vk::ImageAspectFlags::PLANE_1,
                h / 2,
                chroma
                    .iter()
                    .flat_map(|&[cb, cr]| [cb << 6, cr << 6])
                    .collect(),
            ),
        ];
        for (aspect, rows, samples) in planes {
            let layout = device.get_image_subresource_layout(
                image,
                vk::ImageSubresource::default().aspect_mask(aspect),
            );
            let row = samples.len() / rows;
            for (y, line) in samples.chunks_exact(row).enumerate() {
                let bytes: Vec<u8> = line
                    .iter()
                    .flat_map(|sample| sample.to_le_bytes())
                    .collect();
                std::ptr::copy_nonoverlapping(
                    bytes.as_ptr(),
                    mapped.add((layout.offset + y as u64 * layout.row_pitch) as usize),
                    bytes.len(),
                );
            }
        }
        device.unmap_memory(memory);
        let mut commands = Commands::new(context)?;
        commands.begin()?;
        device.cmd_pipeline_barrier(
            commands.handle,
            vk::PipelineStageFlags::HOST,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[vk::ImageMemoryBarrier::default()
                .image(image)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(COLOR)
                        .level_count(1)
                        .layer_count(1),
                )
                .old_layout(vk::ImageLayout::PREINITIALIZED)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .src_access_mask(vk::AccessFlags::HOST_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)],
        );
        commands.submit()?;
        commands.wait()?;
        Ok(Some(source))
    }
}

pub fn painted_snapshot(
    context: &Arc<Context>,
    eye_size: u32,
    colours: [[f32; 4]; 2],
) -> Result<Snapshot> {
    painted_tiles(context, eye_size, 1, |eye, _, _| colours[eye])
}

/// A gamma-encoded stereo snapshot with a separate colour for each quadrant.
pub fn tiled_snapshot(
    context: &Arc<Context>,
    eye_size: u32,
    colours: [[[f32; 4]; 4]; 2],
) -> Result<Snapshot> {
    painted_tiles(context, eye_size, 2, |eye, x, y| {
        colours[eye][(y * 2 + x) as usize]
    })
}

fn painted_tiles(
    context: &Arc<Context>,
    eye_size: u32,
    tiles: u32,
    rgba: impl Fn(usize, u32, u32) -> [f32; 4],
) -> Result<Snapshot> {
    if !eye_size.is_multiple_of(tiles) {
        return Err(io::Error::other("snapshot does not divide into equal tiles").into());
    }
    let device = &context.device;
    let tile = eye_size / tiles;
    let pass = create_render_pass(device, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)?;
    let painted = (|| {
        let width = eye_size
            .checked_mul(2)
            .ok_or_else(|| io::Error::other("snapshot width overflow"))?;
        let target = Target::new(
            context,
            vk::Extent2D {
                width,
                height: eye_size,
            },
            pass,
        )?;
        let mut commands = Commands::new(context)?;
        commands.begin()?;
        // SAFETY: The target and pass are owned here, and the commands are
        // complete before either is used again or destroyed.
        unsafe {
            device.cmd_begin_render_pass(
                commands.handle,
                &vk::RenderPassBeginInfo::default()
                    .render_pass(pass)
                    .framebuffer(target.framebuffer)
                    .render_area(vk::Rect2D {
                        extent: target.extent,
                        ..Default::default()
                    }),
                vk::SubpassContents::INLINE,
            );
            for eye in 0..2 {
                for y in 0..tiles {
                    for x in 0..tiles {
                        device.cmd_clear_attachments(
                            commands.handle,
                            &[vk::ClearAttachment {
                                aspect_mask: COLOR,
                                color_attachment: 0,
                                clear_value: vk::ClearValue {
                                    color: vk::ClearColorValue {
                                        float32: rgba(eye, x, y),
                                    },
                                },
                            }],
                            &[vk::ClearRect {
                                rect: vk::Rect2D {
                                    offset: vk::Offset2D {
                                        x: (eye as u32 * eye_size + x * tile) as i32,
                                        y: (y * tile) as i32,
                                    },
                                    extent: vk::Extent2D {
                                        width: tile,
                                        height: tile,
                                    },
                                },
                                base_array_layer: 0,
                                layer_count: 1,
                            }],
                        );
                    }
                }
            }
            device.cmd_end_render_pass(commands.handle);
        }
        commands.submit()?;
        commands.wait()?;
        Ok(Snapshot {
            target: Arc::new(target),
        })
    })();
    // SAFETY: Every command using the pass has completed or was never sent.
    unsafe { device.destroy_render_pass(pass, None) };
    painted
}

fn pattern() -> Vec<u8> {
    let mut data = Vec::with_capacity(EYE_BYTES * 2);
    for eye in 0..2 {
        for y in 0..SIDE {
            for x in 0..SIDE {
                data.extend_from_slice(&stereo_pixel(x, y, SIDE, SIDE, eye));
            }
        }
    }
    data
}

pub fn run_cadence(duration: Duration, mode: Cadence) -> Result<u32> {
    let split = matches!(mode, Cadence::SeparateEyes(_));
    let mut producer = Producer::new(split)?;
    producer.write(0, 0..2, None)?;
    println!("cadence ready");
    let start = Instant::now();
    let mut frames = 0;
    while start.elapsed() < duration {
        let slot = match mode {
            Cadence::Reuse => 0,
            Cadence::Burst => usize::from(frames % 8 == 0),
            Cadence::Rotate | Cadence::SeparateEyes(_) => frames as usize % SLOTS,
        };
        frames += 1;
        if let Cadence::SeparateEyes(gap) = mode {
            producer.write(frames, 0..1, Some(slot))?;
            thread::sleep(gap);
            producer.write(frames, 1..2, Some(slot))?;
        } else {
            producer.write(frames, 0..2, Some(slot))?;
        }
        let due = start + Duration::from_nanos(u64::from(frames) * 1_000_000_000 / 144);
        if let Some(delay) = due.checked_duration_since(Instant::now()) {
            thread::sleep(delay);
        }
    }
    Ok(frames)
}
