use super::{Context, ImportedImage, Result, gpu::*};
use ash::vk::{self, Handle};
use std::{collections::BTreeMap, io, sync::Arc};

const VERTEX: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/overlay.vert.spv"));
const FRAGMENT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/overlay.frag.spv"));

pub struct SourceView<'a> {
    pub image: &'a Arc<ImportedImage>,
    pub layout: vk::ImageLayout,
    pub layer: u32,
    /// The eye's source rectangle `[x, y, width, height]`, positive size.
    pub crop: [f32; 4],
    /// Read the rectangle's columns (x) or rows (y) in reverse, so the copy
    /// is upright when the source is stored mirrored or bottom row first.
    pub flip: [bool; 2],
    pub external: bool,
}

impl SourceView<'_> {
    /// The copy's source origin and signed size, and the limits that keep
    /// linear filtering on this rectangle's texels.
    fn sampling(&self) -> ([f32; 4], [f32; 4]) {
        let [x, y, width, height] = self.crop;
        let mut signed = self.crop;
        for axis in 0..2 {
            if self.flip[axis] {
                signed[axis] += signed[axis + 2];
                signed[axis + 2] = -signed[axis + 2];
            }
        }
        let description = self.image.description;
        let half = [
            0.5 / description.width.max(1) as f32,
            0.5 / description.height.max(1) as f32,
        ];
        let low = [x + half[0], y + half[1]];
        let high = [
            (x + width - half[0]).max(low[0]),
            (y + height - half[1]).max(low[1]),
        ];
        (signed, [low[0], low[1], high[0], high[1]])
    }

    fn key(&self) -> (u64, u32, i32) {
        (
            self.image.handle().as_raw(),
            self.layer,
            self.layout.as_raw(),
        )
    }

    fn valid(&self, context: &Arc<Context>) -> bool {
        let [x, y, w, h] = self.crop;
        Arc::ptr_eq(self.image.context(), context)
            && self.layer < self.image.description.array_layers
            && matches!(
                self.layout,
                vk::ImageLayout::GENERAL | vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
            )
            && self.crop.into_iter().all(f32::is_finite)
            && x >= 0.0
            && y >= 0.0
            && w > 0.0
            && h > 0.0
            && x + w <= 1.0
            && y + h <= 1.0
    }
}

/// A private packed stereo RGBA image, left eye first, in shader-read layout.
pub struct Snapshot {
    pub(super) target: Arc<Target>,
}

impl Snapshot {
    pub fn image(&self) -> vk::Image {
        self.target.image
    }
    pub fn extent(&self) -> vk::Extent2D {
        self.target.extent
    }
    pub fn allocated_bytes(&self) -> u64 {
        self.target.allocated_bytes
    }
}

pub struct SnapshotRenderer {
    context: Arc<Context>,
    commands: Commands,
    pass: vk::RenderPass,
    vertex: vk::ShaderModule,
    fragment: vk::ShaderModule,
    bindings: BTreeMap<(u64, u32, i32), Binding>,
    measurement_targets: Vec<Option<Snapshot>>,
    measurement_buffer: Option<Buffer>,
}

impl SnapshotRenderer {
    pub fn new(context: Arc<Context>) -> Result<Self> {
        let commands = Commands::new(&context)?;
        let mut renderer = Self {
            context,
            commands,
            pass: vk::RenderPass::null(),
            vertex: vk::ShaderModule::null(),
            fragment: vk::ShaderModule::null(),
            bindings: BTreeMap::new(),
            measurement_targets: Vec::new(),
            measurement_buffer: None,
        };
        let device = &renderer.context.device;
        renderer.pass = create_render_pass(device, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)?;
        renderer.vertex = shader_module(device, VERTEX)?;
        renderer.fragment = shader_module(device, FRAGMENT)?;
        Ok(renderer)
    }

    /// Drop cached views for source images which are no longer published.
    pub fn retain_sources(&mut self, keep: impl Fn(&ImportedImage) -> bool) {
        self.bindings.retain(|_, binding| keep(&binding.image));
    }

    /// Read both source eyes into an owned snapshot without changing the
    /// source image layout. All work finishes before this call returns.
    ///
    /// # Safety
    /// Source image contents must be ready to read in their supplied layouts.
    /// When external is set the producer must have released ownership to
    /// the external queue family. The caller must prevent concurrent writes.
    pub unsafe fn capture(
        &mut self,
        eyes: [SourceView<'_>; 2],
        extent: [u32; 2],
    ) -> Result<Snapshot> {
        let mut output = None;
        // SAFETY: This method carries the same source lifetime contract.
        unsafe {
            self.capture_into(eyes, extent, &mut output)?;
        }
        Ok(output.expect("successful capture provides a snapshot"))
    }

    /// Reuse the snapshot's allocation when its dimensions are unchanged.
    ///
    /// # Safety
    /// The source contract is the same as capture. No external consumer may
    /// use the existing snapshot while it is being replaced.
    pub unsafe fn capture_into(
        &mut self,
        eyes: [SourceView<'_>; 2],
        extent: [u32; 2],
        output: &mut Option<Snapshot>,
    ) -> Result<()> {
        self.commands.begin()?;
        self.record_capture(eyes, extent, output)?;
        self.commands.submit()?;
        self.commands.wait()
    }

    fn record_capture(
        &mut self,
        eyes: [SourceView<'_>; 2],
        extent: [u32; 2],
        output: &mut Option<Snapshot>,
    ) -> Result<()> {
        if !eyes.iter().all(|eye| eye.valid(&self.context)) {
            return Err(io::Error::other("invalid snapshot source view").into());
        }
        let width = extent[0]
            .checked_mul(2)
            .ok_or_else(|| io::Error::other("snapshot extent overflow"))?;
        let target_extent = vk::Extent2D {
            width,
            height: extent[1],
        };
        if output
            .as_ref()
            .is_some_and(|snapshot| !Arc::ptr_eq(snapshot.target.context(), &self.context))
        {
            return Err(io::Error::other("snapshot was allocated on another device").into());
        }
        if output.as_ref().is_none_or(|snapshot| {
            snapshot.extent() != target_extent || Arc::strong_count(&snapshot.target) > 1
        }) {
            *output = Some(Snapshot {
                target: Arc::new(Target::new(&self.context, target_extent, self.pass)?),
            });
        }
        let target = &output.as_ref().expect("snapshot target allocated").target;
        let same_image = eyes[0].image.handle() == eyes[1].image.handle();
        if same_image && (eyes[0].layout != eyes[1].layout || eyes[0].external != eyes[1].external)
        {
            return Err(io::Error::other("views of one image require the same handoff").into());
        }
        for eye in &eyes {
            let key = eye.key();
            if !self.bindings.contains_key(&key) {
                let binding = Binding::new(self, eye)?;
                self.bindings.insert(key, binding);
            }
        }
        let bindings = eyes.each_ref().map(|eye| &self.bindings[&eye.key()]);
        let acquire: Vec<_> = eyes
            .iter()
            .enumerate()
            .filter(|&(i, eye)| eye.external && (i == 0 || !same_image))
            .map(|(_, eye)| {
                vk::ImageMemoryBarrier::default()
                    .old_layout(eye.layout)
                    .new_layout(eye.layout)
                    .image(eye.image.handle())
                    .src_queue_family_index(vk::QUEUE_FAMILY_EXTERNAL)
                    .dst_queue_family_index(self.context.queue_family)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .subresource_range(color_range(eye.image.description.array_layers))
            })
            .collect();
        let release: Vec<_> = acquire
            .iter()
            .map(|barrier| vk::ImageMemoryBarrier {
                src_queue_family_index: barrier.dst_queue_family_index,
                dst_queue_family_index: barrier.src_queue_family_index,
                src_access_mask: vk::AccessFlags::SHADER_READ,
                dst_access_mask: vk::AccessFlags::empty(),
                ..*barrier
            })
            .collect();
        let device = &self.context.device;
        let command = self.commands.handle;
        // SAFETY: The caller supplies the handoff contract. Descriptors,
        // target, and bindings are live and the queue is fenced before return.
        unsafe {
            if !acquire.is_empty() {
                device.cmd_pipeline_barrier(
                    command,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &acquire,
                );
            }
            device.cmd_begin_render_pass(
                command,
                &vk::RenderPassBeginInfo::default()
                    .render_pass(self.pass)
                    .framebuffer(target.framebuffer)
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D::default(),
                        extent: target.extent,
                    }),
                vk::SubpassContents::INLINE,
            );
            for (index, (eye, binding)) in eyes.iter().zip(&bindings).enumerate() {
                let x = index as u32 * extent[0];
                device.cmd_bind_pipeline(
                    command,
                    vk::PipelineBindPoint::GRAPHICS,
                    binding.pipeline,
                );
                device.cmd_bind_descriptor_sets(
                    command,
                    vk::PipelineBindPoint::GRAPHICS,
                    binding.layout,
                    0,
                    &[binding.set],
                    &[],
                );
                device.cmd_set_viewport(
                    command,
                    0,
                    &[vk::Viewport {
                        x: x as f32,
                        y: 0.0,
                        width: extent[0] as f32,
                        height: extent[1] as f32,
                        min_depth: 0.0,
                        max_depth: 1.0,
                    }],
                );
                device.cmd_set_scissor(
                    command,
                    0,
                    &[vk::Rect2D {
                        offset: vk::Offset2D { x: x as i32, y: 0 },
                        extent: vk::Extent2D {
                            width: extent[0],
                            height: extent[1],
                        },
                    }],
                );
                let ([u, v, w, h], limits) = eye.sampling();
                let push = [
                    u,
                    v,
                    w,
                    h,
                    1.0,
                    1.0,
                    1.0,
                    0.5,
                    0.0,
                    0.0,
                    index as f32,
                    f32::from(binding.srgb),
                    limits[0],
                    limits[1],
                    limits[2],
                    limits[3],
                    0.0,
                ];
                device.cmd_push_constants(
                    command,
                    binding.layout,
                    vk::ShaderStageFlags::FRAGMENT,
                    0,
                    bytemuck::bytes_of(&push),
                );
                device.cmd_draw(command, 3, 1, 0, 0);
            }
            device.cmd_end_render_pass(command);
            if !release.is_empty() {
                device.cmd_pipeline_barrier(
                    command,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &release,
                );
            }
        }
        Ok(())
    }

    /// Capture pairs and transfer the pixels in one queue submission.
    ///
    /// # Safety
    /// Each pair must meet the source lifetime and handoff rules of capture.
    pub unsafe fn measure(
        &mut self,
        pairs: Vec<[SourceView<'_>; 2]>,
        extent: [u32; 2],
    ) -> Result<Vec<u8>> {
        if pairs.is_empty() {
            return Ok(Vec::new());
        }
        let bytes_per_pair = (extent[0] as usize)
            .checked_mul(extent[1] as usize)
            .and_then(|area| area.checked_mul(8))
            .filter(|&bytes| bytes != 0)
            .ok_or_else(|| io::Error::other("measurement extent overflow"))?;
        let size = bytes_per_pair
            .checked_mul(pairs.len())
            .ok_or_else(|| io::Error::other("measurement size overflow"))?;
        let mut targets = std::mem::take(&mut self.measurement_targets);
        let mut buffer = self.measurement_buffer.take();
        let result = (|| {
            if buffer.as_ref().is_none_or(|buffer| buffer.size() != size) {
                buffer = Some(Buffer::new(
                    &self.context,
                    size,
                    vk::BufferUsageFlags::TRANSFER_DST,
                )?);
            }
            let buffer = buffer.as_mut().expect("measurement buffer allocated");
            if pairs.len() > targets.len() {
                targets.resize_with(pairs.len(), || None);
            }
            self.commands.begin()?;
            for (i, (pair, target)) in pairs.into_iter().zip(&mut targets).enumerate() {
                self.record_capture(pair, extent, target)?;
                self.record_readback(
                    target.as_ref().expect("measurement target allocated"),
                    buffer.handle,
                    (i * bytes_per_pair) as u64,
                );
            }
            self.commands.submit()?;
            self.commands.wait()?;
            buffer.read()
        })();
        self.measurement_targets = targets;
        self.measurement_buffer = buffer;
        result
    }

    pub fn readback(&mut self, snapshot: &mut Snapshot) -> Result<Vec<u8>> {
        if !Arc::ptr_eq(&self.context, snapshot.target.context()) {
            return Err(io::Error::other("snapshot was allocated on another device").into());
        }
        let extent = snapshot.extent();
        let size = (extent.width as usize * extent.height as usize)
            .checked_mul(4)
            .ok_or_else(|| io::Error::other("snapshot byte count overflow"))?;
        let mut buffer = Buffer::new(&self.context, size, vk::BufferUsageFlags::TRANSFER_DST)?;
        self.commands.begin()?;
        self.record_readback(snapshot, buffer.handle, 0);
        self.commands.submit()?;
        self.commands.wait()?;
        buffer.read()
    }

    fn record_readback(&self, snapshot: &Snapshot, buffer: vk::Buffer, offset: u64) {
        let extent = snapshot.extent();
        let device = &self.context.device;
        let command = self.commands.handle;
        // SAFETY: An exclusive snapshot borrow keeps it alive and prevents
        // another safe operation from using it until the transfer completes.
        unsafe {
            let transfer = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .src_access_mask(
                    vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::SHADER_READ,
                )
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
                .image(snapshot.image())
                .subresource_range(color_range(1));
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[transfer],
            );
            let region = vk::BufferImageCopy::default()
                .buffer_offset(offset)
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(COLOR)
                        .layer_count(1),
                )
                .image_extent(vk::Extent3D {
                    width: extent.width,
                    height: extent.height,
                    depth: 1,
                });
            device.cmd_copy_image_to_buffer(
                command,
                snapshot.image(),
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                buffer,
                &[region],
            );
            let restore = vk::ImageMemoryBarrier::default()
                .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .src_access_mask(vk::AccessFlags::TRANSFER_READ)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .image(snapshot.image())
                .subresource_range(color_range(1));
            let host = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ);
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[host],
                &[],
                &[restore],
            );
        }
    }
}

impl Drop for SnapshotRenderer {
    fn drop(&mut self) {
        let _ = self.commands.wait();
        self.bindings.clear();
        self.measurement_targets.clear();
        // SAFETY: All graphics work is complete and these handles are owned.
        unsafe {
            let device = &self.context.device;
            device.destroy_shader_module(self.fragment, None);
            device.destroy_shader_module(self.vertex, None);
            device.destroy_render_pass(self.pass, None);
        }
    }
}

struct Binding {
    image: Arc<ImportedImage>,
    context: Arc<Context>,
    conversion: vk::SamplerYcbcrConversion,
    sampler: vk::Sampler,
    view: vk::ImageView,
    set_layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    set: vk::DescriptorSet,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    srgb: bool,
}

impl Binding {
    fn new(renderer: &SnapshotRenderer, eye: &SourceView<'_>) -> Result<Self> {
        let context = &renderer.context;
        let device = &context.device;
        let description = eye.image.description;
        let format = vk::Format::from_raw(description.format as i32);
        let srgb = matches!(
            format,
            vk::Format::R8G8B8A8_SRGB
                | vk::Format::B8G8R8A8_SRGB
                | vk::Format::A8B8G8R8_SRGB_PACK32
                | vk::Format::R8G8B8_SRGB
                | vk::Format::B8G8R8_SRGB
        );
        let mut binding = Self {
            image: Arc::clone(eye.image),
            context: Arc::clone(context),
            conversion: vk::SamplerYcbcrConversion::null(),
            sampler: vk::Sampler::null(),
            view: vk::ImageView::null(),
            set_layout: vk::DescriptorSetLayout::null(),
            pool: vk::DescriptorPool::null(),
            set: vk::DescriptorSet::null(),
            layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
            srgb,
        };

        // SAFETY: All handles come from the same context. Binding owns its
        // handles and retires them after the capture fence signals.
        unsafe {
            let mut filter = vk::Filter::LINEAR;
            let mut descriptor_count = 1;
            let ycbcr = description.ycbcr;
            if ycbcr.valid != 0 {
                filter = vk::Filter::from_raw(ycbcr.chroma_filter as i32);
                let model = vk::SamplerYcbcrModelConversion::from_raw(ycbcr.model as i32);
                let identity = matches!(
                    model,
                    vk::SamplerYcbcrModelConversion::RGB_IDENTITY
                        | vk::SamplerYcbcrModelConversion::YCBCR_IDENTITY
                );
                let components = ycbcr
                    .components
                    .map(|value| vk::ComponentSwizzle::from_raw(value as i32));
                let conversion = vk::SamplerYcbcrConversionCreateInfo::default()
                    .format(format)
                    .ycbcr_model(if identity {
                        vk::SamplerYcbcrModelConversion::YCBCR_709
                    } else {
                        model
                    })
                    // The compositor samples stream P010 with an identity
                    // model and converts it itself. On November 14 a ring
                    // converted as limited range measured 1.17 × live − 17,
                    // the limited-to-full stretch, so it uses full range.
                    .ycbcr_range(if identity {
                        vk::SamplerYcbcrRange::ITU_FULL
                    } else {
                        vk::SamplerYcbcrRange::from_raw(ycbcr.range as i32)
                    })
                    .components(vk::ComponentMapping {
                        r: components[0],
                        g: components[1],
                        b: components[2],
                        a: components[3],
                    })
                    .x_chroma_offset(vk::ChromaLocation::from_raw(ycbcr.x_chroma_offset as i32))
                    .y_chroma_offset(vk::ChromaLocation::from_raw(ycbcr.y_chroma_offset as i32))
                    .chroma_filter(filter)
                    .force_explicit_reconstruction(ycbcr.force_explicit_reconstruction != 0);
                binding.conversion = device.create_sampler_ycbcr_conversion(&conversion, None)?;
                descriptor_count = 3;
            } else if (1000156000..=1000156033).contains(&description.format)
                || (1000330000..=1000330003).contains(&description.format)
            {
                return Err(
                    io::Error::other("multi-plane import is missing YCbCr metadata").into(),
                );
            }
            let mut conversion =
                vk::SamplerYcbcrConversionInfo::default().conversion(binding.conversion);
            let sampler = vk::SamplerCreateInfo::default()
                .min_filter(filter)
                .mag_filter(filter)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
            binding.sampler = if binding.conversion == vk::SamplerYcbcrConversion::null() {
                device.create_sampler(&sampler, None)?
            } else {
                device.create_sampler(&sampler.push_next(&mut conversion), None)?
            };
            let mut conversion =
                vk::SamplerYcbcrConversionInfo::default().conversion(binding.conversion);
            let view = vk::ImageViewCreateInfo::default()
                .image(eye.image.handle())
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .subresource_range(color_range(1).base_array_layer(eye.layer));
            binding.view = if binding.conversion == vk::SamplerYcbcrConversion::null() {
                device.create_image_view(&view, None)?
            } else {
                device.create_image_view(&view.push_next(&mut conversion), None)?
            };

            let samplers = [binding.sampler];
            let bindings = [vk::DescriptorSetLayoutBinding::default()
                .binding(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                .immutable_samplers(&samplers)];
            binding.set_layout = device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )?;
            let sizes = [vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count,
            }];
            binding.pool = device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(1)
                    .pool_sizes(&sizes),
                None,
            )?;
            let layouts = [binding.set_layout];
            binding.set = device.allocate_descriptor_sets(
                &vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(binding.pool)
                    .set_layouts(&layouts),
            )?[0];
            let images = [vk::DescriptorImageInfo::default()
                .image_view(binding.view)
                .image_layout(eye.layout)];
            device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(binding.set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&images)],
                &[],
            );
            let ranges = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                .size(68)];
            binding.layout = device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&layouts)
                    .push_constant_ranges(&ranges),
                None,
            )?;
        }
        binding.pipeline = graphics_pipeline(
            device,
            binding.layout,
            renderer.pass,
            renderer.vertex,
            renderer.fragment,
        )?;
        Ok(binding)
    }
}

impl Drop for Binding {
    fn drop(&mut self) {
        // SAFETY: Binding owns these handles and the capture has retired.
        unsafe {
            let device = &self.context.device;
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
            device.destroy_image_view(self.view, None);
            device.destroy_sampler(self.sampler, None);
            device.destroy_sampler_ycbcr_conversion(self.conversion, None);
        }
    }
}
