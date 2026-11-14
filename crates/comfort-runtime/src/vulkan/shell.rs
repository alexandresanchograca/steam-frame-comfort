use super::{Context, Result, Snapshot, gpu::*};
use ash::vk;
use bytemuck::{Pod, Zeroable};
use comfort_core::geometry::{Aperture, Face, Pose, Projection, Shell};
use std::{io, sync::Arc, time::Instant};

const VERTEX: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/overlay.vert.spv"));
const FRAGMENT: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/restframe.frag.spv"));

#[derive(Clone, Copy)]
pub struct Capture<'a> {
    pub snapshot: &'a Snapshot,
    pub eyes: [(Pose, Projection); 2],
}

#[derive(Clone, Copy)]
pub struct Gap {
    pub grid: bool,
    pub background: f32,
    pub lines: f32,
    pub thickness_cm: f32,
    pub spacing: f32,
}

impl Default for Gap {
    fn default() -> Self {
        Self {
            grid: false,
            background: 0.02,
            lines: 0.2,
            thickness_cm: 2.0,
            spacing: 2.0,
        }
    }
}

#[derive(Clone, Copy)]
pub struct ShellParams<'a> {
    pub captures: Option<[Capture<'a>; 2]>,
    pub shell: Shell,
    pub display_eyes: [Pose; 2],
    pub aperture: Aperture,
    pub gap: Gap,
    pub refresh_weight: f32,
    /// The refresh blink dips the picture to black. Grid lines are layered
    /// only by `grid_over_ring`, which stays steady through the dip.
    pub blink: bool,
    pub grid_over_ring: bool,
    pub dim: f32,
    pub faces: u32,
}

impl Default for ShellParams<'_> {
    fn default() -> Self {
        Self {
            captures: None,
            shell: Shell::default(),
            display_eyes: [Pose::default(); 2],
            aperture: Aperture::default(),
            gap: Gap::default(),
            refresh_weight: 0.0,
            blink: false,
            grid_over_ring: false,
            dim: 0.0,
            faces: 0x3f,
        }
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
struct CaptureUniform {
    rows: [[f32; 4]; 3],
    projection: [f32; 4],
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Default, Pod, Zeroable)]
struct Draw {
    center: [f32; 4],
    right: [f32; 4],
    up: [f32; 4],
    captures: [CaptureUniform; 2],
    display: [[f32; 4]; 3],
    mask: [f32; 4],
    style: [f32; 4],
    look: [f32; 4],
    dimensions: [f32; 4],
    effects: [f32; 4],
}
const _: () = assert!(std::mem::size_of::<Draw>() == 304);

struct OutputSet {
    commands: Commands,
    uniform: Buffer,
    descriptor: vk::DescriptorSet,
    faces: [Target; 6],
    snapshots: Option<[Arc<Target>; 2]>,
    rendered: u32,
    timing: Option<RenderTiming>,
}

struct RenderTiming {
    context: Arc<Context>,
    queries: vk::QueryPool,
    submitted: Option<Instant>,
}

impl Drop for RenderTiming {
    fn drop(&mut self) {
        // SAFETY: OutputSet drops its command buffer (and waits for it)
        // before releasing the query pool.
        unsafe { self.context.device.destroy_query_pool(self.queries, None) };
    }
}

pub struct RenderCompletion {
    pub gpu_ms: Option<f64>,
    pub elapsed_ms: f64,
}

const SET_COUNT: usize = 3;

pub struct ShellRenderer {
    context: Arc<Context>,
    commands: Commands,
    size: u32,
    pass: vk::RenderPass,
    vertex: vk::ShaderModule,
    fragment: vk::ShaderModule,
    sampler: vk::Sampler,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    pool: vk::DescriptorPool,
    descriptors: [vk::DescriptorSet; SET_COUNT],
    placeholder: Option<Target>,
    outputs: [Option<OutputSet>; SET_COUNT],
    timing: Option<(u32, f32)>,
}

impl ShellRenderer {
    pub fn new(context: Arc<Context>, face_size: u32) -> Result<Self> {
        let commands = Commands::new(&context)?;
        let mut shell = Self {
            context,
            commands,
            size: face_size,
            pass: vk::RenderPass::null(),
            vertex: vk::ShaderModule::null(),
            fragment: vk::ShaderModule::null(),
            sampler: vk::Sampler::null(),
            set_layout: vk::DescriptorSetLayout::null(),
            layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
            pool: vk::DescriptorPool::null(),
            descriptors: [vk::DescriptorSet::null(); SET_COUNT],
            placeholder: None,
            outputs: std::array::from_fn(|_| None),
            timing: None,
        };
        if face_size < 8 {
            return Err(io::Error::other("face size must be at least 8 pixels").into());
        }
        shell.pass =
            create_render_pass(&shell.context.device, vk::ImageLayout::TRANSFER_SRC_OPTIMAL)?;
        shell.vertex = shader_module(&shell.context.device, VERTEX)?;
        shell.fragment = shader_module(&shell.context.device, FRAGMENT)?;
        shell.create_pipeline()?;
        shell.create_placeholder()?;
        Ok(shell)
    }

    fn create_pipeline(&mut self) -> Result<()> {
        let device = &self.context.device;
        // SAFETY: This renderer owns all created objects and its cleanup runs
        // on partial initialization. All descriptor data is local to the call.
        unsafe {
            let bindings = std::array::from_fn::<_, 3, _>(|index| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(index as u32)
                    .descriptor_type(if index == 2 {
                        vk::DescriptorType::UNIFORM_BUFFER
                    } else {
                        vk::DescriptorType::COMBINED_IMAGE_SAMPLER
                    })
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            });
            self.set_layout = device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                None,
            )?;
            let layouts = [self.set_layout];
            let push = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                .size(4)];
            self.layout = device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&layouts)
                    .push_constant_ranges(&push),
                None,
            )?;
            self.pipeline =
                graphics_pipeline(device, self.layout, self.pass, self.vertex, self.fragment)?;
            self.sampler = device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .min_filter(vk::Filter::LINEAR)
                    .mag_filter(vk::Filter::LINEAR)
                    .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                    .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE),
                None,
            )?;
            let sizes = [
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    descriptor_count: 2 * SET_COUNT as u32,
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: SET_COUNT as u32,
                },
            ];
            self.pool = device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .max_sets(SET_COUNT as u32)
                    .pool_sizes(&sizes),
                None,
            )?;
            self.descriptors = device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(self.pool)
                        .set_layouts(&[self.set_layout; SET_COUNT]),
                )?
                .try_into()
                .map_err(|_| io::Error::other("wrong shell descriptor count"))?;
        }
        Ok(())
    }

    fn create_placeholder(&mut self) -> Result<()> {
        // Plain and Grid never sample the placeholder, but the descriptors
        // must point to valid image views in shader-read layout.
        let target = Target::new(
            &self.context,
            vk::Extent2D {
                width: 1,
                height: 1,
            },
            self.pass,
        )?;
        self.commands.begin()?;
        let barrier = vk::ImageMemoryBarrier::default()
            .image(target.image)
            .subresource_range(color_range(1))
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_access_mask(vk::AccessFlags::SHADER_READ);
        // SAFETY: The image is owned by this renderer and has no prior work.
        unsafe {
            self.context.device.cmd_pipeline_barrier(
                self.commands.handle,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            )
        };
        self.commands.submit()?;
        self.commands.wait()?;
        self.placeholder = Some(target);
        Ok(())
    }

    /// Sample GPU timestamps and host-observed completion for future renders.
    /// Call before allocating shell output sets.
    pub fn enable_timing(&mut self) {
        let instance = &self.context.instance.handle;
        let physical = self.context.physical;
        // SAFETY: The physical device and its selected queue family are live.
        let (bits, period) = unsafe {
            let families = instance.get_physical_device_queue_family_properties(physical);
            let properties = instance.get_physical_device_properties(physical);
            (
                families[self.context.queue_family as usize].timestamp_valid_bits,
                properties.limits.timestamp_period,
            )
        };
        self.timing = Some((bits, period));
    }

    pub fn allocated_bytes(&self) -> u64 {
        self.outputs
            .iter()
            .flatten()
            .flat_map(|output| &output.faces)
            .chain(self.placeholder.iter())
            .map(|face| face.allocated_bytes)
            .sum()
    }

    /// Collect completed submissions without waiting on a GPU fence.
    pub fn completions(&mut self) -> Result<impl Iterator<Item = RenderCompletion>> {
        let mut completed = [None, None, None];
        let Some((bits, period)) = self.timing else {
            return Ok(completed.into_iter().flatten());
        };
        for (slot, output) in completed.iter_mut().zip(&mut self.outputs) {
            let Some(output) = output else { continue };
            let Some(timing) = &mut output.timing else {
                continue;
            };
            if timing.submitted.is_none() || !output.commands.complete()? {
                continue;
            }
            let submission = timing.submitted.take().expect("submission pending");
            let mut values = [0_u64; 2];
            // SAFETY: The fence has signaled and both queries have been written.
            let gpu_ms = (timing.queries != vk::QueryPool::null()
                && unsafe {
                    self.context.device.get_query_pool_results(
                        timing.queries,
                        0,
                        &mut values,
                        vk::QueryResultFlags::TYPE_64,
                    )
                }
                .is_ok())
            .then(|| {
                let mask = u64::MAX >> (64 - bits.min(64));
                (values[1].wrapping_sub(values[0]) & mask) as f64 * f64::from(period) / 1e6
            });
            *slot = Some(RenderCompletion {
                gpu_ms,
                elapsed_ms: submission.elapsed().as_secs_f64() * 1000.0,
            });
        }
        Ok(completed.into_iter().flatten())
    }

    /// Allocate all output sets while the app is waiting for a stick push.
    pub fn prepare_outputs(&mut self) -> Result<()> {
        for set in 0..self.outputs.len() {
            self.prepare_output(set)?;
        }
        Ok(())
    }

    fn prepare_output(&mut self, set: usize) -> Result<()> {
        let output = self
            .outputs
            .get_mut(set)
            .ok_or_else(|| io::Error::other("invalid shell output set"))?;
        if output.is_some() {
            return Ok(());
        }
        let width = self
            .size
            .checked_mul(2)
            .ok_or_else(|| io::Error::other("shell width overflow"))?;
        let extent = vk::Extent2D {
            width,
            height: self.size,
        };
        let faces = (0..6)
            .map(|_| Target::new(&self.context, extent, self.pass))
            .collect::<Result<Vec<_>>>()?;
        let faces = faces
            .try_into()
            .map_err(|_| io::Error::other("wrong shell face count"))?;
        let uniform = Buffer::new(
            &self.context,
            std::mem::size_of::<[Draw; 12]>(),
            vk::BufferUsageFlags::UNIFORM_BUFFER,
        )?;
        let descriptor = self.descriptors[set];
        let buffer = [vk::DescriptorBufferInfo::default()
            .buffer(uniform.handle)
            .range(std::mem::size_of::<[Draw; 12]>() as u64)];
        // SAFETY: This set is fresh and its uniform stays live with the set.
        unsafe {
            self.context.device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(descriptor)
                    .dst_binding(2)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&buffer)],
                &[],
            );
        }
        *output = Some(OutputSet {
            commands: Commands::new(&self.context)?,
            uniform,
            descriptor,
            faces,
            snapshots: None,
            rendered: 0,
            timing: self.timing.map(|(bits, _)| RenderTiming {
                context: Arc::clone(&self.context),
                // SAFETY: The live device owns the new timestamp pool.
                queries: if bits != 0 {
                    unsafe {
                        self.context.device.create_query_pool(
                            &vk::QueryPoolCreateInfo::default()
                                .query_type(vk::QueryType::TIMESTAMP)
                                .query_count(2),
                            None,
                        )
                    }
                    .unwrap_or(vk::QueryPool::null())
                } else {
                    vk::QueryPool::null()
                },
                submitted: None,
            }),
        });
        Ok(())
    }

    /// Render and wait for completion, for callers that read on the host.
    pub fn render(&mut self, params: ShellParams<'_>, set: usize) -> Result<()> {
        self.queue_render(params, set)?;
        self.outputs[set]
            .as_mut()
            .expect("rendered set")
            .commands
            .wait()
    }

    /// Queue the render on the overlay's Vulkan queue. The set stays leased
    /// until retired, and readers must use the same queue or wait for completion.
    pub fn queue_render(&mut self, params: ShellParams<'_>, set: usize) -> Result<()> {
        let draws = parameters(params, self.size, &self.context)?;
        self.prepare_output(set)?;
        let output = &mut self.outputs[set];
        if output.as_ref().is_some_and(|output| output.rendered != 0) {
            return Err(io::Error::other("shell output set is still leased").into());
        }
        let output = output.as_mut().expect("output set allocated above");
        let views = if let Some(captures) = params.captures {
            captures.map(|capture| capture.snapshot.target.view)
        } else {
            [self.placeholder
                .as_ref()
                .expect("placeholder initialized")
                .view; 2]
        };
        let descriptors = views.map(|view| {
            [vk::DescriptorImageInfo::default()
                .sampler(self.sampler)
                .image_view(view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)]
        });
        let writes = std::array::from_fn::<_, 2, _>(|index| {
            vk::WriteDescriptorSet::default()
                .dst_set(output.descriptor)
                .dst_binding(index as u32)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&descriptors[index])
        });
        output.commands.begin()?;
        output.uniform.write(bytemuck::bytes_of(&draws))?;
        let device = &self.context.device;
        let command = output.commands.handle;
        let queries = output
            .timing
            .as_ref()
            .map_or(vk::QueryPool::null(), |timing| timing.queries);
        // SAFETY: This set has no pending work. The renderer keeps both
        // snapshot targets alive until the output's fence signals.
        unsafe {
            if queries != vk::QueryPool::null() {
                device.cmd_reset_query_pool(command, queries, 0, 2);
                device.cmd_write_timestamp(
                    command,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    queries,
                    0,
                );
            }
            device.update_descriptor_sets(&writes, &[]);
            for (index, face) in output.faces.iter().enumerate() {
                if params.faces & (1 << index) == 0 {
                    continue;
                }
                device.cmd_begin_render_pass(
                    command,
                    &vk::RenderPassBeginInfo::default()
                        .render_pass(self.pass)
                        .framebuffer(face.framebuffer)
                        .render_area(vk::Rect2D {
                            extent: face.extent,
                            ..Default::default()
                        }),
                    vk::SubpassContents::INLINE,
                );
                device.cmd_bind_pipeline(command, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
                device.cmd_bind_descriptor_sets(
                    command,
                    vk::PipelineBindPoint::GRAPHICS,
                    self.layout,
                    0,
                    &[output.descriptor],
                    &[],
                );
                for eye in 0..2 {
                    let x = eye * self.size;
                    device.cmd_set_viewport(
                        command,
                        0,
                        &[vk::Viewport {
                            x: x as f32,
                            y: 0.0,
                            width: self.size as f32,
                            height: self.size as f32,
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
                                width: self.size,
                                height: self.size,
                            },
                        }],
                    );
                    let draw = index as u32 * 2 + eye;
                    device.cmd_push_constants(
                        command,
                        self.layout,
                        vk::ShaderStageFlags::FRAGMENT,
                        0,
                        &draw.to_ne_bytes(),
                    );
                    device.cmd_draw(command, 3, 1, 0, 0);
                }
                device.cmd_end_render_pass(command);
            }
            if queries != vk::QueryPool::null() {
                device.cmd_write_timestamp(
                    command,
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                    queries,
                    1,
                );
            }
        }
        output.commands.submit()?;
        if let Some(timing) = &mut output.timing {
            timing.submitted = Some(Instant::now());
        }
        output.snapshots = params
            .captures
            .map(|captures| captures.map(|capture| Arc::clone(&capture.snapshot.target)));
        output.rendered = params.faces;
        Ok(())
    }

    pub fn output_image(&self, set: usize, face: Face) -> Option<vk::Image> {
        Some(self.output(set, face)?.image)
    }

    fn output(&self, set: usize, face: Face) -> Option<&Target> {
        let output = self.outputs.get(set)?.as_ref()?;
        let index = face as usize;
        (output.rendered & (1 << index) != 0).then_some(&output.faces[index])
    }

    /// # Safety
    /// Every external consumer must have finished reading this output set.
    /// A local fence is insufficient to establish OpenVR retirement.
    pub unsafe fn retire(&mut self, set: usize) -> Result<()> {
        if let Some(Some(output)) = self.outputs.get_mut(set) {
            output.commands.wait()?;
            output.snapshots = None;
            output.rendered = 0;
        }
        Ok(())
    }

    pub fn read_face(&mut self, set: usize, face: Face) -> Result<Vec<u8>> {
        let target = self
            .output(set, face)
            .ok_or_else(|| io::Error::other("face was not rendered"))?;
        let (image, extent) = (target.image, target.extent);
        self.outputs[set]
            .as_mut()
            .expect("rendered set")
            .commands
            .wait()?;
        let size = (extent.width as usize * extent.height as usize)
            .checked_mul(4)
            .ok_or_else(|| io::Error::other("face byte count overflow"))?;
        let mut buffer = Buffer::new(&self.context, size, vk::BufferUsageFlags::TRANSFER_DST)?;
        self.commands.begin()?;
        let device = &self.context.device;
        let command = self.commands.handle;
        // SAFETY: The face render has retired into TRANSFER_SRC layout, and
        // both the image and the coherent output buffer live through the wait.
        unsafe {
            let region = vk::BufferImageCopy::default()
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
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                buffer.handle,
                &[region],
            );
            let host = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ);
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[host],
                &[],
                &[],
            );
        }
        self.commands.submit()?;
        self.commands.wait()?;
        buffer.read()
    }
}

impl Drop for ShellRenderer {
    fn drop(&mut self) {
        let _ = self.commands.wait();
        // OutputSet drops Commands first, waiting for each set before freeing
        // its uniform, faces and snapshot references.
        self.outputs = std::array::from_fn(|_| None);
        self.placeholder = None;
        // SAFETY: The renderer owns the handles. The application must release
        // external overlays before destroying this renderer.
        unsafe {
            let device = &self.context.device;
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_pool(self.pool, None);
            device.destroy_descriptor_set_layout(self.set_layout, None);
            device.destroy_sampler(self.sampler, None);
            device.destroy_shader_module(self.fragment, None);
            device.destroy_shader_module(self.vertex, None);
            device.destroy_render_pass(self.pass, None);
        }
    }
}

fn parameters(params: ShellParams<'_>, size: u32, context: &Arc<Context>) -> Result<[Draw; 12]> {
    let ShellParams {
        captures,
        shell,
        display_eyes,
        aperture,
        gap,
        refresh_weight,
        blink,
        grid_over_ring,
        dim,
        faces,
    } = params;
    if !shell.valid()
        || display_eyes.iter().any(|pose| !pose.valid())
        || ![refresh_weight, gap.background, gap.lines, dim]
            .into_iter()
            .all(|v| (0.0..=1.0).contains(&v))
        || ![aperture.feather, aperture.strength, aperture.entry]
            .into_iter()
            .all(|v| (0.0..=1.0).contains(&v))
        || ![aperture.half_width_tan, aperture.half_height_tan]
            .into_iter()
            .all(|v| v.is_finite() && v > 0.0)
        || !gap.thickness_cm.is_finite()
        || gap.thickness_cm < 0.0
        || !gap.spacing.is_finite()
        || gap.spacing <= 0.0
        || faces == 0
        || faces & !0x3f != 0
    {
        return Err(io::Error::other("invalid shell render parameters").into());
    }
    let mut dimensions = [0.0; 4];
    if let Some(captures) = captures {
        for (index, capture) in captures.iter().enumerate() {
            if !Arc::ptr_eq(capture.snapshot.target.context(), context)
                || capture
                    .eyes
                    .iter()
                    .any(|(pose, projection)| !pose.valid() || !projection.valid())
            {
                return Err(io::Error::other("invalid shell snapshot").into());
            }
            let extent = capture.snapshot.extent();
            dimensions[index * 2] = extent.width as f32;
            dimensions[index * 2 + 1] = extent.height as f32;
        }
    }
    let mut draws = [Draw::default(); 12];
    for (index, face) in Face::ALL.into_iter().enumerate() {
        let rows = shell
            .face_matrix(face)
            .ok_or_else(|| io::Error::other("invalid shell face matrix"))?;
        let column = |i: usize, w| [rows[0][i], rows[1][i], rows[2][i], w];
        for (eye, pose) in display_eyes.iter().enumerate() {
            let draw = &mut draws[index * 2 + eye];
            draw.center = column(3, shell.half_size as f32);
            draw.right = column(0, size as f32);
            draw.up = column(1, 0.0);
            draw.display = pose.inverse().matrix();
            if let Some(captures) = captures {
                for (index, capture) in captures.iter().enumerate() {
                    let (pose, p) = capture.eyes[eye];
                    draw.captures[index] = CaptureUniform {
                        rows: pose.inverse().matrix(),
                        projection: [p.left as f32, p.right as f32, p.top as f32, p.bottom as f32],
                    };
                }
            }
            draw.mask = [
                aperture.half_width_tan as f32,
                aperture.half_height_tan as f32,
                aperture.feather as f32,
                f32::from(aperture.rectangle),
            ];
            draw.style = [
                aperture.strength as f32,
                aperture.entry as f32,
                refresh_weight,
                f32::from(gap.grid),
            ];
            draw.look = [
                gap.background,
                gap.lines,
                gap.thickness_cm * 0.01,
                gap.spacing,
            ];
            draw.dimensions = dimensions;
            draw.effects = [
                f32::from(blink),
                // No lines in the blink dip. The shader shared with the C++
                // baseline still accepts a line brightness here.
                0.0,
                f32::from(grid_over_ring),
                (1.0 - dim).powf(2.2),
            ];
        }
    }
    if bytemuck::cast_slice::<Draw, f32>(&draws)
        .iter()
        .any(|value| !value.is_finite())
    {
        return Err(io::Error::other("shell parameter exceeds GPU range").into());
    }
    Ok(draws)
}
