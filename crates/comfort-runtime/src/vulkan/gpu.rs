use super::{Context, Result};
use ash::vk;
use std::{io, sync::Arc};

pub(super) const COLOR: vk::ImageAspectFlags = vk::ImageAspectFlags::COLOR;
pub(super) const RGBA: vk::Format = vk::Format::R8G8B8A8_UNORM;

pub(super) fn color_range(layers: u32) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(COLOR)
        .level_count(1)
        .layer_count(layers)
}

pub(super) struct Target {
    context: Arc<Context>,
    pub image: vk::Image,
    memory: vk::DeviceMemory,
    pub view: vk::ImageView,
    pub framebuffer: vk::Framebuffer,
    pub extent: vk::Extent2D,
    pub(super) allocated_bytes: u64,
}

impl Target {
    pub(super) fn context(&self) -> &Arc<Context> {
        &self.context
    }

    pub fn new(context: &Arc<Context>, extent: vk::Extent2D, pass: vk::RenderPass) -> Result<Self> {
        let device = &context.device;
        let limits = unsafe {
            context
                .instance
                .handle
                .get_physical_device_properties(context.physical)
        }
        .limits;
        if extent.width == 0
            || extent.height == 0
            || extent.width
                > limits
                    .max_framebuffer_width
                    .min(limits.max_image_dimension2_d)
            || extent.height
                > limits
                    .max_framebuffer_height
                    .min(limits.max_image_dimension2_d)
        {
            return Err(io::Error::other("invalid render target extent").into());
        }
        let mut target = Self {
            context: Arc::clone(context),
            image: vk::Image::null(),
            memory: vk::DeviceMemory::null(),
            view: vk::ImageView::null(),
            framebuffer: vk::Framebuffer::null(),
            extent,
            allocated_bytes: 0,
        };
        // SAFETY: The target owns these resources and tears down partial
        // allocations on failure. Its users complete all submissions before drop.
        unsafe {
            target.image = device.create_image(
                &vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(RGBA)
                    .extent(vk::Extent3D {
                        width: extent.width,
                        height: extent.height,
                        depth: 1,
                    })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(
                        vk::ImageUsageFlags::COLOR_ATTACHMENT
                            | vk::ImageUsageFlags::SAMPLED
                            | vk::ImageUsageFlags::TRANSFER_SRC,
                    ),
                None,
            )?;
            let requirement = device.get_image_memory_requirements(target.image);
            let index = context
                .memory_type(
                    requirement.memory_type_bits,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                )
                .ok_or_else(|| io::Error::other("no render target memory type"))?;
            target.memory = device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirement.size)
                    .memory_type_index(index),
                None,
            )?;
            target.allocated_bytes = requirement.size;
            device.bind_image_memory(target.image, target.memory, 0)?;
            target.view = device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(target.image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(RGBA)
                    .subresource_range(color_range(1)),
                None,
            )?;
            let attachments = [target.view];
            target.framebuffer = device.create_framebuffer(
                &vk::FramebufferCreateInfo::default()
                    .render_pass(pass)
                    .attachments(&attachments)
                    .width(extent.width)
                    .height(extent.height)
                    .layers(1),
                None,
            )?;
        }
        Ok(target)
    }
}

impl Drop for Target {
    fn drop(&mut self) {
        // SAFETY: The target owns these handles, and destroying a null handle
        // is permitted. Its users guarantee the target is no longer queued.
        unsafe {
            let device = &self.context.device;
            device.destroy_framebuffer(self.framebuffer, None);
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
            device.free_memory(self.memory, None);
        }
    }
}

pub(super) struct Buffer {
    context: Arc<Context>,
    pub handle: vk::Buffer,
    memory: vk::DeviceMemory,
    size: usize,
}

impl Buffer {
    pub fn size(&self) -> usize {
        self.size
    }

    pub fn new(context: &Arc<Context>, size: usize, usage: vk::BufferUsageFlags) -> Result<Self> {
        let device = &context.device;
        let mut buffer = Self {
            context: Arc::clone(context),
            handle: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
            size,
        };
        // SAFETY: All allocations belong to buffer and are dropped on failure.
        unsafe {
            buffer.handle = device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size as u64)
                    .usage(usage),
                None,
            )?;
            let requirement = device.get_buffer_memory_requirements(buffer.handle);
            let index = context
                .memory_type(
                    requirement.memory_type_bits,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                )
                .ok_or_else(|| io::Error::other("no host coherent buffer memory"))?;
            buffer.memory = device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirement.size)
                    .memory_type_index(index),
                None,
            )?;
            device.bind_buffer_memory(buffer.handle, buffer.memory, 0)?;
        }
        Ok(buffer)
    }

    pub fn write(&mut self, data: &[u8]) -> Result<()> {
        if data.len() != self.size {
            return Err(io::Error::other("buffer write size differs from allocation").into());
        }
        // SAFETY: The buffer is host coherent and exclusively borrowed after
        // the caller completed all earlier uses of its contents.
        unsafe {
            let device = &self.context.device;
            let ptr = device.map_memory(
                self.memory,
                0,
                self.size as u64,
                vk::MemoryMapFlags::empty(),
            )?;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.cast(), self.size);
            device.unmap_memory(self.memory);
        }
        Ok(())
    }

    pub fn read(&mut self) -> Result<Vec<u8>> {
        // SAFETY: The caller has completed the transfer to this host-coherent
        // buffer. The read finishes before unmapping.
        unsafe {
            let device = &self.context.device;
            let ptr = device.map_memory(
                self.memory,
                0,
                self.size as u64,
                vk::MemoryMapFlags::empty(),
            )?;
            let bytes = std::slice::from_raw_parts(ptr.cast::<u8>(), self.size).to_vec();
            device.unmap_memory(self.memory);
            Ok(bytes)
        }
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        // SAFETY: The buffer owns its binding and has no outstanding work.
        unsafe {
            self.context.device.destroy_buffer(self.handle, None);
            self.context.device.free_memory(self.memory, None);
        }
    }
}

pub(super) struct Commands {
    context: Arc<Context>,
    pool: vk::CommandPool,
    pub handle: vk::CommandBuffer,
    fence: vk::Fence,
    pending: bool,
}

impl Commands {
    pub(super) fn complete(&self) -> Result<bool> {
        // SAFETY: The fence is live and belongs to this command buffer.
        Ok(!self.pending || unsafe { self.context.device.get_fence_status(self.fence)? })
    }

    pub fn new(context: &Arc<Context>) -> Result<Self> {
        let device = &context.device;
        let mut commands = Self {
            context: Arc::clone(context),
            pool: vk::CommandPool::null(),
            handle: vk::CommandBuffer::null(),
            fence: vk::Fence::null(),
            pending: false,
        };
        // SAFETY: The pool owns the command buffer and is tied to the queue family.
        unsafe {
            commands.pool = device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                    .queue_family_index(context.queue_family),
                None,
            )?;
            commands.handle = device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(commands.pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )?[0];
            commands.fence = device.create_fence(&vk::FenceCreateInfo::default(), None)?;
        }
        Ok(commands)
    }

    pub fn begin(&mut self) -> Result<()> {
        self.wait()?;
        // SAFETY: The preceding submission is no longer pending.
        unsafe {
            self.context
                .device
                .reset_command_buffer(self.handle, vk::CommandBufferResetFlags::empty())?;
            self.context.device.begin_command_buffer(
                self.handle,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
        }
        Ok(())
    }

    pub fn submit(&mut self) -> Result<()> {
        let handles = [self.handle];
        let info = vk::SubmitInfo::default().command_buffers(&handles);
        // SAFETY: The command is recorded by the caller. The queue lock
        // serializes submissions made by the Rust renderer.
        unsafe {
            self.context.device.end_command_buffer(self.handle)?;
            self.context.device.reset_fences(&[self.fence])?;
            let _queue = self
                .context
                .queue_lock
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            self.context
                .device
                .queue_submit(self.context.queue, &[info], self.fence)?;
        }
        self.pending = true;
        Ok(())
    }

    pub fn wait(&mut self) -> Result<()> {
        if self.pending {
            // SAFETY: A successful submission signaled this fence.
            unsafe {
                self.context
                    .device
                    .wait_for_fences(&[self.fence], true, u64::MAX)?
            };
            self.pending = false;
        }
        Ok(())
    }
}

impl Drop for Commands {
    fn drop(&mut self) {
        let _ = self.wait();
        // SAFETY: The fence and pool belong to this object. If the wait
        // failed, the device is lost and teardown may proceed.
        unsafe {
            self.context.device.destroy_fence(self.fence, None);
            self.context.device.destroy_command_pool(self.pool, None);
        }
    }
}

pub(super) fn create_render_pass(
    device: &ash::Device,
    final_layout: vk::ImageLayout,
) -> Result<vk::RenderPass> {
    let attachments = [vk::AttachmentDescription::default()
        .format(RGBA)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::DONT_CARE)
        .store_op(vk::AttachmentStoreOp::STORE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(final_layout)];
    let colors = [vk::AttachmentReference {
        attachment: 0,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    }];
    let subpasses = [vk::SubpassDescription::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&colors)];
    let dependencies = [
        vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE),
        vk::SubpassDependency::default()
            .src_subpass(0)
            .dst_subpass(vk::SUBPASS_EXTERNAL)
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(
                vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::TRANSFER,
            )
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::TRANSFER_READ),
    ];
    // SAFETY: The pass descriptions borrow live arrays and contain one color target.
    Ok(unsafe {
        device.create_render_pass(
            &vk::RenderPassCreateInfo::default()
                .attachments(&attachments)
                .subpasses(&subpasses)
                .dependencies(&dependencies),
            None,
        )?
    })
}

pub(super) fn shader_module(device: &ash::Device, bytes: &[u8]) -> Result<vk::ShaderModule> {
    let code = ash::util::read_spv(&mut io::Cursor::new(bytes))?;
    // SAFETY: The code was produced by glslc and parsed into aligned words.
    Ok(unsafe {
        device.create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)?
    })
}

pub(super) fn graphics_pipeline(
    device: &ash::Device,
    layout: vk::PipelineLayout,
    pass: vk::RenderPass,
    vertex: vk::ShaderModule,
    fragment: vk::ShaderModule,
) -> Result<vk::Pipeline> {
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex)
            .name(c"main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment)
            .name(c"main"),
    ];
    let input = vk::PipelineVertexInputStateCreateInfo::default();
    let assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .line_width(1.0);
    let samples = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let colors = [vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA)];
    let blend = vk::PipelineColorBlendStateCreateInfo::default().attachments(&colors);
    let dynamic = vk::PipelineDynamicStateCreateInfo::default()
        .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);
    let info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&input)
        .input_assembly_state(&assembly)
        .viewport_state(&viewport)
        .rasterization_state(&raster)
        .multisample_state(&samples)
        .color_blend_state(&blend)
        .dynamic_state(&dynamic)
        .layout(layout)
        .render_pass(pass);
    // SAFETY: The shader modules, pass and layout all belong to this device.
    unsafe {
        match device.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None) {
            Ok(pipelines) => Ok(pipelines[0]),
            Err((partial, error)) => {
                for pipeline in partial {
                    device.destroy_pipeline(pipeline, None);
                }
                Err(error.into())
            }
        }
    }
}
