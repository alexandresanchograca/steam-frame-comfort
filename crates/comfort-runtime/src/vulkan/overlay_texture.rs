use super::{Context, Result, gpu::*};
use ash::vk;
use std::{io, sync::Arc};

/// Two RGBA images an overlay alternates between, so an upload never writes
/// the image SteamVR is showing. Raw overlay uploads flickered on every
/// change. This is the C++ baseline's `OverlayTexture`.
pub struct OverlayTexture {
    context: Arc<Context>,
    commands: Commands,
    staging: Buffer,
    images: [(vk::Image, vk::DeviceMemory); 2],
    extent: vk::Extent2D,
    next: usize,
}

impl OverlayTexture {
    pub fn new(context: Arc<Context>, width: u32, height: u32) -> Result<Self> {
        let size = (width as usize)
            .checked_mul(height as usize)
            .and_then(|pixels| pixels.checked_mul(4))
            .filter(|&bytes| bytes != 0)
            .ok_or_else(|| io::Error::other("invalid overlay texture extent"))?;
        let mut texture = Self {
            commands: Commands::new(&context)?,
            staging: Buffer::new(&context, size, vk::BufferUsageFlags::TRANSFER_SRC)?,
            images: [(vk::Image::null(), vk::DeviceMemory::null()); 2],
            extent: vk::Extent2D { width, height },
            next: 0,
            context,
        };
        for index in 0..texture.images.len() {
            texture.create_image(index)?;
        }
        Ok(texture)
    }

    fn create_image(&mut self, index: usize) -> Result<()> {
        let device = &self.context.device;
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(RGBA)
            .extent(vk::Extent3D {
                width: self.extent.width,
                height: self.extent.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(
                vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::TRANSFER_SRC
                    | vk::ImageUsageFlags::SAMPLED,
            );
        // SAFETY: The texture owns both handles. Each is recorded as soon as
        // it exists, so Drop frees a partial allocation.
        unsafe {
            let image = device.create_image(&info, None)?;
            self.images[index].0 = image;
            let requirements = device.get_image_memory_requirements(image);
            let memory_type = self
                .context
                .memory_type(
                    requirements.memory_type_bits,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                )
                .ok_or_else(|| io::Error::other("no device memory for the overlay texture"))?;
            let memory = device.allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(memory_type),
                None,
            )?;
            self.images[index].1 = memory;
            device.bind_image_memory(image, memory, 0)?;
        }
        Ok(())
    }

    /// Copy `rgba` into the image that is not on show and pass it to
    /// `present`, which submits it to the overlay. The images swap roles only
    /// after a successful submission. The copy has finished before `present`
    /// runs, and the image is left in `TRANSFER_SRC_OPTIMAL` for SteamVR.
    pub fn upload(
        &mut self,
        rgba: &[u8],
        present: impl FnOnce(vk::Image, vk::Extent2D) -> io::Result<()>,
    ) -> Result<()> {
        if rgba.len() != self.staging.size() {
            return Err(io::Error::other("overlay pixels differ from the texture extent").into());
        }
        // Waits for the previous copy, which read the staging buffer.
        self.commands.begin()?;
        self.staging.write(rgba)?;
        let image = self.images[self.next].0;
        let barrier = |old, new, from, to| {
            vk::ImageMemoryBarrier::default()
                .image(image)
                .subresource_range(color_range(1))
                .old_layout(old)
                .new_layout(new)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .src_access_mask(from)
                .dst_access_mask(to)
        };
        let device = &self.context.device;
        let command = self.commands.handle;
        // SAFETY: The command buffer is recording, and the image is not the
        // one SteamVR was last given. Its whole contents are replaced.
        unsafe {
            // SteamVR copied this image on the same queue two uploads ago.
            // The transfer stage orders this write after that read.
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier(
                    vk::ImageLayout::UNDEFINED,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::AccessFlags::empty(),
                    vk::AccessFlags::TRANSFER_WRITE,
                )],
            );
            device.cmd_copy_buffer_to_image(
                command,
                self.staging.handle,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::BufferImageCopy::default()
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(COLOR)
                            .layer_count(1),
                    )
                    .image_extent(vk::Extent3D {
                        width: self.extent.width,
                        height: self.extent.height,
                        depth: 1,
                    })],
            );
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier(
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::AccessFlags::TRANSFER_READ,
                )],
            );
        }
        self.commands.submit()?;
        self.commands.wait()?;
        present(image, self.extent)?;
        self.next ^= 1;
        Ok(())
    }
}

impl Drop for OverlayTexture {
    fn drop(&mut self) {
        let _ = self.commands.wait();
        // SteamVR copies the images on this device's queue.
        let _ = self.context.wait_idle();
        // SAFETY: All work using the images is complete. Destroying a null
        // handle is permitted.
        unsafe {
            let device = &self.context.device;
            for (image, memory) in self.images {
                device.destroy_image(image, None);
                device.free_memory(memory, None);
            }
        }
    }
}
