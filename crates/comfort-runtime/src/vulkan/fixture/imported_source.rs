//! Exercise the real observer's imported-memory path with an immutable image.
use crate::vulkan::{
    Context, ContextOptions, ImportedImage, Result,
    gpu::{Buffer, Commands, RGBA, color_range},
};
use ash::vk;
use comfort_protocol::{ImageAdd, Plane};
use std::{
    io,
    os::fd::{AsFd, FromRawFd, OwnedFd},
    sync::Arc,
};

const SIDE: u32 = 256;
const USAGE: vk::ImageUsageFlags = vk::ImageUsageFlags::from_raw(
    vk::ImageUsageFlags::SAMPLED.as_raw() | vk::ImageUsageFlags::TRANSFER_DST.as_raw(),
);

struct Export {
    context: Arc<Context>,
    image: vk::Image,
    memory: vk::DeviceMemory,
}

impl Drop for Export {
    fn drop(&mut self) {
        // SAFETY: Both resources are ours and all submissions have completed.
        unsafe {
            self.context.device.destroy_image(self.image, None);
            self.context.device.free_memory(self.memory, None);
        }
    }
}

fn context(dma: bool) -> Result<Option<Arc<Context>>> {
    let options = ContextOptions {
        device_extensions: &[ash::khr::external_memory_fd::NAME],
        require_dma_buf: dma,
        ..Default::default()
    };
    match Context::new(options) {
        Ok(context) => Ok(Some(context)),
        // The ordinary opaque test requires a working Vulkan device. Here the
        // selected driver can legitimately lack one of the DMA-BUF extensions.
        Err(error) if dma && error.to_string().starts_with("No suitable GPU:") => Ok(None),
        Err(error) => Err(error),
    }
}

fn export(context: &Arc<Context>, dma: bool) -> Result<Option<(Export, ImageAdd, OwnedFd)>> {
    let handle = if dma {
        vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT
    } else {
        vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD
    };
    let tiling = if dma {
        vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT
    } else {
        vk::ImageTiling::OPTIMAL
    };
    let mut external_format =
        vk::PhysicalDeviceExternalImageFormatInfo::default().handle_type(handle);
    let mut drm_format = vk::PhysicalDeviceImageDrmFormatModifierInfoEXT::default()
        .drm_format_modifier(0) // DRM_FORMAT_MOD_LINEAR
        .sharing_mode(vk::SharingMode::EXCLUSIVE);
    let mut format_info = vk::PhysicalDeviceImageFormatInfo2::default()
        .format(RGBA)
        .ty(vk::ImageType::TYPE_2D)
        .tiling(tiling)
        .usage(USAGE)
        .push_next(&mut external_format);
    if dma {
        format_info = format_info.push_next(&mut drm_format);
    }
    let mut external_properties = vk::ExternalImageFormatProperties::default();
    let mut properties = vk::ImageFormatProperties2::default().push_next(&mut external_properties);
    // SAFETY: The chained parameters and the physical device are live.
    match unsafe {
        context
            .instance
            .handle
            .get_physical_device_image_format_properties2(
                context.physical,
                &format_info,
                &mut properties,
            )
    } {
        Err(vk::Result::ERROR_FORMAT_NOT_SUPPORTED) => return Ok(None),
        result => result?,
    }
    let supported = external_properties
        .external_memory_properties
        .external_memory_features;
    if !supported.contains(
        vk::ExternalMemoryFeatureFlags::IMPORTABLE | vk::ExternalMemoryFeatureFlags::EXPORTABLE,
    ) {
        return Ok(None);
    }

    let mut export = Export {
        context: Arc::clone(context),
        image: vk::Image::null(),
        memory: vk::DeviceMemory::null(),
    };
    let device = &context.device;
    let extent = vk::Extent3D {
        width: SIDE,
        height: SIDE,
        depth: 1,
    };
    let mut external = vk::ExternalMemoryImageCreateInfo::default().handle_types(handle);
    let mut modifiers =
        vk::ImageDrmFormatModifierListCreateInfoEXT::default().drm_format_modifiers(&[0]);
    let mut image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(RGBA)
        .extent(extent)
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(tiling)
        .usage(USAGE)
        .push_next(&mut external);
    if dma {
        image_info = image_info.push_next(&mut modifiers);
    }

    // SAFETY: The exporter owns the image and its memory. All Vulkan
    // structure chains refer to values on this stack for the duration of calls.
    unsafe {
        export.image = device.create_image(&image_info, None)?;
        let mut dedicated = vk::MemoryDedicatedRequirements::default();
        let mut requirements = vk::MemoryRequirements2::default().push_next(&mut dedicated);
        device.get_image_memory_requirements2(
            &vk::ImageMemoryRequirementsInfo2::default().image(export.image),
            &mut requirements,
        );
        let requirements = requirements.memory_requirements;
        let index = context
            .memory_type(
                requirements.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .ok_or_else(|| io::Error::other("no device memory for the source fixture"))?;
        let dedicated = dedicated.requires_dedicated_allocation == vk::TRUE
            || supported.contains(vk::ExternalMemoryFeatureFlags::DEDICATED_ONLY);
        let mut dedicated_info = vk::MemoryDedicatedAllocateInfo::default().image(export.image);
        let mut export_info = vk::ExportMemoryAllocateInfo::default().handle_types(handle);
        let mut allocation = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(index)
            .push_next(&mut export_info);
        if dedicated {
            allocation = allocation.push_next(&mut dedicated_info);
        }
        export.memory = device.allocate_memory(&allocation, None)?;
        device.bind_image_memory(export.image, export.memory, 0)?;

        let mut description = ImageAdd {
            allocation_size: requirements.size,
            handle_type: handle.as_raw(),
            external_handle_types: handle.as_raw(),
            memory_type_index: index,
            format: RGBA.as_raw() as u32,
            width: SIDE,
            height: SIDE,
            mip_levels: 1,
            array_layers: 1,
            samples: vk::SampleCountFlags::TYPE_1.as_raw(),
            tiling: tiling.as_raw() as u32,
            usage: USAGE.as_raw(),
            dedicated: u32::from(dedicated),
            ..Default::default()
        };
        if dma {
            let mut properties = vk::ImageDrmFormatModifierPropertiesEXT::default();
            context
                .modifier
                .as_ref()
                .expect("requested DRM modifiers")
                .get_image_drm_format_modifier_properties(export.image, &mut properties)?;
            assert_eq!(properties.drm_format_modifier, 0);
            let layout = device.get_image_subresource_layout(
                export.image,
                vk::ImageSubresource::default()
                    .aspect_mask(vk::ImageAspectFlags::MEMORY_PLANE_0_EXT),
            );
            description.plane_count = 1;
            description.planes[0] = Plane {
                offset: layout.offset,
                size: layout.size,
                row_pitch: layout.row_pitch,
                array_pitch: layout.array_pitch,
                depth_pitch: layout.depth_pitch,
            };
        }
        let fd = context
            .memory_fd
            .as_ref()
            .expect("requested external memory FD")
            .get_memory_fd(
                &vk::MemoryGetFdInfoKHR::default()
                    .memory(export.memory)
                    .handle_type(handle),
            )?;
        Ok(Some((export, description, OwnedFd::from_raw_fd(fd))))
    }
}

fn barrier(image: vk::Image) -> vk::ImageMemoryBarrier<'static> {
    vk::ImageMemoryBarrier::default()
        .image(image)
        .subresource_range(color_range(1))
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
}

fn paint(context: &Arc<Context>, image: vk::Image) -> Result<()> {
    let pixels: Vec<_> = (0..SIDE * SIDE)
        .flat_map(|i| {
            if i % SIDE < SIDE / 4 {
                [220, 200, 0, 255]
            } else if i / SIDE < SIDE / 4 {
                [0, 200, 0, 255]
            } else {
                [0, 0, 200, 255]
            }
        })
        .collect();
    let mut staging = Buffer::new(context, pixels.len(), vk::BufferUsageFlags::TRANSFER_SRC)?;
    staging.write(&pixels)?;
    let mut commands = Commands::new(context)?;
    commands.begin()?;
    let device = &context.device;
    let command = commands.handle;
    // SAFETY: This command owns the initial write and releases the finished
    // image to EXTERNAL. The host waits before the reimport and acquisition.
    unsafe {
        device.cmd_pipeline_barrier(
            command,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier(image)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)],
        );
        device.cmd_copy_buffer_to_image(
            command,
            staging.handle,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[vk::BufferImageCopy::default()
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .layer_count(1),
                )
                .image_extent(vk::Extent3D {
                    width: SIDE,
                    height: SIDE,
                    depth: 1,
                })],
        );
        device.cmd_pipeline_barrier(
            command,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier(image)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_queue_family_index(context.queue_family)
                .dst_queue_family_index(vk::QUEUE_FAMILY_EXTERNAL)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)],
        );
    }
    commands.submit()?;
    commands.wait()
}

fn publish(context: &Arc<Context>, image: vk::Image) -> Result<()> {
    let mut commands = Commands::new(context)?;
    commands.begin()?;
    let device = &context.device;
    let ownership = barrier(image)
        .old_layout(vk::ImageLayout::GENERAL)
        .new_layout(vk::ImageLayout::GENERAL);
    // SAFETY: paint completed the external release. The alias will remain
    // under external ownership until the observer's client reads it.
    unsafe {
        device.cmd_pipeline_barrier(
            commands.handle,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[ownership
                .src_queue_family_index(vk::QUEUE_FAMILY_EXTERNAL)
                .dst_queue_family_index(context.queue_family)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)],
        );
        device.cmd_pipeline_barrier(
            commands.handle,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[ownership
                .src_queue_family_index(context.queue_family)
                .dst_queue_family_index(vk::QUEUE_FAMILY_EXTERNAL)
                .src_access_mask(vk::AccessFlags::SHADER_READ)],
        );
    }
    commands.submit()?;
    commands.wait()
}

/// Publish one external handoff, then keep the fixture alive until the caller
/// says its consumer has finished. Returns false for an unsupported format.
pub fn publish_imported_source(
    dma: bool,
    wait_for_client: impl FnOnce() -> io::Result<()>,
) -> Result<bool> {
    let Some(context) = context(dma)? else {
        return Ok(false);
    };
    let Some((export, description, fd)) = export(&context, dma)? else {
        return Ok(false);
    };
    paint(&context, export.image)?;
    // SAFETY: The descriptor describes the export above on the same device.
    let imported = unsafe { ImportedImage::import(Arc::clone(&context), description, fd.as_fd())? };
    publish(&context, imported.handle())?;
    wait_for_client()?;
    Ok(true)
}
