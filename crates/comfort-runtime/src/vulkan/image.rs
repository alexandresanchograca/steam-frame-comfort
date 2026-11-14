use super::{Context, Result};
use ash::vk;
use comfort_protocol::ImageAdd;
use std::{
    io,
    os::fd::{AsRawFd, BorrowedFd, IntoRawFd},
    sync::Arc,
};

/// An alias of an observer image. The observer retains its original descriptor
/// and the application owns this VkImage and its Vulkan memory binding.
pub struct ImportedImage {
    context: Arc<Context>,
    handle: vk::Image,
    memory: Option<vk::DeviceMemory>,
    pub description: ImageAdd,
}

impl ImportedImage {
    pub(super) fn context(&self) -> &Arc<Context> {
        &self.context
    }

    pub fn handle(&self) -> vk::Image {
        self.handle
    }

    /// Wrap an image this process created, standing in for an observer image
    /// in GPU tests. The result owns and destroys the image and its memory.
    ///
    /// # Safety
    /// The image must belong to `context`, be bound to `memory`, match
    /// `description`, and have no pending GPU work owned elsewhere.
    #[cfg(feature = "vulkan-fixtures")]
    pub(super) unsafe fn owned(
        context: Arc<Context>,
        handle: vk::Image,
        memory: vk::DeviceMemory,
        description: ImageAdd,
    ) -> Self {
        Self {
            context,
            handle,
            memory: Some(memory),
            description,
        }
    }

    /// Recreate and bind the producer image without recording GPU work.
    ///
    /// # Safety
    /// The description and descriptor must describe a real Vulkan image on
    /// the same physical device and driver. If the returned handle is used
    /// in a submission, those uses must finish before dropping this image.
    pub unsafe fn import(
        context: Arc<Context>,
        description: ImageAdd,
        fd: BorrowedFd<'_>,
    ) -> Result<Self> {
        let d = description;
        let reject = |message| io::Error::other(message);
        if !(2..=16384).contains(&d.width)
            || !(2..=16384).contains(&d.height)
            || d.mip_levels == 0
            || !(1..=2).contains(&d.array_layers)
            || d.samples != vk::SampleCountFlags::TYPE_1.as_raw()
        {
            return Err(reject("unsupported source shape").into());
        }
        let handle_type = vk::ExternalMemoryHandleTypeFlags::from_raw(d.handle_type);
        let dma = handle_type == vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT;
        if !dma && handle_type != vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD {
            return Err(reject("unsupported external handle type").into());
        }
        let memory_fd = context
            .memory_fd
            .as_ref()
            .ok_or_else(|| reject("external memory FD import unavailable"))?;
        let tiling = vk::ImageTiling::from_raw(d.tiling as i32);
        let drm = tiling == vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT;
        let count = d.plane_count as usize;
        if drm && (context.modifier.is_none() || count == 0 || count > d.planes.len()) {
            return Err(reject("DRM-modifier import unavailable").into());
        }
        if d.view_format_count as usize > d.view_formats.len() {
            return Err(reject("invalid view format count").into());
        }
        let layouts = d.planes.map(|plane| vk::SubresourceLayout {
            offset: plane.offset,
            row_pitch: plane.row_pitch,
            array_pitch: if d.array_layers > 1 {
                plane.array_pitch
            } else {
                0
            },
            ..Default::default()
        });
        let formats = d
            .view_formats
            .map(|format| vk::Format::from_raw(format as i32));
        let mut explicit = vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
            .drm_format_modifier(d.modifier)
            .plane_layouts(&layouts[..count.min(layouts.len())]);
        let mut format_list = vk::ImageFormatListCreateInfo::default()
            .view_formats(&formats[..d.view_format_count as usize]);
        let mut external = vk::ExternalMemoryImageCreateInfo::default().handle_types(
            vk::ExternalMemoryHandleTypeFlags::from_raw(if d.external_handle_types == 0 {
                d.handle_type
            } else {
                d.external_handle_types
            }),
        );
        let mut info = vk::ImageCreateInfo::default()
            .flags(vk::ImageCreateFlags::from_raw(d.flags) & !vk::ImageCreateFlags::ALIAS)
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::from_raw(d.format as i32))
            .extent(vk::Extent3D {
                width: d.width,
                height: d.height,
                depth: 1,
            })
            .mip_levels(d.mip_levels)
            .array_layers(d.array_layers)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(tiling)
            .usage(vk::ImageUsageFlags::from_raw(d.usage) | vk::ImageUsageFlags::SAMPLED)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        if drm {
            info = info.push_next(&mut explicit);
        }
        if d.view_format_count != 0 {
            info = info.push_next(&mut format_list);
        }
        info = info.push_next(&mut external);
        let device = &context.device;
        // SAFETY: This recreates the image using the caller's producer
        // descriptor. All chained create parameters are live.
        let handle = unsafe { device.create_image(&info, None)? };
        let mut image = Self {
            context: Arc::clone(&context),
            handle,
            memory: None,
            description,
        };
        if drm {
            let mut actual = vk::ImageDrmFormatModifierPropertiesEXT::default();
            unsafe {
                context
                    .modifier
                    .as_ref()
                    .unwrap()
                    .get_image_drm_format_modifier_properties(handle, &mut actual)?
            };
            if actual.drm_format_modifier != d.modifier {
                return Err(reject("driver chose a different modifier").into());
            }
        }

        let mut dedicated = vk::MemoryDedicatedRequirements::default();
        let requirements_info = vk::ImageMemoryRequirementsInfo2::default().image(handle);
        let mut requirements = vk::MemoryRequirements2::default().push_next(&mut dedicated);
        unsafe { device.get_image_memory_requirements2(&requirements_info, &mut requirements) };
        let requirements = requirements.memory_requirements;
        if requirements.size == 0
            || d.allocation_size
                .checked_sub(d.bind_offset)
                .is_none_or(|size| size < requirements.size)
            || (requirements.alignment != 0
                && !d.bind_offset.is_multiple_of(requirements.alignment))
        {
            return Err(
                reject("imported allocation does not cover the image at its bind offset").into(),
            );
        }
        let mut bits = requirements.memory_type_bits;
        if dma {
            let mut properties = vk::MemoryFdPropertiesKHR::default();
            unsafe {
                memory_fd.get_memory_fd_properties(handle_type, fd.as_raw_fd(), &mut properties)?
            };
            bits &= properties.memory_type_bits;
        }
        let memory_type = if d.memory_type_index < context.memory.memory_type_count
            && bits & (1 << d.memory_type_index) != 0
        {
            d.memory_type_index
        } else if dma {
            context
                .memory_type(bits, vk::MemoryPropertyFlags::empty())
                .ok_or_else(|| reject("no compatible DMA-BUF memory type"))?
        } else {
            return Err(reject("opaque FD requires its original memory type").into());
        };
        let use_dedicated =
            d.dedicated != 0 || dedicated.requires_dedicated_allocation != vk::FALSE;
        if use_dedicated && d.bind_offset != 0 {
            return Err(reject("dedicated import requires offset zero").into());
        }

        let duplicate = fd.try_clone_to_owned()?;
        let mut import = vk::ImportMemoryFdInfoKHR::default()
            .handle_type(handle_type)
            .fd(duplicate.as_raw_fd());
        let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(handle);
        let mut allocation = vk::MemoryAllocateInfo::default()
            .allocation_size(d.allocation_size)
            .memory_type_index(memory_type);
        if use_dedicated {
            allocation = allocation.push_next(&mut dedicated);
        }
        allocation = allocation.push_next(&mut import);
        let memory = unsafe { device.allocate_memory(&allocation, None)? };
        // Vulkan takes ownership only on a successful import.
        let _ = duplicate.into_raw_fd();
        image.memory = Some(memory);
        unsafe { device.bind_image_memory(handle, memory, d.bind_offset)? };
        Ok(image)
    }
}

impl Drop for ImportedImage {
    fn drop(&mut self) {
        // SAFETY: The caller retires GPU uses before drop. This object owns
        // the image and the memory binding, and holds the device alive.
        unsafe {
            self.context.device.destroy_image(self.handle, None);
            if let Some(memory) = self.memory {
                self.context.device.free_memory(memory, None);
            }
        }
    }
}
