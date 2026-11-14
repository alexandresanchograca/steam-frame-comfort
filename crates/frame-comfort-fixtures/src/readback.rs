use ash::{Entry, vk, vk::Handle};
use comfort_runtime::openvr::ffi::VRVulkanTextureData_t;
use std::{error::Error, fs, io, path::Path};

struct Readback<'a> {
    device: &'a ash::Device,
    pool: vk::CommandPool,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
}

impl Drop for Readback<'_> {
    fn drop(&mut self) {
        // The transfer has completed or submission failed. The fixture only
        // owns the command pool and staging allocation.
        unsafe {
            self.device.destroy_command_pool(self.pool, None);
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

/// The caller supplies live Vulkan handles from a completed OpenVR overlay
/// submission. The fixture owns the calling thread's use of the queue.
pub unsafe fn save(data: &VRVulkanTextureData_t, path: &Path) -> Result<(), Box<dyn Error>> {
    let width = data.m_nWidth;
    let height = data.m_nHeight;
    if width == 0
        || height == 0
        || data.m_nSampleCount != 1
        || data.m_nFormat != vk::Format::R8G8B8A8_UNORM.as_raw() as u32
    {
        return Err(io::Error::other("unexpected fixture texture").into());
    }
    let size = u64::from(width) * u64::from(height) * 4;
    // SAFETY: Load dispatch for the supplied instance and device. They remain
    // owned by the app and outlive the borrowed texture readback.
    unsafe {
        let entry = Entry::load()?;
        let instance = ash::Instance::load(
            entry.static_fn(),
            vk::Instance::from_raw(data.m_pInstance as usize as u64),
        );
        let device = ash::Device::load(
            instance.fp_v1_0(),
            vk::Device::from_raw(data.m_pDevice as usize as u64),
        );
        let physical = vk::PhysicalDevice::from_raw(data.m_pPhysicalDevice as usize as u64);
        let queue = vk::Queue::from_raw(data.m_pQueue as usize as u64);
        let mut copy = Readback {
            device: &device,
            pool: vk::CommandPool::null(),
            buffer: vk::Buffer::null(),
            memory: vk::DeviceMemory::null(),
        };
        copy.buffer = device.create_buffer(
            &vk::BufferCreateInfo::default()
                .size(size)
                .usage(vk::BufferUsageFlags::TRANSFER_DST),
            None,
        )?;
        let requirements = device.get_buffer_memory_requirements(copy.buffer);
        let properties = instance.get_physical_device_memory_properties(physical);
        let host = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let index = (0..properties.memory_type_count)
            .find(|&index| {
                requirements.memory_type_bits & (1 << index) != 0
                    && properties.memory_types[index as usize]
                        .property_flags
                        .contains(host)
            })
            .ok_or_else(|| io::Error::other("no coherent readback memory"))?;
        copy.memory = device.allocate_memory(
            &vk::MemoryAllocateInfo::default()
                .allocation_size(requirements.size)
                .memory_type_index(index),
            None,
        )?;
        device.bind_buffer_memory(copy.buffer, copy.memory, 0)?;
        copy.pool = device.create_command_pool(
            &vk::CommandPoolCreateInfo::default().queue_family_index(data.m_nQueueFamilyIndex),
            None,
        )?;
        let command = device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(copy.pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )?[0];
        device.begin_command_buffer(
            command,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;
        device.cmd_copy_image_to_buffer(
            command,
            vk::Image::from_raw(data.m_nImage),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            copy.buffer,
            &[vk::BufferImageCopy::default()
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .layer_count(1),
                )
                .image_extent(vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                })],
        );
        device.end_command_buffer(command)?;
        device.queue_submit(
            queue,
            &[vk::SubmitInfo::default().command_buffers(&[command])],
            vk::Fence::null(),
        )?;
        device.queue_wait_idle(queue)?;
        let bytes = device.map_memory(copy.memory, 0, size, vk::MemoryMapFlags::empty())?;
        let result = fs::write(
            path,
            std::slice::from_raw_parts(bytes.cast::<u8>(), size as usize),
        );
        device.unmap_memory(copy.memory);
        result?;
    }
    Ok(())
}
