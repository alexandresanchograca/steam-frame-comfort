use ash::{Entry, vk};
use std::{
    error::Error,
    ffi::{CStr, c_char},
    io,
    sync::{Arc, Mutex},
};

#[cfg(feature = "vulkan-fixtures")]
#[doc(hidden)]
pub mod fixture;
mod gpu;
mod image;
mod overlay_texture;
#[cfg(feature = "vulkan-fixtures")]
mod pattern;
#[cfg(all(feature = "vulkan-fixtures", any(feature = "scene-fixtures", test)))]
#[doc(hidden)]
pub mod scene;
mod shell;
mod snapshot;
pub use image::ImportedImage;
pub use overlay_texture::OverlayTexture;
pub use shell::{Capture, Gap, ShellParams, ShellRenderer};
pub use snapshot::{Snapshot, SnapshotRenderer, SourceView};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Default)]
pub struct ContextOptions<'a> {
    pub instance_extensions: &'a [&'a CStr],
    pub device_extensions: &'a [&'a CStr],
    pub device_uuid: Option<[u8; 16]>,
    pub require_dma_buf: bool,
    pub validation: bool,
    pub high_priority: bool,
}

struct Instance {
    _entry: Entry,
    handle: ash::Instance,
}

impl Drop for Instance {
    fn drop(&mut self) {
        // SAFETY: The instance outlives every device which references it.
        unsafe { self.handle.destroy_instance(None) };
    }
}

pub struct Context {
    instance: Instance,
    device: ash::Device,
    physical: vk::PhysicalDevice,
    memory: vk::PhysicalDeviceMemoryProperties,
    queue: vk::Queue,
    queue_lock: Mutex<()>,
    queue_family: u32,
    name: String,
    device_uuid: [u8; 16],
    driver_uuid: [u8; 16],
    high_priority: bool,
    memory_fd: Option<ash::khr::external_memory_fd::Device>,
    modifier: Option<ash::ext::image_drm_format_modifier::Device>,
}

impl Context {
    pub fn new(options: ContextOptions<'_>) -> Result<Arc<Self>> {
        // SAFETY: The system loader remains alive in Instance until all
        // devices and their resources are destroyed.
        let entry = unsafe { Entry::load()? };
        let version =
            unsafe { entry.try_enumerate_instance_version()? }.unwrap_or(vk::API_VERSION_1_0);
        if version < vk::API_VERSION_1_1 {
            return Err(io::Error::other("Vulkan 1.1 loader required").into());
        }

        let application = vk::ApplicationInfo::default()
            .application_name(c"VR Window Frame Comfort")
            .api_version(version.min(vk::API_VERSION_1_3));
        let extensions = pointers(options.instance_extensions);
        let layers = if options.validation {
            &[c"VK_LAYER_KHRONOS_validation".as_ptr()][..]
        } else {
            &[]
        };
        let create = vk::InstanceCreateInfo::default()
            .application_info(&application)
            .enabled_extension_names(&extensions)
            .enabled_layer_names(layers);
        // SAFETY: create and its arrays outlive the call.
        let handle = unsafe { entry.create_instance(&create, None)? };
        let instance = Instance {
            _entry: entry,
            handle,
        };
        // SAFETY: Instance is live for physical device queries.
        let devices = unsafe { instance.handle.enumerate_physical_devices()? };
        let mut rejected = Vec::new();
        for physical in devices {
            match DeviceCandidate::new(&instance.handle, physical, &options) {
                Ok(candidate) => return candidate.create(instance, options.high_priority),
                Err(error) => rejected.push(error.to_string()),
            }
        }
        Err(io::Error::other(format!("No suitable GPU: {}", rejected.join("; "))).into())
    }

    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn device_uuid(&self) -> [u8; 16] {
        self.device_uuid
    }
    pub fn driver_uuid(&self) -> [u8; 16] {
        self.driver_uuid
    }
    pub fn high_priority(&self) -> bool {
        self.high_priority
    }
    pub fn supports_dma_buf(&self) -> bool {
        self.modifier.is_some()
    }

    pub fn wait_idle(&self) -> Result<()> {
        let _queue = self
            .queue_lock
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        // SAFETY: The device is live and its single queue is externally synchronized.
        Ok(unsafe { self.device.device_wait_idle() }?)
    }

    pub fn queue_family(&self) -> u32 {
        self.queue_family
    }

    // These handles can be shared with OpenVR. Operations on them remain
    // unsafe and callers must follow Vulkan synchronization and lifetime rules.
    pub fn raw_handles(&self) -> (vk::Instance, vk::PhysicalDevice, vk::Device, vk::Queue) {
        (
            self.instance.handle.handle(),
            self.physical,
            self.device.handle(),
            self.queue,
        )
    }

    fn memory_type(&self, bits: u32, flags: vk::MemoryPropertyFlags) -> Option<u32> {
        (0..self.memory.memory_type_count).find(|&index| {
            let properties = self.memory.memory_types[index as usize].property_flags;
            bits & (1 << index) != 0
                && properties.contains(flags)
                && !properties.contains(vk::MemoryPropertyFlags::PROTECTED)
        })
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        // SAFETY: Every child resource holds an Arc to the device. The queue
        // can be retired before the device and loader are destroyed.
        unsafe {
            let _ = self.device.device_wait_idle();
            self.device.destroy_device(None);
        }
    }
}

struct DeviceCandidate<'a> {
    physical: vk::PhysicalDevice,
    queue_family: u32,
    name: String,
    id: vk::PhysicalDeviceIDProperties<'static>,
    extensions: Vec<&'a CStr>,
    priority_extension: Option<&'static CStr>,
}

impl<'a> DeviceCandidate<'a> {
    fn new(
        instance: &ash::Instance,
        physical: vk::PhysicalDevice,
        options: &ContextOptions<'a>,
    ) -> Result<Self> {
        let mut id = vk::PhysicalDeviceIDProperties::default();
        let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut id);
        // SAFETY: The physical device came from this instance and the next
        // chain consists of live stack values.
        unsafe { instance.get_physical_device_properties2(physical, &mut properties) };
        let properties = properties.properties;
        let name = unsafe { CStr::from_ptr(properties.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        let reject = |message: &str| io::Error::other(format!("{name}: {message}"));
        if properties.api_version < vk::API_VERSION_1_1 {
            return Err(reject("Vulkan 1.1 device required").into());
        }
        if options
            .device_uuid
            .is_some_and(|expected| expected != id.device_uuid)
        {
            return Err(reject("device UUID differs from the source").into());
        }
        let available = unsafe { instance.enumerate_device_extension_properties(physical)? };
        let supports = |extension: &CStr| {
            available.iter().any(|property| {
                // SAFETY: Vulkan returns a NUL-terminated name in this array.
                unsafe { CStr::from_ptr(property.extension_name.as_ptr()) == extension }
            })
        };

        let mut extensions = Vec::new();
        for &extension in options.device_extensions {
            append_extension(&mut extensions, extension);
        }
        if options.require_dma_buf {
            append_extension(&mut extensions, ash::khr::external_memory_fd::NAME);
            append_extension(&mut extensions, ash::ext::external_memory_dma_buf::NAME);
            append_extension(&mut extensions, ash::ext::image_drm_format_modifier::NAME);
        } else if supports(ash::khr::external_memory_fd::NAME) {
            append_extension(&mut extensions, ash::khr::external_memory_fd::NAME);
        }
        if properties.api_version < vk::API_VERSION_1_2
            && extensions.contains(&ash::ext::image_drm_format_modifier::NAME)
        {
            append_extension(&mut extensions, ash::khr::image_format_list::NAME);
        }
        if let Some(missing) = extensions.iter().find(|&&extension| !supports(extension)) {
            return Err(reject(&format!("missing {}", missing.to_string_lossy())).into());
        }

        let priority_extension = [
            ash::khr::global_priority::NAME,
            ash::ext::global_priority::NAME,
        ]
        .into_iter()
        .find(|&extension| supports(extension));
        let mut conversion = vk::PhysicalDeviceSamplerYcbcrConversionFeatures::default();
        let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut conversion);
        unsafe { instance.get_physical_device_features2(physical, &mut features) };
        if conversion.sampler_ycbcr_conversion == vk::FALSE {
            return Err(reject("no sampler YCbCr conversion").into());
        }
        let queue_family = unsafe { instance.get_physical_device_queue_family_properties(physical) }
            .iter()
            .position(|family| {
                family.queue_count != 0
                    && family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                    && !family.queue_flags.contains(vk::QueueFlags::PROTECTED)
            })
            .ok_or_else(|| reject("no graphics queue"))? as u32;
        Ok(Self {
            physical,
            queue_family,
            name,
            id,
            extensions,
            priority_extension,
        })
    }

    fn create(self, instance: Instance, high_priority: bool) -> Result<Arc<Context>> {
        let queue = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(self.queue_family)
            .queue_priorities(&[1.0]);
        let create = |priority: Option<&CStr>| {
            let mut names = self.extensions.clone();
            let mut global = vk::DeviceQueueGlobalPriorityCreateInfoKHR::default()
                .global_priority(vk::QueueGlobalPriorityKHR::HIGH);
            let mut queue = queue;
            if let Some(extension) = priority {
                append_extension(&mut names, extension);
                queue = queue.push_next(&mut global);
            }
            let names = pointers(&names);
            let mut conversion = vk::PhysicalDeviceSamplerYcbcrConversionFeatures::default()
                .sampler_ycbcr_conversion(true);
            let queues = [queue];
            let info = vk::DeviceCreateInfo::default()
                .queue_create_infos(&queues)
                .enabled_extension_names(&names)
                .push_next(&mut conversion);
            // SAFETY: Valid instance/device and all create parameters are live.
            unsafe { instance.handle.create_device(self.physical, &info, None) }
        };

        let priority = if high_priority {
            self.priority_extension
                .and_then(|extension| create(Some(extension)).ok())
        } else {
            None
        };
        let high_priority = priority.is_some();
        let device = match priority {
            Some(device) => device,
            None => create(None)?,
        };
        let queue = unsafe { device.get_device_queue(self.queue_family, 0) };
        let memory = unsafe {
            instance
                .handle
                .get_physical_device_memory_properties(self.physical)
        };
        let memory_fd = self
            .extensions
            .contains(&ash::khr::external_memory_fd::NAME)
            .then(|| ash::khr::external_memory_fd::Device::new(&instance.handle, &device));
        let modifier = self
            .extensions
            .contains(&ash::ext::image_drm_format_modifier::NAME)
            .then(|| ash::ext::image_drm_format_modifier::Device::new(&instance.handle, &device));
        Ok(Arc::new(Context {
            instance,
            device,
            physical: self.physical,
            memory,
            queue,
            queue_lock: Mutex::new(()),
            queue_family: self.queue_family,
            name: self.name,
            device_uuid: self.id.device_uuid,
            driver_uuid: self.id.driver_uuid,
            high_priority,
            memory_fd,
            modifier,
        }))
    }
}

fn append_extension<'a>(names: &mut Vec<&'a CStr>, extension: &'a CStr) {
    if !names.contains(&extension) {
        names.push(extension);
    }
}

fn pointers(names: &[&CStr]) -> Vec<*const c_char> {
    names.iter().map(|name| name.as_ptr()).collect()
}
