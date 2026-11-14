use super::{Context, EyeTarget, Result, ScenePainter};
use crate::{mailbox::monotonic_ns, vulkan::Instance as VulkanInstance};
use ash::{Entry, vk, vk::Handle as _};
use libloading::Library;
use openxr_sys::{self as xr, Handle as _};
use serde_json::{Value, json};
use std::{
    ffi::{CStr, c_void},
    fs,
    io::{self, Write},
    mem::size_of,
    path::{Path, PathBuf},
    ptr,
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};

const STEREO: xr::ViewConfigurationType = xr::ViewConfigurationType::PRIMARY_STEREO;

#[derive(Clone, Copy)]
pub enum Layout {
    PerEye,
    Array,
    SideBySide,
}

impl Layout {
    fn name(self) -> &'static str {
        match self {
            Self::PerEye => "per-eye",
            Self::Array => "array",
            Self::SideBySide => "sbs",
        }
    }

    fn targets(self, width: u32) -> [SwapchainTarget; 2] {
        [
            SwapchainTarget {
                index: 0,
                x: 0,
                layer: 0,
            },
            match self {
                Self::PerEye => SwapchainTarget {
                    index: 1,
                    x: 0,
                    layer: 0,
                },
                Self::Array => SwapchainTarget {
                    index: 0,
                    x: 0,
                    layer: 1,
                },
                Self::SideBySide => SwapchainTarget {
                    index: 0,
                    x: width,
                    layer: 0,
                },
            },
        ]
    }
}

pub struct XrPatternOptions {
    pub manifest: PathBuf,
    pub layout: Layout,
    pub seconds: Duration,
    pub frame_code: bool,
    pub trace_truth: bool,
}

#[derive(Clone, Copy)]
struct SwapchainTarget {
    index: usize,
    x: u32,
    layer: u32,
}

fn check(result: xr::Result, name: &'static str) -> Result<()> {
    if result.into_raw() < 0 {
        Err(io::Error::other(format!("{name}: {result:?}")).into())
    } else {
        Ok(())
    }
}

fn address(
    get: xr::pfn::GetInstanceProcAddr,
    instance: xr::Instance,
    name: &CStr,
) -> Result<xr::pfn::VoidFunction> {
    let mut address = None;
    // SAFETY: The lookup takes a live instance or the null global instance.
    // The SDK specifies the ABI of its NUL-terminated name and output.
    check(
        unsafe { get(instance, name.as_ptr(), &mut address) },
        "xrGetInstanceProcAddr",
    )?;
    address.ok_or_else(|| {
        io::Error::other(format!("{} is unavailable", name.to_string_lossy())).into()
    })
}

macro_rules! command {
    ($get:expr, $instance:expr, $name:ident) => {{
        let name = CStr::from_bytes_with_nul(concat!("xr", stringify!($name), "\0").as_bytes())?;
        let function = address($get, $instance, name)?;
        // SAFETY: Each selected name is paired with its generated SDK type.
        unsafe { std::mem::transmute::<xr::pfn::VoidFunction, xr::pfn::$name>(function) }
    }};
}

macro_rules! functions {
    ($($field:ident: $name:ident),+ $(,)?) => {
        struct Functions {
            $($field: xr::pfn::$name),+
        }

        impl Functions {
            fn load(runtime: &Runtime) -> Result<Self> {
                Ok(Self {
                    $($field: command!(runtime.get_proc, runtime.handle, $name)),+
                })
            }
        }
    };
}

functions! {
    get_system: GetSystem,
    graphics_requirements: GetVulkanGraphicsRequirements2KHR,
    create_vk_instance: CreateVulkanInstanceKHR,
    graphics_device: GetVulkanGraphicsDevice2KHR,
    create_vk_device: CreateVulkanDeviceKHR,
    create_session: CreateSession,
    destroy_session: DestroySession,
    create_space: CreateReferenceSpace,
    destroy_space: DestroySpace,
    view_configuration: EnumerateViewConfigurationViews,
    swapchain_formats: EnumerateSwapchainFormats,
    create_swapchain: CreateSwapchain,
    destroy_swapchain: DestroySwapchain,
    swapchain_images: EnumerateSwapchainImages,
    poll_event: PollEvent,
    begin_session: BeginSession,
    end_session: EndSession,
    request_exit: RequestExitSession,
    wait_frame: WaitFrame,
    begin_frame: BeginFrame,
    end_frame: EndFrame,
    locate_views: LocateViews,
    acquire_image: AcquireSwapchainImage,
    wait_image: WaitSwapchainImage,
    release_image: ReleaseSwapchainImage,
}

struct Runtime {
    _library: Library,
    get_proc: xr::pfn::GetInstanceProcAddr,
    destroy: xr::pfn::DestroyInstance,
    handle: xr::Instance,
}

impl Runtime {
    fn new(manifest: &Path) -> Result<Self> {
        let manifest = manifest.canonicalize()?;
        let config: Value = serde_json::from_slice(&fs::read(&manifest)?)?;
        let runtime = &config["runtime"];
        let library = runtime["library_path"]
            .as_str()
            .ok_or_else(|| io::Error::other("OpenXR manifest has no runtime library_path"))?;
        let library = manifest
            .parent()
            .expect("absolute manifest path")
            .join(library);
        let negotiation = runtime["functions"]["xrNegotiateLoaderRuntimeInterface"]
            .as_str()
            .unwrap_or("xrNegotiateLoaderRuntimeInterface");

        // SAFETY: This process invokes the runtime chosen by the operator or
        // the current user's OpenXR manifest, matching the baseline fixture.
        let module = unsafe { Library::new(&library)? };
        let negotiate = unsafe {
            *module.get::<xr::pfn::NegotiateLoaderRuntimeInterface>(negotiation.as_bytes())?
        };
        let loader = xr::NegotiateLoaderInfo {
            struct_type: xr::LoaderInterfaceStructs::LOADER_INFO,
            struct_version: xr::LOADER_INFO_STRUCT_VERSION as u32,
            struct_size: size_of::<xr::NegotiateLoaderInfo>(),
            min_interface_version: 1,
            max_interface_version: xr::CURRENT_LOADER_RUNTIME_VERSION as u32,
            min_api_version: xr::Version::new(1, 0, 0),
            max_api_version: xr::Version::new(1, 0x3ff, 0xfff),
        };
        let mut request = xr::NegotiateRuntimeRequest {
            struct_type: xr::LoaderInterfaceStructs::RUNTIME_REQUEST,
            struct_version: xr::RUNTIME_INFO_STRUCT_VERSION as u32,
            struct_size: size_of::<xr::NegotiateRuntimeRequest>(),
            runtime_interface_version: 0,
            runtime_api_version: xr::Version::new(0, 0, 0),
            get_instance_proc_addr: None,
        };
        // SAFETY: Header versions, sizes and initialized structs match the
        // runtime negotiation ABI.
        check(
            unsafe { negotiate(&loader, &mut request) },
            "xrNegotiateLoaderRuntimeInterface",
        )?;
        if !(loader.min_interface_version..=loader.max_interface_version)
            .contains(&request.runtime_interface_version)
            || request.runtime_api_version < loader.min_api_version
            || request.runtime_api_version > loader.max_api_version
        {
            return Err(io::Error::other(
                "OpenXR runtime returned an unsupported interface or API version",
            )
            .into());
        }
        let get_proc = request
            .get_instance_proc_addr
            .ok_or_else(|| io::Error::other("OpenXR runtime returned no xrGetInstanceProcAddr"))?;
        println!(
            "{}",
            json!({"runtime":library,"interface":request.runtime_interface_version})
        );

        let create = command!(get_proc, xr::Instance::NULL, CreateInstance);
        let application_info = xr::ApplicationInfo {
            application_name: identifier(b"vrw-comfort-xr-pattern"),
            application_version: 1,
            engine_name: identifier(b"vrw"),
            engine_version: 0,
            api_version: loader.min_api_version,
        };
        let extensions = [c"XR_KHR_vulkan_enable2".as_ptr()];
        let mut handle = xr::Instance::NULL;
        let info = xr::InstanceCreateInfo {
            ty: xr::InstanceCreateInfo::TYPE,
            next: ptr::null(),
            create_flags: xr::InstanceCreateFlags::EMPTY,
            application_info,
            enabled_api_layer_count: 0,
            enabled_api_layer_names: ptr::null(),
            enabled_extension_count: extensions.len() as u32,
            enabled_extension_names: extensions.as_ptr(),
        };
        // SAFETY: The input and output pointers and strings outlive this call.
        check(unsafe { create(&info, &mut handle) }, "xrCreateInstance")?;
        if handle == xr::Instance::NULL {
            return Err(io::Error::other("OpenXR runtime returned no instance").into());
        }
        let destroy = command!(get_proc, handle, DestroyInstance);
        Ok(Self {
            _library: module,
            get_proc,
            destroy,
            handle,
        })
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        // SAFETY: Sessions and Vulkan resources have been destroyed, and the
        // loaded runtime remains alive until after this call.
        unsafe { (self.destroy)(self.handle) };
    }
}

fn identifier<const N: usize>(name: &[u8]) -> [std::ffi::c_char; N] {
    let mut identifier = [0; N];
    for (to, from) in identifier.iter_mut().zip(name) {
        *to = *from as std::ffi::c_char;
    }
    identifier
}

fn vk_pointer(handle: impl ash::vk::Handle) -> *const c_void {
    handle.as_raw() as usize as *const c_void
}

fn vulkan(runtime: &Runtime, xr: &Functions, system: xr::SystemId) -> Result<Arc<Context>> {
    let mut requirements = xr::GraphicsRequirementsVulkanKHR::out(ptr::null_mut());
    // SAFETY: The selected runtime provides XR_KHR_vulkan_enable2. The live
    // instance and system correspond to this context, and out sets the header.
    check(
        unsafe { (xr.graphics_requirements)(runtime.handle, system, requirements.as_mut_ptr()) },
        "xrGetVulkanGraphicsRequirements2KHR",
    )?;
    let requirements = unsafe { requirements.assume_init() };
    let entry = unsafe { Entry::load()? };
    let loader_version =
        unsafe { entry.try_enumerate_instance_version()? }.unwrap_or(vk::API_VERSION_1_0);
    let to_vk = |v: xr::Version| vk::make_api_version(0, v.major() as u32, v.minor() as u32, 0);
    let api_version = vk::API_VERSION_1_1.max(to_vk(requirements.min_api_version_supported));
    if api_version > loader_version || api_version > to_vk(requirements.max_api_version_supported) {
        return Err(
            io::Error::other("runtime and loader have no compatible Vulkan API version").into(),
        );
    }

    let application = vk::ApplicationInfo::default()
        .application_name(c"vrw-comfort-xr-pattern")
        .api_version(api_version);
    let info = vk::InstanceCreateInfo::default().application_info(&application);
    // SAFETY: On Linux, Vulkan's dispatch handles are pointers, and
    // PFN_vkGetInstanceProcAddr in ash and OpenXR has the same C ABI.
    let gipa = unsafe {
        std::mem::transmute::<vk::PFN_vkGetInstanceProcAddr, xr::platform::VkGetInstanceProcAddr>(
            entry.static_fn().get_instance_proc_addr,
        )
    };
    let mut raw = ptr::null();
    let mut vk_result = 0;
    let info = xr::VulkanInstanceCreateInfoKHR {
        ty: xr::VulkanInstanceCreateInfoKHR::TYPE,
        next: ptr::null(),
        system_id: system,
        create_flags: xr::VulkanInstanceCreateFlagsKHR::EMPTY,
        pfn_get_instance_proc_addr: Some(gipa),
        vulkan_create_info: ptr::from_ref(&info).cast(),
        vulkan_allocator: ptr::null(),
    };
    // SAFETY: The runtime receives Vulkan create info and matching loader.
    check(
        unsafe { (xr.create_vk_instance)(runtime.handle, &info, &mut raw, &mut vk_result) },
        "xrCreateVulkanInstanceKHR",
    )?;
    vk::Result::from_raw(vk_result).result()?;
    if raw.is_null() {
        return Err(io::Error::other("OpenXR runtime returned no Vulkan instance").into());
    }
    let instance = VulkanInstance {
        handle: unsafe {
            ash::Instance::load(
                entry.static_fn(),
                vk::Instance::from_raw(raw as usize as u64),
            )
        },
        _entry: entry,
    };
    let info = xr::VulkanGraphicsDeviceGetInfoKHR {
        ty: xr::VulkanGraphicsDeviceGetInfoKHR::TYPE,
        next: ptr::null(),
        system_id: system,
        vulkan_instance: vk_pointer(instance.handle.handle()),
    };
    let mut physical = ptr::null();
    // SAFETY: The runtime created the Vulkan instance and selects its physical device.
    check(
        unsafe { (xr.graphics_device)(runtime.handle, &info, &mut physical) },
        "xrGetVulkanGraphicsDevice2KHR",
    )?;
    if physical.is_null() {
        return Err(io::Error::other("OpenXR runtime returned no Vulkan physical device").into());
    }
    let physical = vk::PhysicalDevice::from_raw(physical as usize as u64);
    let queue_family = unsafe {
        instance
            .handle
            .get_physical_device_queue_family_properties(physical)
    }
    .iter()
    .position(|family| {
        family.queue_count > 0 && family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
    })
    .ok_or_else(|| io::Error::other("OpenXR Vulkan device has no graphics queue"))?
        as u32;
    let queues = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family)
        .queue_priorities(&[1.0])];
    let info = vk::DeviceCreateInfo::default().queue_create_infos(&queues);
    let info = xr::VulkanDeviceCreateInfoKHR {
        ty: xr::VulkanDeviceCreateInfoKHR::TYPE,
        next: ptr::null(),
        system_id: system,
        create_flags: xr::VulkanDeviceCreateFlagsKHR::EMPTY,
        pfn_get_instance_proc_addr: Some(gipa),
        vulkan_physical_device: vk_pointer(physical),
        vulkan_create_info: ptr::from_ref(&info).cast(),
        vulkan_allocator: ptr::null(),
    };
    // SAFETY: The chosen physical device and queue family belong to this instance.
    check(
        unsafe { (xr.create_vk_device)(runtime.handle, &info, &mut raw, &mut vk_result) },
        "xrCreateVulkanDeviceKHR",
    )?;
    vk::Result::from_raw(vk_result).result()?;
    if raw.is_null() {
        return Err(io::Error::other("OpenXR runtime returned no Vulkan device").into());
    }
    // SAFETY: The function just transferred the Vulkan device to us.
    let device = unsafe {
        ash::Device::load(
            instance.handle.fp_v1_0(),
            vk::Device::from_raw(raw as usize as u64),
        )
    };
    let (memory, queue) = unsafe {
        (
            instance
                .handle
                .get_physical_device_memory_properties(physical),
            device.get_device_queue(queue_family, 0),
        )
    };
    Ok(Arc::new(Context {
        instance,
        device,
        physical,
        memory,
        queue,
        queue_lock: Mutex::new(()),
        queue_family,
        name: "OpenXR scene".into(),
        device_uuid: [0; 16],
        driver_uuid: [0; 16],
        high_priority: false,
        memory_fd: None,
        modifier: None,
    }))
}

struct Session<'a> {
    xr: &'a Functions,
    _runtime: &'a Runtime,
    _context: Arc<Context>,
    handle: xr::Session,
    space: xr::Space,
}

impl<'a> Session<'a> {
    fn new(
        runtime: &'a Runtime,
        xr: &'a Functions,
        system: xr::SystemId,
        context: &Arc<Context>,
    ) -> Result<Self> {
        let (instance, physical, device, _) = context.raw_handles();
        let binding = xr::GraphicsBindingVulkanKHR {
            ty: xr::GraphicsBindingVulkanKHR::TYPE,
            next: ptr::null(),
            instance: vk_pointer(instance),
            physical_device: vk_pointer(physical),
            device: vk_pointer(device),
            queue_family_index: context.queue_family(),
            queue_index: 0,
        };
        let info = xr::SessionCreateInfo {
            ty: xr::SessionCreateInfo::TYPE,
            next: ptr::from_ref(&binding).cast(),
            create_flags: xr::SessionCreateFlags::EMPTY,
            system_id: system,
        };
        let mut handle = xr::Session::NULL;
        // SAFETY: Vulkan handles and the runtime stay alive until session teardown.
        check(
            unsafe { (xr.create_session)(runtime.handle, &info, &mut handle) },
            "xrCreateSession",
        )?;
        let mut session = Self {
            xr,
            _runtime: runtime,
            _context: Arc::clone(context),
            handle,
            space: xr::Space::NULL,
        };
        let info = xr::ReferenceSpaceCreateInfo {
            ty: xr::ReferenceSpaceCreateInfo::TYPE,
            next: ptr::null(),
            reference_space_type: xr::ReferenceSpaceType::LOCAL,
            pose_in_reference_space: xr::Posef::IDENTITY,
        };
        // SAFETY: This uses the live session and an identity local transform.
        check(
            unsafe { (xr.create_space)(handle, &info, &mut session.space) },
            "xrCreateReferenceSpace",
        )?;
        Ok(session)
    }
}

impl Drop for Session<'_> {
    fn drop(&mut self) {
        let _ = self._context.wait_idle();
        // SAFETY: All loaded entry points are alive and swapchains have been
        // destroyed. The session owns its local space.
        unsafe {
            if self.space != xr::Space::NULL {
                (self.xr.destroy_space)(self.space);
            }
            (self.xr.destroy_session)(self.handle);
        }
    }
}

struct Swapchain<'a> {
    session: &'a Session<'a>,
    handle: xr::Swapchain,
    images: Vec<vk::Image>,
}

impl<'a> Swapchain<'a> {
    fn new(
        session: &'a Session<'a>,
        width: u32,
        height: u32,
        layers: u32,
        format: vk::Format,
    ) -> Result<Self> {
        let info = xr::SwapchainCreateInfo {
            ty: xr::SwapchainCreateInfo::TYPE,
            next: ptr::null(),
            create_flags: xr::SwapchainCreateFlags::EMPTY,
            usage_flags: xr::SwapchainUsageFlags::COLOR_ATTACHMENT
                | xr::SwapchainUsageFlags::TRANSFER_DST,
            format: i64::from(format.as_raw()),
            sample_count: 1,
            width,
            height,
            face_count: 1,
            array_size: layers,
            mip_count: 1,
        };
        let xr = session.xr;
        let mut handle = xr::Swapchain::NULL;
        // SAFETY: This swapchain belongs to the live session.
        check(
            unsafe { (xr.create_swapchain)(session.handle, &info, &mut handle) },
            "xrCreateSwapchain",
        )?;
        let mut chain = Self {
            session,
            handle,
            images: Vec::new(),
        };
        let mut count = 0;
        // SAFETY: The count query uses zero capacity and no image array.
        check(
            unsafe { (xr.swapchain_images)(handle, 0, &mut count, ptr::null_mut()) },
            "xrEnumerateSwapchainImages",
        )?;
        let mut images = vec![
            xr::SwapchainImageVulkanKHR {
                ty: xr::SwapchainImageVulkanKHR::TYPE,
                next: ptr::null_mut(),
                image: 0,
            };
            count as usize
        ];
        // SAFETY: The concrete image array matches the session's Vulkan binding.
        check(
            unsafe { (xr.swapchain_images)(handle, count, &mut count, images.as_mut_ptr().cast()) },
            "xrEnumerateSwapchainImages",
        )?;
        images.truncate(count as usize);
        chain.images = images
            .iter()
            .map(|image| vk::Image::from_raw(image.image))
            .collect();
        Ok(chain)
    }

    fn acquire(&self) -> Result<u32> {
        let info = xr::SwapchainImageAcquireInfo {
            ty: xr::SwapchainImageAcquireInfo::TYPE,
            next: ptr::null(),
        };
        let mut index = 0;
        // SAFETY: The swapchain is live and has no other acquired image.
        check(
            unsafe { (self.session.xr.acquire_image)(self.handle, &info, &mut index) },
            "xrAcquireSwapchainImage",
        )?;
        let info = xr::SwapchainImageWaitInfo {
            ty: xr::SwapchainImageWaitInfo::TYPE,
            next: ptr::null(),
            timeout: xr::Duration::from_nanos(1_000_000_000),
        };
        let result = unsafe { (self.session.xr.wait_image)(self.handle, &info) };
        check(result, "xrWaitSwapchainImage")?;
        if result == xr::Result::TIMEOUT_EXPIRED {
            return Err(io::Error::other("OpenXR swapchain image wait timed out").into());
        }
        if (index as usize) >= self.images.len() {
            return Err(io::Error::other("invalid OpenXR swapchain image index").into());
        }
        Ok(index)
    }

    fn release(&self) -> Result<()> {
        let info = xr::SwapchainImageReleaseInfo {
            ty: xr::SwapchainImageReleaseInfo::TYPE,
            next: ptr::null(),
        };
        // SAFETY: The caller has completed the transfer to the acquired image.
        check(
            unsafe { (self.session.xr.release_image)(self.handle, &info) },
            "xrReleaseSwapchainImage",
        )
    }
}

impl Drop for Swapchain<'_> {
    fn drop(&mut self) {
        // SAFETY: This object owns the swapchain. The frame's commands have
        // completed or been dropped before it, and its session remains live.
        unsafe { (self.session.xr.destroy_swapchain)(self.handle) };
    }
}

fn extent(runtime: &Runtime, xr: &Functions, system: xr::SystemId) -> Result<vk::Extent2D> {
    let mut count = 0;
    let mut views = [xr::ViewConfigurationView::out(ptr::null_mut()); 2];
    // SAFETY: These two output structs are initialized with the Vulkan
    // runtime's expected headers and the runtime receives capacity two.
    check(
        unsafe {
            (xr.view_configuration)(
                runtime.handle,
                system,
                STEREO,
                2,
                &mut count,
                views.as_mut_ptr().cast(),
            )
        },
        "xrEnumerateViewConfigurationViews",
    )?;
    if count != 2 {
        return Err(io::Error::other("expected two OpenXR stereo views").into());
    }
    let views = views.map(|view| unsafe { view.assume_init() });
    let extent = vk::Extent2D {
        width: views[0].recommended_image_rect_width,
        height: views[0].recommended_image_rect_height,
    };
    if extent.width > (i32::MAX as u32) / 2 || extent.height > i32::MAX as u32 {
        return Err(io::Error::other("invalid OpenXR eye image extent").into());
    }
    Ok(extent)
}

fn format(session: &Session<'_>) -> Result<vk::Format> {
    let mut count = 0;
    let enumerate = session.xr.swapchain_formats;
    // SAFETY: Zero-capacity query returns only the number of formats.
    check(
        unsafe { enumerate(session.handle, 0, &mut count, ptr::null_mut()) },
        "xrEnumerateSwapchainFormats",
    )?;
    let mut formats = vec![0; count as usize];
    // SAFETY: The runtime receives a writable format array of the reported length.
    check(
        unsafe { enumerate(session.handle, count, &mut count, formats.as_mut_ptr()) },
        "xrEnumerateSwapchainFormats",
    )?;
    formats.truncate(count as usize);
    [
        vk::Format::R8G8B8A8_SRGB,
        vk::Format::B8G8R8A8_SRGB,
        vk::Format::R8G8B8A8_UNORM,
        vk::Format::B8G8R8A8_UNORM,
    ]
    .into_iter()
    .find(|format| formats.contains(&i64::from(format.as_raw())))
    .ok_or_else(|| io::Error::other("no 8-bit RGBA or BGRA OpenXR swapchain format").into())
}

fn name(state: xr::SessionState) -> &'static str {
    use xr::SessionState as State;
    match state {
        State::UNKNOWN => "unknown",
        State::IDLE => "idle",
        State::READY => "ready",
        State::SYNCHRONIZED => "synchronized",
        State::VISIBLE => "visible",
        State::FOCUSED => "focused",
        State::STOPPING => "stopping",
        State::LOSS_PENDING => "loss-pending",
        State::EXITING => "exiting",
        _ => "unknown",
    }
}

#[derive(Default)]
struct Timing {
    running: bool,
    exiting: bool,
    state: xr::SessionState,
    frames: u64,
    rendered: u64,
    focused: u64,
    first_code: u32,
    last_code: u32,
    truth_records: usize,
}

impl Timing {
    fn poll(
        &mut self,
        runtime: &Runtime,
        session: &Session<'_>,
        elapsed: Duration,
    ) -> Result<bool> {
        let xr = session.xr;
        loop {
            let mut event = xr::EventDataBuffer {
                ty: xr::EventDataBuffer::TYPE,
                next: ptr::null(),
                varying: [0; 4000],
            };
            // SAFETY: The initialized buffer can hold any runtime event.
            let result = unsafe { (xr.poll_event)(runtime.handle, &mut event) };
            if result == xr::Result::EVENT_UNAVAILABLE {
                return Ok(true);
            }
            check(result, "xrPollEvent")?;
            if event.ty == xr::EventDataInstanceLossPending::TYPE {
                return Ok(false);
            }
            if event.ty != xr::EventDataSessionStateChanged::TYPE {
                continue;
            }
            // SAFETY: The event type identifies the prefix structure filled by OpenXR.
            let changed =
                unsafe { &*ptr::from_ref(&event).cast::<xr::EventDataSessionStateChanged>() };
            if changed.session != session.handle {
                continue;
            }
            self.state = changed.state;
            println!(
                "{}",
                json!({"t":elapsed.as_secs_f64(),"state":name(self.state)})
            );
            match self.state {
                xr::SessionState::READY => {
                    let info = xr::SessionBeginInfo {
                        ty: xr::SessionBeginInfo::TYPE,
                        next: ptr::null(),
                        primary_view_configuration_type: STEREO,
                    };
                    // SAFETY: READY was delivered for this session.
                    check(
                        unsafe { (xr.begin_session)(session.handle, &info) },
                        "xrBeginSession",
                    )?;
                    self.running = true;
                }
                xr::SessionState::STOPPING => {
                    // SAFETY: STOPPING was delivered for this session.
                    check(unsafe { (xr.end_session)(session.handle) }, "xrEndSession")?;
                    self.running = false;
                    if self.exiting {
                        return Ok(false);
                    }
                }
                xr::SessionState::EXITING | xr::SessionState::LOSS_PENDING => return Ok(false),
                _ => {}
            }
        }
    }
}

struct RenderedFrame {
    views: [xr::CompositionLayerProjectionView; 2],
    image_indices: [u32; 2],
    view_flags: xr::ViewStateFlags,
}

struct Scene<'a> {
    session: &'a Session<'a>,
    chains: Vec<Swapchain<'a>>,
    extent: vk::Extent2D,
    targets: [SwapchainTarget; 2],
    painter: ScenePainter,
}

impl Scene<'_> {
    fn render(&mut self, display_time: xr::Time, code: u32) -> Result<Option<RenderedFrame>> {
        let session = self.session;
        let mut view_state = xr::ViewState::out(ptr::null_mut());
        let mut views = [xr::View::out(ptr::null_mut()); 2];
        let mut count = 0;
        let locate = xr::ViewLocateInfo {
            ty: xr::ViewLocateInfo::TYPE,
            next: ptr::null(),
            view_configuration_type: STEREO,
            display_time,
            space: session.space,
        };
        // SAFETY: The live session and space match, and both view output
        // headers have been initialized.
        check(
            unsafe {
                (session.xr.locate_views)(
                    session.handle,
                    &locate,
                    view_state.as_mut_ptr(),
                    2,
                    &mut count,
                    views.as_mut_ptr().cast(),
                )
            },
            "xrLocateViews",
        )?;
        let flags = unsafe { view_state.assume_init() }.view_state_flags;
        let valid = xr::ViewStateFlags::ORIENTATION_VALID | xr::ViewStateFlags::POSITION_VALID;
        if count != 2 || !flags.contains(valid) {
            return Ok(None);
        }
        let views = views.map(|view| unsafe { view.assume_init() });
        let indices = self
            .chains
            .iter()
            .map(Swapchain::acquire)
            .collect::<Result<Vec<_>>>()?;
        let eyes = self.targets.map(|target| EyeTarget {
            image: self.chains[target.index].images[indices[target.index] as usize],
            x: target.x,
            layer: target.layer,
        });
        // SAFETY: The runtime granted access to each image. The selected
        // rectangles/layers cover the images without overlapping.
        unsafe {
            self.painter.paint(
                eyes,
                code,
                self.painter.code.is_none(),
                None,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            )?;
        }
        for chain in &self.chains {
            chain.release()?;
        }
        let views = std::array::from_fn(|eye| {
            let target = self.targets[eye];
            xr::CompositionLayerProjectionView {
                ty: xr::CompositionLayerProjectionView::TYPE,
                next: ptr::null(),
                pose: views[eye].pose,
                fov: views[eye].fov,
                sub_image: xr::SwapchainSubImage {
                    swapchain: self.chains[target.index].handle,
                    image_rect: xr::Rect2Di {
                        offset: xr::Offset2Di {
                            x: target.x as i32,
                            y: 0,
                        },
                        extent: xr::Extent2Di {
                            width: self.extent.width as i32,
                            height: self.extent.height as i32,
                        },
                    },
                    image_array_index: target.layer,
                },
            }
        });
        Ok(Some(RenderedFrame {
            views,
            image_indices: self.targets.map(|target| indices[target.index]),
            view_flags: flags,
        }))
    }

    fn frame(&mut self, timing: &mut Timing, options: &XrPatternOptions) -> Result<()> {
        let xr = self.session.xr;
        let session = self.session.handle;
        let mut state = xr::FrameState::out(ptr::null_mut());
        let wait = xr::FrameWaitInfo {
            ty: xr::FrameWaitInfo::TYPE,
            next: ptr::null(),
        };
        let begin = xr::FrameBeginInfo {
            ty: xr::FrameBeginInfo::TYPE,
            next: ptr::null(),
        };
        // SAFETY: The session is running, and the preceding frame was ended.
        check(
            unsafe { (xr.wait_frame)(session, &wait, state.as_mut_ptr()) },
            "xrWaitFrame",
        )?;
        let state = unsafe { state.assume_init() };
        check(unsafe { (xr.begin_frame)(session, &begin) }, "xrBeginFrame")?;
        timing.frames += 1;
        timing.focused += u64::from(timing.state == xr::SessionState::FOCUSED);
        let code = timing.frames as u32;
        let frame = if bool::from(state.should_render) {
            self.render(state.predicted_display_time, code)
        } else {
            Ok(None)
        };
        // Pair xrBeginFrame with xrEndFrame even when an image render fails.
        let layer = frame.as_ref().ok().and_then(Option::as_ref).map(|frame| {
            xr::CompositionLayerProjection {
                ty: xr::CompositionLayerProjection::TYPE,
                next: ptr::null(),
                layer_flags: xr::CompositionLayerFlags::EMPTY,
                space: self.session.space,
                view_count: 2,
                views: frame.views.as_ptr(),
            }
        });
        let layers = layer
            .as_ref()
            .map(|layer| [ptr::from_ref(layer).cast::<xr::CompositionLayerBaseHeader>()]);
        let info = xr::FrameEndInfo {
            ty: xr::FrameEndInfo::TYPE,
            next: ptr::null(),
            display_time: state.predicted_display_time,
            environment_blend_mode: xr::EnvironmentBlendMode::OPAQUE,
            layer_count: u32::from(layers.is_some()),
            layers: layers.as_ref().map_or(ptr::null(), |value| value.as_ptr()),
        };
        let record = options.trace_truth
            && layer.is_some()
            && timing.truth_records < 256
            && (timing.rendered < 32 || timing.rendered.is_multiple_of(16));
        let submit_ns = if record { monotonic_ns() } else { 0 };
        // SAFETY: The layer array and projection views are borrowed through this call.
        let result = unsafe { (xr.end_frame)(session, &info) };
        if let Some(frame) = frame? {
            if record {
                let eyes = frame.views.map(|view| view.sub_image);
                let eyes: [_; 2] = std::array::from_fn(|eye| {
                    let rect = eyes[eye].image_rect;
                    let view = frame.views[eye];
                    let p = view.pose.position;
                    let q = view.pose.orientation;
                    let fov = view.fov;
                    json!({"eye":eye,"swapchain":self.targets[eye].index,
                        "image_index":frame.image_indices[eye],"array_layer":view.sub_image.image_array_index,
                        "rect":[rect.offset.x,rect.offset.y,rect.extent.width,rect.extent.height],
                        "pose_position_m":[p.x,p.y,p.z],"pose_orientation_xyzw":[q.x,q.y,q.z,q.w],
                        "fov_rad_left_right_up_down":[fov.angle_left,fov.angle_right,fov.angle_up,fov.angle_down]})
                });
                println!(
                    "{}",
                    json!({"fixture_submit":"openxr","frame":timing.frames,
                    "frame_code":options.frame_code.then_some(code),"monotonic_ns":submit_ns,
                    "xr_display_time":info.display_time.as_nanos(),"result":result.into_raw(),
                    "space":"local","view_state_flags":frame.view_flags.into_raw(),"eyes":eyes})
                );
                timing.truth_records += 1;
            }
            if timing.rendered == 0 {
                timing.first_code = code;
            }
            timing.last_code = code;
            timing.rendered += 1;
        }
        check(result, "xrEndFrame")
    }
}

pub fn run_openxr(options: XrPatternOptions) -> Result<()> {
    let runtime = Runtime::new(&options.manifest)?;
    let xr = Functions::load(&runtime)?;
    let get = xr::SystemGetInfo {
        ty: xr::SystemGetInfo::TYPE,
        next: ptr::null(),
        form_factor: xr::FormFactor::HEAD_MOUNTED_DISPLAY,
    };
    let mut system = xr::SystemId::NULL;
    // SAFETY: The runtime instance and input/output pointers are live.
    check(
        unsafe { (xr.get_system)(runtime.handle, &get, &mut system) },
        "xrGetSystem",
    )?;
    let context = vulkan(&runtime, &xr, system)?;
    let session = Session::new(&runtime, &xr, system, &context)?;
    let extent = extent(&runtime, &xr, system)?;
    let format = format(&session)?;
    let layout = options.layout;
    let [count, width, layers] = match layout {
        Layout::PerEye => [2, extent.width, 1],
        Layout::Array => [1, extent.width, 2],
        Layout::SideBySide => [1, 2 * extent.width, 1],
    };
    let mut chains = Vec::new();
    for index in 0..count {
        let chain = Swapchain::new(&session, width, extent.height, layers, format)?;
        println!(
            "{}",
            json!({"swapchain":index,"layout":layout.name(),"width":width,
            "height":extent.height,"arraySize":layers,"format":format.as_raw(),"images":chain.images.len()})
        );
        chains.push(chain);
    }
    let mut scene = Scene {
        session: &session,
        extent,
        chains,
        targets: layout.targets(extent.width),
        painter: ScenePainter::new(&context, extent, format, options.frame_code)?,
    };
    println!(
        "{}",
        json!({"started":true,"layout":layout.name(),"eye":[extent.width,extent.height],
        "left":"red","right":"blue","seconds":options.seconds.as_secs_f64()})
    );
    io::stdout().flush()?;
    let mut timing = Timing::default();
    let start = Instant::now();
    let mut last_report = start;
    while start.elapsed() < options.seconds + Duration::from_secs(5) {
        let elapsed = start.elapsed();
        if !timing.poll(&runtime, &session, elapsed)? {
            break;
        }
        if elapsed >= options.seconds && !timing.exiting {
            if !timing.running {
                break;
            }
            timing.exiting = true;
            // SAFETY: The session is still running and receives the exit request.
            check(
                unsafe { (xr.request_exit)(session.handle) },
                "xrRequestExitSession",
            )?;
        }
        if timing.running {
            scene.frame(&mut timing, &options)?;
        } else {
            thread::sleep(Duration::from_millis(50));
        }
        if last_report.elapsed() > Duration::from_secs(2) {
            last_report = Instant::now();
            println!(
                "{}",
                json!({"t":elapsed.as_secs_f64(),"state":name(timing.state),"frames":timing.frames,
                "rendered":timing.rendered,"focusedFrames":timing.focused})
            );
            io::stdout().flush()?;
        }
    }
    println!(
        "{}",
        json!({"summary":true,"layout":layout.name(),"frames":timing.frames,
        "rendered":timing.rendered,"focusedFrames":timing.focused,"finalState":name(timing.state)})
    );
    if options.frame_code {
        println!(
            "{}",
            json!({"frameCodeRange":{"count":timing.rendered,"first":timing.first_code,
            "last":timing.last_code,"wrapBits":32}})
        );
    }
    Ok(())
}
