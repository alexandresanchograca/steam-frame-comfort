#![cfg(target_os = "linux")]
// Raw Vulkan loader callbacks. The loader supplies live handles and arrays
// for each command and the layer forwards the same arguments to the next layer.
#![allow(unsafe_op_in_unsafe_fn)]

use ash::vk::{self, Handle};
use std::{
    collections::HashMap,
    ffi::{CStr, c_char, c_void},
    mem,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{Arc, Mutex, MutexGuard, OnceLock},
};

#[repr(C)]
pub struct Negotiation {
    kind: u32,
    next: *mut c_void,
    version: u32,
    instance: vk::PFN_vkGetInstanceProcAddr,
    device: vk::PFN_vkGetDeviceProcAddr,
    physical: Option<vk::PFN_vkGetInstanceProcAddr>,
}

#[repr(C)]
struct LayerInfo<T> {
    kind: vk::StructureType,
    next: *const c_void,
    function: u32,
    link: *mut T,
}

#[repr(C)]
struct InstanceLink {
    next: *mut Self,
    gipa: vk::PFN_vkGetInstanceProcAddr,
    physical: Option<vk::PFN_vkGetInstanceProcAddr>,
}

#[repr(C)]
struct DeviceLink {
    next: *mut Self,
    gipa: vk::PFN_vkGetInstanceProcAddr,
    gdpa: vk::PFN_vkGetDeviceProcAddr,
}

unsafe fn layer_link<T>(
    mut node: *const c_void,
    kind: vk::StructureType,
) -> Option<*mut LayerInfo<T>> {
    while let Some(header) = node.cast::<vk::BaseInStructure>().as_ref() {
        if header.s_type == kind {
            let info = node.cast_mut().cast::<LayerInfo<T>>();
            if (*info).function == 0 && !(*info).link.is_null() {
                return Some(info);
            }
        }
        node = header.p_next.cast();
    }
    None
}

unsafe fn next_info<T: vk::TaggedStructure>(mut node: *const c_void) -> Option<*const T> {
    while let Some(header) = node.cast::<vk::BaseInStructure>().as_ref() {
        if header.s_type == T::STRUCTURE_TYPE {
            return Some(node.cast());
        }
        node = header.p_next.cast();
    }
    None
}

unsafe fn dispatch_key(handle: impl Handle) -> usize {
    let address = handle.as_raw() as usize as *const usize;
    if address.is_null() { 0 } else { *address }
}

unsafe fn cast<T: Copy>(pfn: vk::PFN_vkVoidFunction) -> Option<T> {
    pfn.map(|function| {
        debug_assert_eq!(mem::size_of::<T>(), mem::size_of_val(&function));
        mem::transmute_copy(&function)
    })
}

macro_rules! entry {
    ($function:path, $ty:ty) => {
        Some(mem::transmute::<$ty, unsafe extern "system" fn()>(
            $function as $ty,
        ))
    };
}

mod observe;
mod server;

#[derive(Clone, Copy)]
struct Instance {
    handle: vk::Instance,
    next: vk::PFN_vkGetInstanceProcAddr,
}

struct Device {
    handle: vk::Device,
    next: vk::PFN_vkGetDeviceProcAddr,
    active: bool,
    // Cache the next entry points for hooks, which may run for every draw.
    hooks: HashMap<&'static CStr, vk::PFN_vkVoidFunction>,
}

impl Device {
    unsafe fn function<T: Copy>(&self, name: &CStr) -> Option<T> {
        cast(
            self.hooks
                .get(name)
                .copied()
                .unwrap_or_else(|| (self.next)(self.handle, name.as_ptr())),
        )
    }
}

#[derive(Default)]
struct Dispatch {
    instances: HashMap<usize, Instance>,
    devices: HashMap<usize, Arc<Device>>,
}

static DISPATCH: OnceLock<Mutex<Dispatch>> = OnceLock::new();

fn dispatch() -> MutexGuard<'static, Dispatch> {
    DISPATCH
        .get_or_init(Mutex::default)
        .lock()
        .unwrap_or_else(|error| error.into_inner())
}

unsafe fn device(handle: impl Handle) -> Option<Arc<Device>> {
    dispatch().devices.get(&dispatch_key(handle)).cloned()
}

unsafe fn instance(handle: impl Handle) -> Option<Instance> {
    dispatch().instances.get(&dispatch_key(handle)).copied()
}

fn observe_safely(observe: impl FnOnce()) {
    let _ = catch_unwind(AssertUnwindSafe(observe));
}

/// # Safety
/// The loader must supply a valid writable negotiation structure.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkNegotiateLoaderLayerInterfaceVersion(
    info: *mut Negotiation,
) -> vk::Result {
    let Some(info) = info.as_mut() else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    if info.kind != 1 || info.version < 2 {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    info.version = 2;
    info.instance = get_instance_proc;
    info.device = get_device_proc;
    info.physical = None;
    vk::Result::SUCCESS
}

/// # Safety
/// The handle must be null or a live loader instance, and name a valid C string.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkGetInstanceProcAddr(
    handle: vk::Instance,
    name: *const c_char,
) -> vk::PFN_vkVoidFunction {
    get_instance_proc(handle, name)
}

unsafe extern "system" fn get_instance_proc(
    handle: vk::Instance,
    name: *const c_char,
) -> vk::PFN_vkVoidFunction {
    if name.is_null() {
        return None;
    }
    let name = CStr::from_ptr(name);
    match name.to_bytes() {
        b"vkGetInstanceProcAddr" => entry!(get_instance_proc, vk::PFN_vkGetInstanceProcAddr),
        b"vkGetDeviceProcAddr" => entry!(get_device_proc, vk::PFN_vkGetDeviceProcAddr),
        b"vkCreateInstance" => entry!(create_instance, vk::PFN_vkCreateInstance),
        b"vkDestroyInstance" => entry!(destroy_instance, vk::PFN_vkDestroyInstance),
        b"vkCreateDevice" => entry!(create_device, vk::PFN_vkCreateDevice),
        _ => {
            let state = instance(handle)?;
            let next = (state.next)(handle, name.as_ptr())?;
            if name == c"vkDestroyDevice" {
                return entry!(destroy_device, vk::PFN_vkDestroyDevice);
            }
            if server::socket_path().is_some()
                && let Some(hook) = observe::hook(name)
            {
                return Some(hook);
            }
            Some(next)
        }
    }
}

/// # Safety
/// The handle must be a live loader device, and name a valid C string.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkGetDeviceProcAddr(
    handle: vk::Device,
    name: *const c_char,
) -> vk::PFN_vkVoidFunction {
    get_device_proc(handle, name)
}

unsafe extern "system" fn get_device_proc(
    handle: vk::Device,
    name: *const c_char,
) -> vk::PFN_vkVoidFunction {
    if name.is_null() {
        return None;
    }
    let state = device(handle)?;
    let next = (state.next)(handle, name)?;
    let name = CStr::from_ptr(name);
    match name.to_bytes() {
        b"vkGetDeviceProcAddr" => entry!(get_device_proc, vk::PFN_vkGetDeviceProcAddr),
        b"vkDestroyDevice" => entry!(destroy_device, vk::PFN_vkDestroyDevice),
        _ if state.active => observe::hook(name).or(Some(next)),
        _ => Some(next),
    }
}

unsafe extern "system" fn create_instance(
    info: *const vk::InstanceCreateInfo<'_>,
    allocator: *const vk::AllocationCallbacks<'_>,
    output: *mut vk::Instance,
) -> vk::Result {
    let Some(link) = info.as_ref().and_then(|info| {
        layer_link::<InstanceLink>(info.p_next, vk::StructureType::LOADER_INSTANCE_CREATE_INFO)
    }) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let chain = &*(*link).link;
    let next = chain.gipa;
    (*link).link = chain.next;
    let Some(create) =
        cast::<vk::PFN_vkCreateInstance>(next(vk::Instance::null(), c"vkCreateInstance".as_ptr()))
    else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let result = create(info, allocator, output);
    if result == vk::Result::SUCCESS {
        let handle = *output;
        dispatch()
            .instances
            .insert(dispatch_key(handle), Instance { handle, next });
    }
    result
}

unsafe extern "system" fn destroy_instance(
    handle: vk::Instance,
    allocator: *const vk::AllocationCallbacks<'_>,
) {
    let state = dispatch().instances.remove(&dispatch_key(handle));
    if let Some(state) = state
        && let Some(destroy) =
            cast::<vk::PFN_vkDestroyInstance>((state.next)(handle, c"vkDestroyInstance".as_ptr()))
    {
        destroy(handle, allocator);
    }
}

unsafe extern "system" fn create_device(
    physical: vk::PhysicalDevice,
    info: *const vk::DeviceCreateInfo<'_>,
    allocator: *const vk::AllocationCallbacks<'_>,
    output: *mut vk::Device,
) -> vk::Result {
    let Some(instance) = instance(physical) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let Some(link) = info.as_ref().and_then(|info| {
        layer_link::<DeviceLink>(info.p_next, vk::StructureType::LOADER_DEVICE_CREATE_INFO)
    }) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let chain = &*(*link).link;
    let (gipa, next) = (chain.gipa, chain.gdpa);
    (*link).link = chain.next;
    let Some(create) =
        cast::<vk::PFN_vkCreateDevice>(gipa(instance.handle, c"vkCreateDevice".as_ptr()))
    else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let result = create(physical, info, allocator, output);
    if result != vk::Result::SUCCESS {
        return result;
    }
    let handle = *output;
    let path = server::socket_path();
    let active = path.is_some();
    let mut hooks = HashMap::new();
    if active {
        for &name in observe::HOOKS {
            hooks.insert(name, next(handle, name.as_ptr()));
        }
    }
    dispatch().devices.insert(
        dispatch_key(handle),
        Arc::new(Device {
            handle,
            next,
            active,
            hooks,
        }),
    );
    if let Some(path) = path {
        observe_safely(|| {
            let mut id = vk::PhysicalDeviceIDProperties::default();
            let mut props = vk::PhysicalDeviceProperties2::default().push_next(&mut id);
            let query =
                (instance.next)(instance.handle, c"vkGetPhysicalDeviceProperties2".as_ptr())
                    .or_else(|| {
                        (instance.next)(
                            instance.handle,
                            c"vkGetPhysicalDeviceProperties2KHR".as_ptr(),
                        )
                    });
            if let Some(query) = cast::<vk::PFN_vkGetPhysicalDeviceProperties2>(query) {
                query(physical, &mut props);
                server::start(path, id.device_uuid, id.driver_uuid);
            }
        });
    }
    result
}

unsafe extern "system" fn destroy_device(
    handle: vk::Device,
    allocator: *const vk::AllocationCallbacks<'_>,
) {
    let (state, last_active) = {
        let mut dispatch = dispatch();
        let state = dispatch.devices.remove(&dispatch_key(handle));
        (state, !dispatch.devices.values().any(|state| state.active))
    };
    if let Some(state) = state {
        if state.active {
            observe_safely(|| observe::retire_device(handle));
            if last_active {
                server::stop();
            }
        }
        if let Some(destroy) = state.function::<vk::PFN_vkDestroyDevice>(c"vkDestroyDevice") {
            destroy(handle, allocator);
        }
    }
}

unsafe fn slice<'a, T>(ptr: *const T, count: u32, limit: usize) -> &'a [T] {
    if ptr.is_null() || count == 0 {
        &[]
    } else {
        std::slice::from_raw_parts(ptr, (count as usize).min(limit))
    }
}
