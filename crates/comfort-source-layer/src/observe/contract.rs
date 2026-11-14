//! Fake-next contract for the observer, ported from the C++ baseline's
//! `source_layer_contract.cpp`. No driver or GPU is used. Checks forwarding,
//! descriptor duplication and passing, publication, frame reporting, removal,
//! failure paths and the inert process guard.
//!
//! Differences from the baseline: an ineligible image is not tracked at all,
//! and destroying the last device stops the socket thread, so a client sees
//! the connection close where the baseline sent one removal per image.
use super::*;
use comfort_protocol::{ImageRemove, MAX_PACKET_SIZE, Packet, decode};
use rustix::{
    event::{PollFd, PollFlags, Timespec, poll},
    fs::{MemfdFlags, fstat, memfd_create},
    net::{
        AddressFamily, RecvAncillaryBuffer, RecvAncillaryMessage, RecvFlags, SocketAddrUnix,
        SocketFlags, SocketType, connect, recvmsg, socket_with,
    },
};
use std::{
    ffi::c_int,
    io::IoSliceMut,
    mem::MaybeUninit,
    os::fd::{AsRawFd, IntoRawFd},
    path::Path,
    ptr::{null, null_mut},
    sync::atomic::{AtomicI32, AtomicU32, AtomicU64, AtomicUsize, Ordering::Relaxed},
    time::{Duration, Instant},
};

const CHILD: &str = "VRW_COMFORT_CONTRACT_CHILD";
const P010: vk::Format = vk::Format::G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16;
const DMA_BUF: vk::ExternalMemoryHandleTypeFlags = vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT;
const OPAQUE_FD: vk::ExternalMemoryHandleTypeFlags = vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD;

// Dispatchable handles point at their dispatch key. Every fake object shares one.
const KEY: usize = 0xC0FFEE;
static OBJECTS: [usize; 5] = [KEY; 5];

fn object<T: Handle>(index: usize) -> T {
    T::from_raw(&raw const OBJECTS[index] as u64)
}
fn physical() -> vk::PhysicalDevice {
    object(0)
}
fn fake_device() -> vk::Device {
    object(1)
}
fn queue() -> vk::Queue {
    object(2)
}
fn command() -> vk::CommandBuffer {
    object(3)
}
fn secondary() -> vk::CommandBuffer {
    object(4)
}

static NEXT_HANDLE: AtomicU64 = AtomicU64::new(1000);
static ALLOCATE_RESULT: AtomicI32 = AtomicI32::new(0);
static BIND_RESULT: AtomicI32 = AtomicI32::new(0);
static SUBMIT_RESULT: AtomicI32 = AtomicI32::new(0);
static EXPORTED_FD: AtomicI32 = AtomicI32::new(-1);
static LAST_ALLOCATE_INFO: AtomicUsize = AtomicUsize::new(0);
static EXPECTED_BARRIERS: AtomicUsize = AtomicUsize::new(0);
static EXPECTED_SUBMITS: AtomicUsize = AtomicUsize::new(0);
static EXPECTED_WRITES: AtomicUsize = AtomicUsize::new(0);
static EXPECTED_PASS: AtomicUsize = AtomicUsize::new(0);
static EXPECTED_VIEWPORT: AtomicUsize = AtomicUsize::new(0);
static BARRIER_CALLS: AtomicU32 = AtomicU32::new(0);
static SUBMIT_CALLS: AtomicU32 = AtomicU32::new(0);
static DESTROYED_DEVICES: AtomicU32 = AtomicU32::new(0);
static DESTROYED_IMAGES: AtomicU32 = AtomicU32::new(0);
static FREED_MEMORIES: AtomicU32 = AtomicU32::new(0);
static GET_FD_CALLS: AtomicU32 = AtomicU32::new(0);
static BIND_CALLS: AtomicU32 = AtomicU32::new(0);
static PASS_CALLS: AtomicU32 = AtomicU32::new(0);
static VIEWPORT_CALLS: AtomicU32 = AtomicU32::new(0);
static DRAW_CALLS: AtomicU32 = AtomicU32::new(0);
// The fake driver cannot unwind through the layer, so it records what it saw.
static FAULTS: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());
static SERIAL: Mutex<()> = Mutex::new(());

fn fault(message: &'static str) {
    FAULTS
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .push(message);
}

fn faults() -> Vec<&'static str> {
    FAULTS
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .clone()
}

fn new_handle<T: Handle>() -> T {
    T::from_raw(NEXT_HANDLE.fetch_add(1, Relaxed))
}

fn result_of(value: &AtomicI32) -> vk::Result {
    vk::Result::from_raw(value.load(Relaxed))
}

fn pointer<T>(value: &T) -> *const c_void {
    (value as *const T).cast()
}

// ------------------------------------------------------------ fake next ---
unsafe extern "system" fn next_create_device(
    _: vk::PhysicalDevice,
    _: *const vk::DeviceCreateInfo<'_>,
    _: *const vk::AllocationCallbacks<'_>,
    device: *mut vk::Device,
) -> vk::Result {
    *device = fake_device();
    vk::Result::SUCCESS
}
unsafe extern "system" fn next_properties(
    _: vk::PhysicalDevice,
    properties: *mut vk::PhysicalDeviceProperties2<'_>,
) {
    let ids = (*properties)
        .p_next
        .cast::<vk::PhysicalDeviceIDProperties<'_>>();
    for n in 0..16 {
        (*ids).device_uuid[n] = n as u8 + 1;
        (*ids).driver_uuid[n] = 0xA0 + n as u8;
    }
}
unsafe extern "system" fn next_destroy_device(
    _: vk::Device,
    _: *const vk::AllocationCallbacks<'_>,
) {
    DESTROYED_DEVICES.fetch_add(1, Relaxed);
}
unsafe extern "system" fn next_allocate(
    _: vk::Device,
    info: *const vk::MemoryAllocateInfo<'_>,
    _: *const vk::AllocationCallbacks<'_>,
    memory: *mut vk::DeviceMemory,
) -> vk::Result {
    LAST_ALLOCATE_INFO.store(info as usize, Relaxed);
    let result = result_of(&ALLOCATE_RESULT);
    if result != vk::Result::SUCCESS {
        return result;
    }
    // A real driver takes ownership of an imported descriptor on success.
    if let Some(import) = next_info::<vk::ImportMemoryFdInfoKHR<'_>>((*info).p_next) {
        drop(OwnedFd::from_raw_fd((*import).fd));
    }
    *memory = new_handle();
    vk::Result::SUCCESS
}
unsafe extern "system" fn next_free(
    _: vk::Device,
    _: vk::DeviceMemory,
    _: *const vk::AllocationCallbacks<'_>,
) {
    FREED_MEMORIES.fetch_add(1, Relaxed);
}
unsafe extern "system" fn next_get_memory_fd(
    _: vk::Device,
    info: *const vk::MemoryGetFdInfoKHR<'_>,
    fd: *mut c_int,
) -> vk::Result {
    GET_FD_CALLS.fetch_add(1, Relaxed);
    if (*info).handle_type != DMA_BUF {
        fault("export query must prefer DMA-BUF");
    }
    // A new descriptor, owned by the caller.
    match BorrowedFd::borrow_raw(EXPORTED_FD.load(Relaxed)).try_clone_to_owned() {
        Ok(owned) => {
            *fd = owned.into_raw_fd();
            vk::Result::SUCCESS
        }
        Err(_) => vk::Result::ERROR_TOO_MANY_OBJECTS,
    }
}
unsafe extern "system" fn next_create_image(
    _: vk::Device,
    _: *const vk::ImageCreateInfo<'_>,
    _: *const vk::AllocationCallbacks<'_>,
    image: *mut vk::Image,
) -> vk::Result {
    *image = new_handle();
    vk::Result::SUCCESS
}
unsafe extern "system" fn next_destroy_image(
    _: vk::Device,
    _: vk::Image,
    _: *const vk::AllocationCallbacks<'_>,
) {
    DESTROYED_IMAGES.fetch_add(1, Relaxed);
}
unsafe extern "system" fn next_bind(
    _: vk::Device,
    _: vk::Image,
    _: vk::DeviceMemory,
    _: u64,
) -> vk::Result {
    result_of(&BIND_RESULT)
}
unsafe extern "system" fn next_bind2(
    _: vk::Device,
    _: u32,
    _: *const vk::BindImageMemoryInfo<'_>,
) -> vk::Result {
    result_of(&BIND_RESULT)
}
unsafe extern "system" fn next_conversion(
    _: vk::Device,
    _: *const vk::SamplerYcbcrConversionCreateInfo<'_>,
    _: *const vk::AllocationCallbacks<'_>,
    conversion: *mut vk::SamplerYcbcrConversion,
) -> vk::Result {
    *conversion = new_handle();
    vk::Result::SUCCESS
}
unsafe extern "system" fn next_view(
    _: vk::Device,
    _: *const vk::ImageViewCreateInfo<'_>,
    _: *const vk::AllocationCallbacks<'_>,
    view: *mut vk::ImageView,
) -> vk::Result {
    *view = new_handle();
    vk::Result::SUCCESS
}
unsafe extern "system" fn next_destroy_view(
    _: vk::Device,
    _: vk::ImageView,
    _: *const vk::AllocationCallbacks<'_>,
) {
}
unsafe extern "system" fn next_begin(
    _: vk::CommandBuffer,
    _: *const vk::CommandBufferBeginInfo<'_>,
) -> vk::Result {
    vk::Result::SUCCESS
}
unsafe extern "system" fn next_barrier(
    buffer: vk::CommandBuffer,
    _: vk::PipelineStageFlags,
    _: vk::PipelineStageFlags,
    _: vk::DependencyFlags,
    _: u32,
    _: *const vk::MemoryBarrier<'_>,
    _: u32,
    _: *const vk::BufferMemoryBarrier<'_>,
    count: u32,
    images: *const vk::ImageMemoryBarrier<'_>,
) {
    if buffer != command() && buffer != secondary() {
        fault("Barrier command changed");
    }
    if images as usize != EXPECTED_BARRIERS.load(Relaxed) || count < 1 {
        fault("Barrier arguments changed");
    }
    BARRIER_CALLS.fetch_add(1, Relaxed);
}
unsafe extern "system" fn next_execute(_: vk::CommandBuffer, _: u32, _: *const vk::CommandBuffer) {}
unsafe extern "system" fn next_submit(
    target: vk::Queue,
    _: u32,
    submits: *const vk::SubmitInfo<'_>,
    _: vk::Fence,
) -> vk::Result {
    if target != queue() || submits as usize != EXPECTED_SUBMITS.load(Relaxed) {
        fault("Submit arguments changed");
    }
    SUBMIT_CALLS.fetch_add(1, Relaxed);
    result_of(&SUBMIT_RESULT)
}
unsafe extern "system" fn next_allocate_sets(
    _: vk::Device,
    info: *const vk::DescriptorSetAllocateInfo<'_>,
    sets: *mut vk::DescriptorSet,
) -> vk::Result {
    for n in 0..(*info).descriptor_set_count as usize {
        *sets.add(n) = new_handle();
    }
    vk::Result::SUCCESS
}
unsafe extern "system" fn next_free_sets(
    _: vk::Device,
    _: vk::DescriptorPool,
    _: u32,
    _: *const vk::DescriptorSet,
) -> vk::Result {
    vk::Result::SUCCESS
}
unsafe extern "system" fn next_update_sets(
    _: vk::Device,
    _: u32,
    writes: *const vk::WriteDescriptorSet<'_>,
    _: u32,
    _: *const vk::CopyDescriptorSet<'_>,
) {
    if writes as usize != EXPECTED_WRITES.load(Relaxed) {
        fault("Descriptor writes pointer changed");
    }
}
unsafe extern "system" fn next_create_template(
    _: vk::Device,
    _: *const vk::DescriptorUpdateTemplateCreateInfo<'_>,
    _: *const vk::AllocationCallbacks<'_>,
    template: *mut vk::DescriptorUpdateTemplate,
) -> vk::Result {
    *template = new_handle();
    vk::Result::SUCCESS
}
unsafe extern "system" fn next_update_template(
    _: vk::Device,
    _: vk::DescriptorSet,
    _: vk::DescriptorUpdateTemplate,
    _: *const c_void,
) {
}
unsafe extern "system" fn next_bind_sets(
    _: vk::CommandBuffer,
    _: vk::PipelineBindPoint,
    _: vk::PipelineLayout,
    _: u32,
    _: u32,
    _: *const vk::DescriptorSet,
    _: u32,
    _: *const u32,
) {
    BIND_CALLS.fetch_add(1, Relaxed);
}
unsafe extern "system" fn next_push_set(
    _: vk::CommandBuffer,
    _: vk::PipelineBindPoint,
    _: vk::PipelineLayout,
    _: u32,
    _: u32,
    _: *const vk::WriteDescriptorSet<'_>,
) {
}
unsafe extern "system" fn next_begin_pass(
    _: vk::CommandBuffer,
    info: *const vk::RenderPassBeginInfo<'_>,
    _: vk::SubpassContents,
) {
    if info as usize != EXPECTED_PASS.load(Relaxed) {
        fault("Render pass arguments changed");
    }
    PASS_CALLS.fetch_add(1, Relaxed);
}
unsafe extern "system" fn next_end_pass(_: vk::CommandBuffer) {
    PASS_CALLS.fetch_add(1, Relaxed);
}
unsafe extern "system" fn next_set_viewport(
    _: vk::CommandBuffer,
    first: u32,
    count: u32,
    viewports: *const vk::Viewport,
) {
    if first != 0 || count != 1 || viewports as usize != EXPECTED_VIEWPORT.load(Relaxed) {
        fault("Viewport arguments changed");
    }
    VIEWPORT_CALLS.fetch_add(1, Relaxed);
}
unsafe extern "system" fn next_draw_indexed(
    _: vk::CommandBuffer,
    indices: u32,
    instances: u32,
    _: u32,
    _: i32,
    _: u32,
) {
    if indices != 39273 || instances != 1 {
        fault("Draw arguments changed");
    }
    DRAW_CALLS.fetch_add(1, Relaxed);
}

unsafe extern "system" fn next_gdpa(_: vk::Device, name: *const c_char) -> vk::PFN_vkVoidFunction {
    let name = CStr::from_ptr(name).to_bytes();
    macro_rules! next {
        ($($entry:literal => $function:path : $ty:ident),* $(,)?) => {
            $(if name == $entry { return entry!($function, vk::$ty); })*
        };
    }
    next! {
        b"vkDestroyDevice" => next_destroy_device: PFN_vkDestroyDevice,
        b"vkAllocateMemory" => next_allocate: PFN_vkAllocateMemory,
        b"vkFreeMemory" => next_free: PFN_vkFreeMemory,
        b"vkGetMemoryFdKHR" => next_get_memory_fd: PFN_vkGetMemoryFdKHR,
        b"vkCreateImage" => next_create_image: PFN_vkCreateImage,
        b"vkDestroyImage" => next_destroy_image: PFN_vkDestroyImage,
        b"vkBindImageMemory" => next_bind: PFN_vkBindImageMemory,
        b"vkBindImageMemory2" => next_bind2: PFN_vkBindImageMemory2,
        b"vkCreateSamplerYcbcrConversion" => next_conversion: PFN_vkCreateSamplerYcbcrConversion,
        b"vkCreateImageView" => next_view: PFN_vkCreateImageView,
        b"vkDestroyImageView" => next_destroy_view: PFN_vkDestroyImageView,
        b"vkBeginCommandBuffer" => next_begin: PFN_vkBeginCommandBuffer,
        b"vkCmdPipelineBarrier" => next_barrier: PFN_vkCmdPipelineBarrier,
        b"vkCmdExecuteCommands" => next_execute: PFN_vkCmdExecuteCommands,
        b"vkQueueSubmit" => next_submit: PFN_vkQueueSubmit,
        b"vkAllocateDescriptorSets" => next_allocate_sets: PFN_vkAllocateDescriptorSets,
        b"vkFreeDescriptorSets" => next_free_sets: PFN_vkFreeDescriptorSets,
        b"vkUpdateDescriptorSets" => next_update_sets: PFN_vkUpdateDescriptorSets,
        b"vkCreateDescriptorUpdateTemplate" => next_create_template: PFN_vkCreateDescriptorUpdateTemplate,
        b"vkUpdateDescriptorSetWithTemplate" => next_update_template: PFN_vkUpdateDescriptorSetWithTemplate,
        b"vkCmdBindDescriptorSets" => next_bind_sets: PFN_vkCmdBindDescriptorSets,
        b"vkCmdPushDescriptorSetKHR" => next_push_set: PFN_vkCmdPushDescriptorSetKHR,
        b"vkCmdBeginRenderPass" => next_begin_pass: PFN_vkCmdBeginRenderPass,
        b"vkCmdEndRenderPass" => next_end_pass: PFN_vkCmdEndRenderPass,
        b"vkCmdSetViewport" => next_set_viewport: PFN_vkCmdSetViewport,
        b"vkCmdDrawIndexed" => next_draw_indexed: PFN_vkCmdDrawIndexed,
    }
    None
}

unsafe extern "system" fn next_gipa(
    _: vk::Instance,
    name: *const c_char,
) -> vk::PFN_vkVoidFunction {
    match CStr::from_ptr(name).to_bytes() {
        b"vkCreateDevice" => entry!(next_create_device, vk::PFN_vkCreateDevice),
        b"vkGetPhysicalDeviceProperties2" => {
            entry!(next_properties, vk::PFN_vkGetPhysicalDeviceProperties2)
        }
        _ => None,
    }
}

// -------------------------------------------------------------- helpers ---
/// Device and inode of the file a descriptor refers to.
fn identity(fd: impl AsFd) -> (u64, u64) {
    let status = fstat(fd).expect("fstat");
    (status.st_dev, status.st_ino)
}

/// How many of this process's descriptors refer to the file. Counting one
/// file keeps the check independent of descriptors other tests open.
fn descriptors_of(file: (u64, u64)) -> usize {
    std::fs::read_dir("/proc/self/fd")
        .expect("list descriptors")
        .filter_map(|entry| entry.ok()?.file_name().to_str()?.parse::<i32>().ok())
        .filter(|&fd| {
            // SAFETY: A descriptor closed since the listing fails with EBADF.
            fstat(unsafe { BorrowedFd::borrow_raw(fd) })
                .is_ok_and(|status| (status.st_dev, status.st_ino) == file)
        })
        .count()
}

fn eventually(check: impl Fn() -> bool) -> bool {
    let deadline = Instant::now() + Duration::from_secs(2);
    while !check() {
        if Instant::now() >= deadline {
            return false;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    true
}

fn memory_file() -> OwnedFd {
    memfd_create("vrw-source-contract", MemfdFlags::CLOEXEC).expect("memfd")
}

enum Received {
    Packet(Box<Packet>, Option<OwnedFd>),
    Closed,
}

struct Client(OwnedFd);

impl Client {
    fn connect(path: &Path) -> Self {
        let address = SocketAddrUnix::new(path).expect("socket path");
        for _ in 0..200 {
            let socket = socket_with(
                AddressFamily::UNIX,
                SocketType::SEQPACKET,
                SocketFlags::CLOEXEC,
                None,
            )
            .expect("socket");
            if connect(&socket, &address).is_ok() {
                return Self(socket);
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        panic!("Could not connect to the source socket");
    }

    fn receive(&self) -> Received {
        let timeout = Timespec {
            tv_sec: 2,
            tv_nsec: 0,
        };
        let mut fds = [PollFd::from_borrowed_fd(self.0.as_fd(), PollFlags::IN)];
        let ready = poll(&mut fds, Some(&timeout)).expect("poll");
        assert!(ready > 0, "Source message timed out");
        let mut bytes = [0; MAX_PACKET_SIZE];
        let mut space = [MaybeUninit::uninit(); rustix::cmsg_space!(ScmRights(1))];
        let mut control = RecvAncillaryBuffer::new(&mut space);
        let message = recvmsg(
            &self.0,
            &mut [IoSliceMut::new(&mut bytes)],
            &mut control,
            RecvFlags::CMSG_CLOEXEC,
        )
        .expect("recvmsg");
        if message.bytes == 0 {
            return Received::Closed;
        }
        let fd = control.drain().find_map(|message| match message {
            RecvAncillaryMessage::ScmRights(mut fds) => fds.next(),
            _ => None,
        });
        Received::Packet(
            Box::new(decode(&bytes[..message.bytes]).expect("Invalid source datagram")),
            fd,
        )
    }

    fn packet(&self) -> (Packet, Option<OwnedFd>) {
        match self.receive() {
            Received::Packet(packet, fd) => (*packet, fd),
            Received::Closed => panic!("Source connection closed"),
        }
    }

    fn add(&self) -> (ImageAdd, OwnedFd) {
        match self.packet() {
            (Packet::Add(add), Some(fd)) => (add, fd),
            other => panic!("Expected an image with its descriptor, got {:?}", other.0),
        }
    }

    fn frame(&self) -> Frame {
        match self.packet().0 {
            Packet::Frame(frame) => frame,
            other => panic!("Expected a frame, got {other:?}"),
        }
    }

    fn removal(&self) -> ImageRemove {
        match self.packet().0 {
            Packet::Remove(remove) => remove,
            other => panic!("Expected a removal, got {other:?}"),
        }
    }
}

/// Registers the fake instance and creates the fake device through the layer.
unsafe fn create_fake_device() -> vk::Device {
    crate::dispatch().instances.insert(
        KEY,
        crate::Instance {
            handle: object(0),
            next: next_gipa,
        },
    );
    let mut link = crate::DeviceLink {
        next: null_mut(),
        gipa: next_gipa,
        gdpa: next_gdpa,
    };
    let mut layer = crate::LayerInfo {
        kind: vk::StructureType::LOADER_DEVICE_CREATE_INFO,
        next: null(),
        function: 0,
        link: &raw mut link,
    };
    let info = vk::DeviceCreateInfo {
        p_next: (&raw mut layer).cast(),
        ..Default::default()
    };
    let mut device = vk::Device::null();
    assert_eq!(
        crate::create_device(physical(), &info, null(), &mut device),
        vk::Result::SUCCESS
    );
    assert_eq!(device, fake_device(), "Device creation was not forwarded");
    device
}

unsafe fn device_proc(device: vk::Device, name: &CStr) -> Option<usize> {
    crate::get_device_proc(device, name.as_ptr()).map(|function| function as usize)
}

struct Imported {
    memory: vk::DeviceMemory,
    image: vk::Image,
    file: (u64, u64),
}

/// Creates an image, imports a memory file for it and binds them.
unsafe fn import_image(
    device: vk::Device,
    format: vk::Format,
    drm: bool,
    handle_type: vk::ExternalMemoryHandleTypeFlags,
) -> Imported {
    let fd = memory_file();
    let file = identity(&fd);
    let layouts = [
        vk::SubresourceLayout {
            offset: 0,
            size: 100,
            row_pitch: 4608,
            array_pitch: 0,
            depth_pitch: 0,
        },
        vk::SubresourceLayout {
            offset: 10616832,
            size: 50,
            row_pitch: 4608,
            array_pitch: 0,
            depth_pitch: 0,
        },
    ];
    let modifier = vk::ImageDrmFormatModifierExplicitCreateInfoEXT {
        drm_format_modifier: 0x0500_0000_0000_0001,
        drm_format_modifier_plane_count: 2,
        p_plane_layouts: layouts.as_ptr(),
        ..Default::default()
    };
    let external = vk::ExternalMemoryImageCreateInfo {
        handle_types: handle_type,
        p_next: if drm { pointer(&modifier) } else { null() },
        ..Default::default()
    };
    let info = vk::ImageCreateInfo {
        p_next: pointer(&external),
        image_type: vk::ImageType::TYPE_2D,
        format,
        extent: vk::Extent3D {
            width: 2304,
            height: 2304,
            depth: 1,
        },
        mip_levels: 1,
        array_layers: 1,
        samples: vk::SampleCountFlags::TYPE_1,
        tiling: if drm {
            vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT
        } else {
            vk::ImageTiling::OPTIMAL
        },
        usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC,
        ..Default::default()
    };
    let mut image = vk::Image::null();
    assert_eq!(
        create_image(device, &info, null(), &mut image),
        vk::Result::SUCCESS,
        "Image creation was not forwarded"
    );
    let dedicated = vk::MemoryDedicatedAllocateInfo {
        image,
        ..Default::default()
    };
    let import = vk::ImportMemoryFdInfoKHR {
        handle_type,
        fd: fd.into_raw_fd(),
        p_next: pointer(&dedicated),
        ..Default::default()
    };
    let allocation = vk::MemoryAllocateInfo {
        p_next: pointer(&import),
        allocation_size: 15925248,
        memory_type_index: 3,
        ..Default::default()
    };
    let mut memory = vk::DeviceMemory::null();
    assert_eq!(
        allocate_memory(device, &allocation, null(), &mut memory),
        vk::Result::SUCCESS,
        "Allocation was not forwarded"
    );
    assert_eq!(
        LAST_ALLOCATE_INFO.load(Relaxed),
        &raw const allocation as usize,
        "Allocation info pointer changed"
    );
    assert_eq!(
        bind_image(device, image, memory, 0),
        result_of(&BIND_RESULT),
        "Bind result changed"
    );
    Imported {
        memory,
        image,
        file,
    }
}

unsafe fn acquire(buffer: vk::CommandBuffer, image: vk::Image, source_family: u32) {
    let transition = vk::ImageMemoryBarrier {
        image,
        src_queue_family_index: source_family,
        dst_queue_family_index: 0,
        old_layout: vk::ImageLayout::GENERAL,
        new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        ..Default::default()
    };
    EXPECTED_BARRIERS.store(&raw const transition as usize, Relaxed);
    barrier(
        buffer,
        vk::PipelineStageFlags::TOP_OF_PIPE,
        vk::PipelineStageFlags::FRAGMENT_SHADER,
        vk::DependencyFlags::empty(),
        0,
        null(),
        0,
        null(),
        1,
        &transition,
    );
}

unsafe fn submit_commands(buffers: &[vk::CommandBuffer]) -> vk::Result {
    let info = vk::SubmitInfo {
        command_buffer_count: buffers.len() as u32,
        p_command_buffers: buffers.as_ptr(),
        ..Default::default()
    };
    EXPECTED_SUBMITS.store(&raw const info as usize, Relaxed);
    submit(queue(), 1, &info, vk::Fence::null())
}

unsafe fn begin(buffer: vk::CommandBuffer) {
    assert_eq!(
        begin_command(buffer, null()),
        vk::Result::SUCCESS,
        "Begin not forwarded"
    );
}

unsafe fn bind(buffer: vk::CommandBuffer, sets: &[vk::DescriptorSet]) {
    bind_sets(
        buffer,
        vk::PipelineBindPoint::GRAPHICS,
        vk::PipelineLayout::null(),
        0,
        sets.len() as u32,
        sets.as_ptr(),
        0,
        null(),
    );
}

unsafe fn view_of(device: vk::Device, image: vk::Image, layer: u32) -> vk::ImageView {
    let info = vk::ImageViewCreateInfo {
        image,
        subresource_range: vk::ImageSubresourceRange {
            base_array_layer: layer,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut view = vk::ImageView::null();
    assert_eq!(
        create_view(device, &info, null(), &mut view),
        vk::Result::SUCCESS,
        "View not forwarded"
    );
    view
}

fn sampled(view: vk::ImageView, layout: vk::ImageLayout) -> vk::DescriptorImageInfo {
    vk::DescriptorImageInfo {
        sampler: vk::Sampler::null(),
        image_view: view,
        image_layout: layout,
    }
}

fn label(frame: &Frame, id: u64) -> EyeUse {
    let index = frame.touched_ids[..frame.touched_count as usize]
        .iter()
        .position(|&touched| touched == id)
        .expect("Labelled image missing from the frame report");
    frame.touched_eyes[index]
}

fn sequence() -> u64 {
    source().sequence
}

// ---------------------------------------------------------------- tests ---
#[test]
fn forwards_publishes_reports_and_cleans_up() {
    let _serial = SERIAL.lock().unwrap_or_else(|error| error.into_inner());
    let directory =
        std::env::temp_dir().join(format!("vrw-source-contract-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join("source.sock");
    // SAFETY: Tests that read or spawn with the environment hold SERIAL.
    unsafe {
        std::env::set_var("VRW_COMFORT_SOURCE_SOCKET", &path);
        std::env::set_var("VRW_COMFORT_SOURCE_COMM", ""); // Empty: allowed in any process.
    }
    let shader_read = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL.as_raw() as u32;
    let general = vk::ImageLayout::GENERAL.as_raw() as u32;

    // SAFETY: Every pointer handed to the layer outlives its call, and the
    // fake next layer only reads what a driver would.
    unsafe {
        let device = create_fake_device();
        assert!(
            crate::device(device).unwrap().active,
            "Allowed device did not activate the observer"
        );
        assert_eq!(
            device_proc(device, c"vkCmdPipelineBarrier"),
            Some(barrier as usize),
            "Barrier hook missing"
        );
        assert_eq!(
            device_proc(device, c"vkCmdPipelineBarrier2"),
            None,
            "Unavailable entry point was invented"
        );

        // Imports are published before any client connects.
        let stream = import_image(device, P010, true, DMA_BUF);
        assert_eq!(
            descriptors_of(stream.file),
            2,
            "one duplicate for the memory and one for the published image"
        );
        let stream_view = {
            let conversion = vk::SamplerYcbcrConversionCreateInfo {
                format: P010,
                ycbcr_model: vk::SamplerYcbcrModelConversion::YCBCR_709,
                ycbcr_range: vk::SamplerYcbcrRange::ITU_NARROW,
                x_chroma_offset: vk::ChromaLocation::COSITED_EVEN,
                chroma_filter: vk::Filter::LINEAR,
                ..Default::default()
            };
            let mut handle = vk::SamplerYcbcrConversion::null();
            assert_eq!(
                create_conversion(device, &conversion, null(), &mut handle),
                vk::Result::SUCCESS,
                "Conversion was not forwarded"
            );
            let using = vk::SamplerYcbcrConversionInfo {
                conversion: handle,
                ..Default::default()
            };
            let info = vk::ImageViewCreateInfo {
                p_next: pointer(&using),
                image: stream.image,
                ..Default::default()
            };
            let mut view = vk::ImageView::null();
            assert_eq!(
                create_view(device, &info, null(), &mut view),
                vk::Result::SUCCESS,
                "View was not forwarded"
            );
            view
        };
        // Not sampled -> ineligible; never tracked or published.
        {
            let external = vk::ExternalMemoryImageCreateInfo {
                handle_types: DMA_BUF,
                ..Default::default()
            };
            let info = vk::ImageCreateInfo {
                p_next: pointer(&external),
                image_type: vk::ImageType::TYPE_2D,
                format: vk::Format::R8G8B8A8_UNORM,
                extent: vk::Extent3D {
                    width: 64,
                    height: 64,
                    depth: 1,
                },
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                usage: vk::ImageUsageFlags::TRANSFER_DST,
                ..Default::default()
            };
            let mut handle = vk::Image::null();
            assert_eq!(
                create_image(device, &info, null(), &mut handle),
                vk::Result::SUCCESS,
                "Image creation was not forwarded"
            );
            assert!(
                !source().images.contains_key(&(device, handle)),
                "Unsampled image became eligible"
            );
        }

        let client = Client::connect(&path);
        let hello = match client.packet().0 {
            Packet::Hello(hello) => hello,
            other => panic!("Expected the greeting, got {other:?}"),
        };
        assert!(
            hello.pid == std::process::id()
                && hello.device_uuid[0] == 1
                && hello.driver_uuid[15] == 0xAF,
            "Hello identity incorrect"
        );
        let (add, fd) = client.add();
        assert!(
            identity(&fd) == stream.file && add.inode == stream.file.1,
            "Image descriptor was not the imported allocation"
        );
        drop(fd);
        assert!(
            add.format == P010.as_raw() as u32
                && add.width == 2304
                && add.plane_count == 2
                && add.planes[1].offset == 10616832
                && add.planes[1].row_pitch == 4608
                && add.modifier == 0x0500_0000_0000_0001
                && add.allocation_size == 15925248
                && add.memory_type_index == 3
                && add.dedicated == 1
                && add.exported == 0
                && add.handle_type == DMA_BUF.as_raw(),
            "Image metadata incorrect: {add:?}"
        );
        assert!(
            add.ycbcr.valid == 1
                && add.ycbcr.model == vk::SamplerYcbcrModelConversion::YCBCR_709.as_raw() as u32
                && add.ycbcr.range == vk::SamplerYcbcrRange::ITU_NARROW.as_raw() as u32
                && add.ycbcr.x_chroma_offset == vk::ChromaLocation::COSITED_EVEN.as_raw() as u32,
            "Conversion metadata missing"
        );
        let stream_id = add.id;

        // Second, local-game-like opaque image published while connected.
        let opaque = import_image(device, vk::Format::R8G8B8A8_SRGB, false, OPAQUE_FD);
        let (add, fd) = client.add();
        assert!(
            identity(&fd) == opaque.file
                && add.handle_type == OPAQUE_FD.as_raw()
                && add.plane_count == 0,
            "Opaque image not published"
        );
        drop(fd);
        let opaque_id = add.id;

        // Compositor-allocated, exported memory (local apps render into these).
        {
            let exported = memory_file();
            EXPORTED_FD.store(exported.as_raw_fd(), Relaxed);
            let external = vk::ExternalMemoryImageCreateInfo {
                handle_types: DMA_BUF | OPAQUE_FD,
                ..Default::default()
            };
            let info = vk::ImageCreateInfo {
                p_next: pointer(&external),
                image_type: vk::ImageType::TYPE_2D,
                format: vk::Format::R8G8B8A8_SRGB,
                extent: vk::Extent3D {
                    width: 1728,
                    height: 1728,
                    depth: 1,
                },
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
                ..Default::default()
            };
            let mut handle = vk::Image::null();
            assert_eq!(
                create_image(device, &info, null(), &mut handle),
                vk::Result::SUCCESS,
                "Exported image creation was not forwarded"
            );
            let dedicated = vk::MemoryDedicatedAllocateInfo {
                image: handle,
                ..Default::default()
            };
            let export = vk::ExportMemoryAllocateInfo {
                p_next: pointer(&dedicated),
                handle_types: DMA_BUF | OPAQUE_FD,
                ..Default::default()
            };
            let allocation = vk::MemoryAllocateInfo {
                p_next: pointer(&export),
                allocation_size: 12000000,
                memory_type_index: 1,
                ..Default::default()
            };
            let mut memory = vk::DeviceMemory::null();
            assert!(
                allocate_memory(device, &allocation, null(), &mut memory) == vk::Result::SUCCESS
                    && GET_FD_CALLS.load(Relaxed) == 1,
                "Export descriptor query missing"
            );
            assert_eq!(
                bind_image(device, handle, memory, 0),
                vk::Result::SUCCESS,
                "Bind result changed"
            );
            let (add, fd) = client.add();
            assert!(
                identity(&fd) == identity(&exported)
                    && add.exported == 1
                    && add.handle_type == DMA_BUF.as_raw()
                    && add.width == 1728,
                "Exported image not published"
            );
            drop(fd);
            destroy_image(device, handle, null());
            client.removal();
            free_memory(device, memory, null());
            EXPORTED_FD.store(-1, Relaxed);
            assert!(
                eventually(|| descriptors_of(identity(&exported)) == 1),
                "the observer kept a descriptor of freed exported memory"
            );
        }

        // Owned-family barrier and failed submit produce no frame.
        begin(command());
        acquire(command(), stream.image, 0);
        assert_eq!(
            submit_commands(&[command()]),
            vk::Result::SUCCESS,
            "Submit result changed"
        );
        // A transition without an ownership transfer is reported as "touched".
        let touched = client.frame();
        assert!(
            touched.count == 0
                && touched.touched_count == 1
                && touched.touched_ids[0] == stream_id
                && touched.touched_layouts[0] == shader_read
                && touched.sequence == 1,
            "Transition-only use must be reported as touched, not acquired"
        );
        SUBMIT_RESULT.store(vk::Result::ERROR_DEVICE_LOST.as_raw(), Relaxed);
        begin(command());
        acquire(command(), stream.image, vk::QUEUE_FAMILY_EXTERNAL);
        assert_eq!(
            submit_commands(&[command()]),
            vk::Result::ERROR_DEVICE_LOST,
            "Failed submit result changed"
        );
        SUBMIT_RESULT.store(0, Relaxed);
        assert_eq!(sequence(), 1, "A failed submit produced a frame");

        // External and foreign acquires across a primary and an executed secondary.
        begin(command());
        begin(secondary());
        acquire(secondary(), opaque.image, vk::QUEUE_FAMILY_FOREIGN_EXT);
        acquire(command(), stream.image, vk::QUEUE_FAMILY_EXTERNAL);
        execute(command(), 1, &secondary());
        assert_eq!(
            submit_commands(&[command()]),
            vk::Result::SUCCESS,
            "Submit result changed"
        );
        let frame = client.frame();
        assert!(
            frame.count == 2
                && frame.ids[0] == stream_id
                && frame.ids[1] == opaque_id
                && frame.sequence == 2
                && frame.truncated == 0
                && frame.touched_count == 0
                && frame.layouts[0] == general
                && frame.layouts[1] == general,
            "Frame did not report exactly the successful external acquisitions in order"
        );
        assert!(
            BARRIER_CALLS.load(Relaxed) == 4 && SUBMIT_CALLS.load(Relaxed) == 3,
            "Calls were not each forwarded once"
        );

        // A new recording clears old acquisitions.
        begin(command());
        assert_eq!(submit_commands(&[command()]), vk::Result::SUCCESS);
        assert_eq!(sequence(), 2, "A new recording kept old acquisitions");

        // Descriptor use: images sampled through bound sets, templates and push
        // descriptors are reported as touched with the declared layout.
        {
            let pool = vk::DescriptorPool::from_raw(77);
            let allocate = vk::DescriptorSetAllocateInfo {
                descriptor_pool: pool,
                descriptor_set_count: 2,
                ..Default::default()
            };
            let mut sets = [vk::DescriptorSet::null(); 2];
            assert_eq!(
                allocate_sets(device, &allocate, sets.as_mut_ptr()),
                vk::Result::SUCCESS,
                "Set allocation not forwarded"
            );
            let image_info = sampled(stream_view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
            let mut write = vk::WriteDescriptorSet {
                dst_set: sets[0],
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                p_image_info: &image_info,
                ..Default::default()
            };
            EXPECTED_WRITES.store(&raw const write as usize, Relaxed);
            update_sets(device, 1, &write, 0, null());
            begin(command());
            bind(command(), &sets[..1]);
            assert!(
                BIND_CALLS.load(Relaxed) == 1
                    && submit_commands(&[command()]) == vk::Result::SUCCESS,
                "Bind or submit not forwarded"
            );
            let used = client.frame();
            assert!(
                used.count == 0
                    && used.touched_count == 1
                    && used.touched_ids[0] == stream_id
                    && used.touched_layouts[0] == shader_read,
                "Bound descriptor set use not reported"
            );
            // Update template path, on the second set, for the opaque image.
            let opaque_view = view_of(device, opaque.image, 0);
            let entry = vk::DescriptorUpdateTemplateEntry {
                dst_binding: 0,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                offset: 0,
                stride: mem::size_of::<vk::DescriptorImageInfo>(),
            };
            let template_info = vk::DescriptorUpdateTemplateCreateInfo {
                descriptor_update_entry_count: 1,
                p_descriptor_update_entries: &entry,
                ..Default::default()
            };
            let mut template = vk::DescriptorUpdateTemplate::null();
            assert_eq!(
                create_template(device, &template_info, null(), &mut template),
                vk::Result::SUCCESS,
                "Template"
            );
            let template_data = sampled(opaque_view, vk::ImageLayout::GENERAL);
            update_template(device, sets[1], template, pointer(&template_data));
            begin(command());
            bind(command(), &sets[1..]);
            assert_eq!(submit_commands(&[command()]), vk::Result::SUCCESS);
            let used = client.frame();
            assert!(
                used.touched_count == 1
                    && used.touched_ids[0] == opaque_id
                    && used.touched_layouts[0] == general,
                "Template descriptor use not reported"
            );
            // Push descriptor path.
            begin(command());
            push_set_khr(
                command(),
                vk::PipelineBindPoint::GRAPHICS,
                vk::PipelineLayout::null(),
                0,
                1,
                &write,
            );
            assert_eq!(submit_commands(&[command()]), vk::Result::SUCCESS);
            let used = client.frame();
            assert!(
                used.touched_count == 1 && used.touched_ids[0] == stream_id,
                "Push descriptor use not reported"
            );
            // Overwriting with an untracked view, and freeing a set, stop reporting.
            let untracked = sampled(vk::ImageView::from_raw(999999), vk::ImageLayout::GENERAL);
            write.p_image_info = &untracked;
            update_sets(device, 1, &write, 0, null());
            assert_eq!(
                free_sets(device, pool, 1, &sets[1]),
                vk::Result::SUCCESS,
                "Free not forwarded"
            );
            let before = sequence();
            begin(command());
            bind(command(), &sets);
            assert_eq!(submit_commands(&[command()]), vk::Result::SUCCESS);
            assert_eq!(
                sequence(),
                before,
                "Overwritten or freed descriptor sets were still reported"
            );
            // Destroying the opaque view leaves no dangling resolution.
            destroy_view(device, opaque_view, null());
            assert!(
                !source().views.contains_key(&(device, opaque_view)),
                "Destroyed view still tracked"
            );
        }

        // Eye labels: draws into the left and right halves of a 4320-wide
        // render area label the images their bound sets hold with eye, binding
        // and view layer, as the compositor's lens-distortion pass draws a
        // per-eye ring.
        {
            assert!(
                device_proc(device, c"vkCmdDrawIndexed") == Some(draw_indexed as usize)
                    && device_proc(device, c"vkCmdBeginRendering").is_none(),
                "Draw hooks missing or an entry point was invented"
            );
            let left_view = view_of(device, opaque.image, 1);
            let right_view = view_of(device, stream.image, 0);
            let pool = vk::DescriptorPool::from_raw(78);
            let allocate = vk::DescriptorSetAllocateInfo {
                descriptor_pool: pool,
                descriptor_set_count: 2,
                ..Default::default()
            };
            let mut sets = [vk::DescriptorSet::null(); 2];
            assert_eq!(
                allocate_sets(device, &allocate, sets.as_mut_ptr()),
                vk::Result::SUCCESS,
                "Set allocation not forwarded"
            );
            let images = [left_view, right_view]
                .map(|view| sampled(view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL));
            let mut writes = [vk::WriteDescriptorSet::default(); 2];
            for (n, write) in writes.iter_mut().enumerate() {
                write.dst_set = sets[n];
                write.dst_binding = 9 + n as u32;
                write.descriptor_count = 1;
                write.descriptor_type = vk::DescriptorType::COMBINED_IMAGE_SAMPLER;
                write.p_image_info = &images[n];
            }
            EXPECTED_WRITES.store(writes.as_ptr() as usize, Relaxed);
            update_sets(device, 2, writes.as_ptr(), 0, null());

            let record = |left_x: f32, left_width: f32, right_x: f32| {
                begin(command());
                let mut pass = vk::RenderPassBeginInfo::default();
                pass.render_area.extent = vk::Extent2D {
                    width: 4320,
                    height: 2160,
                };
                EXPECTED_PASS.store(&raw const pass as usize, Relaxed);
                begin_pass(command(), &pass, vk::SubpassContents::INLINE);
                for (x, width, set) in [(left_x, left_width, 0), (right_x, 2160.0, 1)] {
                    let half = vk::Viewport {
                        x,
                        y: 2160.0,
                        width,
                        height: -2160.0,
                        min_depth: 0.0,
                        max_depth: 1.0,
                    };
                    EXPECTED_VIEWPORT.store(&raw const half as usize, Relaxed);
                    set_viewport(command(), 0, 1, &half);
                    bind(command(), &sets[set..=set]);
                    draw_indexed(command(), 39273, 1, 0, 0, 0);
                }
                end_pass(command());
                assert_eq!(submit_commands(&[command()]), vk::Result::SUCCESS);
                client.frame()
            };
            let halves = record(0.0, 2160.0, 2160.0);
            let (left, right) = (label(&halves, opaque_id), label(&halves, stream_id));
            assert!(
                left.eyes == 1
                    && left.slot == 9
                    && left.layers[0] == 1
                    && right.eyes == 2
                    && right.slot == 10
                    && right.layers[1] == 0,
                "Left and right half draws were not labelled with eye, binding and layer"
            );
            assert!(
                PASS_CALLS.load(Relaxed) == 2
                    && VIEWPORT_CALLS.load(Relaxed) == 2
                    && DRAW_CALLS.load(Relaxed) == 2,
                "Draw-state calls were not each forwarded once"
            );
            // A full-width draw and an off-centre one label nothing.
            let full = record(0.0, 4320.0, 1000.0);
            assert!(
                label(&full, opaque_id).eyes == 0 && label(&full, stream_id).eyes == 0,
                "A non-half draw produced an eye label"
            );
            // A new recording starts without the previous recording's labels.
            begin(command());
            bind(command(), &sets[..1]);
            assert_eq!(submit_commands(&[command()]), vk::Result::SUCCESS);
            assert_eq!(
                label(&client.frame(), opaque_id).eyes,
                0,
                "Labels outlived their recording"
            );
            destroy_view(device, left_view, null());
            destroy_view(device, right_view, null());
            assert_eq!(
                free_sets(device, pool, 2, sets.as_ptr()),
                vk::Result::SUCCESS,
                "Free not forwarded"
            );
        }

        // Destroy publishes removal and closes our duplicate.
        destroy_image(device, opaque.image, null());
        assert!(
            client.removal().id == opaque_id && DESTROYED_IMAGES.load(Relaxed) == 2,
            "Removal incorrect"
        );
        assert!(
            eventually(|| descriptors_of(opaque.file) == 1),
            "A destroyed image kept its duplicate descriptor"
        );
        free_memory(device, opaque.memory, null());
        assert!(
            FREED_MEMORIES.load(Relaxed) == 2 && descriptors_of(opaque.file) == 0,
            "Freed memory kept its duplicate descriptor"
        );

        // Failed allocation and failed bind leak nothing and publish nothing.
        ALLOCATE_RESULT.store(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY.as_raw(), Relaxed);
        {
            let failing = memory_file();
            let import = vk::ImportMemoryFdInfoKHR {
                handle_type: DMA_BUF,
                fd: failing.as_raw_fd(),
                ..Default::default()
            };
            let allocation = vk::MemoryAllocateInfo {
                p_next: pointer(&import),
                ..Default::default()
            };
            let mut memory = vk::DeviceMemory::null();
            assert_eq!(
                allocate_memory(device, &allocation, null(), &mut memory),
                vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
                "Failure result changed"
            );
            // On failure the application still owns its descriptor.
            assert_eq!(
                descriptors_of(identity(&failing)),
                1,
                "Failed import leaked a duplicate descriptor"
            );
        }
        ALLOCATE_RESULT.store(0, Relaxed);
        BIND_RESULT.store(vk::Result::ERROR_OUT_OF_DEVICE_MEMORY.as_raw(), Relaxed);
        let unbound = import_image(device, vk::Format::R8G8B8A8_UNORM, false, DMA_BUF);
        BIND_RESULT.store(0, Relaxed);
        assert!(
            source()
                .images
                .get(&(device, unbound.image))
                .is_some_and(|image| image.published.is_none()),
            "Failed bind published an image"
        );
        assert_eq!(
            descriptors_of(unbound.file),
            1,
            "an unbound allocation holds only the memory's duplicate"
        );
        assert_eq!(
            faults(),
            Vec::<&str>::new(),
            "the next layer saw changed calls"
        );

        // Device destruction forgets everything and ends the connection.
        crate::destroy_device(device, null());
        assert_eq!(
            DESTROYED_DEVICES.load(Relaxed),
            1,
            "Device destruction was not forwarded"
        );
        loop {
            match client.receive() {
                Received::Closed => break,
                Received::Packet(packet, _) => match *packet {
                    Packet::Remove(_) => {}
                    other => panic!("Unexpected message at teardown: {other:?}"),
                },
            }
        }
        {
            let state = source();
            assert!(
                state.images.is_empty()
                    && state.memory.is_empty()
                    && state.views.is_empty()
                    && state.sets.is_empty()
                    && state.recordings.is_empty()
                    && state.frame.is_none(),
                "Device destruction left tracked state"
            );
        }
        assert!(
            descriptors_of(stream.file) == 0 && descriptors_of(unbound.file) == 0,
            "Device destruction kept duplicate descriptors"
        );
        assert!(!path.exists(), "the socket outlived its last device");
        assert!(
            device_proc(device, c"vkCmdPipelineBarrier").is_none(),
            "a destroyed device still resolved entry points"
        );
    }
    let _ = std::fs::remove_dir(&directory);
}

/// The same binary in a process that is not the configured compositor stays inert.
#[test]
fn stays_inert_outside_the_compositor() {
    let _serial = SERIAL.lock().unwrap_or_else(|error| error.into_inner());
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "observe::contract::inert_child",
            "--ignored",
            "--nocapture",
        ])
        .env("VRW_COMFORT_SOURCE_COMM", "definitely-not-this-process")
        .env(CHILD, "1")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success() && stdout.contains("1 passed"),
        "Inert-process contract failed:\n{stdout}\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
#[ignore = "run in its own process by stays_inert_outside_the_compositor"]
fn inert_child() {
    if std::env::var_os(CHILD).is_none() {
        return;
    }
    // SAFETY: As in the contract test above.
    unsafe {
        let device = create_fake_device();
        assert!(
            !crate::device(device).unwrap().active,
            "Guarded layer became active in another process"
        );
        let imported = import_image(device, P010, true, DMA_BUF);
        assert_eq!(
            descriptors_of(imported.file),
            0,
            "Inert process duplicated a descriptor"
        );
        {
            let state = source();
            assert!(
                state.images.is_empty() && state.memory.is_empty(),
                "Inert process tracked state"
            );
        }
        assert!(
            crate::server::socket_path().is_none(),
            "Inert process would start a server"
        );
        assert_eq!(
            device_proc(device, c"vkCmdPipelineBarrier"),
            Some(next_barrier as usize),
            "Inert process wrapped a command"
        );
        assert_eq!(
            device_proc(device, c"vkDestroyDevice"),
            Some(crate::destroy_device as usize),
            "Inert process lost destroy bookkeeping"
        );
        assert_eq!(faults(), Vec::<&str>::new());
    }
}
