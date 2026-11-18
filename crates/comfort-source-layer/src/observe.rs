use super::*;
use comfort_protocol::{EyeBounds, EyeUse, Frame, Header, ImageAdd, MAX_IMAGES, Plane, Ycbcr};
use std::{
    collections::BTreeMap,
    os::fd::{AsFd, BorrowedFd, FromRawFd, OwnedFd},
};

type MemoryKey = (vk::Device, vk::DeviceMemory);
type ImageKey = (vk::Device, vk::Image);

struct Memory {
    fd: OwnedFd,
    handle: u32,
    index: u32,
    size: u64,
    inode: u64,
    exported: bool,
    dedicated: bool,
}

pub struct Published {
    pub fd: Arc<OwnedFd>,
    pub description: ImageAdd,
}

impl Clone for Published {
    fn clone(&self) -> Self {
        Self {
            fd: Arc::clone(&self.fd),
            description: self.description,
        }
    }
}

struct Image {
    description: ImageAdd,
    published: Option<Published>,
}

#[derive(Clone, Copy)]
struct Seen {
    id: u64,
    layout: u32,
    external: bool,
}

#[derive(Default, Clone)]
struct Recording {
    pool: vk::CommandPool,
    images: Vec<Seen>,
    eye_uses: BTreeMap<u64, EyeUse>,
    bound: Vec<vk::DescriptorSet>,
    graphics: [vk::DescriptorSet; 8],
    pushed: [BTreeMap<(u32, u32), ViewUse>; 8],
    area: Option<vk::Rect2D>,
    viewport: Option<vk::Viewport>,
    pipeline: vk::Pipeline,
    // Dynamic offsets for each graphics set, keyed by binding and element.
    dynamic: [BTreeMap<(u32, u32), u32>; 8],
    pushed_buffers: [BTreeMap<(u32, u32), BufferUse>; 8],
    traced: Vec<trace::TracedDraw>,
    bounds: Vec<BoundsRead>,
}

/// An eye's texture bounds for an image, to read from the compositor's
/// per-submission block when the command buffer is submitted.
#[derive(Clone)]
struct BoundsRead {
    id: u64,
    eye: usize,
    slot: usize,
    block: BoundBuffer,
}

/// A buffer binding resolved to device memory when the draw was recorded.
#[derive(Clone)]
struct BoundBuffer {
    set: u32,
    binding: u32,
    element: u32,
    kind: i32,
    offset: u64,
    dynamic: Option<u32>,
    memory: Option<(vk::DeviceMemory, u64)>,
    range: u64,
}

#[derive(Clone, Copy, Default)]
struct PipelineInfo {
    stages: [u64; 2],
    /// Set 0 bindings of the per-submission block and the scene images.
    scene: Option<(u32, u32)>,
}

#[derive(Clone, Copy)]
struct BufferUse {
    buffer: vk::Buffer,
    offset: u64,
    range: u64,
    kind: vk::DescriptorType,
}

impl BufferUse {
    fn dynamic(&self) -> bool {
        matches!(
            self.kind,
            vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC | vk::DescriptorType::STORAGE_BUFFER_DYNAMIC
        )
    }
}

/// A host mapping of device memory.
#[derive(Clone, Copy)]
struct Mapped {
    pointer: usize,
    offset: u64,
    size: u64,
}

#[derive(Clone, Copy)]
struct ViewUse {
    view: vk::ImageView,
    layout: vk::ImageLayout,
}

#[derive(Default)]
struct State {
    next_id: u64,
    epoch: u64,
    sequence: u64,
    memory: HashMap<MemoryKey, Memory>,
    images: HashMap<ImageKey, Image>,
    recordings: HashMap<(vk::Device, vk::CommandBuffer), Recording>,
    conversions: HashMap<(vk::Device, vk::SamplerYcbcrConversion), Ycbcr>,
    views: HashMap<(vk::Device, vk::ImageView), (vk::Image, u32)>,
    sets: HashMap<(vk::Device, vk::DescriptorSet), Set>,
    templates: HashMap<(vk::Device, vk::DescriptorUpdateTemplate), Template>,
    frame: Option<Frame>,
    // Host-visible bookkeeping used to read the compositor's per-draw blocks.
    sizes: HashMap<(vk::Device, vk::DeviceMemory), u64>,
    maps: HashMap<(vk::Device, vk::DeviceMemory), Mapped>,
    buffers: HashMap<(vk::Device, vk::Buffer), (vk::DeviceMemory, u64)>,
    modules: HashMap<(vk::Device, vk::ShaderModule), spirv::ShaderInfo>,
    pipelines: HashMap<(vk::Device, vk::Pipeline), PipelineInfo>,
}

#[derive(Default, Clone)]
struct Set {
    pool: vk::DescriptorPool,
    bindings: BTreeMap<(u32, u32), ViewUse>,
    buffers: BTreeMap<(u32, u32), BufferUse>,
}

struct Template {
    entries: Vec<vk::DescriptorUpdateTemplateEntry>,
    graphics: bool,
}

static SOURCE: OnceLock<Mutex<State>> = OnceLock::new();
fn source() -> MutexGuard<'static, State> {
    SOURCE
        .get_or_init(Mutex::default)
        .lock()
        .unwrap_or_else(|error| error.into_inner())
}

pub struct Snapshot {
    pub epoch: u64,
    pub frame: Option<Frame>,
    pub images: Option<BTreeMap<u64, Published>>,
}

pub fn snapshot(previous: Option<u64>, sequence: u64) -> Option<Snapshot> {
    let state = source();
    let frame = state.frame.filter(|frame| frame.sequence != sequence);
    let metadata_changed = previous != Some(state.epoch);
    if !metadata_changed && frame.is_none() {
        return None;
    }
    Some(Snapshot {
        epoch: state.epoch,
        frame,
        images: metadata_changed.then(|| {
            state
                .images
                .values()
                .filter_map(|image| {
                    let published = image.published.as_ref()?;
                    Some((published.description.id, published.clone()))
                })
                .collect()
        }),
    })
}

pub fn retire_device(device: vk::Device) {
    let mut state = source();
    state.memory.retain(|(owner, _), _| *owner != device);
    state.images.retain(|(owner, _), _| *owner != device);
    state.recordings.retain(|(owner, _), _| *owner != device);
    state.conversions.retain(|(owner, _), _| *owner != device);
    state.views.retain(|(owner, _), _| *owner != device);
    state.sets.retain(|(owner, _), _| *owner != device);
    state.templates.retain(|(owner, _), _| *owner != device);
    state.sizes.retain(|(owner, _), _| *owner != device);
    state.maps.retain(|(owner, _), _| *owner != device);
    state.buffers.retain(|(owner, _), _| *owner != device);
    state.modules.retain(|(owner, _), _| *owner != device);
    state.pipelines.retain(|(owner, _), _| *owner != device);
    state.frame = None;
    state.metadata_changed();
}

impl State {
    /// The published image set or an image's description changed.
    fn metadata_changed(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
        super::server::notify();
    }

    fn image_id(&self, device: vk::Device, image: vk::Image) -> Option<u64> {
        Some(
            self.images
                .get(&(device, image))?
                .published
                .as_ref()?
                .description
                .id,
        )
    }

    fn used_view(&self, device: vk::Device, view: vk::ImageView) -> Option<(u64, u32)> {
        let &(image, layer) = self.views.get(&(device, view))?;
        Some((self.image_id(device, image)?, layer))
    }

    fn bind(
        &mut self,
        device: vk::Device,
        image: vk::Image,
        memory: vk::DeviceMemory,
        offset: u64,
    ) {
        let Some(memory) = self.memory.get(&(device, memory)) else {
            return;
        };
        let Some(image) = self.images.get_mut(&(device, image)) else {
            return;
        };
        if image.published.is_some() {
            return;
        }
        let Ok(fd) = memory.fd.as_fd().try_clone_to_owned() else {
            return;
        };
        let d = &mut image.description;
        d.handle_type = memory.handle;
        d.memory_type_index = memory.index;
        d.allocation_size = memory.size;
        d.bind_offset = offset;
        d.inode = memory.inode;
        d.dedicated = u32::from(memory.dedicated);
        d.exported = u32::from(memory.exported);
        image.published = Some(Published {
            fd: Arc::new(fd),
            description: *d,
        });
        self.metadata_changed();
    }

    fn barrier(
        &mut self,
        device: vk::Device,
        cmd: vk::CommandBuffer,
        image: vk::Image,
        families: [u32; 2],
        layouts: [vk::ImageLayout; 2],
    ) {
        let Some(id) = self.image_id(device, image) else {
            return;
        };
        let [source, destination] = families;
        let external = matches!(
            source,
            vk::QUEUE_FAMILY_EXTERNAL | vk::QUEUE_FAMILY_FOREIGN_EXT
        ) && destination != source
            && destination != vk::QUEUE_FAMILY_IGNORED;
        let layout = if external { layouts[0] } else { layouts[1] }.as_raw() as u32;
        add_image(
            &mut self.recordings.entry((device, cmd)).or_default().images,
            Seen {
                id,
                layout,
                external,
            },
        );
    }

    fn draw(&mut self, device: vk::Device, command: vk::CommandBuffer) {
        let Some(rec) = self.recordings.get(&(device, command)) else {
            return;
        };
        let (Some(area), Some(viewport)) = (rec.area, rec.viewport) else {
            return;
        };
        let half = f64::from(area.extent.width) / 2.0;
        if half < 1.0
            || !viewport.width.is_finite()
            || (f64::from(viewport.width) - half).abs() > 1.0
        {
            return;
        }
        let x = f64::from(viewport.x) - f64::from(area.offset.x);
        let eye = if x.abs() <= 1.0 {
            0
        } else if (x - half).abs() <= 1.0 {
            1
        } else {
            return;
        };
        let mut labels = Vec::new();
        let mut elements = Vec::new();
        for (index, &set) in rec.graphics.iter().enumerate() {
            let entries = self
                .sets
                .get(&(device, set))
                .into_iter()
                .flat_map(|set| set.bindings.iter())
                .chain(&rec.pushed[index]);
            for (&(binding, element), value) in entries {
                if let Some((id, layer)) = self.used_view(device, value.view) {
                    labels.push((id, binding, layer));
                    elements.push((index, element));
                }
            }
        }
        let reads = self.bounds_reads(device, command, eye, &labels, &elements);
        if let Some(rec) = self.recordings.get_mut(&(device, command)) {
            for read in reads {
                if rec.bounds.len() < 64 {
                    rec.bounds.push(read);
                }
            }
        }
        if !labels.is_empty() && trace::enabled() {
            let traced = self.traced_draw(device, command, eye, &labels, area, viewport);
            if let Some(rec) = self.recordings.get_mut(&(device, command))
                && rec.traced.len() < 16
            {
                rec.traced.push(traced);
            }
        }
        if let Some(rec) = self.recordings.get_mut(&(device, command)) {
            for (id, binding, layer) in labels {
                let mut layers = [0; 2];
                layers[eye] = layer;
                merge_eye(
                    &mut rec.eye_uses,
                    id,
                    EyeUse {
                        eyes: 1 << eye,
                        slot: binding,
                        layers,
                    },
                );
            }
        }
    }

    fn submit(&mut self, device: vk::Device, commands: impl Iterator<Item = vk::CommandBuffer>) {
        let mut images = Vec::new();
        let mut labels = BTreeMap::new();
        let mut traced = Vec::new();
        let mut reads = Vec::new();
        for command in commands {
            let Some(rec) = self.recordings.get(&(device, command)) else {
                continue;
            };
            traced.extend(rec.traced.iter().cloned());
            reads.extend(rec.bounds.iter().cloned());
            for &value in &rec.images {
                add_image(&mut images, value);
            }
            for (&id, &label) in &rec.eye_uses {
                merge_eye(&mut labels, id, label);
            }
            let bound = rec
                .bound
                .iter()
                .filter_map(|&set| self.sets.get(&(device, set)))
                .flat_map(|set| set.bindings.values());
            for value in bound.chain(rec.pushed.iter().flat_map(|set| set.values())) {
                if let Some((id, _)) = self.used_view(device, value.view) {
                    add_image(
                        &mut images,
                        Seen {
                            id,
                            layout: value.layout.as_raw() as u32,
                            external: false,
                        },
                    );
                }
            }
        }
        let mut bounds = BTreeMap::<u64, EyeBounds>::new();
        for read in &reads {
            if let Some(value) = self.read_bounds(device, read) {
                let mut known = EyeBounds::default();
                known.set(read.eye, value);
                bounds.entry(read.id).or_default().merge(&known);
            }
        }
        let now = rustix::time::clock_gettime(rustix::time::ClockId::Monotonic);
        let now_ns = now.tv_sec as u64 * 1_000_000_000 + now.tv_nsec as u64;
        if !traced.is_empty() && trace::due(now_ns) {
            self.trace_submit(device, &traced, now_ns);
        }
        if images.is_empty() {
            return;
        }
        self.sequence = self.sequence.wrapping_add(1);
        let mut frame = Frame {
            header: Header::observer::<Frame>(4),
            sequence: self.sequence,
            monotonic_ns: now_ns,
            ..Default::default()
        };
        for image in images {
            let (count, truncated, ids, layouts, eyes, mapped) = if image.external {
                (
                    &mut frame.count,
                    &mut frame.truncated,
                    &mut frame.ids,
                    &mut frame.layouts,
                    &mut frame.eyes,
                    &mut frame.bounds,
                )
            } else {
                (
                    &mut frame.touched_count,
                    &mut frame.touched_truncated,
                    &mut frame.touched_ids,
                    &mut frame.touched_layouts,
                    &mut frame.touched_eyes,
                    &mut frame.touched_bounds,
                )
            };
            let index = *count as usize;
            if index == MAX_IMAGES {
                *truncated = 1;
                continue;
            }
            ids[index] = image.id;
            layouts[index] = image.layout;
            eyes[index] = labels.get(&image.id).copied().unwrap_or_default();
            mapped[index] = bounds.get(&image.id).copied().unwrap_or_default();
            *count += 1;
        }
        self.frame = Some(frame);
        super::server::notify();
    }
}

fn add_image(images: &mut Vec<Seen>, image: Seen) {
    if let Some(existing) = images.iter_mut().find(|old| old.id == image.id) {
        if image.external && !existing.external {
            *existing = image;
        }
    } else if images.len() < 64 {
        images.push(image);
    }
}

fn merge_eye(uses: &mut BTreeMap<u64, EyeUse>, id: u64, label: EyeUse) {
    if let Some(old) = uses.get_mut(&id) {
        for eye in 0..2 {
            let bit = 1 << eye;
            if label.eyes & bit != 0 && old.eyes & bit == 0 {
                old.layers[eye] = label.layers[eye];
            }
        }
        old.eyes |= label.eyes;
    } else if uses.len() < 64 {
        uses.insert(id, label);
    }
}

impl State {
    /// Resolve a buffer descriptor of a recorded draw to its device memory.
    fn bound_buffer(
        &self,
        device: vk::Device,
        rec: &Recording,
        set: usize,
        (binding, element): (u32, u32),
        value: &BufferUse,
    ) -> BoundBuffer {
        let memory = self.buffers.get(&(device, value.buffer)).copied();
        let range = if value.range == vk::WHOLE_SIZE {
            memory
                .and_then(|(memory, base)| {
                    self.sizes
                        .get(&(device, memory))
                        .map(|size| size.saturating_sub(base + value.offset))
                })
                .unwrap_or(0)
        } else {
            value.range
        };
        BoundBuffer {
            set: set as u32,
            binding,
            element,
            kind: value.kind.as_raw(),
            offset: value.offset,
            dynamic: value
                .dynamic()
                .then(|| rec.dynamic[set].get(&(binding, element)).copied())
                .flatten(),
            memory,
            range,
        }
    }

    /// Host words of a bound buffer from byte `at`, if the memory is mapped.
    fn read_words(
        &self,
        device: vk::Device,
        bound: &BoundBuffer,
        at: u64,
        length: u64,
    ) -> Option<Vec<u32>> {
        let (memory, base) = bound.memory?;
        let mapped = self.maps.get(&(device, memory))?;
        let length = length & !3;
        if length == 0 || at.checked_add(length)? > bound.range {
            return None;
        }
        let start = base
            .checked_add(bound.offset)?
            .checked_add(u64::from(bound.dynamic.unwrap_or(0)))?
            .checked_add(at)?;
        let end = start.checked_add(length)?;
        if start < mapped.offset || end > mapped.offset.checked_add(mapped.size)? {
            return None;
        }
        let mut words = vec![0_u32; (length / 4) as usize];
        // SAFETY: The range lies inside a live host mapping. Unmap and free
        // remove the mapping under the same lock before calling down.
        unsafe {
            std::ptr::copy_nonoverlapping(
                (mapped.pointer + (start - mapped.offset) as usize) as *const u8,
                words.as_mut_ptr().cast::<u8>(),
                length as usize,
            );
        }
        Some(words)
    }

    /// For a recognised scene pipeline, the per-submission block and each
    /// labelled image's submission slot.
    fn bounds_reads(
        &self,
        device: vk::Device,
        command: vk::CommandBuffer,
        eye: usize,
        labels: &[(u64, u32, u32)],
        elements: &[(usize, u32)],
    ) -> Vec<BoundsRead> {
        let Some(rec) = self.recordings.get(&(device, command)) else {
            return Vec::new();
        };
        let Some((block_binding, image_binding)) = self
            .pipelines
            .get(&(device, rec.pipeline))
            .and_then(|pipeline| pipeline.scene)
        else {
            return Vec::new();
        };
        let key = (block_binding, 0);
        let Some(block) = self
            .sets
            .get(&(device, rec.graphics[0]))
            .and_then(|set| set.buffers.get(&key))
            .or_else(|| rec.pushed_buffers[0].get(&key))
            .map(|value| self.bound_buffer(device, rec, 0, key, value))
        else {
            return Vec::new();
        };
        labels
            .iter()
            .zip(elements)
            .filter_map(|(&(id, binding, _), &(set, element))| {
                // The images are three consecutive descriptors from the
                // array's binding, as elements or as following bindings.
                let slot = (set == 0 && binding >= image_binding)
                    .then(|| (binding - image_binding) as usize + element as usize)
                    .filter(|&slot| slot < spirv::SLOTS)?;
                Some(BoundsRead {
                    id,
                    eye,
                    slot,
                    block: block.clone(),
                })
            })
            .collect()
    }

    /// Plausible texture bounds for a queued read.
    fn read_bounds(&self, device: vk::Device, read: &BoundsRead) -> Option<[f32; 4]> {
        let words = self.read_words(
            device,
            &read.block,
            spirv::bounds_offset(read.slot, read.eye),
            64,
        )?;
        let matrix: [f32; 16] = std::array::from_fn(|index| f32::from_bits(words[index]));
        let bounds = spirv::bounds_from_matrix(&matrix);
        let [u0, v0, u1, v1] = bounds;
        let plausible = bounds
            .iter()
            .all(|value| value.is_finite() && value.abs() <= 8.0)
            && [(u1 - u0).abs(), (v1 - v0).abs()]
                .iter()
                .all(|span| (0.05..=8.0).contains(span));
        plausible.then_some(bounds)
    }

    /// The pipeline and the graphics buffer bindings of a labelled draw.
    fn traced_draw(
        &self,
        device: vk::Device,
        command: vk::CommandBuffer,
        eye: usize,
        labels: &[(u64, u32, u32)],
        area: vk::Rect2D,
        viewport: vk::Viewport,
    ) -> trace::TracedDraw {
        let rec = &self.recordings[&(device, command)];
        let mut buffers = Vec::new();
        for (index, &set) in rec.graphics.iter().enumerate().take(4) {
            let entries = self
                .sets
                .get(&(device, set))
                .into_iter()
                .flat_map(|set| set.buffers.iter())
                .chain(&rec.pushed_buffers[index]);
            for (&key, value) in entries {
                buffers.push(self.bound_buffer(device, rec, index, key, value));
            }
        }
        trace::TracedDraw {
            eye,
            labels: labels.to_vec(),
            stages: self
                .pipelines
                .get(&(device, rec.pipeline))
                .map(|pipeline| pipeline.stages)
                .unwrap_or_default(),
            viewport: [viewport.x, viewport.y, viewport.width, viewport.height],
            area: [
                area.offset.x,
                area.offset.y,
                area.extent.width as i32,
                area.extent.height as i32,
            ],
            buffers,
        }
    }
}

unsafe fn observe_memory(
    state: &Device,
    info: &vk::MemoryAllocateInfo<'_>,
    memory: vk::DeviceMemory,
    import: Option<(OwnedFd, u32)>,
) {
    let exported = import.is_none();
    let (fd, handle) = match import {
        Some(import) => import,
        None => {
            let Some(export) = next_info::<vk::ExportMemoryAllocateInfo<'_>>(info.p_next) else {
                return;
            };
            let types = (*export).handle_types;
            let handle = [
                vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT,
                vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD,
            ]
            .into_iter()
            .find(|handle| types.contains(*handle));
            let (Some(handle), Some(get)) = (
                handle,
                state.function::<vk::PFN_vkGetMemoryFdKHR>(c"vkGetMemoryFdKHR"),
            ) else {
                return;
            };
            let get_info = vk::MemoryGetFdInfoKHR::default()
                .memory(memory)
                .handle_type(handle);
            let mut raw = -1;
            if get(state.handle, &get_info, &mut raw) != vk::Result::SUCCESS || raw < 0 {
                return;
            }
            (OwnedFd::from_raw_fd(raw), handle.as_raw())
        }
    };
    let dedicated = next_info::<vk::MemoryDedicatedAllocateInfo<'_>>(info.p_next)
        .is_some_and(|dedicated| (*dedicated).image != vk::Image::null());
    let inode = rustix::fs::fstat(&fd).map_or(0, |metadata| metadata.st_ino);
    let mut global = source();
    if global.memory.len() < 4096 {
        global.memory.insert(
            (state.handle, memory),
            Memory {
                fd,
                handle,
                exported,
                dedicated,
                inode,
                size: info.allocation_size,
                index: info.memory_type_index,
            },
        );
    }
}

unsafe extern "system" fn allocate_memory(
    device_handle: vk::Device,
    info: *const vk::MemoryAllocateInfo<'_>,
    allocator: *const vk::AllocationCallbacks<'_>,
    output: *mut vk::DeviceMemory,
) -> vk::Result {
    let Some(state) = device(device_handle) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let Some(next) = state.function::<vk::PFN_vkAllocateMemory>(c"vkAllocateMemory") else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let import = if state.active {
        info.as_ref()
            .and_then(|info| next_info::<vk::ImportMemoryFdInfoKHR<'_>>(info.p_next))
            .and_then(|import| {
                let import = &*import;
                if import.fd < 0
                    || !matches!(
                        import.handle_type,
                        vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD
                            | vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT
                    )
                {
                    return None;
                }
                Some((
                    BorrowedFd::borrow_raw(import.fd)
                        .try_clone_to_owned()
                        .ok()?,
                    import.handle_type.as_raw(),
                ))
            })
    } else {
        None
    };
    let result = next(device_handle, info, allocator, output);
    if state.active && result == vk::Result::SUCCESS {
        observe_safely(|| {
            if let (Some(info), Some(memory)) = (info.as_ref(), output.as_ref()) {
                {
                    let mut global = source();
                    if global.sizes.len() < 65536 {
                        global
                            .sizes
                            .insert((device_handle, *memory), info.allocation_size);
                    }
                }
                observe_memory(&state, info, *memory, import);
            }
        });
    }
    result
}

unsafe extern "system" fn create_image(
    device_handle: vk::Device,
    info: *const vk::ImageCreateInfo<'_>,
    allocator: *const vk::AllocationCallbacks<'_>,
    output: *mut vk::Image,
) -> vk::Result {
    let Some(state) = device(device_handle) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let Some(next) = state.function::<vk::PFN_vkCreateImage>(c"vkCreateImage") else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let result = next(device_handle, info, allocator, output);
    if state.active && result == vk::Result::SUCCESS {
        observe_safely(|| {
            let (Some(info), Some(&image)) = (info.as_ref(), output.as_ref()) else {
                return;
            };
            let Some(external) = next_info::<vk::ExternalMemoryImageCreateInfo<'_>>(info.p_next)
            else {
                return;
            };
            if (*external).handle_types.is_empty()
                || info.image_type != vk::ImageType::TYPE_2D
                || info.extent.width == 0
                || info.extent.height == 0
                || info.mip_levels == 0
                || !(1..=2).contains(&info.array_layers)
                || info.samples != vk::SampleCountFlags::TYPE_1
                || !info.usage.contains(vk::ImageUsageFlags::SAMPLED)
                || info.flags.intersects(
                    vk::ImageCreateFlags::SPARSE_BINDING
                        | vk::ImageCreateFlags::DISJOINT
                        | vk::ImageCreateFlags::PROTECTED,
                )
            {
                return;
            }
            let mut desc = ImageAdd {
                header: Header::observer::<ImageAdd>(2),
                format: info.format.as_raw() as u32,
                width: info.extent.width,
                height: info.extent.height,
                mip_levels: info.mip_levels,
                array_layers: info.array_layers,
                samples: info.samples.as_raw(),
                tiling: info.tiling.as_raw() as u32,
                usage: info.usage.as_raw(),
                flags: info.flags.as_raw(),
                external_handle_types: (*external).handle_types.as_raw(),
                ..Default::default()
            };
            if let Some(list) = next_info::<vk::ImageFormatListCreateInfo<'_>>(info.p_next) {
                let formats = slice(
                    (*list).p_view_formats,
                    (*list).view_format_count,
                    desc.view_formats.len(),
                );
                desc.view_format_count = formats.len() as u32;
                for (target, format) in desc.view_formats.iter_mut().zip(formats) {
                    *target = format.as_raw() as u32;
                }
            }
            if let Some(layout) =
                next_info::<vk::ImageDrmFormatModifierExplicitCreateInfoEXT<'_>>(info.p_next)
            {
                let layout = &*layout;
                if layout.drm_format_modifier_plane_count as usize > desc.planes.len() {
                    return;
                }
                let planes = slice(
                    layout.p_plane_layouts,
                    layout.drm_format_modifier_plane_count,
                    desc.planes.len(),
                );
                desc.modifier = layout.drm_format_modifier;
                desc.plane_count = planes.len() as u32;
                for (target, plane) in desc.planes.iter_mut().zip(planes) {
                    *target = Plane {
                        offset: plane.offset,
                        size: plane.size,
                        row_pitch: plane.row_pitch,
                        array_pitch: plane.array_pitch,
                        depth_pitch: plane.depth_pitch,
                    };
                }
            }
            if info.tiling == vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT && desc.plane_count == 0 {
                return;
            }
            let mut global = source();
            if global.images.len() < 4096 {
                global.next_id = global.next_id.wrapping_add(1);
                desc.id = global.next_id;
                global.images.insert(
                    (device_handle, image),
                    Image {
                        description: desc,
                        published: None,
                    },
                );
            }
        });
    }
    result
}

unsafe extern "system" fn bind_image(
    device_handle: vk::Device,
    image: vk::Image,
    memory: vk::DeviceMemory,
    offset: u64,
) -> vk::Result {
    let Some(state) = device(device_handle) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let Some(next) = state.function::<vk::PFN_vkBindImageMemory>(c"vkBindImageMemory") else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let result = next(device_handle, image, memory, offset);
    if state.active && result == vk::Result::SUCCESS {
        observe_safely(|| source().bind(device_handle, image, memory, offset));
    }
    result
}

unsafe fn bind_many(
    state: &Device,
    count: u32,
    info: *const vk::BindImageMemoryInfo<'_>,
    result: vk::Result,
) {
    // VkBindMemoryStatus was promoted after the Vulkan header revision used
    // by ash. Its layout and sType are identical to the KHR extension.
    #[repr(C)]
    struct BindStatus {
        kind: vk::StructureType,
        next: *const c_void,
        result: *mut vk::Result,
    }
    unsafe impl vk::TaggedStructure for BindStatus {
        const STRUCTURE_TYPE: vk::StructureType = vk::StructureType::from_raw(1000545002);
    }
    let mut global = source();
    for bind in slice(info, count, 4096) {
        let status =
            next_info::<BindStatus>(bind.p_next).and_then(|status| (*status).result.as_ref());
        if status.copied().unwrap_or(result) == vk::Result::SUCCESS
            && next_info::<vk::BindImagePlaneMemoryInfo<'_>>(bind.p_next).is_none()
        {
            global.bind(state.handle, bind.image, bind.memory, bind.memory_offset);
        }
    }
}

macro_rules! result_hook {
    ($function:ident, $entry:literal, $type:ident, $owner:ident: $owner_type:ty $(, $name:ident: $ty:ty)*;
     |$state:ident, $result:ident| $observe:block) => {
        unsafe extern "system" fn $function($owner: $owner_type $(, $name: $ty)*) -> vk::Result {
            let Some($state) = device($owner) else { return vk::Result::ERROR_INITIALIZATION_FAILED };
            let Some(next) = $state.function::<vk::$type>($entry) else { return vk::Result::ERROR_INITIALIZATION_FAILED };
            let $result = next($owner $(, $name)*);
            if $state.active { observe_safely(|| $observe); }
            $result
        }
    };
}

macro_rules! void_hook {
    ($function:ident, $entry:literal, $type:ident, $owner:ident: $owner_type:ty $(, $name:ident: $ty:ty)*;
     |$state:ident| $observe:block) => {
        unsafe extern "system" fn $function($owner: $owner_type $(, $name: $ty)*) {
            let Some($state) = device($owner) else { return };
            let Some(next) = $state.function::<vk::$type>($entry) else { return };
            next($owner $(, $name)*);
            if $state.active { observe_safely(|| $observe); }
        }
    };
}

unsafe extern "system" fn free_memory(
    dev: vk::Device,
    memory: vk::DeviceMemory,
    allocator: *const vk::AllocationCallbacks<'_>,
) {
    let Some(state) = device(dev) else { return };
    let Some(next) = state.function::<vk::PFN_vkFreeMemory>(c"vkFreeMemory") else {
        return;
    };
    // Freeing unmaps, so forget the mapping before the memory goes away.
    if state.active {
        observe_safely(|| {
            let mut global = source();
            global.memory.remove(&(dev, memory));
            global.maps.remove(&(dev, memory));
            global.sizes.remove(&(dev, memory));
        });
    }
    next(dev, memory, allocator);
}

unsafe extern "system" fn map_memory(
    dev: vk::Device,
    memory: vk::DeviceMemory,
    offset: u64,
    size: u64,
    flags: vk::MemoryMapFlags,
    data: *mut *mut c_void,
) -> vk::Result {
    let Some(state) = device(dev) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let Some(next) = state.function::<vk::PFN_vkMapMemory>(c"vkMapMemory") else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let result = next(dev, memory, offset, size, flags, data);
    if state.active && result == vk::Result::SUCCESS {
        observe_safely(|| record_map(dev, memory, offset, size, data));
    }
    result
}

unsafe fn record_map(
    dev: vk::Device,
    memory: vk::DeviceMemory,
    offset: u64,
    size: u64,
    data: *mut *mut c_void,
) {
    let Some(&pointer) = data.as_ref() else {
        return;
    };
    if pointer.is_null() {
        return;
    }
    let mut global = source();
    let size = if size == vk::WHOLE_SIZE {
        let Some(total) = global.sizes.get(&(dev, memory)) else {
            return;
        };
        total.saturating_sub(offset)
    } else {
        size
    };
    if global.maps.len() < 65536 {
        global.maps.insert(
            (dev, memory),
            Mapped {
                pointer: pointer as usize,
                offset,
                size,
            },
        );
    }
}

unsafe extern "system" fn unmap_memory(dev: vk::Device, memory: vk::DeviceMemory) {
    let Some(state) = device(dev) else { return };
    let Some(next) = state.function::<vk::PFN_vkUnmapMemory>(c"vkUnmapMemory") else {
        return;
    };
    if state.active {
        observe_safely(|| {
            source().maps.remove(&(dev, memory));
        });
    }
    next(dev, memory);
}

unsafe extern "system" fn bind_buffer(
    dev: vk::Device,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    offset: u64,
) -> vk::Result {
    let Some(state) = device(dev) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let Some(next) = state.function::<vk::PFN_vkBindBufferMemory>(c"vkBindBufferMemory") else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    let result = next(dev, buffer, memory, offset);
    if state.active && result == vk::Result::SUCCESS {
        observe_safely(|| {
            let mut global = source();
            if global.buffers.len() < 65536 {
                global.buffers.insert((dev, buffer), (memory, offset));
            }
        });
    }
    result
}

unsafe fn bind_buffers(
    dev: vk::Device,
    count: u32,
    info: *const vk::BindBufferMemoryInfo<'_>,
    result: vk::Result,
) {
    if result != vk::Result::SUCCESS {
        return;
    }
    let mut global = source();
    for bind in slice(info, count, 4096) {
        if global.buffers.len() < 65536 {
            global
                .buffers
                .insert((dev, bind.buffer), (bind.memory, bind.memory_offset));
        }
    }
}

void_hook!(destroy_buffer, c"vkDestroyBuffer", PFN_vkDestroyBuffer, dev: vk::Device, buffer: vk::Buffer, allocator: *const vk::AllocationCallbacks<'_>;
    |_state| { source().buffers.remove(&(dev, buffer)); });
result_hook!(bind_buffer2, c"vkBindBufferMemory2", PFN_vkBindBufferMemory2, dev: vk::Device, count: u32, info: *const vk::BindBufferMemoryInfo<'_>;
    |_state, result| { bind_buffers(dev, count, info, result); });
result_hook!(bind_buffer2_khr, c"vkBindBufferMemory2KHR", PFN_vkBindBufferMemory2, dev: vk::Device, count: u32, info: *const vk::BindBufferMemoryInfo<'_>;
    |_state, result| { bind_buffers(dev, count, info, result); });

unsafe fn shader_info(info: &vk::ShaderModuleCreateInfo<'_>) -> spirv::ShaderInfo {
    if info.p_code.is_null()
        || info.code_size == 0
        || info.code_size > 64 << 20
        || !info.code_size.is_multiple_of(4)
    {
        return spirv::ShaderInfo::default();
    }
    let words = std::slice::from_raw_parts(info.p_code, info.code_size / 4);
    let hash = trace::hash(std::slice::from_raw_parts(
        info.p_code.cast::<u8>(),
        info.code_size,
    ));
    spirv::inspect(words, hash)
}

result_hook!(create_module, c"vkCreateShaderModule", PFN_vkCreateShaderModule, dev: vk::Device,
info: *const vk::ShaderModuleCreateInfo<'_>, allocator: *const vk::AllocationCallbacks<'_>, output: *mut vk::ShaderModule;
|_state, result| {
    if result == vk::Result::SUCCESS && let (Some(info), Some(&module)) = (info.as_ref(), output.as_ref()) {
        let shader = shader_info(info);
        let mut global = source();
        if global.modules.len() < 65536 {
            global.modules.insert((dev, module), shader);
        }
    }
});
void_hook!(destroy_module, c"vkDestroyShaderModule", PFN_vkDestroyShaderModule, dev: vk::Device,
    module: vk::ShaderModule, allocator: *const vk::AllocationCallbacks<'_>;
    |_state| { source().modules.remove(&(dev, module)); });

unsafe fn record_pipelines(
    dev: vk::Device,
    count: u32,
    infos: *const vk::GraphicsPipelineCreateInfo<'_>,
    output: *const vk::Pipeline,
) {
    let mut global = source();
    for (info, &pipeline) in slice(infos, count, 4096)
        .iter()
        .zip(slice(output, count, 4096))
    {
        if pipeline == vk::Pipeline::null() {
            continue;
        }
        let mut shaders = [spirv::ShaderInfo::default(); 2];
        for stage in slice(info.p_stages, info.stage_count, 16) {
            let index = if stage.stage == vk::ShaderStageFlags::VERTEX {
                0
            } else if stage.stage == vk::ShaderStageFlags::FRAGMENT {
                1
            } else {
                continue;
            };
            shaders[index] = if stage.module != vk::ShaderModule::null() {
                global
                    .modules
                    .get(&(dev, stage.module))
                    .copied()
                    .unwrap_or_default()
            } else {
                next_info::<vk::ShaderModuleCreateInfo<'_>>(stage.p_next)
                    .map_or_else(spirv::ShaderInfo::default, |info| shader_info(&*info))
            };
        }
        let [vertex, fragment] = shaders;
        let pipeline_info = PipelineInfo {
            stages: shaders.map(|shader| shader.hash),
            scene: vertex.slot_block.zip(fragment.slot_images),
        };
        if trace::enabled() {
            trace::write_line(&format!(
                "{{\"pipeline\":{},\"vs\":\"{:016x}\",\"fs\":\"{:016x}\",\"scene\":{}}}",
                pipeline.as_raw(),
                vertex.hash,
                fragment.hash,
                pipeline_info.scene.is_some()
            ));
        }
        if global.pipelines.len() < 65536 {
            global.pipelines.insert((dev, pipeline), pipeline_info);
        }
    }
}

result_hook!(create_pipelines, c"vkCreateGraphicsPipelines", PFN_vkCreateGraphicsPipelines, dev: vk::Device,
cache: vk::PipelineCache, count: u32, infos: *const vk::GraphicsPipelineCreateInfo<'_>,
allocator: *const vk::AllocationCallbacks<'_>, output: *mut vk::Pipeline;
|_state, result| {
    if matches!(result, vk::Result::SUCCESS | vk::Result::PIPELINE_COMPILE_REQUIRED) {
        record_pipelines(dev, count, infos, output);
    }
});
void_hook!(destroy_pipeline, c"vkDestroyPipeline", PFN_vkDestroyPipeline, dev: vk::Device,
    pipeline: vk::Pipeline, allocator: *const vk::AllocationCallbacks<'_>;
    |_state| { source().pipelines.remove(&(dev, pipeline)); });
void_hook!(bind_pipeline, c"vkCmdBindPipeline", PFN_vkCmdBindPipeline, cmd: vk::CommandBuffer,
point: vk::PipelineBindPoint, pipeline: vk::Pipeline;
|state| {
    if point == vk::PipelineBindPoint::GRAPHICS {
        source().recordings.entry((state.handle, cmd)).or_default().pipeline = pipeline;
    }
});
void_hook!(destroy_image, c"vkDestroyImage", PFN_vkDestroyImage, dev: vk::Device, image: vk::Image, allocator: *const vk::AllocationCallbacks<'_>;
    |_state| { let mut global = source(); if global.images.remove(&(dev, image)).is_some() { global.metadata_changed(); } });
result_hook!(bind_image2, c"vkBindImageMemory2", PFN_vkBindImageMemory2, dev: vk::Device, count: u32, info: *const vk::BindImageMemoryInfo<'_>;
    |state, result| { bind_many(&state, count, info, result); });
result_hook!(bind_image2_khr, c"vkBindImageMemory2KHR", PFN_vkBindImageMemory2, dev: vk::Device, count: u32, info: *const vk::BindImageMemoryInfo<'_>;
    |state, result| { bind_many(&state, count, info, result); });

result_hook!(allocate_commands, c"vkAllocateCommandBuffers", PFN_vkAllocateCommandBuffers, dev: vk::Device,
info: *const vk::CommandBufferAllocateInfo<'_>, output: *mut vk::CommandBuffer; |_state, result| {
    if result == vk::Result::SUCCESS && let Some(info) = info.as_ref() {
        let mut global = source();
        for &cmd in slice(output, info.command_buffer_count, 65536) {
            global.recordings.insert((dev, cmd), Recording { pool: info.command_pool, ..Default::default() });
        }
    }
});
void_hook!(free_commands, c"vkFreeCommandBuffers", PFN_vkFreeCommandBuffers, dev: vk::Device,
_pool: vk::CommandPool, count: u32, commands: *const vk::CommandBuffer; |_state| {
    let mut global = source();
    for &cmd in slice(commands, count, 65536) { global.recordings.remove(&(dev, cmd)); }
});
void_hook!(destroy_command_pool, c"vkDestroyCommandPool", PFN_vkDestroyCommandPool, dev: vk::Device,
pool: vk::CommandPool, allocator: *const vk::AllocationCallbacks<'_>; |_state| {
    source().recordings.retain(|&(owner, _), record| owner != dev || record.pool != pool);
});
result_hook!(begin_command, c"vkBeginCommandBuffer", PFN_vkBeginCommandBuffer, cmd: vk::CommandBuffer, info: *const vk::CommandBufferBeginInfo<'_>;
|state, result| {
    if result == vk::Result::SUCCESS {
        let mut global = source();
        let rec = global.recordings.entry((state.handle, cmd)).or_default();
        *rec = Recording { pool: rec.pool, ..Default::default() };
    }
});

void_hook!(barrier, c"vkCmdPipelineBarrier", PFN_vkCmdPipelineBarrier, cmd: vk::CommandBuffer,
stages_from: vk::PipelineStageFlags, stages_to: vk::PipelineStageFlags, flags: vk::DependencyFlags,
mem_count: u32, memory: *const vk::MemoryBarrier<'_>, buffer_count: u32, buffers: *const vk::BufferMemoryBarrier<'_>,
count: u32, images: *const vk::ImageMemoryBarrier<'_>; |state| {
    let mut global = source();
    for barrier in slice(images, count, 256) {
        global.barrier(state.handle, cmd, barrier.image,
            [barrier.src_queue_family_index, barrier.dst_queue_family_index], [barrier.old_layout, barrier.new_layout]);
    }
});
unsafe fn dependency(state: &Device, cmd: vk::CommandBuffer, info: *const vk::DependencyInfo<'_>) {
    let Some(info) = info.as_ref() else { return };
    let mut global = source();
    for barrier in slice(
        info.p_image_memory_barriers,
        info.image_memory_barrier_count,
        256,
    ) {
        global.barrier(
            state.handle,
            cmd,
            barrier.image,
            [
                barrier.src_queue_family_index,
                barrier.dst_queue_family_index,
            ],
            [barrier.old_layout, barrier.new_layout],
        );
    }
}
void_hook!(barrier2, c"vkCmdPipelineBarrier2", PFN_vkCmdPipelineBarrier2, cmd: vk::CommandBuffer, info: *const vk::DependencyInfo<'_>;
    |state| { dependency(&state, cmd, info); });
void_hook!(barrier2_khr, c"vkCmdPipelineBarrier2KHR", PFN_vkCmdPipelineBarrier2, cmd: vk::CommandBuffer, info: *const vk::DependencyInfo<'_>;
    |state| { dependency(&state, cmd, info); });
void_hook!(execute, c"vkCmdExecuteCommands", PFN_vkCmdExecuteCommands, cmd: vk::CommandBuffer,
count: u32, children: *const vk::CommandBuffer; |state| {
    let mut global = source();
    for child in slice(children, count, 64) {
        if let Some(record) = global.recordings.get(&(state.handle, *child)).cloned() {
            let parent = global.recordings.entry((state.handle, cmd)).or_default();
            for value in record.images { add_image(&mut parent.images, value); }
        }
    }
});

result_hook!(submit, c"vkQueueSubmit", PFN_vkQueueSubmit, queue: vk::Queue,
count: u32, submits: *const vk::SubmitInfo<'_>, fence: vk::Fence; |state, result| {
    if result == vk::Result::SUCCESS {
        let commands = slice(submits, count, 4096).iter().flat_map(|submit|
            slice(submit.p_command_buffers, submit.command_buffer_count, 4096).iter().copied());
        source().submit(state.handle, commands);
    }
});
unsafe fn submit2_observe(
    state: &Device,
    count: u32,
    submits: *const vk::SubmitInfo2<'_>,
    result: vk::Result,
) {
    if result == vk::Result::SUCCESS {
        let commands = slice(submits, count, 4096).iter().flat_map(|submit| {
            slice(
                submit.p_command_buffer_infos,
                submit.command_buffer_info_count,
                4096,
            )
            .iter()
            .map(|info| info.command_buffer)
        });
        source().submit(state.handle, commands);
    }
}
result_hook!(submit2, c"vkQueueSubmit2", PFN_vkQueueSubmit2, queue: vk::Queue, count: u32, submits: *const vk::SubmitInfo2<'_>, fence: vk::Fence;
    |state, result| { submit2_observe(&state, count, submits, result); });
result_hook!(submit2_khr, c"vkQueueSubmit2KHR", PFN_vkQueueSubmit2, queue: vk::Queue, count: u32, submits: *const vk::SubmitInfo2<'_>, fence: vk::Fence;
    |state, result| { submit2_observe(&state, count, submits, result); });

unsafe fn observe_conversion(
    state: &Device,
    info: *const vk::SamplerYcbcrConversionCreateInfo<'_>,
    output: *mut vk::SamplerYcbcrConversion,
    result: vk::Result,
) {
    if result != vk::Result::SUCCESS {
        return;
    }
    let (Some(info), Some(&handle)) = (info.as_ref(), output.as_ref()) else {
        return;
    };
    let swizzle = info.components;
    let mut global = source();
    if global.conversions.len() < 1024 {
        global.conversions.insert(
            (state.handle, handle),
            Ycbcr {
                valid: 1,
                model: info.ycbcr_model.as_raw() as u32,
                range: info.ycbcr_range.as_raw() as u32,
                chroma_filter: info.chroma_filter.as_raw() as u32,
                x_chroma_offset: info.x_chroma_offset.as_raw() as u32,
                y_chroma_offset: info.y_chroma_offset.as_raw() as u32,
                force_explicit_reconstruction: info.force_explicit_reconstruction,
                reserved: 0,
                components: [swizzle.r, swizzle.g, swizzle.b, swizzle.a]
                    .map(|value| value.as_raw() as u32),
            },
        );
    }
}
result_hook!(create_conversion, c"vkCreateSamplerYcbcrConversion", PFN_vkCreateSamplerYcbcrConversion, dev: vk::Device,
    info: *const vk::SamplerYcbcrConversionCreateInfo<'_>, allocator: *const vk::AllocationCallbacks<'_>, output: *mut vk::SamplerYcbcrConversion;
    |state, result| { observe_conversion(&state, info, output, result); });
result_hook!(create_conversion_khr, c"vkCreateSamplerYcbcrConversionKHR", PFN_vkCreateSamplerYcbcrConversion, dev: vk::Device,
    info: *const vk::SamplerYcbcrConversionCreateInfo<'_>, allocator: *const vk::AllocationCallbacks<'_>, output: *mut vk::SamplerYcbcrConversion;
    |state, result| { observe_conversion(&state, info, output, result); });
void_hook!(destroy_conversion, c"vkDestroySamplerYcbcrConversion", PFN_vkDestroySamplerYcbcrConversion, dev: vk::Device,
    handle: vk::SamplerYcbcrConversion, allocator: *const vk::AllocationCallbacks<'_>;
    |_state| { source().conversions.remove(&(dev, handle)); });
void_hook!(destroy_conversion_khr, c"vkDestroySamplerYcbcrConversionKHR", PFN_vkDestroySamplerYcbcrConversion, dev: vk::Device,
    handle: vk::SamplerYcbcrConversion, allocator: *const vk::AllocationCallbacks<'_>;
    |_state| { source().conversions.remove(&(dev, handle)); });

result_hook!(create_view, c"vkCreateImageView", PFN_vkCreateImageView, dev: vk::Device,
info: *const vk::ImageViewCreateInfo<'_>, allocator: *const vk::AllocationCallbacks<'_>, output: *mut vk::ImageView;
|_state, result| {
    if result != vk::Result::SUCCESS { return; }
    let (Some(info), Some(&view)) = (info.as_ref(), output.as_ref()) else { return };
    let mut global = source();
    if global.images.contains_key(&(dev, info.image)) && global.views.len() < 65536 {
        global.views.insert((dev, view), (info.image, info.subresource_range.base_array_layer));
    }
    let Some(conversion) = next_info::<vk::SamplerYcbcrConversionInfo<'_>>(info.p_next) else { return };
    let Some(&ycbcr) = global.conversions.get(&(dev, (*conversion).conversion)) else { return };
    if let Some(image) = global.images.get_mut(&(dev, info.image))
        && image.description.ycbcr != ycbcr
    {
        image.description.ycbcr = ycbcr;
        if let Some(published) = &mut image.published {
            published.description.ycbcr = ycbcr;
            global.metadata_changed();
        }
    }
});
void_hook!(destroy_view, c"vkDestroyImageView", PFN_vkDestroyImageView, dev: vk::Device,
    view: vk::ImageView, allocator: *const vk::AllocationCallbacks<'_>;
    |_state| { source().views.remove(&(dev, view)); });

fn image_descriptor(ty: vk::DescriptorType) -> bool {
    matches!(
        ty,
        vk::DescriptorType::COMBINED_IMAGE_SAMPLER
            | vk::DescriptorType::SAMPLED_IMAGE
            | vk::DescriptorType::STORAGE_IMAGE
            | vk::DescriptorType::INPUT_ATTACHMENT
    )
}

fn write_descriptor(
    bindings: &mut BTreeMap<(u32, u32), ViewUse>,
    key: (u32, u32),
    info: vk::DescriptorImageInfo,
) {
    if info.image_view == vk::ImageView::null() {
        bindings.remove(&key);
    } else if bindings.len() < 4096 || bindings.contains_key(&key) {
        bindings.insert(
            key,
            ViewUse {
                view: info.image_view,
                layout: info.image_layout,
            },
        );
    }
}

fn buffer_descriptor(ty: vk::DescriptorType) -> bool {
    matches!(
        ty,
        vk::DescriptorType::UNIFORM_BUFFER
            | vk::DescriptorType::STORAGE_BUFFER
            | vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC
            | vk::DescriptorType::STORAGE_BUFFER_DYNAMIC
    )
}

fn write_buffer(
    buffers: &mut BTreeMap<(u32, u32), BufferUse>,
    key: (u32, u32),
    kind: vk::DescriptorType,
    info: vk::DescriptorBufferInfo,
) {
    if info.buffer == vk::Buffer::null() {
        buffers.remove(&key);
    } else if buffers.len() < 256 || buffers.contains_key(&key) {
        buffers.insert(
            key,
            BufferUse {
                buffer: info.buffer,
                offset: info.offset,
                range: info.range,
                kind,
            },
        );
    }
}

unsafe fn write_set_buffers(
    buffers: &mut BTreeMap<(u32, u32), BufferUse>,
    write: &vk::WriteDescriptorSet<'_>,
) {
    if buffer_descriptor(write.descriptor_type) {
        for (index, &info) in slice(write.p_buffer_info, write.descriptor_count, 256)
            .iter()
            .enumerate()
        {
            write_buffer(
                buffers,
                (
                    write.dst_binding,
                    write.dst_array_element.saturating_add(index as u32),
                ),
                write.descriptor_type,
                info,
            );
        }
    }
}

unsafe fn write_set(
    bindings: &mut BTreeMap<(u32, u32), ViewUse>,
    write: &vk::WriteDescriptorSet<'_>,
) {
    if image_descriptor(write.descriptor_type) {
        for (index, &info) in slice(write.p_image_info, write.descriptor_count, 4096)
            .iter()
            .enumerate()
        {
            write_descriptor(
                bindings,
                (
                    write.dst_binding,
                    write.dst_array_element.saturating_add(index as u32),
                ),
                info,
            );
        }
    }
}

result_hook!(allocate_sets, c"vkAllocateDescriptorSets", PFN_vkAllocateDescriptorSets, dev: vk::Device,
info: *const vk::DescriptorSetAllocateInfo<'_>, output: *mut vk::DescriptorSet; |_state, result| {
    if result == vk::Result::SUCCESS && let Some(info) = info.as_ref() {
        let mut global = source();
        for &set in slice(output, info.descriptor_set_count, 65536) {
            if global.sets.len() >= 65536 { break; }
            global.sets.insert((dev, set), Set { pool: info.descriptor_pool, ..Default::default() });
        }
    }
});
result_hook!(free_sets, c"vkFreeDescriptorSets", PFN_vkFreeDescriptorSets, dev: vk::Device,
_pool: vk::DescriptorPool, count: u32, sets: *const vk::DescriptorSet; |_state, result| {
    if result == vk::Result::SUCCESS {
        let mut global = source();
        for &set in slice(sets, count, 65536) { global.sets.remove(&(dev, set)); }
    }
});
result_hook!(reset_sets, c"vkResetDescriptorPool", PFN_vkResetDescriptorPool, dev: vk::Device,
pool: vk::DescriptorPool, _flags: vk::DescriptorPoolResetFlags; |_state, result| {
    if result == vk::Result::SUCCESS {
        source().sets.retain(|&(owner, _), set| owner != dev || set.pool != pool);
    }
});
void_hook!(destroy_sets, c"vkDestroyDescriptorPool", PFN_vkDestroyDescriptorPool, dev: vk::Device,
pool: vk::DescriptorPool, allocator: *const vk::AllocationCallbacks<'_>; |_state| {
    source().sets.retain(|&(owner, _), set| owner != dev || set.pool != pool);
});
void_hook!(update_sets, c"vkUpdateDescriptorSets", PFN_vkUpdateDescriptorSets, dev: vk::Device,
write_count: u32, writes: *const vk::WriteDescriptorSet<'_>,
copy_count: u32, copies: *const vk::CopyDescriptorSet<'_>; |_state| {
    let mut global = source();
    for write in slice(writes, write_count, 65536) {
        if let Some(set) = global.sets.get_mut(&(dev, write.dst_set)) {
            write_set(&mut set.bindings, write);
            write_set_buffers(&mut set.buffers, write);
        }
    }
    for copy in slice(copies, copy_count, 65536) {
        for index in 0..copy.descriptor_count.min(4096) {
            let value = global.sets.get(&(dev, copy.src_set))
                .and_then(|set| set.bindings.get(&(copy.src_binding, copy.src_array_element.saturating_add(index))))
                .copied();
            let buffer = global.sets.get(&(dev, copy.src_set))
                .and_then(|set| set.buffers.get(&(copy.src_binding, copy.src_array_element.saturating_add(index))))
                .copied();
            if let Some(set) = global.sets.get_mut(&(dev, copy.dst_set)) {
                let key = (copy.dst_binding, copy.dst_array_element.saturating_add(index));
                if let Some(value) = value {
                    if set.bindings.len() < 4096 || set.bindings.contains_key(&key) { set.bindings.insert(key, value); }
                } else { set.bindings.remove(&key); }
                if let Some(buffer) = buffer {
                    if set.buffers.len() < 256 || set.buffers.contains_key(&key) { set.buffers.insert(key, buffer); }
                } else { set.buffers.remove(&key); }
            }
        }
    }
});

unsafe fn create_template_record(
    state: &Device,
    info: *const vk::DescriptorUpdateTemplateCreateInfo<'_>,
    output: *mut vk::DescriptorUpdateTemplate,
    result: vk::Result,
) {
    if result != vk::Result::SUCCESS {
        return;
    }
    let (Some(info), Some(&handle)) = (info.as_ref(), output.as_ref()) else {
        return;
    };
    let entries = slice(
        info.p_descriptor_update_entries,
        info.descriptor_update_entry_count,
        4096,
    )
    .to_vec();
    let mut global = source();
    if global.templates.len() < 4096 {
        global.templates.insert(
            (state.handle, handle),
            Template {
                entries,
                graphics: info.pipeline_bind_point == vk::PipelineBindPoint::GRAPHICS,
            },
        );
    }
}

unsafe fn template_values(
    entries: &[vk::DescriptorUpdateTemplateEntry],
    bytes: *const c_void,
) -> Vec<((u32, u32), vk::DescriptorImageInfo)> {
    if bytes.is_null() {
        return Vec::new();
    }
    let mut values = Vec::new();
    for entry in entries {
        if !image_descriptor(entry.descriptor_type) {
            continue;
        }
        for index in 0..entry.descriptor_count.min(4096) {
            let Some(offset) = entry
                .stride
                .checked_mul(index as usize)
                .and_then(|offset| entry.offset.checked_add(offset))
            else {
                break;
            };
            if offset > isize::MAX as usize - mem::size_of::<vk::DescriptorImageInfo>() {
                break;
            }
            // The next Vulkan implementation accepted this template and data.
            let info = bytes
                .cast::<u8>()
                .add(offset)
                .cast::<vk::DescriptorImageInfo>()
                .read_unaligned();
            values.push((
                (
                    entry.dst_binding,
                    entry.dst_array_element.saturating_add(index),
                ),
                info,
            ));
            if values.len() >= 4096 {
                return values;
            }
        }
    }
    values
}

unsafe fn template_buffers(
    entries: &[vk::DescriptorUpdateTemplateEntry],
    bytes: *const c_void,
) -> Vec<((u32, u32), vk::DescriptorType, vk::DescriptorBufferInfo)> {
    if bytes.is_null() {
        return Vec::new();
    }
    let mut values = Vec::new();
    for entry in entries {
        if !buffer_descriptor(entry.descriptor_type) {
            continue;
        }
        for index in 0..entry.descriptor_count.min(256) {
            let Some(offset) = entry
                .stride
                .checked_mul(index as usize)
                .and_then(|offset| entry.offset.checked_add(offset))
            else {
                break;
            };
            if offset > isize::MAX as usize - mem::size_of::<vk::DescriptorBufferInfo>() {
                break;
            }
            // The next Vulkan implementation accepted this template and data.
            let info = bytes
                .cast::<u8>()
                .add(offset)
                .cast::<vk::DescriptorBufferInfo>()
                .read_unaligned();
            values.push((
                (
                    entry.dst_binding,
                    entry.dst_array_element.saturating_add(index),
                ),
                entry.descriptor_type,
                info,
            ));
            if values.len() >= 256 {
                return values;
            }
        }
    }
    values
}

unsafe fn update_template_record(
    dev: vk::Device,
    set: vk::DescriptorSet,
    template: vk::DescriptorUpdateTemplate,
    data: *const c_void,
) {
    let mut global = source();
    let Some(template) = global.templates.get(&(dev, template)) else {
        return;
    };
    let values = template_values(&template.entries, data);
    let buffers = template_buffers(&template.entries, data);
    if let Some(set) = global.sets.get_mut(&(dev, set)) {
        for (key, info) in values {
            write_descriptor(&mut set.bindings, key, info);
        }
        for (key, kind, info) in buffers {
            write_buffer(&mut set.buffers, key, kind, info);
        }
    }
}

result_hook!(create_template, c"vkCreateDescriptorUpdateTemplate", PFN_vkCreateDescriptorUpdateTemplate, dev: vk::Device,
    info: *const vk::DescriptorUpdateTemplateCreateInfo<'_>, allocator: *const vk::AllocationCallbacks<'_>, output: *mut vk::DescriptorUpdateTemplate;
    |state, result| { create_template_record(&state, info, output, result); });
result_hook!(create_template_khr, c"vkCreateDescriptorUpdateTemplateKHR", PFN_vkCreateDescriptorUpdateTemplate, dev: vk::Device,
    info: *const vk::DescriptorUpdateTemplateCreateInfo<'_>, allocator: *const vk::AllocationCallbacks<'_>, output: *mut vk::DescriptorUpdateTemplate;
    |state, result| { create_template_record(&state, info, output, result); });
void_hook!(destroy_template, c"vkDestroyDescriptorUpdateTemplate", PFN_vkDestroyDescriptorUpdateTemplate, dev: vk::Device,
    template: vk::DescriptorUpdateTemplate, allocator: *const vk::AllocationCallbacks<'_>; |_state| { source().templates.remove(&(dev, template)); });
void_hook!(destroy_template_khr, c"vkDestroyDescriptorUpdateTemplateKHR", PFN_vkDestroyDescriptorUpdateTemplate, dev: vk::Device,
    template: vk::DescriptorUpdateTemplate, allocator: *const vk::AllocationCallbacks<'_>; |_state| { source().templates.remove(&(dev, template)); });
void_hook!(update_template, c"vkUpdateDescriptorSetWithTemplate", PFN_vkUpdateDescriptorSetWithTemplate, dev: vk::Device,
    set: vk::DescriptorSet, template: vk::DescriptorUpdateTemplate, data: *const c_void; |_state| { update_template_record(dev, set, template, data); });
void_hook!(update_template_khr, c"vkUpdateDescriptorSetWithTemplateKHR", PFN_vkUpdateDescriptorSetWithTemplate, dev: vk::Device,
    set: vk::DescriptorSet, template: vk::DescriptorUpdateTemplate, data: *const c_void; |_state| { update_template_record(dev, set, template, data); });

void_hook!(bind_sets, c"vkCmdBindDescriptorSets", PFN_vkCmdBindDescriptorSets, cmd: vk::CommandBuffer,
bind_point: vk::PipelineBindPoint, _layout: vk::PipelineLayout, first: u32, count: u32, sets: *const vk::DescriptorSet,
offset_count: u32, offsets: *const u32; |state| {
    if bind_point != vk::PipelineBindPoint::GRAPHICS { return; }
    let mut global = source();
    // Dynamic offsets apply in set order, then binding and element order.
    let mut remaining = slice(offsets, offset_count, 256).iter().copied();
    let dynamic: Vec<BTreeMap<(u32, u32), u32>> = slice(sets, count, 64).iter().map(|set| {
        global.sets.get(&(state.handle, *set)).map(|set| {
            set.buffers.iter().filter(|(_, value)| value.dynamic())
                .filter_map(|(&key, _)| Some((key, remaining.next()?))).collect()
        }).unwrap_or_default()
    }).collect();
    let rec = global.recordings.entry((state.handle, cmd)).or_default();
    for ((index, &set), dynamic) in slice(sets, count, 64).iter().enumerate().zip(dynamic) {
        if rec.bound.len() < 64 && !rec.bound.contains(&set) { rec.bound.push(set); }
        let index = first as usize + index;
        if index < rec.graphics.len() {
            rec.graphics[index] = set;
            rec.pushed[index].clear();
            rec.pushed_buffers[index].clear();
            rec.dynamic[index] = dynamic;
        }
    }
});

unsafe fn push_set_record(
    state: &Device,
    cmd: vk::CommandBuffer,
    point: vk::PipelineBindPoint,
    set: u32,
    count: u32,
    writes: *const vk::WriteDescriptorSet<'_>,
) {
    if point != vk::PipelineBindPoint::GRAPHICS || set >= 8 {
        return;
    }
    let mut global = source();
    let rec = global.recordings.entry((state.handle, cmd)).or_default();
    rec.graphics[set as usize] = vk::DescriptorSet::null();
    for write in slice(writes, count, 4096) {
        write_set(&mut rec.pushed[set as usize], write);
        write_set_buffers(&mut rec.pushed_buffers[set as usize], write);
    }
}

unsafe fn push_template_record(
    state: &Device,
    cmd: vk::CommandBuffer,
    template: vk::DescriptorUpdateTemplate,
    set: u32,
    data: *const c_void,
) {
    if set >= 8 {
        return;
    }
    let mut global = source();
    let Some(template) = global
        .templates
        .get(&(state.handle, template))
        .filter(|template| template.graphics)
    else {
        return;
    };
    let values = template_values(&template.entries, data);
    let buffers = template_buffers(&template.entries, data);
    let rec = global.recordings.entry((state.handle, cmd)).or_default();
    rec.graphics[set as usize] = vk::DescriptorSet::null();
    for (key, info) in values {
        write_descriptor(&mut rec.pushed[set as usize], key, info);
    }
    for (key, kind, info) in buffers {
        write_buffer(&mut rec.pushed_buffers[set as usize], key, kind, info);
    }
}

void_hook!(push_set, c"vkCmdPushDescriptorSet", PFN_vkCmdPushDescriptorSetKHR, cmd: vk::CommandBuffer,
    point: vk::PipelineBindPoint, _layout: vk::PipelineLayout, set: u32, count: u32, writes: *const vk::WriteDescriptorSet<'_>;
    |state| { push_set_record(&state, cmd, point, set, count, writes); });
void_hook!(push_set_khr, c"vkCmdPushDescriptorSetKHR", PFN_vkCmdPushDescriptorSetKHR, cmd: vk::CommandBuffer,
    point: vk::PipelineBindPoint, _layout: vk::PipelineLayout, set: u32, count: u32, writes: *const vk::WriteDescriptorSet<'_>;
    |state| { push_set_record(&state, cmd, point, set, count, writes); });
void_hook!(push_template, c"vkCmdPushDescriptorSetWithTemplate", PFN_vkCmdPushDescriptorSetWithTemplateKHR, cmd: vk::CommandBuffer,
    template: vk::DescriptorUpdateTemplate, _layout: vk::PipelineLayout, set: u32, data: *const c_void;
    |state| { push_template_record(&state, cmd, template, set, data); });
void_hook!(push_template_khr, c"vkCmdPushDescriptorSetWithTemplateKHR", PFN_vkCmdPushDescriptorSetWithTemplateKHR, cmd: vk::CommandBuffer,
    template: vk::DescriptorUpdateTemplate, _layout: vk::PipelineLayout, set: u32, data: *const c_void;
    |state| { push_template_record(&state, cmd, template, set, data); });

fn area(state: &Device, cmd: vk::CommandBuffer, area: Option<vk::Rect2D>) {
    source()
        .recordings
        .entry((state.handle, cmd))
        .or_default()
        .area = area;
}
fn viewport(state: &Device, cmd: vk::CommandBuffer, viewport: vk::Viewport) {
    source()
        .recordings
        .entry((state.handle, cmd))
        .or_default()
        .viewport = Some(viewport);
}

void_hook!(begin_pass, c"vkCmdBeginRenderPass", PFN_vkCmdBeginRenderPass, cmd: vk::CommandBuffer,
    info: *const vk::RenderPassBeginInfo<'_>, _contents: vk::SubpassContents;
    |state| { if let Some(info) = info.as_ref() { area(&state, cmd, Some(info.render_area)); } });
void_hook!(end_pass, c"vkCmdEndRenderPass", PFN_vkCmdEndRenderPass, cmd: vk::CommandBuffer;
    |state| { area(&state, cmd, None); });
void_hook!(begin_pass2, c"vkCmdBeginRenderPass2", PFN_vkCmdBeginRenderPass2, cmd: vk::CommandBuffer,
    info: *const vk::RenderPassBeginInfo<'_>, _begin: *const vk::SubpassBeginInfo<'_>;
    |state| { if let Some(info) = info.as_ref() { area(&state, cmd, Some(info.render_area)); } });
void_hook!(begin_pass2_khr, c"vkCmdBeginRenderPass2KHR", PFN_vkCmdBeginRenderPass2, cmd: vk::CommandBuffer,
    info: *const vk::RenderPassBeginInfo<'_>, _begin: *const vk::SubpassBeginInfo<'_>;
    |state| { if let Some(info) = info.as_ref() { area(&state, cmd, Some(info.render_area)); } });
void_hook!(end_pass2, c"vkCmdEndRenderPass2", PFN_vkCmdEndRenderPass2, cmd: vk::CommandBuffer, _end: *const vk::SubpassEndInfo<'_>;
    |state| { area(&state, cmd, None); });
void_hook!(end_pass2_khr, c"vkCmdEndRenderPass2KHR", PFN_vkCmdEndRenderPass2, cmd: vk::CommandBuffer, _end: *const vk::SubpassEndInfo<'_>;
    |state| { area(&state, cmd, None); });
void_hook!(begin_rendering, c"vkCmdBeginRendering", PFN_vkCmdBeginRendering, cmd: vk::CommandBuffer, info: *const vk::RenderingInfo<'_>;
    |state| { if let Some(info) = info.as_ref() { area(&state, cmd, Some(info.render_area)); } });
void_hook!(begin_rendering_khr, c"vkCmdBeginRenderingKHR", PFN_vkCmdBeginRendering, cmd: vk::CommandBuffer, info: *const vk::RenderingInfo<'_>;
    |state| { if let Some(info) = info.as_ref() { area(&state, cmd, Some(info.render_area)); } });
void_hook!(end_rendering, c"vkCmdEndRendering", PFN_vkCmdEndRendering, cmd: vk::CommandBuffer; |state| { area(&state, cmd, None); });
void_hook!(end_rendering_khr, c"vkCmdEndRenderingKHR", PFN_vkCmdEndRendering, cmd: vk::CommandBuffer; |state| { area(&state, cmd, None); });

void_hook!(set_viewport, c"vkCmdSetViewport", PFN_vkCmdSetViewport, cmd: vk::CommandBuffer,
first: u32, count: u32, values: *const vk::Viewport; |state| {
    if first == 0 && count != 0 && let Some(&first) = values.as_ref() { viewport(&state, cmd, first); }
});
void_hook!(set_viewport_count, c"vkCmdSetViewportWithCount", PFN_vkCmdSetViewportWithCount, cmd: vk::CommandBuffer,
count: u32, values: *const vk::Viewport; |state| {
    if count != 0 && let Some(&first) = values.as_ref() { viewport(&state, cmd, first); }
});
void_hook!(set_viewport_count_ext, c"vkCmdSetViewportWithCountEXT", PFN_vkCmdSetViewportWithCount, cmd: vk::CommandBuffer,
count: u32, values: *const vk::Viewport; |state| {
    if count != 0 && let Some(&first) = values.as_ref() { viewport(&state, cmd, first); }
});

void_hook!(draw, c"vkCmdDraw", PFN_vkCmdDraw, cmd: vk::CommandBuffer, _vertices: u32, _instances: u32, _vertex: u32, _instance: u32;
    |state| { source().draw(state.handle, cmd); });
void_hook!(draw_indexed, c"vkCmdDrawIndexed", PFN_vkCmdDrawIndexed, cmd: vk::CommandBuffer,
    _indices: u32, _instances: u32, _index: u32, _vertex: i32, _instance: u32; |state| { source().draw(state.handle, cmd); });
void_hook!(draw_indirect, c"vkCmdDrawIndirect", PFN_vkCmdDrawIndirect, cmd: vk::CommandBuffer,
    _buffer: vk::Buffer, _offset: u64, _count: u32, _stride: u32; |state| { source().draw(state.handle, cmd); });
void_hook!(draw_indexed_indirect, c"vkCmdDrawIndexedIndirect", PFN_vkCmdDrawIndexedIndirect, cmd: vk::CommandBuffer,
    _buffer: vk::Buffer, _offset: u64, _count: u32, _stride: u32; |state| { source().draw(state.handle, cmd); });

macro_rules! hooks {
    ($($name:literal => $function:ident : $ty:ident),* $(,)?) => {
        pub const HOOKS: &[&CStr] = &[$($name,)* c"vkGetMemoryFdKHR", c"vkDestroyDevice"];
        pub fn hook(name: &CStr) -> vk::PFN_vkVoidFunction {
            // SAFETY: Each hook has the signature of the Vulkan entry point.
            unsafe { $(if name == $name { return entry!($function, vk::$ty); })* }
            None
        }
    };
}

hooks! {
    c"vkAllocateMemory" => allocate_memory: PFN_vkAllocateMemory,
    c"vkFreeMemory" => free_memory: PFN_vkFreeMemory,
    c"vkMapMemory" => map_memory: PFN_vkMapMemory,
    c"vkUnmapMemory" => unmap_memory: PFN_vkUnmapMemory,
    c"vkBindBufferMemory" => bind_buffer: PFN_vkBindBufferMemory,
    c"vkBindBufferMemory2" => bind_buffer2: PFN_vkBindBufferMemory2,
    c"vkBindBufferMemory2KHR" => bind_buffer2_khr: PFN_vkBindBufferMemory2,
    c"vkDestroyBuffer" => destroy_buffer: PFN_vkDestroyBuffer,
    c"vkCreateShaderModule" => create_module: PFN_vkCreateShaderModule,
    c"vkDestroyShaderModule" => destroy_module: PFN_vkDestroyShaderModule,
    c"vkCreateGraphicsPipelines" => create_pipelines: PFN_vkCreateGraphicsPipelines,
    c"vkDestroyPipeline" => destroy_pipeline: PFN_vkDestroyPipeline,
    c"vkCmdBindPipeline" => bind_pipeline: PFN_vkCmdBindPipeline,
    c"vkCreateImage" => create_image: PFN_vkCreateImage,
    c"vkDestroyImage" => destroy_image: PFN_vkDestroyImage,
    c"vkBindImageMemory" => bind_image: PFN_vkBindImageMemory,
    c"vkBindImageMemory2" => bind_image2: PFN_vkBindImageMemory2,
    c"vkBindImageMemory2KHR" => bind_image2_khr: PFN_vkBindImageMemory2,
    c"vkCreateSamplerYcbcrConversion" => create_conversion: PFN_vkCreateSamplerYcbcrConversion,
    c"vkCreateSamplerYcbcrConversionKHR" => create_conversion_khr: PFN_vkCreateSamplerYcbcrConversion,
    c"vkDestroySamplerYcbcrConversion" => destroy_conversion: PFN_vkDestroySamplerYcbcrConversion,
    c"vkDestroySamplerYcbcrConversionKHR" => destroy_conversion_khr: PFN_vkDestroySamplerYcbcrConversion,
    c"vkCreateImageView" => create_view: PFN_vkCreateImageView,
    c"vkDestroyImageView" => destroy_view: PFN_vkDestroyImageView,
    c"vkAllocateDescriptorSets" => allocate_sets: PFN_vkAllocateDescriptorSets,
    c"vkFreeDescriptorSets" => free_sets: PFN_vkFreeDescriptorSets,
    c"vkResetDescriptorPool" => reset_sets: PFN_vkResetDescriptorPool,
    c"vkDestroyDescriptorPool" => destroy_sets: PFN_vkDestroyDescriptorPool,
    c"vkUpdateDescriptorSets" => update_sets: PFN_vkUpdateDescriptorSets,
    c"vkCreateDescriptorUpdateTemplate" => create_template: PFN_vkCreateDescriptorUpdateTemplate,
    c"vkCreateDescriptorUpdateTemplateKHR" => create_template_khr: PFN_vkCreateDescriptorUpdateTemplate,
    c"vkDestroyDescriptorUpdateTemplate" => destroy_template: PFN_vkDestroyDescriptorUpdateTemplate,
    c"vkDestroyDescriptorUpdateTemplateKHR" => destroy_template_khr: PFN_vkDestroyDescriptorUpdateTemplate,
    c"vkUpdateDescriptorSetWithTemplate" => update_template: PFN_vkUpdateDescriptorSetWithTemplate,
    c"vkUpdateDescriptorSetWithTemplateKHR" => update_template_khr: PFN_vkUpdateDescriptorSetWithTemplate,
    c"vkCmdBindDescriptorSets" => bind_sets: PFN_vkCmdBindDescriptorSets,
    c"vkCmdPushDescriptorSet" => push_set: PFN_vkCmdPushDescriptorSetKHR,
    c"vkCmdPushDescriptorSetKHR" => push_set_khr: PFN_vkCmdPushDescriptorSetKHR,
    c"vkCmdPushDescriptorSetWithTemplate" => push_template: PFN_vkCmdPushDescriptorSetWithTemplateKHR,
    c"vkCmdPushDescriptorSetWithTemplateKHR" => push_template_khr: PFN_vkCmdPushDescriptorSetWithTemplateKHR,
    c"vkAllocateCommandBuffers" => allocate_commands: PFN_vkAllocateCommandBuffers,
    c"vkFreeCommandBuffers" => free_commands: PFN_vkFreeCommandBuffers,
    c"vkBeginCommandBuffer" => begin_command: PFN_vkBeginCommandBuffer,
    c"vkDestroyCommandPool" => destroy_command_pool: PFN_vkDestroyCommandPool,
    c"vkCmdPipelineBarrier" => barrier: PFN_vkCmdPipelineBarrier,
    c"vkCmdPipelineBarrier2" => barrier2: PFN_vkCmdPipelineBarrier2,
    c"vkCmdPipelineBarrier2KHR" => barrier2_khr: PFN_vkCmdPipelineBarrier2,
    c"vkCmdExecuteCommands" => execute: PFN_vkCmdExecuteCommands,
    c"vkQueueSubmit" => submit: PFN_vkQueueSubmit,
    c"vkQueueSubmit2" => submit2: PFN_vkQueueSubmit2,
    c"vkQueueSubmit2KHR" => submit2_khr: PFN_vkQueueSubmit2,
    c"vkCmdBeginRenderPass" => begin_pass: PFN_vkCmdBeginRenderPass,
    c"vkCmdBeginRenderPass2" => begin_pass2: PFN_vkCmdBeginRenderPass2,
    c"vkCmdBeginRenderPass2KHR" => begin_pass2_khr: PFN_vkCmdBeginRenderPass2,
    c"vkCmdEndRenderPass" => end_pass: PFN_vkCmdEndRenderPass,
    c"vkCmdEndRenderPass2" => end_pass2: PFN_vkCmdEndRenderPass2,
    c"vkCmdEndRenderPass2KHR" => end_pass2_khr: PFN_vkCmdEndRenderPass2,
    c"vkCmdBeginRendering" => begin_rendering: PFN_vkCmdBeginRendering,
    c"vkCmdBeginRenderingKHR" => begin_rendering_khr: PFN_vkCmdBeginRendering,
    c"vkCmdEndRendering" => end_rendering: PFN_vkCmdEndRendering,
    c"vkCmdEndRenderingKHR" => end_rendering_khr: PFN_vkCmdEndRendering,
    c"vkCmdSetViewport" => set_viewport: PFN_vkCmdSetViewport,
    c"vkCmdSetViewportWithCount" => set_viewport_count: PFN_vkCmdSetViewportWithCount,
    c"vkCmdSetViewportWithCountEXT" => set_viewport_count_ext: PFN_vkCmdSetViewportWithCount,
    c"vkCmdDraw" => draw: PFN_vkCmdDraw,
    c"vkCmdDrawIndexed" => draw_indexed: PFN_vkCmdDrawIndexed,
    c"vkCmdDrawIndirect" => draw_indirect: PFN_vkCmdDrawIndirect,
    c"vkCmdDrawIndexedIndirect" => draw_indexed_indirect: PFN_vkCmdDrawIndexedIndirect,
}

mod spirv;
mod trace;

#[cfg(test)]
mod contract;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_eye_binding_and_array_layer_in_successful_submissions() {
        let device = vk::Device::from_raw(1);
        let cmd = vk::CommandBuffer::from_raw(2);
        let image = vk::Image::from_raw(3);
        let left = vk::ImageView::from_raw(4);
        let right = vk::ImageView::from_raw(5);
        let descriptor = vk::DescriptorSet::from_raw(6);
        let fd: OwnedFd = std::fs::File::open("/dev/null").unwrap().into();
        let description = ImageAdd {
            id: 11,
            ..Default::default()
        };
        let mut global = State::default();
        global.images.insert(
            (device, image),
            Image {
                description,
                published: Some(Published {
                    fd: Arc::new(fd),
                    description,
                }),
            },
        );
        global.views.insert((device, left), (image, 1));
        global.views.insert((device, right), (image, 0));

        let record = global.recordings.entry((device, cmd)).or_default();
        record.area = Some(vk::Rect2D {
            extent: vk::Extent2D {
                width: 4320,
                height: 2160,
            },
            ..Default::default()
        });
        record.viewport = Some(vk::Viewport {
            width: 2160.0,
            ..Default::default()
        });
        record.graphics[0] = descriptor;
        record.bound.push(descriptor);
        let entry = |view| ViewUse {
            view,
            layout: vk::ImageLayout::GENERAL,
        };
        global
            .sets
            .entry((device, descriptor))
            .or_default()
            .bindings
            .insert((9, 0), entry(left));
        global.draw(device, cmd);

        global
            .recordings
            .get_mut(&(device, cmd))
            .unwrap()
            .viewport
            .as_mut()
            .unwrap()
            .x = 2160.0;
        global
            .sets
            .get_mut(&(device, descriptor))
            .unwrap()
            .bindings
            .insert((9, 0), entry(right));
        global.draw(device, cmd);
        global.submit(device, [cmd].into_iter());
        let frame = global.frame.unwrap();
        assert_eq!(frame.touched_count, 1);
        assert_eq!(frame.touched_ids[0], 11);
        assert_eq!(
            frame.touched_eyes[0],
            EyeUse {
                eyes: 3,
                slot: 9,
                layers: [1, 0]
            }
        );

        let record = global.recordings.get_mut(&(device, cmd)).unwrap();
        record.eye_uses.clear();
        record.viewport = Some(vk::Viewport {
            width: 4320.0,
            ..Default::default()
        });
        global.draw(device, cmd);
        global.submit(device, [cmd].into_iter());
        assert_eq!(global.frame.unwrap().touched_eyes[0], EyeUse::default());
    }

    /// One array image sampled by a recognised scene pipeline, both eyes,
    /// with the compositor's per-submission block in host memory.
    fn scene(block: &mut [u32], scene: bool) -> State {
        let device = vk::Device::from_raw(1);
        let cmd = vk::CommandBuffer::from_raw(2);
        let image = vk::Image::from_raw(3);
        let [left, right] = [4, 5].map(vk::ImageView::from_raw);
        let descriptor = vk::DescriptorSet::from_raw(6);
        let [buffer, memory, pipeline] = [7_u64, 8, 9];
        let fd: OwnedFd = std::fs::File::open("/dev/null").unwrap().into();
        let description = ImageAdd {
            id: 11,
            ..Default::default()
        };
        let mut global = State::default();
        global.images.insert(
            (device, image),
            Image {
                description,
                published: Some(Published {
                    fd: Arc::new(fd),
                    description,
                }),
            },
        );
        global.views.insert((device, left), (image, 0));
        global.views.insert((device, right), (image, 1));
        let memory = vk::DeviceMemory::from_raw(memory);
        let buffer = vk::Buffer::from_raw(buffer);
        global.buffers.insert((device, buffer), (memory, 64));
        global.maps.insert(
            (device, memory),
            Mapped {
                pointer: block.as_mut_ptr() as usize - 64,
                offset: 0,
                size: 64 + (block.len() * 4) as u64,
            },
        );
        let pipeline = vk::Pipeline::from_raw(pipeline);
        global.pipelines.insert(
            (device, pipeline),
            PipelineInfo {
                stages: [1, 2],
                scene: scene.then_some((2, 9)),
            },
        );
        let set = global.sets.entry((device, descriptor)).or_default();
        set.buffers.insert(
            (2, 0),
            BufferUse {
                buffer,
                offset: 0,
                range: (block.len() * 4) as u64,
                kind: vk::DescriptorType::UNIFORM_BUFFER,
            },
        );
        let record = global.recordings.entry((device, cmd)).or_default();
        record.area = Some(vk::Rect2D {
            extent: vk::Extent2D {
                width: 4320,
                height: 2160,
            },
            ..Default::default()
        });
        record.graphics[0] = descriptor;
        record.bound.push(descriptor);
        record.pipeline = pipeline;
        for (x, view) in [(0.0, left), (2160.0, right)] {
            global.recordings.get_mut(&(device, cmd)).unwrap().viewport = Some(vk::Viewport {
                x,
                width: 2160.0,
                y: 2160.0,
                height: -2160.0,
                ..Default::default()
            });
            global
                .sets
                .get_mut(&(device, descriptor))
                .unwrap()
                .bindings
                .insert(
                    // The third submission slot, as a following binding.
                    (11, 0),
                    ViewUse {
                        view,
                        layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    },
                );
            global.draw(device, cmd);
        }
        global.submit(device, [cmd].into_iter());
        global
    }

    fn block_with(bounds: [[f32; 4]; 2]) -> Vec<u32> {
        let mut block = vec![0_u32; 704];
        for (eye, bounds) in bounds.iter().enumerate() {
            let at = (spirv::bounds_offset(2, eye) / 4) as usize;
            for (index, value) in [1, 5, 9, 13].into_iter().zip(bounds) {
                block[at + index] = value.to_bits();
            }
        }
        block
    }

    #[test]
    fn scene_draws_report_each_eyes_texture_bounds() {
        let crop = [0.0, 0.0, 0.867, 1.0];
        let flip = [0.0, 1.0, 1.0, 0.0];
        let mut block = block_with([crop, flip]);
        let frame = scene(&mut block, true).frame.unwrap();
        assert_eq!(frame.touched_ids[0], 11);
        assert_eq!(frame.touched_eyes[0].layers, [0, 1]);
        assert_eq!(frame.touched_bounds[0].get(0), Some(crop));
        assert_eq!(frame.touched_bounds[0].get(1), Some(flip));

        // An unrecognised pipeline leaves the mapping unknown.
        let frame = scene(&mut block, false).frame.unwrap();
        assert_eq!(frame.touched_bounds[0], EyeBounds::default());

        // Implausible values are not reported.
        let mut block = block_with([[f32::NAN, 0.0, 1.0, 1.0], [0.5, 0.5, 0.5, 1.0]]);
        let frame = scene(&mut block, true).frame.unwrap();
        assert_eq!(frame.touched_bounds[0], EyeBounds::default());
    }
}
