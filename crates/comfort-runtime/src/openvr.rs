use libloading::Library;
use std::{
    error::Error,
    ffi::{CStr, CString, c_char, c_void},
    io,
    marker::PhantomData,
    path::Path,
    ptr::{self, NonNull},
    rc::Rc,
    sync::atomic::{AtomicBool, Ordering},
};

#[allow(
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    dead_code
)]
#[doc(hidden)]
pub mod ffi {
    include!(concat!(env!("OUT_DIR"), "/openvr.rs"));
}

type Init =
    unsafe extern "C" fn(*mut ffi::EVRInitError, ffi::EVRApplicationType, *const c_char) -> u32;
type Shutdown = unsafe extern "C" fn();
type GetInterface = unsafe extern "C" fn(*const c_char, *mut ffi::EVRInitError) -> *mut c_void;
type DescribeError = unsafe extern "C" fn(ffi::EVRInitError) -> *const c_char;

static INITIALIZED: AtomicBool = AtomicBool::new(false);

struct SessionGuard;
impl SessionGuard {
    fn acquire() -> io::Result<Self> {
        if INITIALIZED.swap(true, Ordering::AcqRel) {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "OpenVR already initialized",
            ));
        }
        Ok(Self)
    }
}

impl Drop for SessionGuard {
    fn drop(&mut self) {
        INITIALIZED.store(false, Ordering::Release);
    }
}

pub struct OpenVr {
    _library: Library,
    shutdown: Shutdown,
    get_interface: GetInterface,
    describe_error: DescribeError,
    _session: SessionGuard,
    _thread_bound: PhantomData<Rc<()>>,
}

impl OpenVr {
    /// Loads the OpenVR client and creates one overlay session in this process.
    ///
    /// # Safety
    /// The path must point to a trusted OpenVR library implementing the pinned ABI.
    pub unsafe fn init(path: &Path) -> Result<Self, Box<dyn Error>> {
        unsafe {
            Self::init_as(
                path,
                ffi::EVRApplicationType_EVRApplicationType_VRApplication_Overlay,
            )
        }
    }

    /// Initializes a read-only OpenVR client which does not start SteamVR.
    ///
    /// # Safety
    /// The path must point to a trusted OpenVR library implementing the pinned ABI.
    pub unsafe fn init_background(path: &Path) -> Result<Self, Box<dyn Error>> {
        unsafe {
            Self::init_as(
                path,
                ffi::EVRApplicationType_EVRApplicationType_VRApplication_Background,
            )
        }
    }

    /// Initializes an OpenVR scene application which can take scene focus.
    ///
    /// # Safety
    /// The path must point to a trusted OpenVR library implementing the pinned ABI.
    pub unsafe fn init_scene(path: &Path) -> Result<Self, Box<dyn Error>> {
        unsafe {
            Self::init_as(
                path,
                ffi::EVRApplicationType_EVRApplicationType_VRApplication_Scene,
            )
        }
    }

    unsafe fn init_as(
        path: &Path,
        application: ffi::EVRApplicationType,
    ) -> Result<Self, Box<dyn Error>> {
        let session = SessionGuard::acquire()?;
        // SAFETY: Only the installed OpenVR client (or a trusted fixture) should
        // be passed here. The pinned SDK specifies each symbol's ABI.
        unsafe {
            let library = Library::new(path)?;
            let init = *library.get::<Init>(b"VR_InitInternal2\0")?;
            let shutdown = *library.get::<Shutdown>(b"VR_ShutdownInternal\0")?;
            let get_interface = *library.get::<GetInterface>(b"VR_GetGenericInterface\0")?;
            let describe_error =
                *library.get::<DescribeError>(b"VR_GetVRInitErrorAsEnglishDescription\0")?;
            let mut error = 0;
            init(&mut error, application, ptr::null());
            if error != 0 {
                return Err(Self::init_error(describe_error, error).into());
            }
            Ok(Self {
                _library: library,
                shutdown,
                get_interface,
                describe_error,
                _session: session,
                _thread_bound: PhantomData,
            })
        }
    }

    fn init_error(describe: DescribeError, error: ffi::EVRInitError) -> io::Error {
        // SAFETY: OpenVR returns a static NUL-terminated description, or null.
        let detail = unsafe {
            describe(error)
                .as_ref()
                .map(|p| CStr::from_ptr(p).to_string_lossy())
        };
        io::Error::other(format!(
            "OpenVR error {error}: {}",
            detail.as_deref().unwrap_or("unknown error")
        ))
    }

    /// The caller selects the exact table type for the pinned interface version.
    unsafe fn table<T>(&self, version: &[u8]) -> io::Result<&T> {
        let version = CStr::from_bytes_with_nul(version).map_err(io::Error::other)?;
        let name = CString::new([b"FnTable:".as_slice(), version.to_bytes()].concat())?;
        let mut error = 0;
        // SAFETY: The name and error pointer are valid for this synchronous call.
        let table = unsafe { (self.get_interface)(name.as_ptr(), &mut error) };
        if error != 0 {
            return Err(Self::init_error(self.describe_error, error));
        }
        let table = NonNull::new(table.cast::<T>())
            .ok_or_else(|| io::Error::other("OpenVR returned no interface"))?;
        // SAFETY: The versioned table is owned by the active OpenVR session.
        Ok(unsafe { table.as_ref() })
    }

    pub fn recommended_size(&self) -> io::Result<[u32; 2]> {
        // SAFETY: The generated type corresponds to this pinned interface name.
        let table = unsafe { self.table::<ffi::VR_IVRSystem_FnTable>(ffi::IVRSystem_Version)? };
        let size = table
            .GetRecommendedRenderTargetSize
            .ok_or_else(|| io::Error::other("OpenVR render size function missing"))?;
        let (mut width, mut height) = (0, 0);
        // SAFETY: Both output pointers are live for the duration of the call.
        unsafe { size(&mut width, &mut height) };
        Ok([width, height])
    }

    pub fn create_overlay(&self, key: &CStr, name: &CStr) -> io::Result<Overlay<'_>> {
        // SAFETY: The generated type corresponds to this pinned interface name.
        let table = unsafe { self.table::<ffi::VR_IVROverlay_FnTable>(ffi::IVROverlay_Version)? };
        let create = table
            .CreateOverlay
            .ok_or_else(|| io::Error::other("OpenVR create overlay function missing"))?;
        let destroy = table
            .DestroyOverlay
            .ok_or_else(|| io::Error::other("OpenVR destroy overlay function missing"))?;
        let mut handle = 0;
        // The flat header uses mutable pointers for these strings. Give OpenVR
        // owned writable storage for the call.
        let mut key = key.to_bytes_with_nul().to_vec();
        let mut name = name.to_bytes_with_nul().to_vec();
        // SAFETY: All pointers are live for the duration of the call.
        let result = unsafe {
            create(
                key.as_mut_ptr().cast(),
                name.as_mut_ptr().cast(),
                &mut handle,
            )
        };
        if result != 0 {
            return Err(io::Error::other(format!(
                "OpenVR could not create overlay: {result}"
            )));
        }
        Ok(Overlay {
            _session: self,
            handle,
            destroy,
        })
    }
}

impl Drop for OpenVr {
    fn drop(&mut self) {
        // SAFETY: The client library is still loaded and the session guard is held.
        unsafe { (self.shutdown)() };
    }
}

pub struct Overlay<'vr> {
    _session: &'vr OpenVr,
    handle: ffi::VROverlayHandle_t,
    destroy: unsafe extern "C" fn(ffi::VROverlayHandle_t) -> ffi::EVROverlayError,
}

impl Drop for Overlay<'_> {
    fn drop(&mut self) {
        // SAFETY: This overlay owns the handle and borrows the live session.
        unsafe { (self.destroy)(self.handle) };
    }
}

pub fn vulkan_device_names() -> Result<Vec<String>, Box<dyn Error>> {
    // SAFETY: The Vulkan loader and the instance follow the ash ABI. The
    // instance is destroyed after enumeration, including when enumeration fails.
    unsafe {
        let entry = ash::Entry::load()?;
        let instance = entry.create_instance(&ash::vk::InstanceCreateInfo::default(), None)?;
        let devices = instance.enumerate_physical_devices().map(|devices| {
            devices
                .into_iter()
                .map(|device| {
                    let properties = instance.get_physical_device_properties(device);
                    CStr::from_ptr(properties.device_name.as_ptr())
                        .to_string_lossy()
                        .into_owned()
                })
                .collect()
        });
        instance.destroy_instance(None);
        Ok(devices?)
    }
}

use crate::vulkan::Context;
use ash::vk::{self, Handle};
use comfort_core::geometry::{Matrix34, Pose, Projection};
use std::mem::{MaybeUninit, size_of};

const STANDING: ffi::ETrackingUniverseOrigin =
    ffi::ETrackingUniverseOrigin_ETrackingUniverseOrigin_TrackingUniverseStanding;

fn overlay_result(code: ffi::EVROverlayError) -> io::Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(io::Error::other(format!("OpenVR overlay error {code}")))
    }
}

fn required<T>(entry: Option<T>, name: &'static str) -> io::Result<T> {
    entry.ok_or_else(|| io::Error::other(format!("OpenVR function missing: {name}")))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SystemEvent {
    Quit,
    TrackingChanged,
    Other,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OverlayEvent {
    Move([f32; 2]),
    Press([f32; 2]),
    Leave,
    Other,
}

pub struct Tracking {
    pub head: Pose,
    pub speed_deg_per_sec: f32,
}

pub struct RawEyeGeometry {
    pub eye_to_head: [[f32; 4]; 3],
    /// Left, right, raw top, raw bottom, in OpenVR's convention.
    pub tangents: [f32; 4],
}

impl OpenVr {
    fn system(&self) -> io::Result<&ffi::VR_IVRSystem_FnTable> {
        // SAFETY: Generated from the SDK that supplies the interface version.
        unsafe { self.table(ffi::IVRSystem_Version) }
    }

    fn overlay(&self) -> io::Result<&ffi::VR_IVROverlay_FnTable> {
        // SAFETY: Generated from the SDK that supplies the interface version.
        unsafe { self.table(ffi::IVROverlay_Version) }
    }

    fn compositor(&self) -> io::Result<&ffi::VR_IVRCompositor_FnTable> {
        // SAFETY: Generated from the SDK that supplies the interface version.
        unsafe { self.table(ffi::IVRCompositor_Version) }
    }

    pub fn tracking_space(&self) -> io::Result<ffi::ETrackingUniverseOrigin> {
        let get = required(self.compositor()?.GetTrackingSpace, "GetTrackingSpace")?;
        // SAFETY: The OpenVR session is live.
        Ok(unsafe { get() })
    }

    pub fn wait_scene_poses(
        &self,
    ) -> io::Result<(ffi::EVRCompositorError, ffi::TrackedDevicePose_t)> {
        let wait = required(self.compositor()?.WaitGetPoses, "WaitGetPoses")?;
        let mut poses = [unsafe { std::mem::zeroed() }; ffi::k_unMaxTrackedDeviceCount as usize];
        // SAFETY: OpenVR gets a writable array of the requested size.
        let result = unsafe { wait(poses.as_mut_ptr(), poses.len() as u32, ptr::null_mut(), 0) };
        Ok((result, poses[0]))
    }

    /// # Safety
    /// The caller must supply a live single-sample image in TRANSFER_SRC_OPTIMAL
    /// and preserve it and its Vulkan context until the runtime is done with them.
    pub unsafe fn submit_scene_eye(
        &self,
        eye: u32,
        context: &Context,
        image: vk::Image,
        extent: vk::Extent2D,
        format: vk::Format,
    ) -> io::Result<ffi::EVRCompositorError> {
        if eye > 1 {
            return Err(io::Error::other("invalid OpenVR eye index"));
        }
        let submit = required(self.compositor()?.Submit, "Submit")?;
        let (instance, physical, device, queue) = context.raw_handles();
        let mut data = ffi::VRVulkanTextureData_t {
            m_nImage: image.as_raw(),
            m_pDevice: device.as_raw() as usize as *mut _,
            m_pPhysicalDevice: physical.as_raw() as usize as *mut _,
            m_pInstance: instance.as_raw() as usize as *mut _,
            m_pQueue: queue.as_raw() as usize as *mut _,
            m_nQueueFamilyIndex: context.queue_family(),
            m_nWidth: extent.width,
            m_nHeight: extent.height,
            m_nFormat: format.as_raw() as u32,
            m_nSampleCount: 1,
        };
        let mut texture = ffi::Texture_t {
            handle: (&mut data as *mut ffi::VRVulkanTextureData_t).cast(),
            eType: ffi::ETextureType_ETextureType_TextureType_Vulkan,
            eColorSpace: ffi::EColorSpace_EColorSpace_ColorSpace_Auto,
        };
        // SAFETY: The caller supplies live resources and the metadata outlives the call.
        Ok(unsafe { submit(eye, &mut texture, ptr::null_mut(), 0) })
    }

    pub fn request_stereo_screenshot(&self, preview: &CStr, stereo: &CStr) -> io::Result<u32> {
        // SAFETY: The table matches the pinned interface version.
        let screenshots: &ffi::VR_IVRScreenshots_FnTable =
            unsafe { self.table(ffi::IVRScreenshots_Version)? };
        let request = required(screenshots.RequestScreenshot, "RequestScreenshot")?;
        let mut handle = 0;
        // The SDK uses non-const pointers for these filename arguments.
        let mut preview = preview.to_bytes_with_nul().to_vec();
        let mut stereo = stereo.to_bytes_with_nul().to_vec();
        // SAFETY: Both filenames are writable NUL-terminated strings for this call.
        let result = unsafe {
            request(
                &mut handle,
                ffi::EVRScreenshotType_EVRScreenshotType_VRScreenshotType_Stereo,
                preview.as_mut_ptr().cast(),
                stereo.as_mut_ptr().cast(),
            )
        };
        if result == 0 {
            Ok(handle)
        } else {
            Err(io::Error::other(format!(
                "OpenVR screenshot error {result}"
            )))
        }
    }

    pub fn raw_eye_geometry(&self, eye: u32) -> io::Result<RawEyeGeometry> {
        if eye > 1 {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "invalid eye"));
        }
        let system = self.system()?;
        let transform = required(system.GetEyeToHeadTransform, "GetEyeToHeadTransform")?;
        let projection = required(system.GetProjectionRaw, "GetProjectionRaw")?;
        let (mut left, mut right, mut top, mut bottom) = (0.0, 0.0, 0.0, 0.0);
        // SAFETY: The eye index is valid and all output pointers are live.
        unsafe { projection(eye, &mut left, &mut right, &mut top, &mut bottom) };
        Ok(RawEyeGeometry {
            eye_to_head: unsafe { transform(eye) }.m,
            tangents: [left, right, top, bottom],
        })
    }

    pub fn projection_matrix(&self, eye: u32) -> io::Result<[[f32; 4]; 4]> {
        if eye > 1 {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "invalid eye"));
        }
        let matrix = required(self.system()?.GetProjectionMatrix, "GetProjectionMatrix")?;
        // SAFETY: The index is valid and the clipping distances are positive.
        Ok(unsafe { matrix(eye, 0.1, 100.0) }.m)
    }

    pub fn eye_geometry(&self) -> io::Result<[(Pose, Projection); 2]> {
        let eye = |index| {
            let geometry = self.raw_eye_geometry(index)?;
            let [left, right, top, bottom] = geometry.tangents.map(f64::from);
            let raw = Projection::from_openvr_raw(left, right, top, bottom);
            let pose = Pose::from_matrix(geometry.eye_to_head)
                .ok_or_else(|| io::Error::other("invalid OpenVR eye transform"))?;
            if !raw.valid() {
                return Err(io::Error::other("invalid OpenVR projection"));
            }
            Ok((pose, raw))
        };
        Ok([eye(0)?, eye(1)?])
    }

    pub fn raw_tracking(&self, prediction: f32) -> io::Result<ffi::TrackedDevicePose_t> {
        let get = required(
            self.system()?.GetDeviceToAbsoluteTrackingPose,
            "GetDeviceToAbsoluteTrackingPose",
        )?;
        let mut pose = MaybeUninit::<ffi::TrackedDevicePose_t>::zeroed();
        // SAFETY: The runtime fills the one allocated HMD pose.
        Ok(unsafe {
            get(STANDING, prediction, pose.as_mut_ptr(), 1);
            pose.assume_init()
        })
    }

    pub fn tracking(&self, prediction: f32) -> Option<Tracking> {
        let pose = self.raw_tracking(prediction).ok()?;
        if !pose.bPoseIsValid
            || !pose.bDeviceIsConnected
            || pose.eTrackingResult
                != ffi::ETrackingResult_ETrackingResult_TrackingResult_Running_OK
        {
            return None;
        }
        let head = Pose::from_matrix(pose.mDeviceToAbsoluteTracking.m)?;
        let [x, y, z] = pose.vAngularVelocity.v;
        let speed = x.hypot(y).hypot(z).to_degrees();
        Some(Tracking {
            head,
            speed_deg_per_sec: if speed.is_finite() { speed } else { 1000.0 },
        })
    }

    pub fn display_timing(&self) -> (f32, f32) {
        let query = |property| {
            let read = self.system().ok()?.GetFloatTrackedDeviceProperty?;
            let mut error = 0;
            // SAFETY: HMD is tracked device zero and error points to a live value.
            let value = unsafe { read(0, property, &mut error) };
            (error == 0 && value.is_finite()).then_some(value)
        };
        let hz =
            query(ffi::ETrackedDeviceProperty_ETrackedDeviceProperty_Prop_DisplayFrequency_Float)
                .filter(|value| (30.0..=500.0).contains(value))
                .unwrap_or(90.0);
        let photons = query(
            ffi::ETrackedDeviceProperty_ETrackedDeviceProperty_Prop_SecondsFromVsyncToPhotons_Float,
        )
        .filter(|value| (0.0..0.1).contains(value))
        .unwrap_or(0.0);
        (1000.0 / hz, photons)
    }

    pub fn since_vsync(&self) -> f32 {
        self.vsync().map_or(0.0, |(since, _)| since)
    }

    pub fn vsync(&self) -> Option<(f32, u64)> {
        let get = self.system().ok()?.GetTimeSinceLastVsync?;
        let (mut since, mut counter) = (0.0_f32, 0);
        // SAFETY: The runtime writes two stack variables.
        (unsafe { get(&mut since, &mut counter) } && since.is_finite()).then_some((since, counter))
    }

    #[allow(non_upper_case_globals)] // Names come from the pinned Valve SDK.
    pub fn poll_system_event(&self) -> Option<SystemEvent> {
        let system = self.system().ok()?;
        let poll = system.PollNextEvent?;
        let mut event = MaybeUninit::<ffi::VREvent_t>::zeroed();
        // SAFETY: OpenVR receives the exact size of the generated event type.
        if !unsafe { poll(event.as_mut_ptr(), size_of::<ffi::VREvent_t>() as u32) } {
            return None;
        }
        let kind = unsafe { event.assume_init() }.eventType;
        use ffi::*;
        Some(match kind {
            EVREventType_EVREventType_VREvent_Quit => {
                if let Some(acknowledge) = system.AcknowledgeQuit_Exiting {
                    unsafe { acknowledge() };
                }
                SystemEvent::Quit
            }
            EVREventType_EVREventType_VREvent_ChaperoneUniverseHasChanged
            | EVREventType_EVREventType_VREvent_StandingZeroPoseReset
            | EVREventType_EVREventType_VREvent_SeatedZeroPoseReset
            | EVREventType_EVREventType_VREvent_IpdChanged => SystemEvent::TrackingChanged,
            _ => SystemEvent::Other,
        })
    }

    pub fn wait_frame(&self, timeout_ms: u32) -> io::Result<()> {
        let wait = required(self.overlay()?.WaitFrameSync, "WaitFrameSync")?;
        // SAFETY: The runtime session is live.
        overlay_result(unsafe { wait(timeout_ms) })
    }

    pub fn dashboard_visible(&self) -> io::Result<bool> {
        let visible = required(self.overlay()?.IsDashboardVisible, "IsDashboardVisible")?;
        // SAFETY: The runtime session is live.
        Ok(unsafe { visible() })
    }

    pub fn instance_extensions(&self) -> io::Result<Vec<CString>> {
        let get = required(
            self.compositor()?.GetVulkanInstanceExtensionsRequired,
            "GetVulkanInstanceExtensionsRequired",
        )?;
        extension_names(|bytes, size| {
            // SAFETY: extension_names supplies null with size zero or a valid writable buffer.
            unsafe { get(bytes, size) }
        })
    }

    /// # Safety
    /// The device must belong to a live Vulkan instance and represent the
    /// physical device to use with this OpenVR session.
    pub unsafe fn device_extensions(
        &self,
        physical: vk::PhysicalDevice,
    ) -> io::Result<Vec<CString>> {
        let get = required(
            self.compositor()?.GetVulkanDeviceExtensionsRequired,
            "GetVulkanDeviceExtensionsRequired",
        )?;
        extension_names(|bytes, size| unsafe {
            get(physical.as_raw() as usize as *mut _, bytes, size)
        })
    }

    pub fn dashboard(
        &self,
        key: &CStr,
        name: &CStr,
        width: u32,
        height: u32,
    ) -> io::Result<[Overlay<'_>; 2]> {
        let api = self.overlay()?;
        let create = required(api.CreateDashboardOverlay, "CreateDashboardOverlay")?;
        let destroy = required(api.DestroyOverlay, "DestroyOverlay")?;
        let mut key = key.to_bytes_with_nul().to_vec();
        let mut name = name.to_bytes_with_nul().to_vec();
        let [mut handle, mut thumbnail] = [0, 0];
        // SAFETY: All pointers reference live writable storage during the call.
        overlay_result(unsafe {
            create(
                key.as_mut_ptr().cast(),
                name.as_mut_ptr().cast(),
                &mut handle,
                &mut thumbnail,
            )
        })?;
        let panel = Overlay {
            _session: self,
            handle,
            destroy,
        };
        let thumbnail = Overlay {
            _session: self,
            handle: thumbnail,
            destroy,
        };
        let input = required(api.SetOverlayInputMethod, "SetOverlayInputMethod")?;
        let scale = required(api.SetOverlayMouseScale, "SetOverlayMouseScale")?;
        let mut pixels = ffi::HmdVector2_t {
            v: [width as f32, height as f32],
        };
        overlay_result(unsafe {
            input(handle, ffi::VROverlayInputMethod_VROverlayInputMethod_Mouse)
        })?;
        overlay_result(unsafe { scale(handle, &mut pixels) })?;
        panel.set_width(2.8)?;
        Ok([panel, thumbnail])
    }

    pub fn show_dashboard(&self, key: &CStr) -> io::Result<()> {
        let show = required(self.overlay()?.ShowDashboard, "ShowDashboard")?;
        let mut key = key.to_bytes_with_nul().to_vec();
        // SAFETY: The key is a writable, NUL-terminated buffer.
        unsafe { show(key.as_mut_ptr().cast()) };
        Ok(())
    }
}

fn extension_names(query: impl Fn(*mut c_char, u32) -> u32) -> io::Result<Vec<CString>> {
    let size = query(ptr::null_mut(), 0);
    if size == 0 {
        return Ok(Vec::new());
    }
    if size > 64 * 1024 {
        return Err(io::Error::other("OpenVR extension list too large"));
    }
    let mut bytes = vec![0; size as usize];
    if query(bytes.as_mut_ptr().cast(), size) > size {
        return Err(io::Error::other("OpenVR extension list changed"));
    }
    let names = CStr::from_bytes_until_nul(&bytes)
        .map_err(io::Error::other)?
        .to_str()
        .map_err(io::Error::other)?;
    names
        .split_ascii_whitespace()
        .map(|name| CString::new(name).map_err(io::Error::other))
        .collect()
}

impl Overlay<'_> {
    pub fn set_width(&self, width: f32) -> io::Result<()> {
        let set = required(
            self._session.overlay()?.SetOverlayWidthInMeters,
            "SetOverlayWidthInMeters",
        )?;
        overlay_result(unsafe { set(self.handle, width) })
    }

    pub fn stereo(&self, sort_order: u32) -> io::Result<()> {
        let api = self._session.overlay()?;
        let flag = required(api.SetOverlayFlag, "SetOverlayFlag")?;
        let order = required(api.SetOverlaySortOrder, "SetOverlaySortOrder")?;
        overlay_result(unsafe {
            flag(
                self.handle,
                ffi::VROverlayFlags_VROverlayFlags_SideBySide_Parallel,
                true,
            )
        })?;
        overlay_result(unsafe { order(self.handle, sort_order) })
    }

    pub fn set_sort_order(&self, sort_order: u32) -> io::Result<()> {
        let order = required(
            self._session.overlay()?.SetOverlaySortOrder,
            "SetOverlaySortOrder",
        )?;
        // SAFETY: This is a live overlay in the active session.
        overlay_result(unsafe { order(self.handle, sort_order) })
    }

    pub fn set_transform(&self, matrix: Matrix34, head_relative: bool) -> io::Result<()> {
        let api = self._session.overlay()?;
        let mut matrix = ffi::HmdMatrix34_t { m: matrix };
        // SAFETY: This is a live overlay and the transform stays valid during the call.
        let code = unsafe {
            if head_relative {
                required(
                    api.SetOverlayTransformTrackedDeviceRelative,
                    "SetOverlayTransformTrackedDeviceRelative",
                )?(self.handle, 0, &mut matrix)
            } else {
                required(
                    api.SetOverlayTransformAbsolute,
                    "SetOverlayTransformAbsolute",
                )?(self.handle, STANDING, &mut matrix)
            }
        };
        overlay_result(code)
    }

    pub fn visible(&self, visible: bool) -> io::Result<()> {
        let api = self._session.overlay()?;
        let show = if visible {
            required(api.ShowOverlay, "ShowOverlay")?
        } else {
            required(api.HideOverlay, "HideOverlay")?
        };
        overlay_result(unsafe { show(self.handle) })
    }

    pub fn clear(&self) -> io::Result<()> {
        let clear = required(
            self._session.overlay()?.ClearOverlayTexture,
            "ClearOverlayTexture",
        )?;
        overlay_result(unsafe { clear(self.handle) })
    }

    pub fn upload_rgba(&self, pixels: &mut [[u8; 4]], width: u32, height: u32) -> io::Result<()> {
        if width as usize * height as usize != pixels.len() {
            return Err(io::Error::other(
                "overlay pixel count differs from its extent",
            ));
        }
        let upload = required(self._session.overlay()?.SetOverlayRaw, "SetOverlayRaw")?;
        // SAFETY: The pixel array has exactly the byte count OpenVR consumes.
        overlay_result(unsafe { upload(self.handle, pixels.as_mut_ptr().cast(), width, height, 4) })
    }

    /// # Safety
    /// The image must belong to context and satisfy OpenVR's Vulkan texture
    /// contract. It must stay alive until OpenVR releases it. Submissions
    /// using the shared queue must be externally serialized.
    pub unsafe fn submit_image(
        &self,
        context: &crate::vulkan::Context,
        image: vk::Image,
        extent: vk::Extent2D,
    ) -> io::Result<()> {
        let submit = required(
            self._session.overlay()?.SetOverlayTexture,
            "SetOverlayTexture",
        )?;
        let (instance, physical, device, queue) = context.raw_handles();
        let mut data = ffi::VRVulkanTextureData_t {
            m_nImage: image.as_raw(),
            m_pDevice: device.as_raw() as usize as *mut _,
            m_pPhysicalDevice: physical.as_raw() as usize as *mut _,
            m_pInstance: instance.as_raw() as usize as *mut _,
            m_pQueue: queue.as_raw() as usize as *mut _,
            m_nQueueFamilyIndex: context.queue_family(),
            m_nWidth: extent.width,
            m_nHeight: extent.height,
            m_nFormat: vk::Format::R8G8B8A8_UNORM.as_raw() as u32,
            m_nSampleCount: 1,
        };
        let mut texture = ffi::Texture_t {
            handle: (&mut data as *mut ffi::VRVulkanTextureData_t).cast(),
            eType: ffi::ETextureType_ETextureType_TextureType_Vulkan,
            eColorSpace: ffi::EColorSpace_EColorSpace_ColorSpace_Gamma,
        };
        // SAFETY: Texture metadata is live during the call. The caller owns the GPU lifetime.
        overlay_result(unsafe { submit(self.handle, &mut texture) })
    }

    #[allow(non_upper_case_globals)] // Names come from the pinned Valve SDK.
    pub fn poll_dashboard_event(&self) -> Option<OverlayEvent> {
        let api = self._session.overlay().ok()?;
        let poll = api.PollNextOverlayEvent?;
        let mut event = MaybeUninit::<ffi::VREvent_t>::zeroed();
        if !unsafe {
            poll(
                self.handle,
                event.as_mut_ptr(),
                size_of::<ffi::VREvent_t>() as u32,
            )
        } {
            return None;
        }
        let event = unsafe { event.assume_init() };
        use ffi::*;
        Some(match event.eventType {
            EVREventType_EVREventType_VREvent_MouseMove => {
                let mouse = unsafe { event.data.mouse };
                OverlayEvent::Move([mouse.x, mouse.y])
            }
            EVREventType_EVREventType_VREvent_MouseButtonDown => {
                let mouse = unsafe { event.data.mouse };
                let active = api
                    .IsActiveDashboardOverlay
                    .is_some_and(|is_active| unsafe { is_active(self.handle) });
                if mouse.button == EVRMouseButton_EVRMouseButton_VRMouseButton_Left && active {
                    OverlayEvent::Press([mouse.x, mouse.y])
                } else {
                    OverlayEvent::Other
                }
            }
            EVREventType_EVREventType_VREvent_FocusLeave
            | EVREventType_EVREventType_VREvent_OverlayHidden => OverlayEvent::Leave,
            _ => OverlayEvent::Other,
        })
    }
}
