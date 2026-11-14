#![cfg(target_os = "linux")]
#![allow(non_snake_case)] // OpenVR C ABI.

mod readback;

use comfort_runtime::openvr::ffi::*;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    env,
    ffi::{CStr, c_char, c_void},
    fs::OpenOptions,
    io::Write,
    ptr,
    sync::{Mutex, OnceLock},
    thread,
    time::{Duration, Instant},
};

const PERIOD: f64 = 1.0 / 90.0;
const BAD_PARAMETER: EVROverlayError =
    EVROverlayError_EVROverlayError_VROverlayError_InvalidParameter;
const CONTROLS: &str = "org.vrwindow.comfort.controls";

#[derive(Default)]
struct State {
    next: u64,
    overlays: BTreeMap<u64, Overlay>,
    panel_clicks: usize,
    mouse_moved: bool,
}

struct Overlay {
    key: String,
    submissions: u32,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

fn with_state<T>(f: impl FnOnce(&mut State) -> T) -> T {
    let mut state = STATE.lock().unwrap_or_else(|error| error.into_inner());
    f(state.get_or_insert_with(State::default))
}

fn now() -> f64 {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_secs_f64()
}

fn log(mut event: Value) {
    event["t"] = json!(now());
    if let Some(path) = env::var_os("FAKE_OPENVR_LOG")
        && let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path)
    {
        let _ = writeln!(file, "{event}");
    }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn VR_InitInternal2(
    error: *mut EVRInitError,
    kind: EVRApplicationType,
    _: *const c_char,
) -> u32 {
    unsafe { error.write(0) };
    log(json!({"call":"VR_Init","type":kind}));
    1
}

#[unsafe(no_mangle)]
extern "C" fn VR_ShutdownInternal() {
    log(json!({"call":"VR_Shutdown"}));
}

#[unsafe(no_mangle)]
extern "C" fn VR_GetVRInitErrorAsEnglishDescription(_: EVRInitError) -> *const c_char {
    c"Frame Comfort OpenVR fixture".as_ptr()
}

#[unsafe(no_mangle)]
unsafe extern "C" fn VR_GetGenericInterface(
    version: *const c_char,
    error: *mut EVRInitError,
) -> *mut c_void {
    let version = unsafe { CStr::from_ptr(version) };
    let name = version.to_bytes_with_nul().strip_prefix(b"FnTable:");
    let table = match name {
        Some(name) if name == IVRSystem_Version => ptr::from_ref(system()).cast(),
        Some(name) if name == IVRCompositor_Version => ptr::from_ref(compositor()).cast(),
        Some(name) if name == IVROverlay_Version => ptr::from_ref(overlay()).cast(),
        _ => ptr::null::<c_void>(),
    };
    if !error.is_null() {
        unsafe {
            error.write(if table.is_null() {
                EVRInitError_EVRInitError_VRInitError_Init_InterfaceNotFound
            } else {
                0
            })
        };
    }
    table.cast_mut()
}

fn system() -> &'static VR_IVRSystem_FnTable {
    static TABLE: OnceLock<VR_IVRSystem_FnTable> = OnceLock::new();
    TABLE.get_or_init(|| VR_IVRSystem_FnTable {
        GetRecommendedRenderTargetSize: Some(recommended_size),
        GetProjectionRaw: Some(projection),
        GetProjectionMatrix: Some(projection_matrix),
        GetEyeToHeadTransform: Some(eye_transform),
        GetTimeSinceLastVsync: Some(vsync),
        GetDeviceToAbsoluteTrackingPose: Some(tracking),
        GetFloatTrackedDeviceProperty: Some(float_property),
        PollNextEvent: Some(system_event),
        AcknowledgeQuit_Exiting: Some(acknowledge),
        // All remaining members are nullable function pointers.
        ..unsafe { std::mem::zeroed() }
    })
}

unsafe extern "C" fn recommended_size(width: *mut u32, height: *mut u32) {
    unsafe {
        width.write(1728);
        height.write(1728);
    }
}

unsafe extern "C" fn projection(
    eye: EVREye,
    left: *mut f32,
    right: *mut f32,
    top: *mut f32,
    bottom: *mut f32,
) {
    let [l, r, t, b] = if eye == 0 {
        [-1.66486, 1.21300, -1.72748, 1.17198]
    } else {
        [-1.21868, 1.65399, -1.72807, 1.17120]
    };
    unsafe {
        left.write(l);
        right.write(r);
        top.write(t);
        bottom.write(b);
    }
}

unsafe extern "C" fn eye_transform(eye: EVREye) -> HmdMatrix34_t {
    let x = if eye == 0 { -0.032574467 } else { 0.032574467 };
    HmdMatrix34_t {
        m: [
            [1.0, 0.0, 0.0, x],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
    }
}

unsafe extern "C" fn projection_matrix(eye: EVREye, near: f32, far: f32) -> HmdMatrix44_t {
    let [mut left, mut right, mut top, mut bottom] = [0.0; 4];
    // SAFETY: Each output points to a distinct stack value.
    unsafe { projection(eye, &mut left, &mut right, &mut top, &mut bottom) };
    let width = right - left;
    let height = bottom - top;
    HmdMatrix44_t {
        m: [
            [2.0 / width, 0.0, (right + left) / width, 0.0],
            [0.0, 2.0 / height, (bottom + top) / height, 0.0],
            [0.0, 0.0, -far / (far - near), -far * near / (far - near)],
            [0.0, 0.0, -1.0, 0.0],
        ],
    }
}

unsafe extern "C" fn vsync(seconds: *mut f32, frame: *mut u64) -> bool {
    let t = now();
    unsafe {
        seconds.write((t % PERIOD) as f32);
        if !frame.is_null() {
            frame.write((t / PERIOD) as u64);
        }
    }
    true
}

unsafe extern "C" fn tracking(
    origin: ETrackingUniverseOrigin,
    prediction: f32,
    poses: *mut TrackedDevicePose_t,
    count: u32,
) {
    if count == 0 {
        return;
    }
    unsafe { poses.write_bytes(0, count as usize) };
    if origin != ETrackingUniverseOrigin_ETrackingUniverseOrigin_TrackingUniverseStanding {
        return;
    }
    let phase = std::f32::consts::TAU * 0.1 * (now() as f32 + prediction);
    let (s, c) = (12.0_f32.to_radians() * phase.sin()).sin_cos();
    let yaw_speed = 12.0_f32.to_radians() * std::f32::consts::TAU * 0.1 * phase.cos();
    let pose = unsafe { &mut *poses };
    pose.mDeviceToAbsoluteTracking.m = [[c, 0.0, s, 0.2], [0.0, 1.0, 0.0, 1.6], [-s, 0.0, c, -0.1]];
    pose.vAngularVelocity.v = [0.0, yaw_speed, 0.0];
    pose.eTrackingResult = ETrackingResult_ETrackingResult_TrackingResult_Running_OK;
    pose.bPoseIsValid = true;
    pose.bDeviceIsConnected = true;
}

#[allow(non_upper_case_globals)]
unsafe extern "C" fn float_property(
    _: u32,
    property: ETrackedDeviceProperty,
    error: *mut ETrackedPropertyError,
) -> f32 {
    let (value, code) = match property {
        ETrackedDeviceProperty_ETrackedDeviceProperty_Prop_DisplayFrequency_Float => (90.0, 0),
        ETrackedDeviceProperty_ETrackedDeviceProperty_Prop_SecondsFromVsyncToPhotons_Float => {
            (0.011, 0)
        }
        _ => (
            0.0,
            ETrackedPropertyError_ETrackedPropertyError_TrackedProp_UnknownProperty,
        ),
    };
    if !error.is_null() {
        unsafe { error.write(code) };
    }
    value
}

unsafe extern "C" fn system_event(_: *mut VREvent_t, _: u32) -> bool {
    false
}
unsafe extern "C" fn acknowledge() {}

fn compositor() -> &'static VR_IVRCompositor_FnTable {
    static TABLE: OnceLock<VR_IVRCompositor_FnTable> = OnceLock::new();
    TABLE.get_or_init(|| VR_IVRCompositor_FnTable {
        GetVulkanInstanceExtensionsRequired: Some(extensions),
        GetVulkanDeviceExtensionsRequired: Some(device_extensions),
        // All remaining members are nullable function pointers.
        ..unsafe { std::mem::zeroed() }
    })
}

unsafe extern "C" fn extensions(out: *mut c_char, size: u32) -> u32 {
    if size != 0 && !out.is_null() {
        unsafe { out.write(0) };
    }
    1
}

unsafe extern "C" fn device_extensions(
    _: *mut VkPhysicalDevice_T,
    out: *mut c_char,
    size: u32,
) -> u32 {
    unsafe { extensions(out, size) }
}

fn overlay() -> &'static VR_IVROverlay_FnTable {
    static TABLE: OnceLock<VR_IVROverlay_FnTable> = OnceLock::new();
    TABLE.get_or_init(|| VR_IVROverlay_FnTable {
        CreateOverlay: Some(create_overlay),
        CreateDashboardOverlay: Some(create_dashboard),
        DestroyOverlay: Some(destroy),
        ShowOverlay: Some(show),
        HideOverlay: Some(hide),
        ClearOverlayTexture: Some(clear),
        SetOverlaySortOrder: Some(sort_order),
        SetOverlayFlag: Some(flag),
        SetOverlayWidthInMeters: Some(width),
        SetOverlayInputMethod: Some(input_method),
        SetOverlayMouseScale: Some(mouse_scale),
        SetOverlayTransformAbsolute: Some(transform_absolute),
        SetOverlayTransformTrackedDeviceRelative: Some(transform_relative),
        SetOverlayTexture: Some(set_texture),
        SetOverlayRaw: Some(set_raw),
        PollNextOverlayEvent: Some(poll_overlay),
        IsDashboardVisible: Some(dashboard_visible),
        IsActiveDashboardOverlay: Some(active_dashboard),
        ShowDashboard: Some(show_dashboard),
        WaitFrameSync: Some(wait_frame),
        // All remaining members are nullable function pointers.
        ..unsafe { std::mem::zeroed() }
    })
}

fn allocate(key: String) -> u64 {
    with_state(|state| {
        state.next += 1;
        state.overlays.insert(
            state.next,
            Overlay {
                key,
                submissions: 0,
            },
        );
        state.next
    })
}

unsafe extern "C" fn create_overlay(
    key: *mut c_char,
    _: *mut c_char,
    handle: *mut u64,
) -> EVROverlayError {
    let key = unsafe { CStr::from_ptr(key) }
        .to_string_lossy()
        .into_owned();
    let value = allocate(key.clone());
    unsafe { handle.write(value) };
    log(json!({"call":"CreateOverlay","key":key,"overlay":value}));
    0
}

unsafe extern "C" fn create_dashboard(
    key: *mut c_char,
    _: *mut c_char,
    handle: *mut u64,
    thumb: *mut u64,
) -> EVROverlayError {
    let key = unsafe { CStr::from_ptr(key) }
        .to_string_lossy()
        .into_owned();
    let value = allocate(key.clone());
    unsafe {
        handle.write(value);
        thumb.write(allocate(format!("{key}.thumb")));
    }
    log(json!({"call":"CreateDashboardOverlay","key":key,"overlay":value}));
    0
}

fn event(handle: u64, mut fields: Value) -> EVROverlayError {
    let key = with_state(|state| {
        state
            .overlays
            .get(&handle)
            .map(|overlay| overlay.key.clone())
    });
    if let Some(key) = key {
        fields["key"] = json!(key);
        log(fields);
        0
    } else {
        BAD_PARAMETER
    }
}

unsafe extern "C" fn destroy(handle: u64) -> EVROverlayError {
    let status = event(handle, json!({"call":"DestroyOverlay"}));
    with_state(|state| state.overlays.remove(&handle));
    status
}
unsafe extern "C" fn show(handle: u64) -> EVROverlayError {
    event(handle, json!({"call":"ShowOverlay"}))
}
unsafe extern "C" fn hide(handle: u64) -> EVROverlayError {
    event(handle, json!({"call":"HideOverlay"}))
}
unsafe extern "C" fn clear(handle: u64) -> EVROverlayError {
    event(handle, json!({"call":"ClearOverlayTexture"}))
}
unsafe extern "C" fn sort_order(handle: u64, order: u32) -> EVROverlayError {
    event(handle, json!({"call":"SetOverlaySortOrder","order":order}))
}
unsafe extern "C" fn flag(handle: u64, flag: VROverlayFlags, enabled: bool) -> EVROverlayError {
    event(
        handle,
        json!({"call":"SetOverlayFlag","flag":flag,"on":enabled}),
    )
}
unsafe extern "C" fn width(handle: u64, width: f32) -> EVROverlayError {
    event(
        handle,
        json!({"call":"SetOverlayWidthInMeters","width":width}),
    )
}
unsafe extern "C" fn input_method(handle: u64, method: VROverlayInputMethod) -> EVROverlayError {
    event(
        handle,
        json!({"call":"SetOverlayInputMethod","method":method}),
    )
}
unsafe extern "C" fn mouse_scale(handle: u64, scale: *mut HmdVector2_t) -> EVROverlayError {
    let [w, h] = unsafe { (*scale).v };
    event(handle, json!({"call":"SetOverlayMouseScale","w":w,"h":h}))
}
unsafe extern "C" fn transform_absolute(
    handle: u64,
    origin: ETrackingUniverseOrigin,
    matrix: *mut HmdMatrix34_t,
) -> EVROverlayError {
    let matrix = unsafe { (*matrix).m };
    event(
        handle,
        json!({"call":"SetOverlayTransformAbsolute","origin":origin,"m":matrix}),
    )
}
unsafe extern "C" fn transform_relative(
    handle: u64,
    device: u32,
    matrix: *mut HmdMatrix34_t,
) -> EVROverlayError {
    let matrix = unsafe { (*matrix).m };
    event(
        handle,
        json!({"call":"SetOverlayTransformTrackedDeviceRelative","device":device,"m":matrix}),
    )
}

fn submission(handle: u64) -> Option<(String, u32)> {
    with_state(|state| {
        let overlay = state.overlays.get_mut(&handle)?;
        overlay.submissions += 1;
        Some((overlay.key.clone(), overlay.submissions))
    })
}

unsafe extern "C" fn set_texture(handle: u64, texture: *mut Texture_t) -> EVROverlayError {
    let Some((key, n)) = submission(handle) else {
        return BAD_PARAMETER;
    };
    let texture = unsafe { &*texture };
    if texture.eType != ETextureType_ETextureType_TextureType_Vulkan || texture.handle.is_null() {
        return BAD_PARAMETER;
    }
    let data = unsafe { &*texture.handle.cast::<VRVulkanTextureData_t>() };
    if n == 1 || n.is_multiple_of(90) {
        log(
            json!({"call":"SetOverlayTexture","key":key,"n":n,"w":data.m_nWidth,"h":data.m_nHeight,
            "format":data.m_nFormat,"color":texture.eColorSpace}),
        );
    }
    if key == CONTROLS
        && let Some(mut prefix) = env::var_os("FAKE_OPENVR_PANEL_DUMP")
        && let Some(tab) = ["opening", "timing", "extras"].get(n as usize - 1)
    {
        prefix.push(format!("-{tab}-{}x{}.rgba", data.m_nWidth, data.m_nHeight));
        if let Err(error) = unsafe { readback::save(data, prefix.as_ref()) } {
            log(json!({"call":"readBack","error":error.to_string()}));
            return BAD_PARAMETER;
        }
    }
    if let Some(prefix) = env::var_os("FAKE_OPENVR_DUMP") {
        let suffix = if key.contains("restframe.0.") && matches!(n, 30 | 120) {
            Some(n.to_string())
        } else if key.contains(".nose.") && n == 1 {
            Some("nose".into())
        } else {
            None
        };
        if let Some(suffix) = suffix {
            let mut path = prefix;
            path.push(format!(
                "-{suffix}-{}x{}.rgba",
                data.m_nWidth, data.m_nHeight
            ));
            if let Err(error) = unsafe { readback::save(data, path.as_ref()) } {
                log(json!({"call":"readBack","error":error.to_string()}));
                return BAD_PARAMETER;
            }
        }
    }
    0
}

unsafe extern "C" fn set_raw(
    handle: u64,
    _pixels: *mut c_void,
    w: u32,
    h: u32,
    bytes: u32,
) -> EVROverlayError {
    let Some((key, n)) = submission(handle) else {
        return BAD_PARAMETER;
    };
    log(json!({"call":"SetOverlayRaw","key":key,"n":n,"w":w,"h":h,"bytesPerPixel":bytes}));
    0
}

unsafe extern "C" fn dashboard_visible() -> bool {
    (env::var_os("FAKE_OPENVR_DASHBOARD").is_some()
        || env::var_os("FAKE_OPENVR_PANEL_DUMP").is_some())
        && (4.9..5.4).contains(&now())
}
unsafe extern "C" fn active_dashboard(handle: u64) -> bool {
    (unsafe { dashboard_visible() })
        && with_state(|state| {
            state
                .overlays
                .get(&handle)
                .is_some_and(|overlay| overlay.key == CONTROLS)
        })
}
unsafe extern "C" fn show_dashboard(_: *mut c_char) {
    log(json!({"call":"ShowDashboard"}));
}
unsafe extern "C" fn poll_overlay(handle: u64, out: *mut VREvent_t, size: u32) -> bool {
    if size as usize != size_of::<VREvent_t>()
        || env::var_os("FAKE_OPENVR_PANEL_DUMP").is_none()
        || !unsafe { active_dashboard(handle) }
    {
        return false;
    }
    with_state(|state| {
        let Some(&x) = [261.0, 421.0].get(state.panel_clicks) else {
            return false;
        };
        if state.overlays[&handle].submissions as usize != state.panel_clicks + 1 {
            return false;
        }
        // Emit cursor movement separately, as real runtimes may omit click coordinates.
        let mut value: VREvent_t = unsafe { std::mem::zeroed() };
        value.data.mouse = VREvent_Mouse_t {
            x: 0.0,
            y: 0.0,
            button: 0,
            cursorIndex: 0,
        };
        if state.mouse_moved {
            value.eventType = EVREventType_EVREventType_VREvent_MouseButtonDown;
            value.data.mouse.button = EVRMouseButton_EVRMouseButton_VRMouseButton_Left;
            state.panel_clicks += 1;
        } else {
            value.eventType = EVREventType_EVREventType_VREvent_MouseMove;
            value.data.mouse = VREvent_Mouse_t {
                x,
                y: 539.0,
                button: 0,
                cursorIndex: 0,
            };
        }
        unsafe { out.write(value) };
        state.mouse_moved = !state.mouse_moved;
        true
    })
}

unsafe extern "C" fn wait_frame(timeout_ms: u32) -> EVROverlayError {
    let delta = PERIOD - now() % PERIOD;
    if delta > f64::from(timeout_ms) / 1000.0 {
        return EVROverlayError_EVROverlayError_VROverlayError_TimedOut;
    }
    thread::sleep(Duration::from_secs_f64(delta));
    0
}
