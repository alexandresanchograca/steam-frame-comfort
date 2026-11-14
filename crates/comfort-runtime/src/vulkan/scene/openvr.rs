use super::{Context, EyeTarget, MARKER_WIDTH, Result, SceneImage, ScenePainter};
use crate::{
    mailbox::monotonic_ns,
    openvr::{OpenVr, SystemEvent},
    vulkan::ContextOptions,
};
use ash::vk;
use serde_json::json;
use std::{
    io::{self, Write},
    path::PathBuf,
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

pub struct EyePatternOptions {
    pub library: PathBuf,
    pub seconds: Duration,
    pub bar: bool,
    pub markers: bool,
    pub frame_code: bool,
    pub trace_truth: bool,
}

fn context(vr: &OpenVr) -> Result<Arc<Context>> {
    let instance_extensions = vr.instance_extensions()?;
    let instance_extensions: Vec<_> = instance_extensions
        .iter()
        .map(|ext| ext.as_c_str())
        .collect();
    let context = Context::new(ContextOptions {
        instance_extensions: &instance_extensions,
        ..Default::default()
    })?;
    let (_, physical, _, _) = context.raw_handles();
    // SAFETY: This physical device belongs to the live context.
    let device_extensions = unsafe { vr.device_extensions(physical)? };
    if device_extensions.is_empty() {
        return Ok(context);
    }
    let device_extensions: Vec<_> = device_extensions.iter().map(|ext| ext.as_c_str()).collect();
    Context::new(ContextOptions {
        instance_extensions: &instance_extensions,
        device_extensions: &device_extensions,
        device_uuid: Some(context.device_uuid()),
        ..Default::default()
    })
}

pub fn run_openvr(options: EyePatternOptions) -> Result<()> {
    // Keep both images and their device alive until OpenVR has released them,
    // including when a frame fails after submission.
    let images;
    // SAFETY: The standalone diagnostic uses the installed SteamVR client
    // or the OpenVR library explicitly chosen by its operator.
    let vr = unsafe { OpenVr::init_scene(&options.library)? };
    let context = context(&vr)?;
    let [width, height] = vr.recommended_size()?;
    let extent = vk::Extent2D { width, height };
    let format = vk::Format::R8G8B8A8_SRGB;
    images = [
        SceneImage::new(&context, extent, 1, format)?,
        SceneImage::new(&context, extent, 1, format)?,
    ];
    let mut painter = ScenePainter::new(&context, extent, format, options.frame_code)?;
    let projection = if options.markers || options.trace_truth {
        [
            vr.raw_eye_geometry(0)?.tangents,
            vr.raw_eye_geometry(1)?.tangents,
        ]
    } else {
        [[0.0; 4]; 2]
    };
    let tracking_space = options
        .trace_truth
        .then(|| vr.tracking_space())
        .transpose()?;
    let markers = options.markers.then(|| {
        std::array::from_fn(|eye| {
            let [left, right, top, bottom] = projection[eye];
            let u = -left / (right - left);
            let v = bottom / (bottom - top);
            let offset = |fraction: f32, size: u32| {
                ((fraction * size as f32).round() as i64 - i64::from(MARKER_WIDTH / 2))
                    .clamp(0, i64::from(size - MARKER_WIDTH)) as u32
            };
            let [x, y] = [offset(u, width), offset(v, height)];
            println!(
                "{}",
                json!({"eye":if eye == 0 {"left"} else {"right"},"projection":projection[eye],
                "forwardU":u,"forwardV":v,"markerX":x,"markerY":y})
            );
            [x, y]
        })
    });
    let eyes = images.each_ref().map(|image| EyeTarget {
        image: image.handle,
        x: 0,
        layer: 0,
    });
    println!(
        "local eye pattern: {width}x{height} per eye, red LEFT, blue RIGHT, {:.0}s",
        options.seconds.as_secs_f64()
    );
    io::stdout().flush()?;
    let end = Instant::now() + options.seconds;
    let mut frames = 0_u32;
    let mut errors = 0_u32;
    let mut truth_records = 0;
    let release = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
    let dynamic = options.bar || options.markers || options.frame_code;
    while Instant::now() < end {
        if std::iter::from_fn(|| vr.poll_system_event())
            .any(|event| matches!(event, SystemEvent::Quit))
        {
            break;
        }
        let (wait_result, pose) = vr.wait_scene_poses()?;
        if frames == 0 || dynamic {
            let old = if frames == 0 {
                vk::ImageLayout::UNDEFINED
            } else {
                release
            };
            // SAFETY: These are our images. OpenVR's prior reads were
            // submitted on the same queue and the barrier orders the rewrite.
            unsafe { painter.paint(eyes, frames, options.bar, markers, old, release)? };
        }
        let record = options.trace_truth
            && truth_records < 256
            && (frames < 32 || frames.is_multiple_of(16));
        let mut submit_ns = [0; 2];
        let mut results = [0; 2];
        for (eye, target) in eyes.iter().enumerate() {
            if record {
                submit_ns[eye] = monotonic_ns();
            }
            // SAFETY: The transfers completed and left both images in
            // TRANSFER_SRC_OPTIMAL. Resources outlive OpenVR.
            results[eye] =
                unsafe { vr.submit_scene_eye(eye as u32, &context, target.image, extent, format)? };
            errors += u32::from(results[eye] != 0);
        }
        if record {
            let valid = wait_result == 0 && pose.bPoseIsValid;
            let eyes: [_; 2] = std::array::from_fn(|eye| {
                json!({"eye":eye,"image_index":eye,"array_layer":0,"rect":[0,0,width,height],
                    "monotonic_ns":submit_ns[eye],"result":results[eye],
                    "runtime_projection_raw_left_right_top_bottom":projection[eye]})
            });
            println!(
                "{}",
                json!({"fixture_submit":"openvr","frame":frames,
                "frame_code":options.frame_code.then_some(frames), "monotonic_ns":submit_ns[0],
                "initial_tracking_space":tracking_space,"wait_get_poses_result":wait_result,
                "wait_hmd_pose_valid":valid,
                "wait_hmd_to_tracking_3x4":valid.then_some(pose.mDeviceToAbsoluteTracking.m),"eyes":eyes})
            );
            truth_records += 1;
        }
        frames = frames.wrapping_add(1);
        if wait_result != 0 {
            // A scene without focus can return immediately from WaitGetPoses.
            thread::sleep(Duration::from_millis(10));
        }
    }
    context.wait_idle()?;
    println!("frames={frames} submitErrors={errors}");
    if options.frame_code {
        println!(
            "{}",
            json!({"frameCodeRange":{"count":frames,"first":0,"last":frames.saturating_sub(1),"wrapBits":32}})
        );
    }
    if errors > 0 {
        return Err(io::Error::other(format!("OpenVR reported {errors} submission errors")).into());
    }
    Ok(())
}
