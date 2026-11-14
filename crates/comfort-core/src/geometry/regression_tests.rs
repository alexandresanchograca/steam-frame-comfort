use super::*;

fn rotation(axis: Vec3, degrees: f64) -> Quaternion {
    let (sin, cos) = (degrees.to_radians() / 2.0).sin_cos();
    Quaternion {
        w: cos,
        axis: axis * sin,
    }
}

fn frame_projections() -> [Projection; 2] {
    [
        Projection::from_openvr_raw(-1.66486, 1.21300, -1.72748, 1.17198),
        Projection::from_openvr_raw(-1.21868, 1.65399, -1.72807, 1.17120),
    ]
}

#[test]
fn culling_keeps_every_face_reached_by_either_eye() {
    const ALL: u32 = 0x3f;
    let bit = |face: Face| 1 << face as u32;
    let shell = Shell {
        pose: Pose {
            rotation: rotation(Y, 20.0),
            position: Vec3(0.3, 1.6, -0.2),
        },
        half_size: 4.0,
    };
    let eyes = |orientation, offset| {
        let head = Pose {
            rotation: shell.pose.rotation.then_local(orientation),
            position: shell.pose.position + offset,
        };
        let projections = frame_projections();
        std::array::from_fn(|eye| {
            let position = if eye == 0 { -0.0326 } else { 0.0326 };
            (
                head.compose(Pose {
                    position: Vec3(position, 0.0, 0.0),
                    ..Pose::default()
                }),
                projections[eye],
            )
        })
    };

    let forward = eyes(Quaternion::default(), Vec3::default());
    let backward = eyes(rotation(Y, 180.0), Vec3::default());
    let up = eyes(rotation(X, 50.0), Vec3::default());
    assert_eq!(shell.faces_in_view(forward, 0.0), ALL & !bit(Face::Back));
    assert_eq!(shell.faces_in_view(forward, 30.0), ALL & !bit(Face::Back));
    assert_eq!(shell.faces_in_view(backward, 30.0), ALL & !bit(Face::Front));
    assert_eq!(
        shell.faces_in_view(up, 30.0) & bit(Face::Top),
        bit(Face::Top)
    );
    assert_eq!(
        shell.faces_in_view(up, 30.0) & (bit(Face::Bottom) | bit(Face::Back)),
        0
    );
    assert_eq!(shell.faces_in_view(forward, f64::NAN), ALL);
    let mut invalid = forward;
    invalid[1].0.position.0 = f64::NAN;
    assert_eq!(shell.faces_in_view(invalid, 30.0), ALL);

    for yaw in (0..10).map(|step| f64::from(step) * 37.0) {
        for pitch in (-4..=4).map(|step| f64::from(step) * 20.0) {
            for roll in [-30.0, 0.0, 30.0] {
                let orientation =
                    rotation(Y, yaw).then_local(rotation(X, pitch).then_local(rotation(Z, roll)));
                let views = eyes(orientation, Vec3(0.4, -0.3, 0.5));
                let exact = shell.faces_in_view(views, 0.0);
                let widened = shell.faces_in_view(views, 30.0);
                assert_eq!(
                    exact & widened,
                    exact,
                    "yaw {yaw}, pitch {pitch}, roll {roll}"
                );
                for (eye, projection) in views {
                    let [width, height] = projection.span();
                    for x in 0..=8 {
                        for y in 0..=8 {
                            let ray = Vec3(
                                projection.left + width * f64::from(x) / 8.0,
                                -projection.top - height * f64::from(y) / 8.0,
                                -1.0,
                            );
                            let hit = shell
                                .intersect(eye.position, eye.rotation.rotate(ray))
                                .unwrap();
                            assert_ne!(
                                exact & bit(hit.face),
                                0,
                                "{hit:?}, yaw {yaw}, pitch {pitch}, roll {roll}"
                            );
                            let point = shell.face_point(hit.face, hit.uv).unwrap();
                            assert!((point - hit.point).length() < 1e-9, "{hit:?}");
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn matched_snapshots_respect_source_resolution_and_memory_limits() {
    let frame = frame_projections();
    assert_eq!(matched_face_size(frame, [1152, 1152], 256, 2048), Some(816));
    assert_eq!(matched_face_size(frame, [1152, 1152], 256, 512), Some(512));
    assert_eq!(matched_face_size(frame, [0, 1152], 256, 2048), None);
    assert_eq!(snapshot_eye_extent(frame, [2160, 2160], 816), [1184, 1184]);
    assert_eq!(snapshot_eye_extent(frame, [1152, 2160], 816), [1152, 1184]);
    assert_eq!(rgba_storage_bytes([1152; 2], [512; 2], 3), Some(58_982_400));
    assert_eq!(rgba_storage_bytes([0, 1], [1; 2], 1), None);
    assert_eq!(rgba_storage_bytes([u32::MAX; 2], [1; 2], 1), None);
}

#[test]
fn aperture_stays_symmetric_as_its_opacity_changes() {
    let head = Pose {
        rotation: rotation(Y, 46.0).then_local(rotation(X, 12.0)),
        position: Vec3(2.0, 1.6, 3.0),
    };
    let mut aperture = Aperture {
        half_width_tan: 0.6,
        half_height_tan: 0.5,
        feather: 0.1,
        strength: 0.4,
        ..Aperture::default()
    };
    for entry in [0.0, 0.5, 1.0] {
        aperture.entry = entry;
        for ipd in [-0.0326, 0.0326] {
            let eye = head.head_aligned_eye(Vec3(ipd, 0.0, 0.0));
            for sign in [-1.0, 1.0] {
                for point in [Vec3(sign * 2.0, 0.0, -4.0), Vec3(0.0, sign * 1.6, -4.0)] {
                    assert_eq!(
                        aperture.peripheral_alpha(eye, eye.transform(point)),
                        Some(0.0)
                    );
                }
                for point in [Vec3(sign * 3.0, 0.0, -4.0), Vec3(0.0, sign * 2.1, -4.0)] {
                    assert_eq!(
                        aperture.peripheral_alpha(eye, eye.transform(point)),
                        Some(0.4 * entry)
                    );
                }
            }
        }
    }
    let corner = Vec3(2.0, 1.8, -4.0);
    assert_eq!(
        aperture.peripheral_alpha(Pose::default(), corner),
        Some(0.4)
    );
    aperture.rectangle = true;
    assert_eq!(
        aperture.peripheral_alpha(Pose::default(), corner),
        Some(0.0)
    );
}

#[test]
fn tracked_pose_history_uses_sample_time_and_rejects_recentered_poses() {
    const MS: u64 = 1_000_000;
    let mut history = PoseHistory::new(120 * MS, 3);
    assert!(history.add(10 * MS, Pose::default(), 1));
    assert!(history.add(
        30 * MS,
        Pose {
            rotation: rotation(Y, 1.0),
            position: Vec3(0.2, 0.0, 0.0)
        },
        1
    ));
    assert!((history.angular_speed(30 * MS, 20 * MS, 1).unwrap() - 50.0).abs() < 1e-7);
    assert_eq!(
        estimated_pose_timestamp(30 * MS, 10 * MS as i64),
        Some(20 * MS)
    );
    assert_eq!(
        estimated_pose_timestamp(30 * MS, -(10 * MS as i64)),
        Some(40 * MS)
    );
    assert_eq!(estimated_pose_timestamp(30 * MS, 40 * MS as i64), None);
    assert_eq!(
        history.sample(20 * MS, 1).unwrap().position,
        Vec3(0.1, 0.0, 0.0)
    );
    assert!(history.sample(40 * MS, 1).is_none());

    assert!(history.add(40 * MS, Pose::default(), 2));
    assert!(history.sample(20 * MS, 1).is_none());
    assert!(history.sample(20 * MS, 2).is_none());
    assert!(history.angular_speed(40 * MS, 20 * MS, 2).is_none());
}
