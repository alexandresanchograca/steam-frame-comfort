//! World-space geometry. Camera forward is -Z and image V increases downward.
use std::{
    collections::VecDeque,
    ops::{Add, Mul, Sub},
};

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Vec3(pub f64, pub f64, pub f64);

impl Add for Vec3 {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0, self.1 + other.1, self.2 + other.2)
    }
}
impl Sub for Vec3 {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self(self.0 - other.0, self.1 - other.1, self.2 - other.2)
    }
}
impl Mul<f64> for Vec3 {
    type Output = Self;
    fn mul(self, scale: f64) -> Self {
        Self(self.0 * scale, self.1 * scale, self.2 * scale)
    }
}

impl Vec3 {
    pub fn dot(self, other: Self) -> f64 {
        self.0 * other.0 + self.1 * other.1 + self.2 * other.2
    }
    pub fn cross(self, other: Self) -> Self {
        Self(
            self.1 * other.2 - self.2 * other.1,
            self.2 * other.0 - self.0 * other.2,
            self.0 * other.1 - self.1 * other.0,
        )
    }
    pub fn length(self) -> f64 {
        self.0.hypot(self.1).hypot(self.2)
    }
    pub fn finite(self) -> bool {
        self.0.is_finite() && self.1.is_finite() && self.2.is_finite()
    }
    fn max_abs(self) -> f64 {
        self.0.abs().max(self.1.abs()).max(self.2.abs())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quaternion {
    pub w: f64,
    pub axis: Vec3,
}

impl Default for Quaternion {
    fn default() -> Self {
        Self {
            w: 1.0,
            axis: Vec3::default(),
        }
    }
}

impl Quaternion {
    pub fn dot(self, other: Self) -> f64 {
        self.w * other.w + self.axis.dot(other.axis)
    }
    pub fn valid(self) -> bool {
        (self.dot(self) - 1.0).abs() <= 1e-5
    }
    pub fn normalized(self) -> Option<Self> {
        let norm = self.dot(self).sqrt();
        (norm.is_finite() && norm >= 1e-12).then(|| self.scaled(1.0 / norm))
    }
    fn scaled(self, scale: f64) -> Self {
        Self {
            w: self.w * scale,
            axis: self.axis * scale,
        }
    }
    pub fn conjugate(self) -> Self {
        Self {
            w: self.w,
            axis: self.axis * -1.0,
        }
    }
    pub fn rotate(self, point: Vec3) -> Vec3 {
        let t = self.axis.cross(point) * 2.0;
        point + t * self.w + self.axis.cross(t)
    }
    pub fn then_local(self, local: Self) -> Self {
        Self {
            w: self.w * local.w - self.axis.dot(local.axis),
            axis: local.axis * self.w + self.axis * local.w + self.axis.cross(local.axis),
        }
    }
    pub fn angle_to(self, other: Self) -> Option<f64> {
        (self.valid() && other.valid()).then(|| 2.0 * self.dot(other).abs().clamp(0.0, 1.0).acos())
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Pose {
    pub rotation: Quaternion,
    pub position: Vec3,
}

pub type Matrix34 = [[f32; 4]; 3];

impl Pose {
    pub fn valid(self) -> bool {
        self.rotation.valid() && self.position.finite()
    }
    pub fn transform(self, point: Vec3) -> Vec3 {
        self.rotation.rotate(point) + self.position
    }
    pub fn inverse_transform(self, point: Vec3) -> Vec3 {
        self.rotation.conjugate().rotate(point - self.position)
    }

    pub fn compose(self, local: Self) -> Self {
        Self {
            rotation: self.rotation.then_local(local.rotation),
            position: self.transform(local.position),
        }
    }

    pub fn inverse(self) -> Self {
        let rotation = self.rotation.conjugate();
        Self {
            rotation,
            position: rotation.rotate(self.position) * -1.0,
        }
    }

    pub fn interpolate(self, other: Self, t: f64) -> Option<Self> {
        if !self.valid() || !other.valid() || !(0.0..=1.0).contains(&t) {
            return None;
        }
        let mut second = other.rotation;
        let mut cosine = self.rotation.dot(second);
        if cosine < 0.0 {
            second = second.scaled(-1.0);
            cosine = -cosine;
        }
        let (u, v) = if cosine < 0.9995 {
            let angle = cosine.clamp(-1.0, 1.0).acos();
            (
                ((1.0 - t) * angle).sin() / angle.sin(),
                (t * angle).sin() / angle.sin(),
            )
        } else {
            (1.0 - t, t)
        };
        let rotation = Quaternion {
            w: u * self.rotation.w + v * second.w,
            axis: self.rotation.axis * u + second.axis * v,
        };
        Some(Self {
            rotation: rotation.normalized()?,
            position: self.position * (1.0 - t) + other.position * t,
        })
    }

    pub fn from_matrix(matrix: Matrix34) -> Option<Self> {
        if !matrix.iter().flatten().all(|v| v.is_finite()) {
            return None;
        }
        let r = matrix.map(|row| row.map(f64::from));
        let columns: [Vec3; 3] = std::array::from_fn(|c| Vec3(r[0][c], r[1][c], r[2][c]));
        for (i, &a) in columns.iter().enumerate() {
            for (j, &b) in columns.iter().enumerate().skip(i) {
                let expected = f64::from(i == j);
                if (a.dot(b) - expected).abs() > 1e-3 {
                    return None;
                }
            }
        }
        if (columns[0].dot(columns[1].cross(columns[2])) - 1.0).abs() > 1e-3 {
            return None;
        }
        let trace = r[0][0] + r[1][1] + r[2][2];
        let (w, x, y, z) = if trace > 0.0 {
            let s = (trace + 1.0).sqrt() * 2.0;
            (
                0.25 * s,
                (r[2][1] - r[1][2]) / s,
                (r[0][2] - r[2][0]) / s,
                (r[1][0] - r[0][1]) / s,
            )
        } else if r[0][0] > r[1][1] && r[0][0] > r[2][2] {
            let s = (1.0 + r[0][0] - r[1][1] - r[2][2]).sqrt() * 2.0;
            (
                (r[2][1] - r[1][2]) / s,
                0.25 * s,
                (r[0][1] + r[1][0]) / s,
                (r[0][2] + r[2][0]) / s,
            )
        } else if r[1][1] > r[2][2] {
            let s = (1.0 + r[1][1] - r[0][0] - r[2][2]).sqrt() * 2.0;
            (
                (r[0][2] - r[2][0]) / s,
                (r[0][1] + r[1][0]) / s,
                0.25 * s,
                (r[1][2] + r[2][1]) / s,
            )
        } else {
            let s = (1.0 + r[2][2] - r[0][0] - r[1][1]).sqrt() * 2.0;
            (
                (r[1][0] - r[0][1]) / s,
                (r[0][2] + r[2][0]) / s,
                (r[1][2] + r[2][1]) / s,
                0.25 * s,
            )
        };
        Some(Self {
            rotation: Quaternion {
                w,
                axis: Vec3(x, y, z),
            }
            .normalized()?,
            position: Vec3(r[0][3], r[1][3], r[2][3]),
        })
    }

    pub fn matrix(self) -> Matrix34 {
        matrix_columns(
            self.rotation.rotate(X),
            self.rotation.rotate(Y),
            self.rotation.rotate(Z),
            self.position,
        )
    }

    pub fn head_aligned_eye(self, eye_position: Vec3) -> Self {
        Self {
            rotation: self.rotation,
            position: self.transform(eye_position),
        }
    }
}

const X: Vec3 = Vec3(1.0, 0.0, 0.0);
const Y: Vec3 = Vec3(0.0, 1.0, 0.0);
const Z: Vec3 = Vec3(0.0, 0.0, 1.0);

fn matrix_columns(x: Vec3, y: Vec3, z: Vec3, p: Vec3) -> Matrix34 {
    [
        [x.0 as f32, y.0 as f32, z.0 as f32, p.0 as f32],
        [x.1 as f32, y.1 as f32, z.1 as f32, p.1 as f32],
        [x.2 as f32, y.2 as f32, z.2 as f32, p.2 as f32],
    ]
}

pub struct PoseHistory {
    samples: VecDeque<(u64, Pose)>,
    duration_ns: u64,
    capacity: usize,
    epoch: u64,
}

impl Default for PoseHistory {
    fn default() -> Self {
        Self::new(1_000_000_000, 512)
    }
}

impl PoseHistory {
    pub fn new(duration_ns: u64, capacity: usize) -> Self {
        Self {
            samples: VecDeque::new(),
            duration_ns,
            capacity: capacity.max(2),
            epoch: 0,
        }
    }

    pub fn reset(&mut self, epoch: u64) {
        self.invalidate();
        self.epoch = epoch;
    }

    pub fn invalidate(&mut self) {
        self.samples.clear();
    }

    pub fn add(&mut self, ns: u64, pose: Pose, epoch: u64) -> bool {
        if epoch != self.epoch {
            self.reset(epoch);
        }
        if !pose.valid() || self.samples.back().is_some_and(|&(last, _)| ns <= last) {
            self.invalidate();
            return false;
        }
        self.samples.push_back((ns, pose));
        while self.samples.len() > self.capacity
            || (self.samples.len() > 1
                && self
                    .samples
                    .front()
                    .is_some_and(|&(first, _)| ns - first > self.duration_ns))
        {
            self.samples.pop_front();
        }
        true
    }

    pub fn sample(&self, ns: u64, epoch: u64) -> Option<Pose> {
        if epoch != self.epoch || ns < self.samples.front()?.0 || ns > self.samples.back()?.0 {
            return None;
        }
        for (index, &(time, pose)) in self.samples.iter().enumerate() {
            if time == ns {
                return Some(pose);
            }
            if time > ns {
                let (before, prior) = self.samples[index - 1];
                return prior.interpolate(pose, (ns - before) as f64 / (time - before) as f64);
            }
        }
        None
    }

    pub fn angular_speed(&self, ns: u64, window_ns: u64, epoch: u64) -> Option<f64> {
        if window_ns == 0 {
            return None;
        }
        let before = self.sample(ns.checked_sub(window_ns)?, epoch)?;
        let current = self.sample(ns, epoch)?;
        Some(before.rotation.angle_to(current.rotation)?.to_degrees() / (window_ns as f64 * 1e-9))
    }
}

pub fn estimated_pose_timestamp(report_ns: u64, offset_ns: i64) -> Option<u64> {
    if offset_ns >= 0 {
        report_ns.checked_sub(offset_ns as u64)
    } else {
        report_ns.checked_add(offset_ns.unsigned_abs())
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Projection {
    pub left: f64,
    pub right: f64,
    pub top: f64,
    pub bottom: f64,
}

impl Default for Projection {
    fn default() -> Self {
        Self {
            left: -1.0,
            right: 1.0,
            top: -1.0,
            bottom: 1.0,
        }
    }
}

impl Projection {
    pub fn from_openvr_raw(left: f64, right: f64, raw_top: f64, raw_bottom: f64) -> Self {
        Self {
            left,
            right,
            top: -raw_bottom,
            bottom: -raw_top,
        }
    }

    pub fn valid(self) -> bool {
        [self.left, self.right, self.top, self.bottom]
            .into_iter()
            .all(f64::is_finite)
            && self
                .span()
                .into_iter()
                .all(|span| span.is_finite() && span > 1e-8)
    }

    fn span(self) -> [f64; 2] {
        [self.right - self.left, self.bottom - self.top]
    }

    /// The part of this projection covering `field`, given as `[x0, y0, x1,
    /// y1]` fractions of its width and height from the top-left.
    pub fn restrict(self, field: [f32; 4]) -> Self {
        let [x0, y0, x1, y1] = field.map(f64::from);
        let [width, height] = self.span();
        Self {
            left: self.left + width * x0,
            right: self.left + width * x1,
            top: self.top + height * y0,
            bottom: self.top + height * y1,
        }
    }

    pub fn project(self, world_from_eye: Pose, world_point: Vec3) -> Option<[f64; 2]> {
        if !self.valid() || !world_from_eye.valid() || !world_point.finite() {
            return None;
        }
        let point = world_from_eye.inverse_transform(world_point);
        if !point.finite() || point.2 >= -1e-8 {
            return None;
        }
        let [width, height] = self.span();
        let uv = [
            (point.0 / -point.2 - self.left) / width,
            (-point.1 / -point.2 - self.top) / height,
        ];
        uv.into_iter()
            .all(|v| (0.0..=1.0).contains(&v))
            .then_some(uv)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Face {
    Front,
    Back,
    Left,
    Right,
    Bottom,
    Top,
}

impl Face {
    pub const ALL: [Self; 6] = [
        Self::Front,
        Self::Back,
        Self::Left,
        Self::Right,
        Self::Bottom,
        Self::Top,
    ];

    fn basis(self, half_size: f64) -> [Vec3; 4] {
        let (right, up, inward) = match self {
            Self::Front => (X, Y, Z),
            Self::Back => (X * -1.0, Y, Z * -1.0),
            Self::Left => (Z * -1.0, Y, X),
            Self::Right => (Z, Y, X * -1.0),
            Self::Bottom => (X, Z * -1.0, Y),
            Self::Top => (X, Z, Y * -1.0),
        };
        [inward * -half_size, right, up, inward]
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Shell {
    pub pose: Pose,
    pub half_size: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct ShellHit {
    pub face: Face,
    pub uv: [f64; 2],
    pub point: Vec3,
    pub distance: f64,
}

impl Default for Shell {
    fn default() -> Self {
        Self {
            pose: Pose::default(),
            half_size: 4.0,
        }
    }
}

impl Shell {
    pub fn valid(self) -> bool {
        self.pose.valid() && self.half_size.is_finite() && self.half_size > 1e-6
    }

    pub fn contains(self, point: Vec3, margin: f64) -> bool {
        self.valid()
            && point.finite()
            && (0.0..self.half_size).contains(&margin)
            && self.pose.inverse_transform(point).max_abs() <= self.half_size - margin
    }

    pub fn face_point(self, face: Face, uv: [f64; 2]) -> Option<Vec3> {
        if !self.valid() || !uv.into_iter().all(|v| (0.0..=1.0).contains(&v)) {
            return None;
        }
        let [center, right, up, _] = face.basis(self.half_size);
        Some(self.pose.transform(
            center
                + right * ((2.0 * uv[0] - 1.0) * self.half_size)
                + up * ((1.0 - 2.0 * uv[1]) * self.half_size),
        ))
    }

    pub fn face_matrix(self, face: Face) -> Option<Matrix34> {
        if !self.valid() {
            return None;
        }
        let [center, right, up, inward] = face.basis(self.half_size);
        let rotation = self.pose.rotation;
        Some(matrix_columns(
            rotation.rotate(right),
            rotation.rotate(up),
            rotation.rotate(inward),
            self.pose.transform(center),
        ))
    }

    pub fn intersect(self, origin: Vec3, direction: Vec3) -> Option<ShellHit> {
        let length = direction.length();
        if !self.contains(origin, 0.0) || !direction.finite() || length < 1e-12 {
            return None;
        }
        let local_origin = self.pose.inverse_transform(origin);
        let local_direction = self.pose.rotation.conjugate().rotate(direction);
        let mut hit: Option<ShellHit> = None;
        for face in Face::ALL {
            let [center, right, up, inward] = face.basis(self.half_size);
            let denom = local_direction.dot(inward);
            if denom >= -1e-12 {
                continue;
            }
            let t = (center - local_origin).dot(inward) / denom;
            if t < 0.0 || !t.is_finite() {
                continue;
            }
            let point = local_origin + local_direction * t;
            let from_center = point - center;
            let u = 0.5 + from_center.dot(right) / (2.0 * self.half_size);
            let v = 0.5 - from_center.dot(up) / (2.0 * self.half_size);
            let tolerance = -1e-10..=1.0 + 1e-10;
            if !tolerance.contains(&u) || !tolerance.contains(&v) {
                continue;
            }
            let distance = length * t;
            if hit.is_none_or(|previous| distance < previous.distance) {
                hit = Some(ShellHit {
                    face,
                    uv: [u.clamp(0.0, 1.0), v.clamp(0.0, 1.0)],
                    point: self.pose.transform(point),
                    distance,
                });
            }
        }
        hit
    }

    pub fn faces_in_view(self, eyes: [(Pose, Projection); 2], margin_deg: f64) -> u32 {
        const ALL_FACES: u32 = 0x3f;
        if !self.valid()
            || !margin_deg.is_finite()
            || margin_deg < 0.0
            || eyes
                .iter()
                .any(|(pose, projection)| !pose.valid() || !projection.valid())
        {
            return ALL_FACES;
        }
        let edge =
            |tangent: f64| (tangent.atan() + margin_deg.to_radians()).min(179_f64.to_radians());
        let tolerance = 1e-9 * self.half_size;
        let mut faces = 0;
        for (pose, projection) in eyes {
            let [right, left, up, down] = [
                projection.right,
                -projection.left,
                -projection.top,
                projection.bottom,
            ]
            .map(edge);
            let normals = [
                Vec3(right.cos(), 0.0, right.sin()),
                Vec3(-left.cos(), 0.0, left.sin()),
                Vec3(0.0, up.cos(), up.sin()),
                Vec3(0.0, -down.cos(), down.sin()),
            ];
            for face in Face::ALL {
                let bit = 1 << face as u32;
                if faces & bit != 0 {
                    continue;
                }
                let mut polygon = [Vec3::default(); 8];
                for (index, uv) in [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
                    .into_iter()
                    .enumerate()
                {
                    let Some(point) = self.face_point(face, uv) else {
                        return ALL_FACES;
                    };
                    polygon[index] = pose.inverse_transform(point);
                }
                let mut count = 4;
                for normal in normals {
                    let mut clipped = [Vec3::default(); 8];
                    let mut kept = 0;
                    for index in 0..count {
                        let a = polygon[index];
                        let b = polygon[(index + 1) % count];
                        let da = normal.dot(a);
                        let db = normal.dot(b);
                        if da <= tolerance {
                            if kept == clipped.len() {
                                return ALL_FACES;
                            }
                            clipped[kept] = a;
                            kept += 1;
                        }
                        if (da <= tolerance) != (db <= tolerance) {
                            if kept == clipped.len() {
                                return ALL_FACES;
                            }
                            clipped[kept] = a + (b - a) * (da / (da - db));
                            kept += 1;
                        }
                    }
                    if kept == 0 {
                        count = 0;
                        break;
                    }
                    polygon = clipped;
                    count = kept;
                }
                if count != 0 {
                    faces |= bit;
                }
            }
        }
        faces
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Aperture {
    pub rectangle: bool,
    pub half_width_tan: f64,
    pub half_height_tan: f64,
    pub feather: f64,
    pub strength: f64,
    pub entry: f64,
}

impl Default for Aperture {
    fn default() -> Self {
        Self {
            rectangle: false,
            half_width_tan: 1.0,
            half_height_tan: 1.0,
            feather: 0.25,
            strength: 1.0,
            entry: 1.0,
        }
    }
}

impl Aperture {
    pub fn peripheral_alpha(self, eye: Pose, point: Vec3) -> Option<f64> {
        if !eye.valid()
            || !point.finite()
            || ![self.half_width_tan, self.half_height_tan]
                .into_iter()
                .all(|v| v.is_finite() && v > 0.0)
            || ![self.feather, self.strength, self.entry]
                .into_iter()
                .all(|v| (0.0..=1.0).contains(&v))
        {
            return None;
        }
        let level = self.strength * self.entry;
        if level == 0.0 {
            return Some(0.0);
        }
        let point = eye.inverse_transform(point);
        if !point.finite() {
            return None;
        }
        if point.2 >= -1e-8 {
            return Some(level);
        }
        let x = point.0 / -point.2 / self.half_width_tan;
        let y = point.1 / -point.2 / self.half_height_tan;
        let radius = if self.rectangle {
            x.abs().max(y.abs())
        } else {
            x.hypot(y)
        };
        if !radius.is_finite() {
            return None;
        }
        if self.feather < 1e-6 {
            return Some(if radius >= 1.0 { level } else { 0.0 });
        }
        let t = ((radius - (1.0 - self.feather)) / self.feather).clamp(0.0, 1.0);
        Some(level * t * t * (3.0 - 2.0 * t))
    }
}

pub fn matched_face_size(
    source: [Projection; 2],
    extent: [u32; 2],
    minimum: u32,
    maximum: u32,
) -> Option<u32> {
    if extent.contains(&0) || minimum > maximum || source.iter().any(|p| !p.valid()) {
        return None;
    }
    let density = source
        .into_iter()
        .flat_map(|projection| {
            let [width, height] = projection.span();
            [f64::from(extent[0]) / width, f64::from(extent[1]) / height]
        })
        .fold(0.0, f64::max);
    let needed = (2.0 * density / 16.0).ceil() * 16.0;
    needed
        .is_finite()
        .then(|| needed.clamp(f64::from(minimum), f64::from(maximum)) as u32)
}

pub fn snapshot_eye_extent(source: [Projection; 2], extent: [u32; 2], face_size: u32) -> [u32; 2] {
    let [left, right] = source.map(Projection::span);
    std::array::from_fn(|axis| {
        let span = left[axis].max(right[axis]).max(0.0);
        let wanted = (f64::from(face_size) * span / 32.0).ceil() * 16.0;
        wanted.min(f64::from(extent[axis])) as u32
    })
}

pub fn rgba_storage_bytes(snapshot: [u32; 2], face: [u32; 2], output_sets: u32) -> Option<u64> {
    if snapshot.contains(&0) || face.contains(&0) || output_sets == 0 {
        return None;
    }
    let snapshots = (u64::from(snapshot[0]) * u64::from(snapshot[1])).checked_mul(16)?;
    let outputs = (u64::from(face[0]) * u64::from(face[1]))
        .checked_mul(48)?
        .checked_mul(u64::from(output_sets))?;
    snapshots.checked_add(outputs)
}

#[cfg(test)]
mod regression_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_restricted_projection_keeps_the_same_tangents_for_the_same_pixels() {
        let eye = Projection::from_openvr_raw(-1.665, 1.213, -1.727, 1.172);
        assert_eq!(eye.restrict([0.0, 0.0, 1.0, 1.0]), eye);
        let part = eye.restrict([0.25, 0.5, 0.75, 1.0]);
        assert!((part.left - (-1.665 + 0.25 * 2.878)).abs() < 1e-9);
        assert!((part.right - (-1.665 + 0.75 * 2.878)).abs() < 1e-9);
        assert!((part.top - (-1.172 + 0.5 * 2.899)).abs() < 1e-9);
        assert!((part.bottom - 1.727).abs() < 1e-9);
    }

    #[test]
    fn openvr_vertical_bounds_and_front_shell_match() {
        let projection = Projection::from_openvr_raw(-1.0, 1.0, -1.727, 1.172);
        assert_eq!(projection.top, -1.172);
        assert_eq!(projection.bottom, 1.727);
        let eye = Pose::default();
        assert_eq!(
            projection.project(eye, Vec3(0.0, 1.172, -1.0)),
            Some([0.5, 0.0])
        );
        assert_eq!(
            projection.project(eye, Vec3(0.0, -1.727, -1.0)),
            Some([0.5, 1.0])
        );
        let shell = Shell::default();
        let hit = shell.intersect(eye.position, Vec3(0.0, 0.0, -1.0)).unwrap();
        assert_eq!(hit.face, Face::Front);
        assert_eq!(hit.uv, [0.5, 0.5]);
        assert_eq!(shell.face_point(hit.face, hit.uv), Some(hit.point));
        let faces = shell.faces_in_view([(eye, projection); 2], 0.0);
        assert_eq!(faces & (1 << Face::Front as u32), 1);
        assert_eq!(faces & (1 << Face::Back as u32), 0);
    }

    #[test]
    fn invalid_tracking_breaks_pose_interpolation() {
        let mut history = PoseHistory::default();
        let first = Pose::default();
        let last = Pose {
            position: Vec3(1.0, 2.0, 3.0),
            ..first
        };
        assert!(history.add(1, first, 1));
        assert!(history.add(3, last, 1));
        assert_eq!(history.sample(2, 1).unwrap().position, Vec3(0.5, 1.0, 1.5));
        assert!(history.sample(2, 2).is_none());
        assert!(!history.add(3, first, 1));
        assert!(history.sample(2, 1).is_none());
        let mut reflected = first.matrix();
        reflected[0][0] = -1.0;
        assert!(Pose::from_matrix(reflected).is_none());
        assert_eq!(Pose::from_matrix(last.matrix()), Some(last));
    }
}
