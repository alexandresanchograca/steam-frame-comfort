//! The little-endian wire format shared with the C++ Vulkan observer.
#![forbid(unsafe_code)]

use bytemuck::{Pod, Zeroable};
use std::mem::{offset_of, size_of};

#[cfg(not(target_endian = "little"))]
compile_error!("Frame Comfort's local wire format requires a little-endian host");

pub const MAGIC: u32 = 0x46435256;
pub const VERSION: u16 = 6;
pub const MAX_IMAGES: usize = 16;

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable, PartialEq, Eq)]
pub struct Header {
    pub magic: u32,
    pub version: u16,
    pub kind: u16,
    pub size: u32,
    pub reserved: u32,
}

impl Header {
    pub fn observer<T>(kind: u16) -> Self {
        Self {
            magic: MAGIC,
            version: VERSION,
            kind,
            size: size_of::<T>() as u32,
            reserved: 0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable, PartialEq, Eq)]
pub struct Hello {
    pub header: Header,
    pub pid: u32,
    pub reserved: u32,
    pub device_uuid: [u8; 16],
    pub driver_uuid: [u8; 16],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable, PartialEq, Eq)]
pub struct Plane {
    pub offset: u64,
    pub size: u64,
    pub row_pitch: u64,
    pub array_pitch: u64,
    pub depth_pitch: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable, PartialEq, Eq)]
pub struct Ycbcr {
    pub valid: u32,
    pub model: u32,
    pub range: u32,
    pub chroma_filter: u32,
    pub x_chroma_offset: u32,
    pub y_chroma_offset: u32,
    pub force_explicit_reconstruction: u32,
    pub reserved: u32,
    pub components: [u32; 4],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable, PartialEq, Eq)]
pub struct ImageAdd {
    pub header: Header,
    pub id: u64,
    pub inode: u64,
    pub allocation_size: u64,
    pub bind_offset: u64,
    pub modifier: u64,
    pub handle_type: u32,
    pub memory_type_index: u32,
    pub format: u32,
    pub width: u32,
    pub height: u32,
    pub mip_levels: u32,
    pub array_layers: u32,
    pub samples: u32,
    pub tiling: u32,
    pub usage: u32,
    pub flags: u32,
    pub plane_count: u32,
    pub dedicated: u32,
    pub exported: u32,
    pub external_handle_types: u32,
    pub view_format_count: u32,
    pub view_formats: [u32; 8],
    pub planes: [Plane; 4],
    pub ycbcr: Ycbcr,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable, PartialEq, Eq)]
pub struct ImageRemove {
    pub header: Header,
    pub id: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable, PartialEq, Eq)]
pub struct EyeUse {
    pub eyes: u32,
    pub slot: u32,
    pub layers: [u32; 2],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable, PartialEq, Eq)]
pub struct Frame {
    pub header: Header,
    pub sequence: u64,
    pub monotonic_ns: u64,
    pub count: u32,
    pub truncated: u32,
    pub ids: [u64; MAX_IMAGES],
    pub layouts: [u32; MAX_IMAGES],
    pub touched_count: u32,
    pub touched_truncated: u32,
    pub touched_ids: [u64; MAX_IMAGES],
    pub touched_layouts: [u32; MAX_IMAGES],
    pub eyes: [EyeUse; MAX_IMAGES],
    pub touched_eyes: [EyeUse; MAX_IMAGES],
}

pub const FRAME_SIZE_V5: usize = offset_of!(Frame, eyes);
pub const MAX_PACKET_SIZE: usize = size_of::<Frame>();
const _: () = {
    assert!(size_of::<Hello>() == 56);
    assert!(offset_of!(ImageAdd, planes) == 152);
    assert!(size_of::<ImageAdd>() == 360);
    assert!(size_of::<ImageRemove>() == 24);
    assert!(FRAME_SIZE_V5 == 432);
    assert!(MAX_PACKET_SIZE == 944);
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
// Keep decoded compositor reports on the stack without a per-frame allocation.
#[allow(clippy::large_enum_variant)]
pub enum Packet {
    Hello(Hello),
    Add(ImageAdd),
    Remove(ImageRemove),
    Frame(Frame),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecodeError {
    Header,
    FrameCount,
}

pub fn decode(bytes: &[u8]) -> Result<Packet, DecodeError> {
    let header = bytes
        .get(..size_of::<Header>())
        .ok_or(DecodeError::Header)?;
    let header = bytemuck::pod_read_unaligned::<Header>(header);
    if header.magic != MAGIC
        || !(5..=VERSION).contains(&header.version)
        || header.size as usize != bytes.len()
    {
        return Err(DecodeError::Header);
    }
    match header.kind {
        1 if bytes.len() == size_of::<Hello>() => {
            Ok(Packet::Hello(bytemuck::pod_read_unaligned(bytes)))
        }
        2 if bytes.len() == size_of::<ImageAdd>() => {
            Ok(Packet::Add(bytemuck::pod_read_unaligned(bytes)))
        }
        3 if bytes.len() == size_of::<ImageRemove>() => {
            Ok(Packet::Remove(bytemuck::pod_read_unaligned(bytes)))
        }
        4 => {
            let frame: Frame = match (header.version, bytes.len()) {
                (5, FRAME_SIZE_V5) => {
                    let mut padded = [0; MAX_PACKET_SIZE];
                    padded[..FRAME_SIZE_V5].copy_from_slice(bytes);
                    bytemuck::pod_read_unaligned(&padded)
                }
                (VERSION, MAX_PACKET_SIZE) => bytemuck::pod_read_unaligned(bytes),
                _ => return Err(DecodeError::Header),
            };
            if frame.count as usize > MAX_IMAGES || frame.touched_count as usize > MAX_IMAGES {
                return Err(DecodeError::FrameCount);
            }
            Ok(Packet::Frame(frame))
        }
        _ => Err(DecodeError::Header),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_labelled_and_unlabelled_frames_with_their_exact_sizes() {
        let mut frame = Frame {
            header: Header::observer::<Frame>(4),
            count: 1,
            ..Default::default()
        };
        frame.ids[0] = 92;
        frame.eyes[0] = EyeUse {
            eyes: 2,
            slot: 3,
            layers: [0, 1],
        };
        assert_eq!(decode(bytemuck::bytes_of(&frame)), Ok(Packet::Frame(frame)));
        frame.header.version = 5;
        frame.header.size = FRAME_SIZE_V5 as u32;
        assert_eq!(decode(bytemuck::bytes_of(&frame)), Err(DecodeError::Header));
        let mut expected = frame;
        expected.eyes.fill(EyeUse::default());
        assert_eq!(
            decode(&bytemuck::bytes_of(&frame)[..FRAME_SIZE_V5]),
            Ok(Packet::Frame(expected))
        );
        frame.count = 17;
        assert_eq!(
            decode(&bytemuck::bytes_of(&frame)[..FRAME_SIZE_V5]),
            Err(DecodeError::FrameCount)
        );
    }
}
