//! Recognises the compositor's scene shaders by their resource layout.
//!
//! SteamVR's distortion vertex shaders read, at descriptor set 0, a uniform
//! block holding the scene layer's last three submissions. Each submission
//! has eight words of bookkeeping and, for each eye, seven row-major 4×4
//! matrices. The seventh matrix's second column is the eye's texture bounds,
//! which the shader applies as `mix(bounds.xy, bounds.zw, uv)`. The matching
//! fragment shaders sample the three submissions from an array of three 2D
//! images at set 0. Only this exact layout is decoded; anything else leaves
//! the compositor's mapping unknown.
use std::collections::HashMap;

pub const SLOTS: usize = 3;
/// Byte layout of the per-submission block.
pub const SLOT_OFFSET: u64 = 16;
pub const SLOT_STRIDE: u64 = 928;
const EYE_OFFSET: u64 = 32;
const EYE_STRIDE: u64 = 448;
const MATRIX_SIZE: u64 = 64;
#[cfg(test)]
const BLOCK_SIZE: u64 = SLOT_OFFSET + SLOT_STRIDE * SLOTS as u64;

/// Byte offset of the first float of an eye's bounds matrix in the block.
pub fn bounds_offset(slot: usize, eye: usize) -> u64 {
    SLOT_OFFSET + SLOT_STRIDE * slot as u64 + EYE_OFFSET + EYE_STRIDE * eye as u64 + 6 * MATRIX_SIZE
}

/// The bounds `[u0, v0, u1, v1]` from a row-major bounds matrix: column 1.
pub fn bounds_from_matrix(matrix: &[f32; 16]) -> [f32; 4] {
    [matrix[1], matrix[5], matrix[9], matrix[13]]
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ShaderInfo {
    pub hash: u64,
    /// Set 0 binding of the per-submission block (vertex shaders).
    pub slot_block: Option<u32>,
    /// Set 0 binding of the three-image scene array (fragment shaders).
    pub slot_images: Option<u32>,
}

#[derive(Clone, Debug)]
enum Type {
    Int {
        width: u32,
    },
    Float {
        width: u32,
    },
    Vector {
        component: u32,
        count: u32,
    },
    Matrix {
        column: u32,
        columns: u32,
    },
    Image {
        dim: u32,
        arrayed: u32,
        sampled: u32,
    },
    SampledImage {
        image: u32,
    },
    Array {
        element: u32,
        length: u32,
    },
    Struct {
        members: Vec<u32>,
    },
    Pointer {
        storage: u32,
        pointee: u32,
    },
}

#[derive(Default)]
struct Module {
    types: HashMap<u32, Type>,
    constants: HashMap<u32, u32>,
    variables: Vec<(u32, u32)>, // pointer type, result
    bindings: HashMap<u32, (Option<u32>, Option<u32>)>, // set, binding
    array_strides: HashMap<u32, u32>,
    offsets: HashMap<(u32, u32), u32>,
    row_major: HashMap<(u32, u32), bool>,
    matrix_strides: HashMap<(u32, u32), u32>,
}

const OP_TYPE_INT: u32 = 21;
const OP_TYPE_FLOAT: u32 = 22;
const OP_TYPE_VECTOR: u32 = 23;
const OP_TYPE_MATRIX: u32 = 24;
const OP_TYPE_IMAGE: u32 = 25;
const OP_TYPE_SAMPLED_IMAGE: u32 = 27;
const OP_TYPE_ARRAY: u32 = 28;
const OP_TYPE_STRUCT: u32 = 30;
const OP_TYPE_POINTER: u32 = 32;
const OP_CONSTANT: u32 = 43;
const OP_VARIABLE: u32 = 59;
const OP_DECORATE: u32 = 71;
const OP_MEMBER_DECORATE: u32 = 72;

const ROW_MAJOR: u32 = 4;
const ARRAY_STRIDE: u32 = 6;
const MATRIX_STRIDE: u32 = 7;
const BINDING: u32 = 33;
const DESCRIPTOR_SET: u32 = 34;
const OFFSET: u32 = 35;

const STORAGE_UNIFORM_CONSTANT: u32 = 0;
const STORAGE_UNIFORM: u32 = 2;
const DIM_2D: u32 = 1;

fn parse(words: &[u32]) -> Option<Module> {
    if words.len() < 5 || words[0] != 0x0723_0203 {
        return None;
    }
    let mut module = Module::default();
    let mut at = 5;
    while at < words.len() {
        let count = (words[at] >> 16) as usize;
        let op = words[at] & 0xffff;
        if count == 0 || at + count > words.len() {
            return None;
        }
        let w = &words[at + 1..at + count];
        let get = |index: usize| w.get(index).copied();
        match op {
            OP_TYPE_INT => {
                module.types.insert(get(0)?, Type::Int { width: get(1)? });
            }
            OP_TYPE_FLOAT => {
                module.types.insert(get(0)?, Type::Float { width: get(1)? });
            }
            OP_TYPE_VECTOR => {
                module.types.insert(
                    get(0)?,
                    Type::Vector {
                        component: get(1)?,
                        count: get(2)?,
                    },
                );
            }
            OP_TYPE_MATRIX => {
                module.types.insert(
                    get(0)?,
                    Type::Matrix {
                        column: get(1)?,
                        columns: get(2)?,
                    },
                );
            }
            OP_TYPE_IMAGE => {
                module.types.insert(
                    get(0)?,
                    Type::Image {
                        dim: get(2)?,
                        arrayed: get(4)?,
                        sampled: get(6)?,
                    },
                );
            }
            OP_TYPE_SAMPLED_IMAGE => {
                module
                    .types
                    .insert(get(0)?, Type::SampledImage { image: get(1)? });
            }
            OP_TYPE_ARRAY => {
                module.types.insert(
                    get(0)?,
                    Type::Array {
                        element: get(1)?,
                        length: get(2)?,
                    },
                );
            }
            OP_TYPE_STRUCT => {
                module.types.insert(
                    get(0)?,
                    Type::Struct {
                        members: w.get(1..)?.to_vec(),
                    },
                );
            }
            OP_TYPE_POINTER => {
                module.types.insert(
                    get(0)?,
                    Type::Pointer {
                        storage: get(1)?,
                        pointee: get(2)?,
                    },
                );
            }
            // 32-bit constants only; array lengths in these shaders are.
            OP_CONSTANT if w.len() == 3 => {
                module.constants.insert(get(1)?, get(2)?);
            }
            OP_VARIABLE => module.variables.push((get(0)?, get(1)?)),
            OP_DECORATE => match get(1)? {
                BINDING => module.bindings.entry(get(0)?).or_default().1 = Some(get(2)?),
                DESCRIPTOR_SET => module.bindings.entry(get(0)?).or_default().0 = Some(get(2)?),
                ARRAY_STRIDE => {
                    module.array_strides.insert(get(0)?, get(2)?);
                }
                _ => {}
            },
            OP_MEMBER_DECORATE => {
                let key = (get(0)?, get(1)?);
                match get(2)? {
                    OFFSET => {
                        module.offsets.insert(key, get(3)?);
                    }
                    ROW_MAJOR => {
                        module.row_major.insert(key, true);
                    }
                    MATRIX_STRIDE => {
                        module.matrix_strides.insert(key, get(3)?);
                    }
                    _ => {}
                }
            }
            _ => {}
        }
        at += count;
    }
    Some(module)
}

impl Module {
    fn ty(&self, id: u32) -> Option<&Type> {
        self.types.get(&id)
    }

    fn is_scalar(&self, id: u32, float: bool) -> bool {
        matches!(
            (self.ty(id), float),
            (Some(Type::Int { width: 32 }), false) | (Some(Type::Float { width: 32 }), true)
        )
    }

    fn array(&self, id: u32, length: u32) -> Option<u32> {
        match self.ty(id)? {
            Type::Array {
                element,
                length: length_id,
            } if self.constants.get(length_id) == Some(&length) => Some(*element),
            _ => None,
        }
    }

    fn members(&self, id: u32) -> Option<&[u32]> {
        match self.ty(id)? {
            Type::Struct { members } => Some(members),
            _ => None,
        }
    }

    fn offset(&self, id: u32, member: u32) -> Option<u32> {
        self.offsets.get(&(id, member)).copied()
    }

    fn mat4(&self, id: u32) -> bool {
        let Some(Type::Matrix { column, columns: 4 }) = self.ty(id) else {
            return false;
        };
        matches!(self.ty(*column), Some(Type::Vector { component, count: 4 }) if self.is_scalar(*component, true))
    }

    /// Seven row-major 4×4 matrices, 64 bytes apart.
    fn eye_matrices(&self, id: u32) -> bool {
        let Some(members) = self.members(id) else {
            return false;
        };
        members.len() == 7
            && members.iter().enumerate().all(|(index, &member)| {
                let index = index as u32;
                self.mat4(member)
                    && self.offset(id, index) == Some(64 * index)
                    && self.row_major.get(&(id, index)) == Some(&true)
                    && self.matrix_strides.get(&(id, index)) == Some(&16)
            })
    }

    /// Eight words of bookkeeping, then two eyes 448 bytes apart.
    fn slot(&self, id: u32) -> bool {
        let Some(members) = self.members(id) else {
            return false;
        };
        members.len() == 9
            && members[..8].iter().enumerate().all(|(index, &member)| {
                self.is_scalar(member, false)
                    && self.offset(id, index as u32) == Some(4 * index as u32)
            })
            && self.offset(id, 8) == Some(EYE_OFFSET as u32)
            && self.array_strides.get(&members[8]) == Some(&(EYE_STRIDE as u32))
            && self
                .array(members[8], 2)
                .is_some_and(|eye| self.eye_matrices(eye))
    }

    fn slot_block(&self, id: u32) -> bool {
        let Some(members) = self.members(id) else {
            return false;
        };
        members.len() == 2
            && self.is_scalar(members[0], false)
            && self.offset(id, 0) == Some(0)
            && self.offset(id, 1) == Some(SLOT_OFFSET as u32)
            && self.array_strides.get(&members[1]) == Some(&(SLOT_STRIDE as u32))
            && self
                .array(members[1], SLOTS as u32)
                .is_some_and(|slot| self.slot(slot))
    }

    fn image_2d(&self, id: u32) -> bool {
        match self.ty(id) {
            Some(Type::Image {
                dim: DIM_2D,
                arrayed: 0,
                sampled: 1,
            }) => true,
            Some(Type::SampledImage { image }) => self.image_2d(*image),
            _ => false,
        }
    }

    fn set_zero_binding(&self, variable: u32) -> Option<u32> {
        match self.bindings.get(&variable)? {
            (Some(0), Some(binding)) => Some(*binding),
            _ => None,
        }
    }
}

pub fn inspect(words: &[u32], hash: u64) -> ShaderInfo {
    let mut info = ShaderInfo {
        hash,
        ..ShaderInfo::default()
    };
    let Some(module) = parse(words) else {
        return info;
    };
    for &(pointer, variable) in &module.variables {
        let Some(Type::Pointer { storage, pointee }) = module.ty(pointer) else {
            continue;
        };
        let Some(binding) = module.set_zero_binding(variable) else {
            continue;
        };
        match *storage {
            STORAGE_UNIFORM if module.slot_block(*pointee) => {
                info.slot_block.get_or_insert(binding);
            }
            STORAGE_UNIFORM_CONSTANT
                if module
                    .array(*pointee, SLOTS as u32)
                    .is_some_and(|element| module.image_2d(element)) =>
            {
                info.slot_images.get_or_insert(binding);
            }
            _ => {}
        }
    }
    info
}

#[cfg(test)]
pub mod tests {
    use super::*;

    /// Assembles a minimal module with the compositor's block at
    /// `block_binding` and the image array at `image_binding`.
    pub fn scene_module(block_binding: u32, image_binding: u32, slot_stride: u32) -> Vec<u32> {
        let mut words = vec![0x0723_0203, 0x0001_0000, 0, 100, 0];
        let mut op = |code: u32, operands: &[u32]| {
            words.push(((operands.len() as u32 + 1) << 16) | code);
            words.extend_from_slice(operands);
        };
        let (uint, float, vec4, mat4, eye, two, eyes, slot, three, slots, block, pointer) =
            (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12);
        let (variable, image, images, image_pointer, image_variable) = (13, 14, 15, 16, 17);
        op(OP_DECORATE, &[variable, DESCRIPTOR_SET, 0]);
        op(OP_DECORATE, &[variable, BINDING, block_binding]);
        op(OP_DECORATE, &[image_variable, DESCRIPTOR_SET, 0]);
        op(OP_DECORATE, &[image_variable, BINDING, image_binding]);
        op(OP_DECORATE, &[eyes, ARRAY_STRIDE, EYE_STRIDE as u32]);
        op(OP_DECORATE, &[slots, ARRAY_STRIDE, slot_stride]);
        for member in 0..7 {
            op(OP_MEMBER_DECORATE, &[eye, member, OFFSET, 64 * member]);
            op(OP_MEMBER_DECORATE, &[eye, member, ROW_MAJOR]);
            op(OP_MEMBER_DECORATE, &[eye, member, MATRIX_STRIDE, 16]);
        }
        for member in 0..8 {
            op(OP_MEMBER_DECORATE, &[slot, member, OFFSET, 4 * member]);
        }
        op(OP_MEMBER_DECORATE, &[slot, 8, OFFSET, EYE_OFFSET as u32]);
        op(OP_MEMBER_DECORATE, &[block, 0, OFFSET, 0]);
        op(OP_MEMBER_DECORATE, &[block, 1, OFFSET, SLOT_OFFSET as u32]);
        op(OP_TYPE_INT, &[uint, 32, 0]);
        op(OP_TYPE_FLOAT, &[float, 32]);
        op(OP_TYPE_VECTOR, &[vec4, float, 4]);
        op(OP_TYPE_MATRIX, &[mat4, vec4, 4]);
        op(
            OP_TYPE_STRUCT,
            &[eye, mat4, mat4, mat4, mat4, mat4, mat4, mat4],
        );
        op(OP_CONSTANT, &[uint, two, 2]);
        op(OP_TYPE_ARRAY, &[eyes, eye, two]);
        op(
            OP_TYPE_STRUCT,
            &[slot, uint, uint, uint, uint, uint, uint, uint, uint, eyes],
        );
        op(OP_CONSTANT, &[uint, three, 3]);
        op(OP_TYPE_ARRAY, &[slots, slot, three]);
        op(OP_TYPE_STRUCT, &[block, uint, slots]);
        op(OP_TYPE_POINTER, &[pointer, STORAGE_UNIFORM, block]);
        op(OP_VARIABLE, &[pointer, variable, STORAGE_UNIFORM]);
        op(OP_TYPE_IMAGE, &[image, float, DIM_2D, 0, 0, 0, 1, 0]);
        op(OP_TYPE_ARRAY, &[images, image, three]);
        op(
            OP_TYPE_POINTER,
            &[image_pointer, STORAGE_UNIFORM_CONSTANT, images],
        );
        op(
            OP_VARIABLE,
            &[image_pointer, image_variable, STORAGE_UNIFORM_CONSTANT],
        );
        words
    }

    #[test]
    fn recognises_the_slot_block_and_image_array() {
        let info = inspect(&scene_module(2, 9, SLOT_STRIDE as u32), 7);
        assert_eq!(info.slot_block, Some(2));
        assert_eq!(info.slot_images, Some(9));
        assert_eq!(info.hash, 7);
    }

    #[test]
    fn a_changed_layout_is_not_decoded() {
        let info = inspect(&scene_module(2, 9, 944), 7);
        assert_eq!(info.slot_block, None);
        assert_eq!(info.slot_images, Some(9));
        let mut truncated = scene_module(2, 9, SLOT_STRIDE as u32);
        truncated.truncate(40);
        assert_eq!(inspect(&truncated, 1).slot_block, None);
        assert_eq!(
            inspect(&[1, 2, 3], 1),
            ShaderInfo {
                hash: 1,
                ..ShaderInfo::default()
            }
        );
    }

    #[test]
    fn bounds_offsets_follow_the_shader_layout() {
        assert_eq!(bounds_offset(0, 0), 16 + 32 + 384);
        assert_eq!(bounds_offset(2, 1), 16 + 2 * 928 + 32 + 448 + 384);
        assert!(bounds_offset(2, 1) + 64 <= BLOCK_SIZE);
        let mut matrix = [0.0; 16];
        matrix[1] = 0.134;
        matrix[5] = 1.0;
        matrix[9] = 1.0;
        matrix[13] = 0.0;
        assert_eq!(bounds_from_matrix(&matrix), [0.134, 1.0, 1.0, 0.0]);
    }
}
