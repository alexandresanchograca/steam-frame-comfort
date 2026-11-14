#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(target_os = "linux")]
pub mod source_client;

#[cfg(all(target_os = "linux", feature = "steamvr"))]
pub mod openvr;

#[cfg(all(target_os = "linux", feature = "steamvr"))]
pub mod vulkan;

#[cfg(target_os = "linux")]
pub mod mailbox;
