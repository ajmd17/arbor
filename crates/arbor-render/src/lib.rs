//! Standalone OpenGL renderer for arbor geometry, independent of any UI toolkit.
//!
//! Works on a [`glow::Context`] you supply, so it can be hosted by egui, winit, or a
//! browser canvas alike.

pub mod gpu;
pub mod hdri;
pub mod ibl;
pub mod lighting;
pub mod mipmap;
pub mod render;
pub mod shaders;
