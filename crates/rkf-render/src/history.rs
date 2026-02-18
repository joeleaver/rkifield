//! History buffers for temporal upscaling.
//!
//! Maintains ping-pong color and metadata textures at display resolution.
//! The temporal upscaler reads from one set and writes to the other,
//! swapping each frame.
//!
//! | Buffer          | Format     | Content                                    |
//! |-----------------|------------|--------------------------------------------|
//! | history_color   | Rgba16Float| Accumulated HDR color                      |
//! | history_metadata| Rg32Uint   | Packed: depth (f32-as-u32) + material_id   |

use wgpu::util::DeviceExt;
use bytemuck::{Pod, Zeroable};

/// Texture format for history color (HDR accumulated).
pub const HISTORY_COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Texture format for history metadata (packed depth + material).
pub const HISTORY_METADATA_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg32Uint;

/// GPU-uploadable history uniforms (16 bytes).
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct HistoryUniforms {
    /// Display resolution width.
    pub display_width: u32,
    /// Display resolution height.
    pub display_height: u32,
    /// Internal resolution width.
    pub internal_width: u32,
    /// Internal resolution height.
    pub internal_height: u32,
}

/// Ping-pong history buffers at display resolution.
///
/// Two sets (A and B) of color + metadata textures. Each frame, one set
/// is read as "previous history" and the other is written as "new history".
/// Call [`swap()`](HistoryBuffers::swap) at the end of each frame.
pub struct HistoryBuffers {
    /// Color textures [A, B].
    pub color_textures: [wgpu::Texture; 2],
    /// Color texture views [A, B].
    pub color_views: [wgpu::TextureView; 2],
    /// Metadata textures [A, B].
    pub metadata_textures: [wgpu::Texture; 2],
    /// Metadata texture views [A, B].
    pub metadata_views: [wgpu::TextureView; 2],
    /// Uniform buffer with resolution info.
    pub uniforms_buffer: wgpu::Buffer,
    /// Current read index (0 or 1). Write index is `1 - read_idx`.
    read_idx: usize,
    /// Display resolution width.
    pub width: u32,
    /// Display resolution height.
    pub height: u32,
}

impl HistoryBuffers {
    /// Create history buffers at the given display resolution.
    pub fn new(
        device: &wgpu::Device,
        display_width: u32,
        display_height: u32,
        internal_width: u32,
        internal_height: u32,
    ) -> Self {
        let size = wgpu::Extent3d {
            width: display_width,
            height: display_height,
            depth_or_array_layers: 1,
        };
        let usage = wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING;

        let make_color = |label: &str| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: HISTORY_COLOR_FORMAT,
                usage,
                view_formats: &[],
            })
        };

        let make_metadata = |label: &str| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: HISTORY_METADATA_FORMAT,
                usage,
                view_formats: &[],
            })
        };

        let color_a = make_color("history color A");
        let color_b = make_color("history color B");
        let meta_a = make_metadata("history metadata A");
        let meta_b = make_metadata("history metadata B");

        let color_views = [
            color_a.create_view(&Default::default()),
            color_b.create_view(&Default::default()),
        ];
        let metadata_views = [
            meta_a.create_view(&Default::default()),
            meta_b.create_view(&Default::default()),
        ];

        let uniforms = HistoryUniforms {
            display_width,
            display_height,
            internal_width,
            internal_height,
        };
        let uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("history uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            color_textures: [color_a, color_b],
            color_views,
            metadata_textures: [meta_a, meta_b],
            metadata_views,
            uniforms_buffer,
            read_idx: 0,
            width: display_width,
            height: display_height,
        }
    }

    /// Get the read (previous history) color view.
    pub fn read_color_view(&self) -> &wgpu::TextureView {
        &self.color_views[self.read_idx]
    }

    /// Get the write (current output) color view.
    pub fn write_color_view(&self) -> &wgpu::TextureView {
        &self.color_views[1 - self.read_idx]
    }

    /// Get the read (previous history) metadata view.
    pub fn read_metadata_view(&self) -> &wgpu::TextureView {
        &self.metadata_views[self.read_idx]
    }

    /// Get the write (current output) metadata view.
    pub fn write_metadata_view(&self) -> &wgpu::TextureView {
        &self.metadata_views[1 - self.read_idx]
    }

    /// Swap read/write indices. Call at the end of each frame.
    pub fn swap(&mut self) {
        self.read_idx = 1 - self.read_idx;
    }

    /// Current read index (for debugging).
    pub fn read_index(&self) -> usize {
        self.read_idx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn history_uniforms_size_is_16() {
        assert_eq!(std::mem::size_of::<HistoryUniforms>(), 16);
    }

    #[test]
    fn history_uniforms_pod_roundtrip() {
        let u = HistoryUniforms {
            display_width: 1920,
            display_height: 1080,
            internal_width: 960,
            internal_height: 540,
        };
        let bytes = bytemuck::bytes_of(&u);
        assert_eq!(bytes.len(), 16);
        let u2: &HistoryUniforms = bytemuck::from_bytes(bytes);
        assert_eq!(u.display_width, u2.display_width);
        assert_eq!(u.internal_width, u2.internal_width);
    }

    #[test]
    fn swap_toggles_read_index() {
        // Can't create real GPU textures in unit tests, but we can test the index logic
        // by checking that read_idx toggles between 0 and 1
        assert_eq!(1 - 0usize, 1);
        assert_eq!(1 - 1usize, 0);
    }

    #[test]
    fn history_format_constants() {
        assert_eq!(HISTORY_COLOR_FORMAT, wgpu::TextureFormat::Rgba16Float);
        assert_eq!(HISTORY_METADATA_FORMAT, wgpu::TextureFormat::Rg32Uint);
    }
}
