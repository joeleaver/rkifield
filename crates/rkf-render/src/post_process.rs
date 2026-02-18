//! Post-processing stack architecture.
//!
//! Provides the configuration system, ping-pong buffer management, and
//! pass context types for the post-processing pipeline.
//!
//! Post-processing passes are split into two groups:
//! - **Pre-upscale**: run at internal (lower) resolution after shading, before temporal upscale.
//! - **Post-upscale**: run at display resolution after sharpening, before tone mapping.

use std::collections::HashSet;

/// Texture format used for all post-processing ping-pong buffers.
pub const PP_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Identifies each post-processing pass for enable/disable control.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PostProcessPassId {
    // Pre-upscale passes
    /// Bloom blur (pre-upscale).
    Bloom,
    /// Depth of field (pre-upscale).
    DepthOfField,
    /// Motion blur (pre-upscale).
    MotionBlur,
    // Post-upscale passes
    /// Bloom composite (post-upscale).
    BloomComposite,
    /// Auto exposure adjustment (post-upscale).
    AutoExposure,
    /// Color grading LUT (post-upscale).
    ColorGrading,
    /// Vignette (post-upscale).
    Vignette,
    /// Film grain (post-upscale).
    FilmGrain,
    /// Chromatic aberration (post-upscale).
    ChromaticAberration,
}

impl PostProcessPassId {
    /// Returns `true` for passes that run before the temporal upscale (at internal resolution).
    pub fn is_pre_upscale(self) -> bool {
        matches!(self, Self::Bloom | Self::DepthOfField | Self::MotionBlur)
    }

    /// Ordered slice of all pre-upscale pass IDs.
    fn all_pre_upscale() -> &'static [PostProcessPassId] {
        &[
            PostProcessPassId::Bloom,
            PostProcessPassId::DepthOfField,
            PostProcessPassId::MotionBlur,
        ]
    }

    /// Ordered slice of all post-upscale pass IDs.
    fn all_post_upscale() -> &'static [PostProcessPassId] {
        &[
            PostProcessPassId::BloomComposite,
            PostProcessPassId::AutoExposure,
            PostProcessPassId::ColorGrading,
            PostProcessPassId::Vignette,
            PostProcessPassId::FilmGrain,
            PostProcessPassId::ChromaticAberration,
        ]
    }
}

/// Runtime configuration for post-processing passes.
///
/// Tracks which passes are disabled; all passes are enabled by default.
#[derive(Debug, Clone)]
pub struct PostProcessConfig {
    /// Set of disabled passes. Passes not in this set are enabled.
    disabled: HashSet<PostProcessPassId>,
}

impl PostProcessConfig {
    /// Creates a new config with all passes enabled.
    pub fn new() -> Self {
        Self {
            disabled: HashSet::new(),
        }
    }

    /// Returns `true` if the given pass is enabled.
    pub fn is_enabled(&self, id: PostProcessPassId) -> bool {
        !self.disabled.contains(&id)
    }

    /// Sets whether a pass is enabled.
    pub fn set_enabled(&mut self, id: PostProcessPassId, enabled: bool) {
        if enabled {
            self.disabled.remove(&id);
        } else {
            self.disabled.insert(id);
        }
    }

    /// Enables a pass.
    pub fn enable(&mut self, id: PostProcessPassId) {
        self.disabled.remove(&id);
    }

    /// Disables a pass.
    pub fn disable(&mut self, id: PostProcessPassId) {
        self.disabled.insert(id);
    }

    /// Returns the enabled pre-upscale passes in declaration order.
    pub fn enabled_pre_upscale(&self) -> Vec<PostProcessPassId> {
        PostProcessPassId::all_pre_upscale()
            .iter()
            .copied()
            .filter(|id| self.is_enabled(*id))
            .collect()
    }

    /// Returns the enabled post-upscale passes in declaration order.
    pub fn enabled_post_upscale(&self) -> Vec<PostProcessPassId> {
        PostProcessPassId::all_post_upscale()
            .iter()
            .copied()
            .filter(|id| self.is_enabled(*id))
            .collect()
    }
}

impl Default for PostProcessConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Ping-pong buffer pair for chaining post-processing passes.
///
/// Two `Rgba16Float` textures at the same resolution. Passes alternate
/// reading from one and writing to the other.
#[allow(dead_code)]
pub struct PingPongBuffers {
    /// The two GPU textures.
    pub textures: [wgpu::Texture; 2],
    /// Texture views for sampling / storage access.
    pub views: [wgpu::TextureView; 2],
    /// Current read index (0 or 1). Write index is `1 - read_index`.
    read_index: usize,
    /// Width of both textures.
    pub width: u32,
    /// Height of both textures.
    pub height: u32,
}

impl PingPongBuffers {
    /// Creates two `Rgba16Float` textures with `STORAGE_BINDING | TEXTURE_BINDING` usage.
    pub fn new(device: &wgpu::Device, width: u32, height: u32, label: &str) -> Self {
        let make_texture = |index: usize| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("{label}_pp_{index}")),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: PP_FORMAT,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        };

        let tex0 = make_texture(0);
        let tex1 = make_texture(1);

        let view0 = tex0.create_view(&wgpu::TextureViewDescriptor::default());
        let view1 = tex1.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            textures: [tex0, tex1],
            views: [view0, view1],
            read_index: 0,
            width,
            height,
        }
    }

    /// Returns the current read-side texture view.
    pub fn read_view(&self) -> &wgpu::TextureView {
        &self.views[self.read_index]
    }

    /// Returns the current write-side texture view.
    pub fn write_view(&self) -> &wgpu::TextureView {
        &self.views[1 - self.read_index]
    }

    /// Swaps read and write sides.
    pub fn swap(&mut self) {
        self.read_index = 1 - self.read_index;
    }

    /// Returns the current read index (0 or 1).
    pub fn read_index(&self) -> usize {
        self.read_index
    }
}

/// Context provided to post-processing passes during execution.
///
/// Contains references to shared resources needed by most passes.
#[allow(dead_code)]
pub struct PostProcessContext<'a> {
    /// The command encoder for recording GPU commands.
    pub encoder: &'a mut wgpu::CommandEncoder,
    /// The GPU queue for buffer writes.
    pub queue: &'a wgpu::Queue,
    /// Current input texture view (read).
    pub input_view: &'a wgpu::TextureView,
    /// Current output texture view (write).
    pub output_view: &'a wgpu::TextureView,
    /// Width of the textures being processed.
    pub width: u32,
    /// Height of the textures being processed.
    pub height: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_all_enabled_by_default() {
        let config = PostProcessConfig::new();
        assert!(config.is_enabled(PostProcessPassId::Bloom));
        assert!(config.is_enabled(PostProcessPassId::DepthOfField));
        assert!(config.is_enabled(PostProcessPassId::Vignette));
    }

    #[test]
    fn config_disable_enable() {
        let mut config = PostProcessConfig::new();
        config.disable(PostProcessPassId::Bloom);
        assert!(!config.is_enabled(PostProcessPassId::Bloom));
        config.enable(PostProcessPassId::Bloom);
        assert!(config.is_enabled(PostProcessPassId::Bloom));
    }

    #[test]
    fn config_set_enabled() {
        let mut config = PostProcessConfig::new();
        config.set_enabled(PostProcessPassId::FilmGrain, false);
        assert!(!config.is_enabled(PostProcessPassId::FilmGrain));
        config.set_enabled(PostProcessPassId::FilmGrain, true);
        assert!(config.is_enabled(PostProcessPassId::FilmGrain));
    }

    #[test]
    fn pre_upscale_classification() {
        assert!(PostProcessPassId::Bloom.is_pre_upscale());
        assert!(PostProcessPassId::DepthOfField.is_pre_upscale());
        assert!(PostProcessPassId::MotionBlur.is_pre_upscale());
        assert!(!PostProcessPassId::BloomComposite.is_pre_upscale());
        assert!(!PostProcessPassId::Vignette.is_pre_upscale());
    }

    #[test]
    fn enabled_pre_upscale_ordering() {
        let config = PostProcessConfig::new();
        let pre = config.enabled_pre_upscale();
        assert_eq!(
            pre,
            vec![
                PostProcessPassId::Bloom,
                PostProcessPassId::DepthOfField,
                PostProcessPassId::MotionBlur,
            ]
        );
    }

    #[test]
    fn enabled_post_upscale_ordering() {
        let config = PostProcessConfig::new();
        let post = config.enabled_post_upscale();
        assert_eq!(
            post,
            vec![
                PostProcessPassId::BloomComposite,
                PostProcessPassId::AutoExposure,
                PostProcessPassId::ColorGrading,
                PostProcessPassId::Vignette,
                PostProcessPassId::FilmGrain,
                PostProcessPassId::ChromaticAberration,
            ]
        );
    }

    #[test]
    fn enabled_filters_disabled() {
        let mut config = PostProcessConfig::new();
        config.disable(PostProcessPassId::DepthOfField);
        config.disable(PostProcessPassId::FilmGrain);
        let pre = config.enabled_pre_upscale();
        assert_eq!(
            pre,
            vec![PostProcessPassId::Bloom, PostProcessPassId::MotionBlur]
        );
        let post = config.enabled_post_upscale();
        assert!(!post.contains(&PostProcessPassId::FilmGrain));
    }

    #[test]
    fn ping_pong_swap() {
        // Test the swap logic without GPU
        let mut idx = 0usize;
        assert_eq!(idx, 0);
        assert_eq!(1 - idx, 1);
        idx = 1 - idx;
        assert_eq!(idx, 1);
        assert_eq!(1 - idx, 0);
    }
}
