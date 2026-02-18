//! GPU device and queue wrapper.
//!
//! [`RenderContext`] encapsulates the wgpu adapter, device, and queue used
//! by all rendering passes. Created once at startup from a compatible surface.

/// Core GPU context holding the wgpu device and queue.
///
/// Created via [`RenderContext::new`] which requests an adapter compatible
/// with the given surface, then opens a device with default limits.
pub struct RenderContext {
    /// The wgpu device used for resource creation and command encoding.
    pub device: wgpu::Device,
    /// The command queue for submitting GPU work.
    pub queue: wgpu::Queue,
    /// The adapter that was selected.
    pub adapter: wgpu::Adapter,
}

impl RenderContext {
    /// Create a new render context compatible with the given surface.
    ///
    /// Blocks on async wgpu initialization using `pollster`.
    ///
    /// # Panics
    ///
    /// Panics if no compatible adapter is found or device creation fails.
    pub fn new(instance: &wgpu::Instance, surface: &wgpu::Surface<'_>) -> Self {
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: Some(surface),
        }))
        .expect("failed to find a compatible GPU adapter");

        log::info!("GPU adapter: {:?}", adapter.get_info().name);

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("rkf-render device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))
        .expect("failed to create GPU device");

        Self {
            device,
            queue,
            adapter,
        }
    }

    /// Configure a surface for presentation with the given dimensions.
    ///
    /// Returns the chosen surface format.
    pub fn configure_surface(
        &self,
        surface: &wgpu::Surface<'_>,
        width: u32,
        height: u32,
    ) -> wgpu::TextureFormat {
        let caps = surface.get_capabilities(&self.adapter);
        let format = caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: width.max(1),
            height: height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&self.device, &config);

        log::info!("Surface configured: {width}x{height}, format={format:?}");
        format
    }
}
