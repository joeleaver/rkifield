//! DLSS integration layer.
//!
//! Provides an interface for NVIDIA DLSS upscaling. Currently a stub that
//! reports DLSS as unavailable, falling back to the custom temporal upscaler.
//! When the DLSS SDK is integrated (via `dlss_wgpu` or direct SDK bindings),
//! this module will handle initialization, resource management, and dispatch.
//!
//! # Feature gating
//!
//! DLSS support is behind the `dlss` feature flag. When disabled (default),
//! `is_available()` always returns `false` and no NVIDIA SDK code is compiled.

/// DLSS quality modes matching NVIDIA's preset levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DlssQuality {
    /// Maximum quality (minimal upscaling, close to native).
    UltraQuality,
    /// High quality (slight upscaling).
    Quality,
    /// Balanced quality/performance.
    Balanced,
    /// Maximum performance (aggressive upscaling).
    Performance,
    /// Ultra performance (maximum upscaling ratio).
    UltraPerformance,
}

impl DlssQuality {
    /// Render resolution scale factor for this quality mode.
    ///
    /// Returns the ratio of internal resolution to display resolution.
    pub fn scale_factor(self) -> f32 {
        match self {
            DlssQuality::UltraQuality => 0.77,
            DlssQuality::Quality => 0.67,
            DlssQuality::Balanced => 0.58,
            DlssQuality::Performance => 0.5,
            DlssQuality::UltraPerformance => 0.33,
        }
    }

    /// Compute internal resolution for a given display resolution.
    pub fn internal_resolution(self, display_width: u32, display_height: u32) -> (u32, u32) {
        let scale = self.scale_factor();
        let w = ((display_width as f32 * scale).round() as u32).max(1);
        let h = ((display_height as f32 * scale).round() as u32).max(1);
        (w, h)
    }
}

/// DLSS feature context — handles initialization and capability queries.
///
/// Currently a stub. When DLSS SDK is integrated, this will hold the
/// DLSS feature context handle and GPU resource descriptors.
pub struct DlssContext {
    /// Whether DLSS is actually available (GPU + driver + SDK).
    available: bool,
}

impl DlssContext {
    /// Create a new DLSS context, probing for hardware/driver support.
    ///
    /// Currently always reports DLSS as unavailable. When the SDK is
    /// integrated, this will:
    /// 1. Check for NVIDIA GPU via adapter info
    /// 2. Verify driver version meets minimum requirements
    /// 3. Initialize the DLSS SDK feature evaluator
    /// 4. Query supported quality modes
    pub fn new(_device: &wgpu::Device, _adapter_info: &wgpu::AdapterInfo) -> Self {
        // TODO: When dlss_wgpu or direct SDK bindings are available:
        // 1. Check adapter_info.vendor == NVIDIA_VENDOR_ID (0x10DE)
        // 2. Initialize DLSS SDK
        // 3. Query feature support
        log::info!("DLSS: not available (SDK not integrated), using custom temporal upscaler");
        Self { available: false }
    }

    /// Check if DLSS is available on this system.
    pub fn is_available(&self) -> bool {
        self.available
    }
}

/// Check if the adapter is an NVIDIA GPU.
///
/// Quick check based on adapter vendor ID. Returns `false` if
/// the adapter is not NVIDIA or if the `dlss` feature is disabled.
pub fn is_nvidia_gpu(adapter_info: &wgpu::AdapterInfo) -> bool {
    // NVIDIA vendor ID
    const NVIDIA_VENDOR_ID: u32 = 0x10DE;
    adapter_info.vendor == NVIDIA_VENDOR_ID
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quality_scale_factors_decrease() {
        let modes = [
            DlssQuality::UltraQuality,
            DlssQuality::Quality,
            DlssQuality::Balanced,
            DlssQuality::Performance,
            DlssQuality::UltraPerformance,
        ];
        for i in 0..modes.len() - 1 {
            assert!(
                modes[i].scale_factor() > modes[i + 1].scale_factor(),
                "{:?} scale should be > {:?} scale",
                modes[i],
                modes[i + 1]
            );
        }
    }

    #[test]
    fn quality_internal_resolution() {
        let (w, h) = DlssQuality::Performance.internal_resolution(1920, 1080);
        assert_eq!(w, 960);
        assert_eq!(h, 540);
    }

    #[test]
    fn ultra_performance_is_aggressive() {
        let (w, h) = DlssQuality::UltraPerformance.internal_resolution(1920, 1080);
        assert!(w < 700, "ultra performance should be very aggressive: got {w}");
        assert!(h < 400, "ultra performance should be very aggressive: got {h}");
    }

    #[test]
    fn internal_resolution_minimum_one() {
        let (w, h) = DlssQuality::UltraPerformance.internal_resolution(1, 1);
        assert!(w >= 1);
        assert!(h >= 1);
    }

    #[test]
    fn nvidia_vendor_id_check() {
        let make_info = |vendor: u32| wgpu::AdapterInfo {
            name: String::new(),
            vendor,
            device: 0,
            device_type: wgpu::DeviceType::DiscreteGpu,
            driver: String::new(),
            driver_info: String::new(),
            backend: wgpu::Backend::Vulkan,
        };

        assert!(is_nvidia_gpu(&make_info(0x10DE)));
        assert!(!is_nvidia_gpu(&make_info(0x1002)));
    }
}
