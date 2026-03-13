//! Renderer construction — `new`, `from_shared`, and `build`.

use std::sync::Arc;

use glam::Vec3;
use winit::window::Window as WinitWindow;

#[allow(deprecated)]
use rkf_render::material_table::{create_test_materials, MaterialTable};
use rkf_render::radiance_inject::RadianceInjectPass;
use rkf_render::radiance_mip::RadianceMipPass;
use rkf_render::{
    BlitPass, BloomCompositePass, BloomPass, Camera, CloudShadowPass,
    CoarseField, ColorGradePass, CosmeticsPass, DebugViewPass, DofPass, GBuffer,
    GodRaysBlurPass, GpuSceneV2, Light, LightBuffer, MotionBlurPass,
    RadianceVolume, RayMarchPass, RenderContext, ShadingPass,
    SharpenPass, TileObjectCullPass, ToneMapPass, VolCompositePass, VolMarchPass, VolShadowPass,
    VolUpscalePass, WireframePass, AutoExposurePass, COARSE_VOXEL_SIZE,
};

use super::helpers::{create_offscreen_target, create_readback_buffer};
use super::{Renderer, RendererConfig, RenderEnvironment, OFFSCREEN_FORMAT};

impl Renderer {
    /// Create a new Renderer with its own GPU device, attached to a window.
    ///
    /// Returns the Renderer and the configured `wgpu::Surface` for frame
    /// presentation. Pass the surface to [`render_to_surface`] each frame.
    pub fn new(window: Arc<dyn WinitWindow>, config: RendererConfig) -> (Self, wgpu::Surface<'static>) {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let surface = instance
            .create_surface(window.clone())
            .expect("create surface");
        let ctx = RenderContext::new(&instance, &surface);

        let surface_format =
            ctx.configure_surface(&surface, config.display_width, config.display_height);

        (Self::build(ctx, surface_format, config, false), surface)
    }

    /// Create a new Renderer using a shared GPU device (e.g. from an editor compositor).
    ///
    /// Sets up an offscreen render target at `display_width x display_height` for
    /// compositor integration. Use [`offscreen_view`] to get the texture view.
    pub fn from_shared(
        device: wgpu::Device,
        queue: wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        config: RendererConfig,
    ) -> Self {
        let ctx = RenderContext::from_shared(device, queue);
        Self::build(ctx, surface_format, config, true)
    }

    fn build(
        ctx: RenderContext,
        surface_format: wgpu::TextureFormat,
        config: RendererConfig,
        offscreen: bool,
    ) -> Self {
        let iw = config.internal_width;
        let ih = config.internal_height;
        let dw = config.display_width.max(64);
        let dh = config.display_height.max(64);

        // Create empty brick pool buffer.
        let brick_pool_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brick_pool"),
            size: 8,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let gpu_scene = GpuSceneV2::new(&ctx.device, brick_pool_buffer);

        let brick_pool_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brick_pool_shadow"),
            size: 8,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let gbuffer = GBuffer::new(&ctx.device, iw, ih);
        let tile_cull = TileObjectCullPass::new(&ctx.device, &gpu_scene, iw, ih);

        let coarse_field =
            CoarseField::from_scene_aabbs(&ctx.device, &[], COARSE_VOXEL_SIZE, 1.0);

        let ray_march =
            RayMarchPass::new(&ctx.device, &gpu_scene, &gbuffer, &tile_cull, &coarse_field);
        let debug_view = DebugViewPass::new(&ctx.device, &gbuffer);

        let materials = create_test_materials();
        let material_table = MaterialTable::upload(&ctx.device, &materials);

        // Lights — start with a default sun.
        let world_lights = vec![Light::directional(
            [0.5, 1.0, 0.3],
            [1.0, 0.95, 0.85],
            3.0,
            true,
        )];
        let light_buffer = LightBuffer::upload(&ctx.device, &world_lights);

        // GI
        let radiance_volume = RadianceVolume::new(&ctx.device);
        let radiance_inject = RadianceInjectPass::new(
            &ctx.device,
            &gpu_scene,
            &material_table.buffer,
            &light_buffer,
            &radiance_volume,
            &coarse_field,
        );
        let radiance_mip = RadianceMipPass::new(&ctx.device, &radiance_volume);

        // Brush overlay (empty placeholder — runtime doesn't use sculpting)
        let brush_overlay = rkf_render::BrushOverlay::empty(&ctx.device);
        // Color pool (empty placeholder — runtime doesn't use painting)
        let gpu_color_pool = rkf_render::GpuColorPool::empty(&ctx.device);

        // Shading
        let shading_pass = ShadingPass::new(
            &ctx.device,
            &gbuffer,
            &gpu_scene,
            &light_buffer,
            &coarse_field,
            &radiance_volume,
            &material_table.buffer,
            iw,
            ih,
            None,
            &brush_overlay,
            &gpu_color_pool,
        );

        // Volumetrics
        let vol_shadow =
            VolShadowPass::new(&ctx.device, &ctx.queue, &coarse_field.bind_group_layout);
        let cloud_shadow = CloudShadowPass::new(&ctx.device);
        let half_w = iw / 2;
        let half_h = ih / 2;
        let vol_march = VolMarchPass::new(
            &ctx.device,
            &ctx.queue,
            &gbuffer.position_view,
            &vol_shadow.shadow_view,
            &cloud_shadow.shadow_view,
            half_w,
            half_h,
            iw,
            ih,
        );
        let vol_upscale = VolUpscalePass::new(
            &ctx.device,
            &vol_march.output_view,
            &gbuffer.position_view,
            iw,
            ih,
            half_w,
            half_h,
        );
        let vol_composite = VolCompositePass::new(
            &ctx.device,
            &shading_pass.hdr_view,
            &vol_upscale.output_view,
            iw,
            ih,
        );

        // Post-processing
        let god_rays_blur = GodRaysBlurPass::new(
            &ctx.device,
            &vol_composite.output_view,
            &gbuffer.position_view,
            iw,
            ih,
        );
        let bloom = BloomPass::new(&ctx.device, &god_rays_blur.output_view, iw, ih);
        let auto_exposure =
            AutoExposurePass::new(&ctx.device, &god_rays_blur.output_view, iw, ih);
        let dof = DofPass::new(
            &ctx.device,
            &god_rays_blur.output_view,
            &gbuffer.position_view,
            iw,
            ih,
        );
        let motion_blur = MotionBlurPass::new(
            &ctx.device,
            &dof.output_view,
            &gbuffer.motion_view,
            iw,
            ih,
        );
        let bloom_composite = BloomCompositePass::new(
            &ctx.device,
            &motion_blur.output_view,
            bloom.mip_views(),
            iw,
            ih,
        );
        let tone_map = ToneMapPass::new_with_exposure(
            &ctx.device,
            &bloom_composite.output_view,
            iw,
            ih,
            Some(auto_exposure.get_exposure_buffer()),
        );
        let color_grade = ColorGradePass::new(
            &ctx.device,
            &ctx.queue,
            &tone_map.ldr_view,
            iw,
            ih,
        );
        let cosmetics = CosmeticsPass::new(&ctx.device, &color_grade.output_view, iw, ih);
        let sharpen = SharpenPass::new(
            &ctx.device,
            &vol_composite.output_view,
            &gbuffer,
            iw,
            ih,
        );

        // Blit target format depends on mode.
        let blit_format = if offscreen { OFFSCREEN_FORMAT } else { surface_format };
        let blit = BlitPass::new(&ctx.device, &cosmetics.output_view, blit_format);

        // Wireframe pass (targets offscreen or swapchain format).
        let wireframe_format = if offscreen { OFFSCREEN_FORMAT } else { surface_format };
        let wireframe_pass = WireframePass::new(&ctx.device, wireframe_format);

        // Offscreen render target (compositor path only).
        let (offscreen_texture, offscreen_view, offscreen_blit) = if offscreen {
            let (tex, view) = create_offscreen_target(&ctx.device, dw, dh);
            let os_blit = BlitPass::new(&ctx.device, &cosmetics.output_view, OFFSCREEN_FORMAT);
            (Some(tex), Some(view), Some(os_blit))
        } else {
            (None, None, None)
        };

        // Camera
        let mut camera = Camera::new(Vec3::new(0.0, 2.5, 5.0));
        camera.pitch = -0.15;
        camera.move_speed = 5.0;

        // Readback buffers
        let readback_width = if offscreen { dw } else { iw };
        let readback_height = if offscreen { dh } else { ih };
        let readback_buffer = create_readback_buffer(&ctx.device, readback_width, readback_height);

        // Pick readback — 1 pixel from R32Uint material texture (256-byte aligned).
        let pick_readback_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pick_readback"),
            size: 256,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Brush hit readback — 1 pixel from Rgba32Float position (16 bytes, 256-aligned).
        let brush_readback_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("brush_readback"),
            size: 256,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        Self {
            ctx,
            gpu_scene,
            brick_pool_buffer,
            gbuffer,
            tile_cull,
            coarse_field,
            ray_march,
            debug_view,
            shading_pass,
            brush_overlay,
            gpu_color_pool,
            material_table,
            radiance_volume,
            radiance_inject,
            radiance_mip,
            vol_shadow,
            cloud_shadow,
            vol_march,
            vol_upscale,
            vol_composite,
            god_rays_blur,
            bloom,
            auto_exposure,
            dof,
            motion_blur,
            bloom_composite,
            tone_map,
            color_grade,
            cosmetics,
            sharpen,
            blit,
            wireframe_pass,
            offscreen_texture,
            offscreen_view,
            offscreen_blit,
            readback_buffer,
            pick_readback_buffer,
            brush_readback_buffer,
            camera,
            world_lights,
            materials,
            light_buffer,
            render_env: RenderEnvironment::default(),
            frame_index: 0,
            prev_vp: [[0.0; 4]; 4],
            shade_debug_mode: 0,
            accumulated_time: 0.0,
            internal_width: iw,
            internal_height: ih,
            display_width: dw,
            display_height: dh,
            surface_format,
        }
    }
}
