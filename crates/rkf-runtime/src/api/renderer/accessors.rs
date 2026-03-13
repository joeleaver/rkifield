//! Public accessor methods on the Renderer.

use glam::{Mat4, Vec3};

use rkf_core::material::Material;
use rkf_render::{BlitPass, Camera, GpuSceneV2, Light, LineVertex};
#[allow(deprecated)]
use rkf_render::material_table::MaterialTable;

use super::{Renderer, RenderEnvironment};
use crate::api::error::WorldError;
use crate::api::world::World;
use crate::components::CameraComponent;

impl Renderer {
    // ── Camera ─────────────────────────────────────────────────────────────

    /// Get a shared reference to the camera.
    pub fn camera(&self) -> &Camera {
        &self.camera
    }

    /// Get a mutable reference to the camera.
    pub fn camera_mut(&mut self) -> &mut Camera {
        &mut self.camera
    }

    /// Set the camera world position.
    pub fn set_camera_position(&mut self, pos: Vec3) {
        self.camera.position = pos;
    }

    /// Set the camera orientation (yaw and pitch in radians).
    pub fn set_camera_orientation(&mut self, yaw: f32, pitch: f32) {
        self.camera.yaw = yaw;
        self.camera.pitch = pitch;
    }

    /// Set the camera field of view (in degrees).
    pub fn set_camera_fov(&mut self, fov_degrees: f32) {
        self.camera.fov_degrees = fov_degrees;
    }

    /// Copy a camera entity's state to the viewport (rendering) camera.
    pub fn snap_camera_to(
        &mut self,
        world: &World,
        entity_id: uuid::Uuid,
    ) -> Result<(), WorldError> {
        let pos = world.position(entity_id)?;
        let cam = world
            .get::<CameraComponent>(entity_id)
            .map_err(|_| WorldError::MissingComponent(entity_id, "CameraComponent"))?;
        self.camera.position = pos.to_vec3();
        self.camera.yaw = cam.yaw.to_radians();
        self.camera.pitch = cam.pitch.to_radians();
        self.camera.fov_degrees = cam.fov_degrees;
        Ok(())
    }

    /// Camera-relative view-projection matrix for overlay rendering.
    pub fn view_projection(&self) -> Mat4 {
        self.camera
            .view_projection(self.internal_width, self.internal_height)
    }

    /// Current camera position in world space.
    pub fn camera_position(&self) -> Vec3 {
        self.camera.position
    }

    // ── Materials ──────────────────────────────────────────────────────────

    /// Set a material at the given index.
    pub fn set_material(&mut self, index: u16, material: Material) {
        let idx = index as usize;
        if idx >= self.materials.len() {
            self.materials.resize(idx + 1, Material::default());
        }
        self.materials[idx] = material;
        self.material_table = MaterialTable::upload(&self.ctx.device, &self.materials);
    }

    /// Set all materials at once.
    pub fn set_materials(&mut self, materials: &[Material]) {
        self.materials = materials.to_vec();
        self.material_table = MaterialTable::upload(&self.ctx.device, &self.materials);
    }

    /// Get the current materials.
    pub fn materials(&self) -> &[Material] {
        &self.materials
    }

    // ── Lights ─────────────────────────────────────────────────────────────

    /// Add a light and return its index.
    pub fn add_light(&mut self, light: Light) -> usize {
        let idx = self.world_lights.len();
        self.world_lights.push(light);
        self.light_buffer = rkf_render::LightBuffer::upload(&self.ctx.device, &self.world_lights);
        idx
    }

    /// Remove a light by index.
    pub fn remove_light(&mut self, index: usize) {
        if index < self.world_lights.len() {
            self.world_lights.remove(index);
            self.light_buffer = rkf_render::LightBuffer::upload(&self.ctx.device, &self.world_lights);
        }
    }

    /// Get a shared reference to a light.
    pub fn light(&self, index: usize) -> Option<&Light> {
        self.world_lights.get(index)
    }

    /// Get a mutable reference to a light.
    pub fn light_mut(&mut self, index: usize) -> Option<&mut Light> {
        self.world_lights.get_mut(index)
    }

    /// Replace all lights.
    pub fn set_lights(&mut self, lights: Vec<Light>) {
        self.world_lights = lights;
        self.light_buffer = rkf_render::LightBuffer::upload(&self.ctx.device, &self.world_lights);
    }

    /// Get all lights.
    pub fn lights(&self) -> &[Light] {
        &self.world_lights
    }

    // ── Environment ────────────────────────────────────────────────────────

    /// Set the render environment (sun, fog, clouds, post-processing).
    pub fn set_render_environment(&mut self, env: RenderEnvironment) {
        self.apply_post_process_settings(&env);
        self.render_env = env;
    }

    /// Get the current render environment.
    pub fn render_environment(&self) -> &RenderEnvironment {
        &self.render_env
    }

    /// Get a mutable reference to the render environment.
    pub fn render_environment_mut(&mut self) -> &mut RenderEnvironment {
        &mut self.render_env
    }

    /// Apply post-processing settings from the render environment to GPU passes.
    pub(super) fn apply_post_process_settings(&mut self, env: &RenderEnvironment) {
        let queue = &self.ctx.queue;

        // Bloom
        let t = env.bloom_threshold;
        self.bloom.set_threshold(queue, t, t * 0.5);
        self.bloom_composite.set_intensity(queue, env.bloom_intensity);

        // Tone mapping
        let mode = if env.tone_map_mode == 1 {
            rkf_render::ToneMapMode::AgX
        } else {
            rkf_render::ToneMapMode::Aces
        };
        self.tone_map.set_mode(queue, mode);
        self.tone_map.set_exposure(queue, env.exposure);

        // Depth of field
        if env.dof_enabled {
            self.dof.update_focus(queue, env.dof_focus_distance, env.dof_focus_range, env.dof_max_coc);
        } else {
            self.dof.update_focus(queue, env.dof_focus_distance, env.dof_focus_range, 0.0);
        }

        // Other post-FX
        self.sharpen.set_strength(queue, env.sharpen_strength);
        self.motion_blur.set_intensity(queue, env.motion_blur_intensity);
        self.god_rays_blur.set_intensity(queue, env.god_rays_intensity);
        self.cosmetics.set_vignette(queue, env.vignette_intensity);
        self.cosmetics.set_grain(queue, env.grain_intensity);
        self.cosmetics.set_chromatic_aberration(queue, env.chromatic_aberration);
    }

    // ── Debug / quality ────────────────────────────────────────────────────

    /// Set the debug visualization mode.
    pub fn set_debug_mode(&mut self, mode: u32) {
        self.shade_debug_mode = mode;
    }

    /// Get the current debug mode.
    pub fn debug_mode(&self) -> u32 {
        self.shade_debug_mode
    }

    // ── Wireframe ─────────────────────────────────────────────────────────

    /// Draw wireframe lines onto the offscreen render target.
    pub fn draw_wireframe(&mut self, vertices: &[LineVertex]) {
        if vertices.is_empty() {
            return;
        }
        let target = match self.offscreen_view.as_ref() {
            Some(v) => v,
            None => return,
        };
        let vp_matrix = self.view_projection();
        let viewport = (
            0.0,
            0.0,
            self.display_width as f32,
            self.display_height as f32,
        );
        let mut encoder = self.ctx.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("wireframe"),
            },
        );
        self.wireframe_pass.draw(
            &self.ctx.device,
            &self.ctx.queue,
            &mut encoder,
            target,
            vp_matrix,
            viewport,
            vertices,
        );
        self.ctx.queue.submit(std::iter::once(encoder.finish()));
    }

    // ── Stats ──────────────────────────────────────────────────────────────

    /// Get the current frame index.
    pub fn frame_index(&self) -> u32 {
        self.frame_index
    }

    /// Get the internal render resolution.
    pub fn internal_resolution(&self) -> (u32, u32) {
        (self.internal_width, self.internal_height)
    }

    /// Get the display/viewport resolution.
    pub fn display_resolution(&self) -> (u32, u32) {
        (self.display_width, self.display_height)
    }

    // ── GPU access (advanced) ──────────────────────────────────────────────

    /// Get a reference to the GPU device.
    pub fn device(&self) -> &wgpu::Device {
        &self.ctx.device
    }

    /// Get a reference to the GPU queue.
    pub fn queue(&self) -> &wgpu::Queue {
        &self.ctx.queue
    }

    /// Get the surface format used for presentation.
    pub fn surface_format(&self) -> wgpu::TextureFormat {
        self.surface_format
    }

    /// Get the final LDR compute output texture view (before offscreen blit).
    pub fn output_view(&self) -> &wgpu::TextureView {
        &self.cosmetics.output_view
    }

    /// Get the offscreen texture view for compositor integration.
    pub fn offscreen_view(&self) -> Option<&wgpu::TextureView> {
        self.offscreen_view.as_ref()
    }

    /// Get a reference to the blit pass (for custom target rendering).
    pub fn blit_pass(&self) -> &BlitPass {
        &self.blit
    }

    /// Access the GPU scene (for direct brick pool / brick map operations).
    pub fn gpu_scene(&self) -> &GpuSceneV2 {
        &self.gpu_scene
    }

    /// Access the GPU scene mutably.
    pub fn gpu_scene_mut(&mut self) -> &mut GpuSceneV2 {
        &mut self.gpu_scene
    }
}
