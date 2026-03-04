//! Environment settings application for the editor engine.

use super::EditorEngine;

impl EditorEngine {
    /// Apply environment settings to the render pipeline.
    ///
    /// Updates volumetric params (sun, fog) and all post-processing passes
    /// from the editor's `EnvironmentState`. Only writes to the GPU when
    /// the environment is marked dirty, then clears the flag.
    pub fn apply_environment(&mut self, env: &mut crate::environment::EnvironmentState) {
        if !env.is_dirty() {
            return;
        }

        // Cache vol params for use in render_frame_inner.
        let atmo = &env.atmosphere;
        self.env_sun_dir = [atmo.sun_direction.x, atmo.sun_direction.y, atmo.sun_direction.z];

        // Tint sun color based on elevation: at low angles the light path
        // through the atmosphere is much longer, scattering away blue/green
        // and leaving orange/red (Rayleigh extinction approximation).
        // Uses fixed gentle coefficients — the user's rayleigh/mie_scale
        // control sky scattering, not direct sun color.
        let sun_elevation = atmo.sun_direction.y.asin(); // radians
        let base_color = atmo.sun_color;
        let tinted_color = {
            let path = (1.0 / sun_elevation.max(0.02).sin()).min(12.0);
            let tau = glam::Vec3::new(0.02, 0.06, 0.15);
            let extinction = glam::Vec3::new(
                (-tau.x * path).exp(),
                (-tau.y * path).exp(),
                (-tau.z * path).exp(),
            );
            base_color * extinction
        };
        let sc = tinted_color * atmo.sun_intensity;
        self.env_sun_color = [sc.x, sc.y, sc.z];
        self.env_sun_intensity = atmo.sun_intensity;
        self.env_sun_color_raw = [tinted_color.x, tinted_color.y, tinted_color.z];
        self.env_rayleigh_scale = atmo.rayleigh_scale;
        self.env_mie_scale = atmo.mie_scale;
        self.env_atmosphere_enabled = atmo.enabled;

        let fog = &env.fog;
        self.env_fog_color = [fog.color.x, fog.color.y, fog.color.z];
        self.env_fog_density = if fog.enabled { fog.density } else { 0.0 };
        self.env_fog_height_falloff = fog.height_falloff;
        self.env_ambient_dust = fog.ambient_dust_density;
        self.env_dust_g = fog.dust_asymmetry;

        // Cloud settings → GPU CloudParams.
        let clouds = &env.clouds;
        self.env_cloud_settings.procedural_enabled = clouds.enabled;
        self.env_cloud_settings.cloud_min = clouds.altitude;
        self.env_cloud_settings.cloud_max = clouds.altitude + clouds.thickness;
        self.env_cloud_settings.cloud_density_scale = clouds.density;
        // Coverage → threshold: the FBM product (shape * weather * height_gradient)
        // is concentrated near 0 (three [0,1] values multiplied), so threshold
        // must drop quickly to reveal clouds. A power curve (coverage^0.35)
        // makes the slider response perceptually linear instead of bunching
        // all visible change into the last 20% of the range.
        self.env_cloud_settings.cloud_threshold =
            (1.0 - clouds.coverage.clamp(0.0, 1.0).powf(0.35)) * 0.4;
        self.env_cloud_settings.wind_direction = [clouds.wind_direction.x, clouds.wind_direction.z];
        self.env_cloud_settings.wind_speed = clouds.wind_speed;
        self.env_cloud_settings.shadow_enabled = clouds.enabled;

        let queue = &self.ctx.queue;
        let pp = &env.post_process;

        // Bloom.
        let t = pp.bloom_threshold;
        self.bloom.set_threshold(queue, t, t * 0.5);
        self.bloom_composite.set_intensity(queue, pp.bloom_intensity);

        // Tone mapping.
        let mode = if pp.tone_map_mode == 1 {
            rkf_render::ToneMapMode::AgX
        } else {
            rkf_render::ToneMapMode::Aces
        };
        self.tone_map.set_mode(queue, mode);
        self.tone_map.set_exposure(queue, pp.exposure);

        // Depth of field — max_coc=0 disables the blur.
        if pp.dof_enabled {
            self.dof.update_focus(queue, pp.dof_focus_distance, pp.dof_focus_range, pp.dof_max_coc);
        } else {
            self.dof.update_focus(queue, pp.dof_focus_distance, pp.dof_focus_range, 0.0);
        }

        // Sharpen.
        self.sharpen.set_strength(queue, pp.sharpen_strength);

        // Motion blur.
        self.motion_blur.set_intensity(queue, pp.motion_blur_intensity);

        // God rays blur.
        self.god_rays_blur.set_intensity(queue, pp.god_rays_intensity);

        // Cosmetics (vignette, grain, chromatic aberration).
        self.cosmetics.set_vignette(queue, pp.vignette_intensity);
        self.cosmetics.set_grain(queue, pp.grain_intensity);
        self.cosmetics.set_chromatic_aberration(queue, pp.chromatic_aberration);

        env.clear_dirty();
    }

    /// Apply environment settings from a snapshot (no dirty flag management).
    ///
    /// Used by the engine thread after snapshotting EnvironmentState under the
    /// editor_state lock and releasing the lock before doing GPU work.
    pub fn apply_environment_snapshot(&mut self, env: &crate::environment::EnvironmentState) {
        let atmo = &env.atmosphere;
        self.env_sun_dir = [atmo.sun_direction.x, atmo.sun_direction.y, atmo.sun_direction.z];

        let sun_elevation = atmo.sun_direction.y.asin();
        let base_color = atmo.sun_color;
        let tinted_color = {
            let path = (1.0 / sun_elevation.max(0.02).sin()).min(12.0);
            let tau = glam::Vec3::new(0.02, 0.06, 0.15);
            let extinction = glam::Vec3::new(
                (-tau.x * path).exp(),
                (-tau.y * path).exp(),
                (-tau.z * path).exp(),
            );
            base_color * extinction
        };
        let sc = tinted_color * atmo.sun_intensity;
        self.env_sun_color = [sc.x, sc.y, sc.z];
        self.env_sun_intensity = atmo.sun_intensity;
        self.env_sun_color_raw = [tinted_color.x, tinted_color.y, tinted_color.z];
        self.env_rayleigh_scale = atmo.rayleigh_scale;
        self.env_mie_scale = atmo.mie_scale;
        self.env_atmosphere_enabled = atmo.enabled;

        let fog = &env.fog;
        self.env_fog_color = [fog.color.x, fog.color.y, fog.color.z];
        self.env_fog_density = if fog.enabled { fog.density } else { 0.0 };
        self.env_fog_height_falloff = fog.height_falloff;
        self.env_ambient_dust = fog.ambient_dust_density;
        self.env_dust_g = fog.dust_asymmetry;

        let clouds = &env.clouds;
        self.env_cloud_settings.procedural_enabled = clouds.enabled;
        self.env_cloud_settings.cloud_min = clouds.altitude;
        self.env_cloud_settings.cloud_max = clouds.altitude + clouds.thickness;
        self.env_cloud_settings.cloud_density_scale = clouds.density;
        self.env_cloud_settings.cloud_threshold =
            (1.0 - clouds.coverage.clamp(0.0, 1.0).powf(0.35)) * 0.4;
        self.env_cloud_settings.wind_direction = [clouds.wind_direction.x, clouds.wind_direction.z];
        self.env_cloud_settings.wind_speed = clouds.wind_speed;
        self.env_cloud_settings.shadow_enabled = clouds.enabled;

        let queue = &self.ctx.queue;
        let pp = &env.post_process;

        let t = pp.bloom_threshold;
        self.bloom.set_threshold(queue, t, t * 0.5);
        self.bloom_composite.set_intensity(queue, pp.bloom_intensity);

        let mode = if pp.tone_map_mode == 1 {
            rkf_render::ToneMapMode::AgX
        } else {
            rkf_render::ToneMapMode::Aces
        };
        self.tone_map.set_mode(queue, mode);
        self.tone_map.set_exposure(queue, pp.exposure);

        if pp.dof_enabled {
            self.dof.update_focus(queue, pp.dof_focus_distance, pp.dof_focus_range, pp.dof_max_coc);
        } else {
            self.dof.update_focus(queue, pp.dof_focus_distance, pp.dof_focus_range, 0.0);
        }

        self.sharpen.set_strength(queue, pp.sharpen_strength);
        self.motion_blur.set_intensity(queue, pp.motion_blur_intensity);
        self.god_rays_blur.set_intensity(queue, pp.god_rays_intensity);
        self.cosmetics.set_vignette(queue, pp.vignette_intensity);
        self.cosmetics.set_grain(queue, pp.grain_intensity);
        self.cosmetics.set_chromatic_aberration(queue, pp.chromatic_aberration);
    }
}
