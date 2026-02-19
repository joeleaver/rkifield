//! Screen-space particle projection and rendering.
//!
//! Screen-space particles are a lightweight post-process overlay for weather
//! effects (rain, snow, dust). They project to screen coordinates, depth-test
//! against the G-buffer, and draw as oriented streaks or dots.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4, Vec4Swizzles};
use half::f16;

use crate::particle::{flags, Particle, RenderType};

/// GPU screen-space particle -- 48 bytes.
///
/// Matches the WGSL `ScreenParticle` struct in `particle_screen.wgsl`.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct ScreenParticle {
    /// Projected screen position in `[0, 1]` (top-left origin).
    pub screen_pos: [f32; 2],
    /// Linear depth for occlusion testing.
    pub depth: f32,
    /// Screen-space size in pixels.
    pub size: f32,
    /// Linear RGB color.
    pub color: [f32; 3],
    /// Opacity (fades with age when `FADE_OUT` flag is set).
    pub alpha: f32,
    /// Screen-space velocity for streak orientation.
    pub velocity_screen: [f32; 2],
    /// Emission intensity.
    pub emission: f32,
    /// Padding to 48 bytes.
    pub _pad: f32,
}

unsafe impl Zeroable for ScreenParticle {}
unsafe impl Pod for ScreenParticle {}

/// Projection parameters for screen-space particle generation.
#[derive(Debug, Clone, Copy)]
pub struct ProjectionParams {
    /// Combined view-projection matrix.
    pub view_proj: Mat4,
    /// Viewport width in pixels.
    pub screen_width: f32,
    /// Viewport height in pixels.
    pub screen_height: f32,
    /// Near clip plane distance.
    pub near: f32,
    /// Far clip plane distance.
    pub far: f32,
}

/// Small time step for computing screen-space velocity via finite difference.
const VELOCITY_DT: f32 = 1.0 / 60.0;

/// Project world-space particles to screen-space descriptors.
///
/// Filters for `render_type == ScreenSpace`, projects through `view_proj`,
/// discards particles behind the camera or outside the frustum, and computes
/// screen-space velocity for streak orientation.
pub fn project_particles(particles: &[Particle], params: &ProjectionParams) -> Vec<ScreenParticle> {
    let mut result = Vec::new();

    for p in particles {
        if p.render_type != RenderType::ScreenSpace as u8 {
            continue;
        }
        if p.flags & flags::ALIVE == 0 {
            continue;
        }

        let world_pos = Vec3::from(p.position);
        let clip = params.view_proj * Vec4::new(world_pos.x, world_pos.y, world_pos.z, 1.0);

        // Behind camera -- discard.
        if clip.w <= 0.0 {
            continue;
        }

        let ndc = clip.xyz() / clip.w;

        // Outside NDC cube -- discard.
        if ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0 || ndc.z < -1.0 || ndc.z > 1.0 {
            continue;
        }

        // NDC to screen [0, 1] (y-flipped: NDC +1 = top, screen 0 = top).
        let screen_x = (ndc.x + 1.0) * 0.5;
        let screen_y = (1.0 - ndc.y) * 0.5;

        // Linear depth from clip.w (view-space Z).
        let linear_depth = clip.w;

        // Decode particle size and compute perspective-scaled screen pixels.
        let size_world = f16::from_bits(p.size).to_f32();
        let size_pixels = if linear_depth > 1e-6 {
            size_world / linear_depth * params.screen_height
        } else {
            0.0
        };

        // Decode color.
        let r = f16::from_bits(p.color_emission[0]).to_f32();
        let g = f16::from_bits(p.color_emission[1]).to_f32();
        let b = f16::from_bits(p.color_emission[2]).to_f32();
        let emission = f16::from_bits(p.color_emission[3]).to_f32();

        // Alpha: 1.0 normally, fades with age if FADE_OUT set.
        let alpha = if p.flags & flags::FADE_OUT != 0 {
            if p.max_lifetime > 0.0 {
                (p.lifetime / p.max_lifetime).clamp(0.0, 1.0)
            } else {
                0.0
            }
        } else {
            1.0
        };

        // Compute screen-space velocity by projecting a future position.
        let vel = Vec3::from(p.velocity);
        let future_pos = world_pos + vel * VELOCITY_DT;
        let future_clip = params.view_proj * Vec4::new(future_pos.x, future_pos.y, future_pos.z, 1.0);

        let velocity_screen = if future_clip.w > 0.0 {
            let future_ndc = future_clip.xyz() / future_clip.w;
            let future_sx = (future_ndc.x + 1.0) * 0.5;
            let future_sy = (1.0 - future_ndc.y) * 0.5;
            [
                (future_sx - screen_x) * params.screen_width,
                (future_sy - screen_y) * params.screen_height,
            ]
        } else {
            [0.0, 0.0]
        };

        result.push(ScreenParticle {
            screen_pos: [screen_x, screen_y],
            depth: linear_depth,
            size: size_pixels,
            color: [r, g, b],
            alpha,
            velocity_screen,
            emission,
            _pad: 0.0,
        });
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::particle::{flags, Particle, RenderType};
    use glam::Mat4;
    use half::f16;

    /// Helper: create a ScreenSpace particle at `pos` with `size` and optional flags.
    fn make_screen_particle(pos: [f32; 3], size: f32, lifetime: f32, max_lifetime: f32, fade: bool) -> Particle {
        let mut f = flags::ALIVE;
        if fade {
            f |= flags::FADE_OUT;
        }
        Particle {
            position: pos,
            lifetime,
            velocity: [0.0, 0.0, -1.0], // moving into screen
            max_lifetime,
            color_emission: [
                f16::from_f32(1.0).to_bits(),
                f16::from_f32(0.5).to_bits(),
                f16::from_f32(0.2).to_bits(),
                f16::from_f32(0.0).to_bits(),
            ],
            size: f16::from_f32(size).to_bits(),
            render_type: RenderType::ScreenSpace as u8,
            flags: f,
            material_id: 0,
            _pad: 0,
        }
    }

    /// Simple perspective projection looking down -Z.
    fn test_projection_params() -> ProjectionParams {
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_4, 16.0 / 9.0, 0.1, 100.0);
        let view = Mat4::look_at_rh(
            Vec3::new(0.0, 0.0, 5.0),  // eye
            Vec3::new(0.0, 0.0, 0.0),  // target
            Vec3::new(0.0, 1.0, 0.0),  // up
        );
        ProjectionParams {
            view_proj: proj * view,
            screen_width: 1920.0,
            screen_height: 1080.0,
            near: 0.1,
            far: 100.0,
        }
    }

    #[test]
    fn test_screen_particle_size() {
        assert_eq!(std::mem::size_of::<ScreenParticle>(), 48);
    }

    #[test]
    fn test_project_visible() {
        // Particle at origin, camera at (0,0,5) looking at origin -- should be visible.
        let p = make_screen_particle([0.0, 0.0, 0.0], 0.1, 1.0, 1.0, false);
        let params = test_projection_params();
        let result = project_particles(&[p], &params);

        assert_eq!(result.len(), 1);
        let sp = &result[0];
        // Should be roughly center of screen.
        assert!((sp.screen_pos[0] - 0.5).abs() < 0.01, "x: {}", sp.screen_pos[0]);
        assert!((sp.screen_pos[1] - 0.5).abs() < 0.01, "y: {}", sp.screen_pos[1]);
        assert!(sp.depth > 0.0, "depth: {}", sp.depth);
        assert!(sp.size > 0.0, "size: {}", sp.size);
    }

    #[test]
    fn test_project_behind_camera() {
        // Particle behind the camera (z > 5, camera at z=5 looking -Z).
        let p = make_screen_particle([0.0, 0.0, 10.0], 0.1, 1.0, 1.0, false);
        let params = test_projection_params();
        let result = project_particles(&[p], &params);

        assert!(result.is_empty(), "particle behind camera should be filtered");
    }

    #[test]
    fn test_project_outside_frustum() {
        // Particle far to the side -- outside frustum.
        let p = make_screen_particle([100.0, 0.0, 0.0], 0.1, 1.0, 1.0, false);
        let params = test_projection_params();
        let result = project_particles(&[p], &params);

        assert!(result.is_empty(), "particle outside frustum should be filtered");
    }

    #[test]
    fn test_project_alpha_fade() {
        // With FADE_OUT, alpha = lifetime / max_lifetime.
        let p_full = make_screen_particle([0.0, 0.0, 0.0], 0.1, 2.0, 2.0, true);
        let p_half = make_screen_particle([0.0, 0.0, 0.0], 0.1, 1.0, 2.0, true);
        let p_none = make_screen_particle([0.0, 0.0, 0.0], 0.1, 2.0, 2.0, false);

        let params = test_projection_params();

        let r_full = project_particles(&[p_full], &params);
        let r_half = project_particles(&[p_half], &params);
        let r_none = project_particles(&[p_none], &params);

        assert!((r_full[0].alpha - 1.0).abs() < 1e-5, "full: {}", r_full[0].alpha);
        assert!((r_half[0].alpha - 0.5).abs() < 1e-5, "half: {}", r_half[0].alpha);
        assert!((r_none[0].alpha - 1.0).abs() < 1e-5, "no fade: {}", r_none[0].alpha);
    }

    #[test]
    fn test_project_filters_screen_type() {
        let screen = make_screen_particle([0.0, 0.0, 0.0], 0.1, 1.0, 1.0, false);
        let mut volumetric = screen;
        volumetric.render_type = RenderType::Volumetric as u8;
        let mut micro = screen;
        micro.render_type = RenderType::SdfMicro as u8;

        let params = test_projection_params();
        let result = project_particles(&[screen, volumetric, micro], &params);

        assert_eq!(result.len(), 1, "only ScreenSpace type should be projected");
    }

    #[test]
    fn test_project_skips_dead() {
        let mut p = make_screen_particle([0.0, 0.0, 0.0], 0.1, 1.0, 1.0, false);
        p.flags &= !flags::ALIVE;

        let params = test_projection_params();
        let result = project_particles(&[p], &params);

        assert!(result.is_empty());
    }

    #[test]
    fn test_project_velocity_screen() {
        // Particle moving in world space should have non-zero screen velocity.
        let mut p = make_screen_particle([0.0, 0.0, 0.0], 0.1, 1.0, 1.0, false);
        p.velocity = [1.0, 0.0, 0.0]; // moving right

        let params = test_projection_params();
        let result = project_particles(&[p], &params);

        assert_eq!(result.len(), 1);
        // Moving right in world -> positive screen X velocity.
        assert!(result[0].velocity_screen[0].abs() > 0.01, "should have screen velocity x: {}", result[0].velocity_screen[0]);
    }
}
