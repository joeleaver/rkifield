//! SDF micro-object particles.
//!
//! SDF micro-objects are small SDF primitives (spheres, capsules) registered
//! as temporary entries for the ray marcher. They receive full shading,
//! shadows, and GI -- unlike volumetric splats which only contribute density.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use half::f16;

use crate::particle::{flags, Particle, RenderType};

/// Maximum number of micro-objects extracted per frame.
pub const MAX_MICRO_OBJECTS: usize = 1000;

/// Shape type for SDF micro-objects.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MicroShape {
    /// Simple sphere SDF.
    Sphere,
    /// Capsule (line segment + radius). `end_offset` is the offset from
    /// the particle position to the second endpoint.
    Capsule {
        /// Offset from position to second endpoint.
        end_offset: Vec3,
    },
}

/// GPU-friendly SDF micro-object descriptor -- 48 bytes.
///
/// Matches the WGSL `SdfMicroObject` struct in `particle_micro_sdf.wgsl`.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct SdfMicroObject {
    /// World-space center position.
    pub position: [f32; 3],
    /// SDF radius (sphere radius or capsule thickness).
    pub radius: f32,
    /// Offset to second endpoint (capsule only; zero for sphere).
    pub end_offset: [f32; 3],
    /// Material ID (u32 for alignment; actual ID is u16 range).
    pub material_id: u32,
    /// Linear RGB color from particle.
    pub color: [f32; 3],
    /// Emission intensity.
    pub emission: f32,
}

unsafe impl Zeroable for SdfMicroObject {}
unsafe impl Pod for SdfMicroObject {}

/// Result of evaluating micro-objects at a point -- the closest hit.
#[derive(Debug, Clone, Copy)]
pub struct MicroHit {
    /// Signed distance to the closest micro-object surface.
    pub distance: f32,
    /// Material ID of the hit object.
    pub material_id: u16,
    /// Linear RGB color.
    pub color: Vec3,
    /// Emission intensity.
    pub emission: f32,
}

/// Extract GPU micro-object descriptors from alive particles.
///
/// Filters for `render_type == SdfMicro`, decodes f16 fields, and caps
/// output at [`MAX_MICRO_OBJECTS`].
pub fn extract_micro_objects(particles: &[Particle]) -> Vec<SdfMicroObject> {
    let mut result = Vec::with_capacity(MAX_MICRO_OBJECTS.min(particles.len()));

    for p in particles {
        if result.len() >= MAX_MICRO_OBJECTS {
            break;
        }
        if p.render_type != RenderType::SdfMicro as u8 {
            continue;
        }
        if p.flags & flags::ALIVE == 0 {
            continue;
        }

        let radius = f16::from_bits(p.size).to_f32();
        let r = f16::from_bits(p.color_emission[0]).to_f32();
        let g = f16::from_bits(p.color_emission[1]).to_f32();
        let b = f16::from_bits(p.color_emission[2]).to_f32();
        let emission = f16::from_bits(p.color_emission[3]).to_f32();

        result.push(SdfMicroObject {
            position: p.position,
            radius,
            end_offset: [0.0; 3], // sphere by default; capsule set by caller
            material_id: p.material_id as u32,
            color: [r, g, b],
            emission,
        });
    }

    result
}

/// Signed distance from `point` to a sphere centered at `center` with `radius`.
pub fn sdf_sphere(point: Vec3, center: Vec3, radius: f32) -> f32 {
    point.distance(center) - radius
}

/// Signed distance from `point` to a capsule defined by endpoints `a`, `b`
/// with `radius`.
pub fn sdf_capsule(point: Vec3, a: Vec3, b: Vec3, radius: f32) -> f32 {
    let pa = point - a;
    let ba = b - a;
    let len_sq = ba.length_squared();
    let h = if len_sq < 1e-10 {
        // Degenerate capsule (a == b) -- treat as sphere.
        0.0
    } else {
        (pa.dot(ba) / len_sq).clamp(0.0, 1.0)
    };
    (pa - ba * h).length() - radius
}

/// Evaluate all micro-objects at `pos` and return the closest hit (if any
/// surface is within positive distance threshold, or the point is inside).
///
/// Returns `None` if no micro-object is close enough to matter (all
/// distances > 0 and very far).
pub fn evaluate_micro_objects(pos: Vec3, objects: &[SdfMicroObject]) -> Option<MicroHit> {
    let mut best_dist = f32::MAX;
    let mut best_idx: Option<usize> = None;

    for (i, obj) in objects.iter().enumerate() {
        let center = Vec3::from(obj.position);
        let end_offset = Vec3::from(obj.end_offset);

        let d = if end_offset.length_squared() < 1e-10 {
            // Sphere.
            sdf_sphere(pos, center, obj.radius)
        } else {
            // Capsule.
            sdf_capsule(pos, center, center + end_offset, obj.radius)
        };

        if d < best_dist {
            best_dist = d;
            best_idx = Some(i);
        }
    }

    best_idx.map(|i| {
        let obj = &objects[i];
        MicroHit {
            distance: best_dist,
            material_id: obj.material_id as u16,
            color: Vec3::from(obj.color),
            emission: obj.emission,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::particle::{flags, Particle, RenderType};
    use half::f16;

    /// Helper: create an SdfMicro particle at `pos` with `size` radius.
    fn make_micro_particle(pos: [f32; 3], size: f32, material_id: u16) -> Particle {
        Particle {
            position: pos,
            lifetime: 1.0,
            velocity: [0.0; 3],
            max_lifetime: 1.0,
            color_emission: [
                f16::from_f32(0.8).to_bits(),
                f16::from_f32(0.2).to_bits(),
                f16::from_f32(0.5).to_bits(),
                f16::from_f32(2.0).to_bits(),
            ],
            size: f16::from_f32(size).to_bits(),
            render_type: RenderType::SdfMicro as u8,
            flags: flags::ALIVE,
            material_id,
            _pad: 0,
        }
    }

    #[test]
    fn test_micro_object_size() {
        assert_eq!(std::mem::size_of::<SdfMicroObject>(), 48);
    }

    #[test]
    fn test_extract_filters_sdf_micro() {
        let micro = make_micro_particle([1.0, 0.0, 0.0], 0.1, 5);
        let mut volumetric = Particle::new([2.0, 0.0, 0.0], [0.0; 3], 1.0);
        volumetric.render_type = RenderType::Volumetric as u8;
        let mut screen = Particle::new([3.0, 0.0, 0.0], [0.0; 3], 1.0);
        screen.render_type = RenderType::ScreenSpace as u8;

        let particles = [micro, volumetric, screen];
        let objects = extract_micro_objects(&particles);

        assert_eq!(objects.len(), 1);
        assert_eq!(objects[0].position, [1.0, 0.0, 0.0]);
        assert_eq!(objects[0].material_id, 5);
    }

    #[test]
    fn test_extract_budget_cap() {
        let particles: Vec<Particle> = (0..1500)
            .map(|i| make_micro_particle([i as f32, 0.0, 0.0], 0.05, 0))
            .collect();

        let objects = extract_micro_objects(&particles);
        assert_eq!(objects.len(), MAX_MICRO_OBJECTS);
    }

    #[test]
    fn test_extract_skips_dead() {
        let mut p = make_micro_particle([0.0; 3], 0.1, 0);
        p.flags &= !flags::ALIVE;

        let objects = extract_micro_objects(&[p]);
        assert!(objects.is_empty());
    }

    #[test]
    fn test_extract_decodes_f16_fields() {
        let p = make_micro_particle([1.0, 2.0, 3.0], 0.25, 7);
        let objects = extract_micro_objects(&[p]);
        assert_eq!(objects.len(), 1);

        let obj = &objects[0];
        assert!((obj.radius - 0.25).abs() < 0.01);
        assert!((obj.color[0] - 0.8).abs() < 0.01);
        assert!((obj.color[1] - 0.2).abs() < 0.01);
        assert!((obj.color[2] - 0.5).abs() < 0.01);
        assert!((obj.emission - 2.0).abs() < 0.01);
        assert_eq!(obj.material_id, 7);
    }

    #[test]
    fn test_sdf_sphere() {
        // Point at center -- distance should be -radius (inside).
        let d = sdf_sphere(Vec3::ZERO, Vec3::ZERO, 1.0);
        assert!((d - (-1.0)).abs() < 1e-6);

        // Point on surface -- distance should be ~0.
        let d = sdf_sphere(Vec3::new(1.0, 0.0, 0.0), Vec3::ZERO, 1.0);
        assert!(d.abs() < 1e-6);

        // Point outside -- distance should be positive.
        let d = sdf_sphere(Vec3::new(3.0, 0.0, 0.0), Vec3::ZERO, 1.0);
        assert!((d - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_sdf_capsule() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(0.0, 2.0, 0.0);
        let radius = 0.5;

        // Point at midpoint of axis, offset by radius -- on surface.
        let d = sdf_capsule(Vec3::new(0.5, 1.0, 0.0), a, b, radius);
        assert!(d.abs() < 1e-6, "on surface: {}", d);

        // Point on axis midpoint -- inside by radius.
        let d = sdf_capsule(Vec3::new(0.0, 1.0, 0.0), a, b, radius);
        assert!((d - (-0.5)).abs() < 1e-6, "on axis: {}", d);

        // Point beyond endpoint a -- distance from a minus radius.
        let d = sdf_capsule(Vec3::new(0.0, -1.0, 0.0), a, b, radius);
        assert!((d - 0.5).abs() < 1e-6, "beyond a: {}", d);

        // Point beyond endpoint b.
        let d = sdf_capsule(Vec3::new(0.0, 3.0, 0.0), a, b, radius);
        assert!((d - 0.5).abs() < 1e-6, "beyond b: {}", d);
    }

    #[test]
    fn test_sdf_capsule_degenerate() {
        // When a == b, capsule degenerates to sphere.
        let a = Vec3::new(1.0, 1.0, 1.0);
        let radius = 0.5;
        let d = sdf_capsule(Vec3::new(1.0, 1.0, 1.0), a, a, radius);
        assert!((d - (-0.5)).abs() < 1e-6);
    }

    #[test]
    fn test_evaluate_closest_hit() {
        let objects = vec![
            SdfMicroObject {
                position: [5.0, 0.0, 0.0],
                radius: 0.5,
                end_offset: [0.0; 3],
                material_id: 1,
                color: [1.0, 0.0, 0.0],
                emission: 0.0,
            },
            SdfMicroObject {
                position: [1.0, 0.0, 0.0],
                radius: 0.5,
                end_offset: [0.0; 3],
                material_id: 2,
                color: [0.0, 1.0, 0.0],
                emission: 3.0,
            },
        ];

        let hit = evaluate_micro_objects(Vec3::new(0.8, 0.0, 0.0), &objects).unwrap();
        // Closer to object at (1,0,0) with radius 0.5.
        assert_eq!(hit.material_id, 2);
        assert!((hit.color.y - 1.0).abs() < 1e-6);
        assert!((hit.emission - 3.0).abs() < 1e-6);
        // Distance: |0.8 - 1.0| - 0.5 = 0.2 - 0.5 = -0.3 (inside)
        assert!((hit.distance - (-0.3)).abs() < 1e-5, "distance: {}", hit.distance);
    }

    #[test]
    fn test_evaluate_empty() {
        let result = evaluate_micro_objects(Vec3::ZERO, &[]);
        assert!(result.is_none());
    }

    #[test]
    fn test_evaluate_capsule_micro_object() {
        let objects = vec![SdfMicroObject {
            position: [0.0, 0.0, 0.0],
            radius: 0.5,
            end_offset: [0.0, 2.0, 0.0],
            material_id: 10,
            color: [0.5, 0.5, 0.5],
            emission: 1.0,
        }];

        // Point near the middle of the capsule axis, offset by radius.
        let hit = evaluate_micro_objects(Vec3::new(0.5, 1.0, 0.0), &objects).unwrap();
        assert_eq!(hit.material_id, 10);
        assert!(hit.distance.abs() < 1e-5, "on surface: {}", hit.distance);
    }
}
