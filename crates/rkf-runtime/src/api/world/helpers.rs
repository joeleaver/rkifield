//! Free-standing helper functions for scene I/O serialization.

use glam::Vec3;

#[allow(dead_code)]
pub(super) fn parse_analytical_primitive(
    name: &str,
    params: Option<&[f32]>,
) -> rkf_core::scene_node::SdfPrimitive {
    use rkf_core::scene_node::SdfPrimitive;

    match name.to_lowercase().as_str() {
        "sphere" => {
            let radius = params.and_then(|p| p.first().copied()).unwrap_or(0.5);
            SdfPrimitive::Sphere { radius }
        }
        "box" => {
            let half = params
                .map(|p| {
                    Vec3::new(
                        *p.first().unwrap_or(&0.5),
                        *p.get(1).unwrap_or(&0.5),
                        *p.get(2).unwrap_or(&0.5),
                    )
                })
                .unwrap_or(Vec3::splat(0.5));
            SdfPrimitive::Box {
                half_extents: half,
            }
        }
        "capsule" => {
            let radius = params.and_then(|p| p.first().copied()).unwrap_or(0.2);
            let half_height = params.and_then(|p| p.get(1).copied()).unwrap_or(0.5);
            SdfPrimitive::Capsule {
                radius,
                half_height,
            }
        }
        "torus" => {
            let major = params.and_then(|p| p.first().copied()).unwrap_or(0.5);
            let minor = params.and_then(|p| p.get(1).copied()).unwrap_or(0.1);
            SdfPrimitive::Torus {
                major_radius: major,
                minor_radius: minor,
            }
        }
        "cylinder" => {
            let radius = params.and_then(|p| p.first().copied()).unwrap_or(0.3);
            let half_height = params.and_then(|p| p.get(1).copied()).unwrap_or(0.5);
            SdfPrimitive::Cylinder {
                radius,
                half_height,
            }
        }
        "plane" => {
            let normal = params
                .map(|p| {
                    Vec3::new(
                        *p.first().unwrap_or(&0.0),
                        *p.get(1).unwrap_or(&1.0),
                        *p.get(2).unwrap_or(&0.0),
                    )
                })
                .unwrap_or(Vec3::Y);
            let distance = params.and_then(|p| p.get(3).copied()).unwrap_or(0.0);
            SdfPrimitive::Plane { normal, distance }
        }
        _ => SdfPrimitive::Sphere { radius: 0.5 },
    }
}

#[allow(dead_code)]
pub(super) fn primitive_to_analytical(
    primitive: &rkf_core::scene_node::SdfPrimitive,
) -> (String, Vec<f32>) {
    use rkf_core::scene_node::SdfPrimitive;

    match primitive {
        SdfPrimitive::Sphere { radius } => ("sphere".to_string(), vec![*radius]),
        SdfPrimitive::Box { half_extents } => (
            "box".to_string(),
            vec![half_extents.x, half_extents.y, half_extents.z],
        ),
        SdfPrimitive::Capsule {
            radius,
            half_height,
        } => ("capsule".to_string(), vec![*radius, *half_height]),
        SdfPrimitive::Torus {
            major_radius,
            minor_radius,
        } => ("torus".to_string(), vec![*major_radius, *minor_radius]),
        SdfPrimitive::Cylinder {
            radius,
            half_height,
        } => ("cylinder".to_string(), vec![*radius, *half_height]),
        SdfPrimitive::Plane { normal, distance } => (
            "plane".to_string(),
            vec![normal.x, normal.y, normal.z, *distance],
        ),
    }
}
