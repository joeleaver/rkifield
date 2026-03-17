//! Editor wireframe helpers — gizmo rendering and node tree AABB computation.
//!
//! Generic wireframe types (`WireframePass`, `LineVertex`) and shape helpers
//! (`obb_wireframe`, `aabb_wireframe`, `sphere_wireframe`, etc.) live in
//! `rkf_render::wireframe`. This module re-exports them and adds editor-specific
//! gizmo wireframes and node tree utilities.

// Re-export generic wireframe types from rkf-render.
pub use rkf_render::wireframe::{
    aabb_wireframe, circle_wireframe, ground_grid_wireframe, obb_wireframe,
    point_light_wireframe, spot_light_wireframe, LineVertex, WireframePass,
};
// Also re-export less commonly used helpers (suppress unused-import warnings).
#[allow(unused_imports)]
pub use rkf_render::wireframe::{crosshair, directional_light_wireframe};

use glam::{Mat4, Vec3};

// ── Transform gizmo wireframes ──────────────────────────────────────────────

/// Axis colors: R = X, G = Y, B = Z.
const GIZMO_X_COLOR: [f32; 4] = [1.0, 0.2, 0.2, 1.0];
const GIZMO_Y_COLOR: [f32; 4] = [0.2, 1.0, 0.2, 1.0];
const GIZMO_Z_COLOR: [f32; 4] = [0.3, 0.3, 1.0, 1.0];

/// Bright saturated colors for hovered axes.
const GIZMO_X_HOVER: [f32; 4] = [1.0, 0.4, 0.4, 1.0];
const GIZMO_Y_HOVER: [f32; 4] = [0.4, 1.0, 0.4, 1.0];
const GIZMO_Z_HOVER: [f32; 4] = [0.4, 0.4, 1.0, 1.0];

/// Dimmed colors for non-hovered axes (when another axis is hovered).
const GIZMO_X_DIM: [f32; 4] = [0.4, 0.08, 0.08, 0.4];
const GIZMO_Y_DIM: [f32; 4] = [0.08, 0.4, 0.08, 0.4];
const GIZMO_Z_DIM: [f32; 4] = [0.12, 0.12, 0.4, 0.4];

use crate::gizmo::GizmoAxis;

/// Pick the color for a gizmo axis based on hover state.
///
/// - No hover active (`hovered == None`): normal brightness
/// - This axis is hovered: bright saturated
/// - Another axis is hovered: dimmed
fn gizmo_axis_color(axis_idx: usize, hovered: GizmoAxis) -> [f32; 4] {
    let normal = [GIZMO_X_COLOR, GIZMO_Y_COLOR, GIZMO_Z_COLOR];
    let bright = [GIZMO_X_HOVER, GIZMO_Y_HOVER, GIZMO_Z_HOVER];
    let dim = [GIZMO_X_DIM, GIZMO_Y_DIM, GIZMO_Z_DIM];

    let hovered_idx = match hovered {
        GizmoAxis::X => Some(0),
        GizmoAxis::Y => Some(1),
        GizmoAxis::Z => Some(2),
        GizmoAxis::None => None,
        _ => None,
    };

    match hovered_idx {
        None => normal[axis_idx],
        Some(hi) if hi == axis_idx => bright[axis_idx],
        Some(_) => dim[axis_idx],
    }
}

/// Build a translate gizmo: 3 axis arrows from `center` with length `size`.
///
/// Each axis is a line + arrowhead in its respective color (R=X, G=Y, B=Z).
/// `size` should be proportional to camera distance for constant screen size.
/// `hovered` controls highlight: hovered axis brightens + thickens, others dim.
/// `cam_pos` is used to compute perpendicular offset for thickness simulation.
pub fn translate_gizmo_wireframe(
    center: Vec3, size: f32, hovered: GizmoAxis, cam_pos: Vec3,
) -> Vec<LineVertex> {
    let mut verts = Vec::new();
    let head_len = size * 0.2;
    let head_radius = size * 0.06;
    let to_cam = (cam_pos - center).normalize_or_zero();

    for (axis_idx, axis_dir) in [(0, Vec3::X), (1, Vec3::Y), (2, Vec3::Z)] {
        let color = gizmo_axis_color(axis_idx, hovered);
        let is_hovered = matches!(
            (axis_idx, hovered),
            (0, GizmoAxis::X) | (1, GizmoAxis::Y) | (2, GizmoAxis::Z)
        );

        let tip = center + axis_dir * size;
        // Shaft.
        verts.push(LineVertex { position: center.to_array(), color });
        verts.push(LineVertex { position: tip.to_array(), color });

        // Thickness: 2 extra parallel lines offset perpendicular to axis and camera.
        if is_hovered {
            let perp = axis_dir.cross(to_cam).normalize_or_zero();
            let offset = size * 0.004;
            for sign in [-1.0f32, 1.0] {
                let off = perp * (offset * sign);
                verts.push(LineVertex { position: (center + off).to_array(), color });
                verts.push(LineVertex { position: (tip + off).to_array(), color });
            }
        }

        // Arrowhead: 4 lines from tip to a base ring.
        let tangent = if axis_dir.dot(Vec3::Y).abs() < 0.99 {
            axis_dir.cross(Vec3::Y).normalize()
        } else {
            axis_dir.cross(Vec3::X).normalize()
        };
        let bitangent = axis_dir.cross(tangent);
        let base = tip - axis_dir * head_len;

        let step = std::f32::consts::TAU / 4.0;
        for i in 0..4 {
            let a = step * i as f32;
            let p = base + (tangent * a.cos() + bitangent * a.sin()) * head_radius;
            verts.push(LineVertex { position: tip.to_array(), color });
            verts.push(LineVertex { position: p.to_array(), color });
        }
    }
    verts
}

/// Build a rotate gizmo: 3 axis rings at `center` with radius `size`.
///
/// `hovered` controls highlight: hovered axis brightens + thickens, others dim.
/// `cam_pos` is used to compute perpendicular offset for thickness simulation.
pub fn rotate_gizmo_wireframe(
    center: Vec3, size: f32, hovered: GizmoAxis, cam_pos: Vec3,
) -> Vec<LineVertex> {
    let segs = 48;
    let to_cam = (cam_pos - center).normalize_or_zero();
    let offset_mag = size * 0.004;

    let mut verts = Vec::new();
    for (axis_idx, normal) in [(0, Vec3::X), (1, Vec3::Y), (2, Vec3::Z)] {
        let color = gizmo_axis_color(axis_idx, hovered);
        let is_hovered = matches!(
            (axis_idx, hovered),
            (0, GizmoAxis::X) | (1, GizmoAxis::Y) | (2, GizmoAxis::Z)
        );

        verts.extend(circle_wireframe(center, normal, size, color, segs));

        // Thickness: 2 extra rings at slightly different radii.
        if is_hovered {
            // Offset along the radial direction (expand/shrink radius).
            // Also offset along the ring normal for visible thickening from any angle.
            let perp = normal.cross(to_cam).normalize_or_zero();
            for sign in [-1.0f32, 1.0] {
                let off_center = center + perp * (offset_mag * sign);
                verts.extend(circle_wireframe(off_center, normal, size, color, segs));
            }
        }
    }
    verts
}

/// Build a scale gizmo: 3 axis lines with small cubes at the ends.
///
/// `hovered` controls highlight: hovered axis brightens + thickens, others dim.
/// `cam_pos` is used to compute perpendicular offset for thickness simulation.
pub fn scale_gizmo_wireframe(
    center: Vec3, size: f32, hovered: GizmoAxis, cam_pos: Vec3,
) -> Vec<LineVertex> {
    let cube_half = size * 0.06;
    let to_cam = (cam_pos - center).normalize_or_zero();
    let mut verts = Vec::new();

    for (axis_idx, axis_dir) in [(0, Vec3::X), (1, Vec3::Y), (2, Vec3::Z)] {
        let color = gizmo_axis_color(axis_idx, hovered);
        let is_hovered = matches!(
            (axis_idx, hovered),
            (0, GizmoAxis::X) | (1, GizmoAxis::Y) | (2, GizmoAxis::Z)
        );

        let tip = center + axis_dir * size;
        // Shaft.
        verts.push(LineVertex { position: center.to_array(), color });
        verts.push(LineVertex { position: tip.to_array(), color });

        // Thickness: 2 extra parallel lines.
        if is_hovered {
            let perp = axis_dir.cross(to_cam).normalize_or_zero();
            let offset = size * 0.004;
            for sign in [-1.0f32, 1.0] {
                let off = perp * (offset * sign);
                verts.push(LineVertex { position: (center + off).to_array(), color });
                verts.push(LineVertex { position: (tip + off).to_array(), color });
            }
        }

        // Small cube at the tip.
        let min = tip - Vec3::splat(cube_half);
        let max = tip + Vec3::splat(cube_half);
        verts.extend(aabb_wireframe(min, max, color));
    }

    // Center cube for uniform scale (highlight if View is hovered).
    let center_color = if hovered == GizmoAxis::View {
        [1.0, 1.0, 1.0, 1.0]
    } else if hovered != GizmoAxis::None {
        [0.4, 0.4, 0.4, 0.4]
    } else {
        [0.9, 0.9, 0.9, 1.0]
    };
    let cc = Vec3::splat(cube_half * 1.2);
    verts.extend(aabb_wireframe(center - cc, center + cc, center_color));
    verts
}

/// Compute the world-space AABB of a scene node tree by recursing through all children.
///
/// `parent_world` is the accumulated world transform (translation + rotation + scale)
/// for the current node. For root calls, pass the object's world transform:
/// `Mat4::from_scale_rotation_translation(scale, rotation, world_pos)`.
pub fn compute_node_tree_aabb(
    node: &rkf_core::scene_node::SceneNode,
    parent_world: Mat4,
) -> Option<(Vec3, Vec3)> {
    use rkf_core::scene_node::SdfSource;

    let local = Mat4::from_scale_rotation_translation(
        node.local_transform.scale,
        node.local_transform.rotation,
        node.local_transform.position,
    );
    let world = parent_world * local;

    let mut aabb_min = Vec3::splat(f32::MAX);
    let mut aabb_max = Vec3::splat(f32::MIN);
    let mut has_bounds = false;

    // Get local bounds for this node's SDF.
    let local_bounds = match &node.sdf_source {
        SdfSource::Analytical { primitive, .. } => {
            Some(local_bounds_for_primitive(primitive))
        }
        SdfSource::Voxelized { brick_map_handle, voxel_size, .. } => {
            let brick_extent = voxel_size * 8.0;
            let grid_half = Vec3::new(
                brick_map_handle.dims.x as f32 * brick_extent * 0.5,
                brick_map_handle.dims.y as f32 * brick_extent * 0.5,
                brick_map_handle.dims.z as f32 * brick_extent * 0.5,
            );
            Some((-grid_half, grid_half))
        }
        SdfSource::None => None,
    };

    // Transform local bounds corners to world space and expand AABB.
    if let Some((lmin, lmax)) = local_bounds {
        let local_corners = [
            Vec3::new(lmin.x, lmin.y, lmin.z),
            Vec3::new(lmax.x, lmin.y, lmin.z),
            Vec3::new(lmax.x, lmax.y, lmin.z),
            Vec3::new(lmin.x, lmax.y, lmin.z),
            Vec3::new(lmin.x, lmin.y, lmax.z),
            Vec3::new(lmax.x, lmin.y, lmax.z),
            Vec3::new(lmax.x, lmax.y, lmax.z),
            Vec3::new(lmin.x, lmax.y, lmax.z),
        ];
        for c in &local_corners {
            let wc = world.transform_point3(*c);
            aabb_min = aabb_min.min(wc);
            aabb_max = aabb_max.max(wc);
        }
        has_bounds = true;
    }

    // Recurse into children.
    for child in &node.children {
        if let Some((cmin, cmax)) = compute_node_tree_aabb(child, world) {
            aabb_min = aabb_min.min(cmin);
            aabb_max = aabb_max.max(cmax);
            has_bounds = true;
        }
    }

    if has_bounds {
        Some((aabb_min, aabb_max))
    } else {
        None
    }
}

/// Given a scene tree entity ID, find the matching core SceneNode and the
/// world transform of its **parent**. Returns `None` if the entity doesn't
/// belong to this object.
///
/// The returned `parent_world` is suitable for passing directly to
/// [`compute_node_tree_aabb`], which applies the node's own `local_transform`.
///
/// Entity IDs follow the scheme: root = `obj.id`, children = `parent_id * 1000 + child_index`.
/// We decode by repeatedly dividing by 1000 to build a path of child indices,
/// then walk the core SceneNode tree along that path.
pub fn find_child_node_and_transform<'a>(
    entity_id: u64,
    obj_id: u64,
    root_node: &'a rkf_core::scene_node::SceneNode,
    root_world: Mat4,
) -> Option<(&'a rkf_core::scene_node::SceneNode, Mat4)> {
    // Decode the entity_id into a path of child indices.
    let mut path = Vec::new();
    let mut eid = entity_id;
    while eid != obj_id {
        if eid < 1000 {
            return None; // Not a child of this object.
        }
        let child_idx = (eid % 1000) as usize;
        eid /= 1000;
        path.push(child_idx);
    }
    // path is in reverse order (deepest child first), reverse it.
    path.reverse();

    // Walk the core SceneNode tree along the path, accumulating the
    // parent's world transform. We stop one step short so the returned
    // matrix does NOT include the target node's own local_transform
    // (compute_node_tree_aabb will apply it).
    let mut node = root_node;
    let mut parent_world = root_world;

    // First, include the root node's own transform (root_world is the object transform).
    let root_local = Mat4::from_scale_rotation_translation(
        node.local_transform.scale,
        node.local_transform.rotation,
        node.local_transform.position,
    );
    let mut world = root_world * root_local;

    for (i, &idx) in path.iter().enumerate() {
        if idx >= node.children.len() {
            return None;
        }
        parent_world = world; // world before the child's transform
        node = &node.children[idx];
        if i + 1 < path.len() {
            // Intermediate node: accumulate its transform.
            let local = Mat4::from_scale_rotation_translation(
                node.local_transform.scale,
                node.local_transform.rotation,
                node.local_transform.position,
            );
            world = parent_world * local;
        }
        // For the final node (target), don't apply its transform —
        // compute_node_tree_aabb will do that.
    }

    Some((node, parent_world))
}

/// Compute local-space bounding box for an SDF primitive.
pub fn local_bounds_for_primitive(primitive: &rkf_core::scene_node::SdfPrimitive) -> (Vec3, Vec3) {
    use rkf_core::scene_node::SdfPrimitive;
    match *primitive {
        SdfPrimitive::Sphere { radius } => {
            (Vec3::splat(-radius), Vec3::splat(radius))
        }
        SdfPrimitive::Box { half_extents } => {
            (-half_extents, half_extents)
        }
        SdfPrimitive::Capsule { radius, half_height } => {
            let h = half_height + radius;
            (Vec3::new(-radius, -h, -radius), Vec3::new(radius, h, radius))
        }
        SdfPrimitive::Torus { major_radius, minor_radius } => {
            let r = major_radius + minor_radius;
            (Vec3::new(-r, -minor_radius, -r), Vec3::new(r, minor_radius, r))
        }
        SdfPrimitive::Cylinder { radius, half_height } => {
            (Vec3::new(-radius, -half_height, -radius), Vec3::new(radius, half_height, radius))
        }
        SdfPrimitive::Plane { .. } => {
            // Infinite plane — use a large representative extent.
            let e = 100.0;
            (Vec3::new(-e, -0.01, -e), Vec3::new(e, 0.01, e))
        }
    }
}
