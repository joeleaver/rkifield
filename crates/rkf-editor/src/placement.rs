//! AABB computation for scene objects and SDF primitives.

#![allow(dead_code)]

/// Compute a local-space AABB for an analytical SDF primitive.
pub fn primitive_local_aabb(primitive: &rkf_core::scene_node::SdfPrimitive) -> (glam::Vec3, glam::Vec3) {
    use rkf_core::scene_node::SdfPrimitive;
    match *primitive {
        SdfPrimitive::Sphere { radius } => (
            glam::Vec3::splat(-radius),
            glam::Vec3::splat(radius),
        ),
        SdfPrimitive::Box { half_extents } => (-half_extents, half_extents),
        SdfPrimitive::Capsule { radius, half_height } => (
            glam::Vec3::new(-radius, -(half_height + radius), -radius),
            glam::Vec3::new(radius, half_height + radius, radius),
        ),
        SdfPrimitive::Torus { major_radius, minor_radius } => {
            let r = major_radius + minor_radius;
            (glam::Vec3::new(-r, -minor_radius, -r), glam::Vec3::new(r, minor_radius, r))
        }
        SdfPrimitive::Cylinder { radius, half_height } => (
            glam::Vec3::new(-radius, -half_height, -radius),
            glam::Vec3::new(radius, half_height, radius),
        ),
        SdfPrimitive::Plane { .. } => {
            // Infinite planes get a large but finite AABB.
            (glam::Vec3::splat(-1000.0), glam::Vec3::splat(1000.0))
        }
    }
}

/// Compute a local-space AABB for a SceneObject by walking its entire node tree.
///
/// Recursively traverses all child nodes, applying their local transforms,
/// and takes the union of all primitive AABBs. This correctly handles group
/// nodes (SdfSource::None) that have analytical/voxelized children.
///
/// Returns an AABB in the object's local space (no world position applied).
pub fn compute_object_local_aabb(obj: &rkf_core::scene::SceneObject) -> rkf_core::aabb::Aabb {
    let mut lo = glam::Vec3::splat(f32::MAX);
    let mut hi = glam::Vec3::splat(f32::MIN);

    fn walk_node(
        node: &rkf_core::scene_node::SceneNode,
        parent_matrix: glam::Mat4,
        lo: &mut glam::Vec3,
        hi: &mut glam::Vec3,
    ) {
        use rkf_core::scene_node::SdfSource;

        let local = node.local_transform.to_matrix();
        let world = parent_matrix * local;

        // Get this node's local AABB (if it has geometry).
        let bounds = match &node.sdf_source {
            SdfSource::Analytical { primitive, .. } => Some(primitive_local_aabb(primitive)),
            SdfSource::Voxelized { aabb, .. } => Some((aabb.min, aabb.max)),
            SdfSource::None => None,
        };

        if let Some((bmin, bmax)) = bounds {
            // Transform all 8 corners through the accumulated matrix.
            let corners = [
                glam::Vec3::new(bmin.x, bmin.y, bmin.z),
                glam::Vec3::new(bmax.x, bmin.y, bmin.z),
                glam::Vec3::new(bmin.x, bmax.y, bmin.z),
                glam::Vec3::new(bmax.x, bmax.y, bmin.z),
                glam::Vec3::new(bmin.x, bmin.y, bmax.z),
                glam::Vec3::new(bmax.x, bmin.y, bmax.z),
                glam::Vec3::new(bmin.x, bmax.y, bmax.z),
                glam::Vec3::new(bmax.x, bmax.y, bmax.z),
            ];
            for c in &corners {
                let p = world.transform_point3(*c);
                *lo = lo.min(p);
                *hi = hi.max(p);
            }
        }

        for child in &node.children {
            walk_node(child, world, lo, hi);
        }
    }

    walk_node(&obj.root_node, glam::Mat4::IDENTITY, &mut lo, &mut hi);

    // If no geometry was found, fall back to a unit cube.
    if lo.x > hi.x {
        lo = glam::Vec3::splat(-0.5);
        hi = glam::Vec3::splat(0.5);
    }

    rkf_core::aabb::Aabb::new(lo, hi)
}
