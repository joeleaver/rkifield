//! Bounding volume hierarchy over scene objects.
//!
//! [`Bvh`] provides a CPU-side binary BVH over object AABBs for spatial
//! acceleration. The ray marcher uses it (via GPU upload) to quickly
//! find candidate objects for each ray.
//!
//! # Build
//!
//! [`Bvh::build`] constructs the tree top-down using Surface Area Heuristic
//! (SAH) for split decisions. Each internal node stores an AABB enclosing all
//! descendants; each leaf stores a single object index.
//!
//! # Queries
//!
//! - [`query_ray`](Bvh::query_ray) — ray-AABB intersection test, returns leaf object indices
//! - [`query_aabb`](Bvh::query_aabb) — AABB overlap, returns intersecting object indices
//! - [`query_sphere`](Bvh::query_sphere) — sphere overlap
//!
//! # Refit
//!
//! [`Bvh::refit`] updates AABBs bottom-up when objects move, preserving tree
//! topology. Use this when objects change position but are not added/removed.

use glam::Vec3;

use crate::aabb::Aabb;

/// Sentinel for invalid node indices (no child / no parent).
pub const INVALID: u32 = u32::MAX;

/// A single node in the BVH.
#[derive(Debug, Clone, Copy)]
pub struct BvhNode {
    /// Bounding box enclosing this node (and all descendants).
    pub aabb: Aabb,
    /// Left child index, or [`INVALID`] if leaf.
    pub left: u32,
    /// Right child index, or [`INVALID`] if leaf.
    pub right: u32,
    /// Object index if leaf, or [`INVALID`] if internal.
    pub object_index: u32,
}

impl BvhNode {
    /// Returns `true` if this is a leaf node (has an object, no children).
    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.left == INVALID && self.right == INVALID
    }
}

/// CPU-side bounding volume hierarchy over scene objects.
#[derive(Debug, Clone)]
pub struct Bvh {
    /// Flat array of BVH nodes.
    pub nodes: Vec<BvhNode>,
}

impl Bvh {
    /// Build a BVH over the given objects using SAH.
    ///
    /// `objects` is a slice of `(object_id, aabb)` pairs. The returned BVH's
    /// leaf `object_index` values correspond to the `object_id` values.
    ///
    /// Returns an empty BVH if `objects` is empty.
    pub fn build(objects: &[(u32, Aabb)]) -> Self {
        if objects.is_empty() {
            return Self { nodes: Vec::new() };
        }

        let mut indices: Vec<usize> = (0..objects.len()).collect();
        let mut nodes = Vec::with_capacity(objects.len() * 2);

        build_recursive(objects, &mut indices, &mut nodes);

        Self { nodes }
    }

    /// Refit AABBs bottom-up using updated object AABBs.
    ///
    /// `objects` must contain the same IDs as the original build, with
    /// potentially updated AABBs. Tree topology is preserved.
    pub fn refit(&mut self, objects: &[(u32, Aabb)]) {
        if self.nodes.is_empty() {
            return;
        }

        // Build a map from object_id to aabb for O(1) lookup.
        let map: std::collections::HashMap<u32, Aabb> =
            objects.iter().map(|&(id, aabb)| (id, aabb)).collect();

        refit_recursive(&mut self.nodes, 0, &map);
    }

    /// Find all objects whose AABBs are hit by the ray.
    ///
    /// Returns a vec of object indices (from the original build input).
    pub fn query_ray(&self, origin: Vec3, dir: Vec3, max_t: f32) -> Vec<u32> {
        let mut result = Vec::new();
        if !self.nodes.is_empty() {
            let inv_dir = Vec3::new(
                if dir.x.abs() > 1e-10 { 1.0 / dir.x } else { f32::MAX },
                if dir.y.abs() > 1e-10 { 1.0 / dir.y } else { f32::MAX },
                if dir.z.abs() > 1e-10 { 1.0 / dir.z } else { f32::MAX },
            );
            query_ray_recursive(&self.nodes, 0, origin, inv_dir, max_t, &mut result);
        }
        result
    }

    /// Find all objects whose AABBs overlap the given AABB.
    pub fn query_aabb(&self, aabb: &Aabb) -> Vec<u32> {
        let mut result = Vec::new();
        if !self.nodes.is_empty() {
            query_aabb_recursive(&self.nodes, 0, aabb, &mut result);
        }
        result
    }

    /// Find all objects whose AABBs overlap the given sphere.
    pub fn query_sphere(&self, center: Vec3, radius: f32) -> Vec<u32> {
        let mut result = Vec::new();
        if !self.nodes.is_empty() {
            query_sphere_recursive(&self.nodes, 0, center, radius * radius, &mut result);
        }
        result
    }

    /// Number of nodes in the BVH.
    #[inline]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of leaf nodes (= number of objects).
    pub fn leaf_count(&self) -> usize {
        self.nodes.iter().filter(|n| n.is_leaf()).count()
    }

    /// Returns true if the BVH has no nodes.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

// ── SAH build ──────────────────────────────────────────────────────────────

/// Cost of traversing one BVH node (relative to cost of intersecting a leaf).
const TRAVERSAL_COST: f32 = 1.0;
/// Cost of intersecting one leaf (relative unit).
const LEAF_COST: f32 = 1.0;

fn build_recursive(
    objects: &[(u32, Aabb)],
    indices: &mut [usize],
    nodes: &mut Vec<BvhNode>,
) -> u32 {
    // Compute enclosing AABB.
    let mut enclosing = objects[indices[0]].1;
    for &idx in indices.iter().skip(1) {
        enclosing = enclosing.expand_aabb(&objects[idx].1);
    }

    // Base case: single object → leaf.
    if indices.len() == 1 {
        let node_idx = nodes.len() as u32;
        nodes.push(BvhNode {
            aabb: enclosing,
            left: INVALID,
            right: INVALID,
            object_index: objects[indices[0]].0,
        });
        return node_idx;
    }

    // Two objects → split directly.
    if indices.len() == 2 {
        let node_idx = nodes.len() as u32;
        // Reserve space for internal node.
        nodes.push(BvhNode {
            aabb: enclosing,
            left: INVALID,
            right: INVALID,
            object_index: INVALID,
        });

        let left = {
            let mut left_slice = [indices[0]];
            build_recursive(objects, &mut left_slice, nodes)
        };
        let right = {
            let mut right_slice = [indices[1]];
            build_recursive(objects, &mut right_slice, nodes)
        };

        nodes[node_idx as usize].left = left;
        nodes[node_idx as usize].right = right;
        return node_idx;
    }

    // SAH: try splits along each axis.
    let enclosing_sa = enclosing.surface_area().max(1e-10);
    let mut best_cost = f32::MAX;
    let mut best_axis = 0;
    let mut best_split = indices.len() / 2;

    for axis in 0..3 {
        // Sort indices by centroid along this axis.
        indices.sort_by(|&a, &b| {
            let ca = centroid_axis(&objects[a].1, axis);
            let cb = centroid_axis(&objects[b].1, axis);
            ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Evaluate SAH cost at each split position.
        for split in 1..indices.len() {
            let left_aabb = enclosing_of(objects, &indices[..split]);
            let right_aabb = enclosing_of(objects, &indices[split..]);

            let cost = TRAVERSAL_COST
                + LEAF_COST
                    * (left_aabb.surface_area() * split as f32
                        + right_aabb.surface_area() * (indices.len() - split) as f32)
                    / enclosing_sa;

            if cost < best_cost {
                best_cost = cost;
                best_axis = axis;
                best_split = split;
            }
        }
    }

    // Sort by best axis for the final split.
    indices.sort_by(|&a, &b| {
        let ca = centroid_axis(&objects[a].1, best_axis);
        let cb = centroid_axis(&objects[b].1, best_axis);
        ca.partial_cmp(&cb).unwrap_or(std::cmp::Ordering::Equal)
    });

    // Reserve internal node.
    let node_idx = nodes.len() as u32;
    nodes.push(BvhNode {
        aabb: enclosing,
        left: INVALID,
        right: INVALID,
        object_index: INVALID,
    });

    let (left_indices, right_indices) = indices.split_at_mut(best_split);
    let left = build_recursive(objects, left_indices, nodes);
    let right = build_recursive(objects, right_indices, nodes);

    nodes[node_idx as usize].left = left;
    nodes[node_idx as usize].right = right;

    node_idx
}

fn centroid_axis(aabb: &Aabb, axis: usize) -> f32 {
    let c = aabb.center();
    match axis {
        0 => c.x,
        1 => c.y,
        _ => c.z,
    }
}

fn enclosing_of(objects: &[(u32, Aabb)], indices: &[usize]) -> Aabb {
    let mut aabb = objects[indices[0]].1;
    for &idx in indices.iter().skip(1) {
        aabb = aabb.expand_aabb(&objects[idx].1);
    }
    aabb
}

// ── Refit ──────────────────────────────────────────────────────────────────

fn refit_recursive(
    nodes: &mut [BvhNode],
    idx: u32,
    map: &std::collections::HashMap<u32, Aabb>,
) -> Aabb {
    let node = nodes[idx as usize];

    if node.is_leaf() {
        if let Some(&aabb) = map.get(&node.object_index) {
            nodes[idx as usize].aabb = aabb;
            return aabb;
        }
        return node.aabb;
    }

    let left_aabb = refit_recursive(nodes, node.left, map);
    let right_aabb = refit_recursive(nodes, node.right, map);
    let combined = left_aabb.expand_aabb(&right_aabb);
    nodes[idx as usize].aabb = combined;
    combined
}

// ── Queries ────────────────────────────────────────────────────────────────

fn ray_aabb_intersect(aabb: &Aabb, origin: Vec3, inv_dir: Vec3, max_t: f32) -> bool {
    let t1 = (aabb.min - origin) * inv_dir;
    let t2 = (aabb.max - origin) * inv_dir;

    let tmin = t1.min(t2);
    let tmax = t1.max(t2);

    let tmin_max = tmin.x.max(tmin.y).max(tmin.z);
    let tmax_min = tmax.x.min(tmax.y).min(tmax.z);

    tmax_min >= tmin_max.max(0.0) && tmin_max <= max_t
}

fn query_ray_recursive(
    nodes: &[BvhNode],
    idx: u32,
    origin: Vec3,
    inv_dir: Vec3,
    max_t: f32,
    result: &mut Vec<u32>,
) {
    let node = &nodes[idx as usize];

    if !ray_aabb_intersect(&node.aabb, origin, inv_dir, max_t) {
        return;
    }

    if node.is_leaf() {
        result.push(node.object_index);
        return;
    }

    query_ray_recursive(nodes, node.left, origin, inv_dir, max_t, result);
    query_ray_recursive(nodes, node.right, origin, inv_dir, max_t, result);
}

fn query_aabb_recursive(
    nodes: &[BvhNode],
    idx: u32,
    query: &Aabb,
    result: &mut Vec<u32>,
) {
    let node = &nodes[idx as usize];

    if !node.aabb.intersects(query) {
        return;
    }

    if node.is_leaf() {
        result.push(node.object_index);
        return;
    }

    query_aabb_recursive(nodes, node.left, query, result);
    query_aabb_recursive(nodes, node.right, query, result);
}

fn sphere_aabb_intersect(aabb: &Aabb, center: Vec3, radius_sq: f32) -> bool {
    // Find closest point on AABB to sphere center.
    let closest = Vec3::new(
        center.x.clamp(aabb.min.x, aabb.max.x),
        center.y.clamp(aabb.min.y, aabb.max.y),
        center.z.clamp(aabb.min.z, aabb.max.z),
    );
    (closest - center).length_squared() <= radius_sq
}

fn query_sphere_recursive(
    nodes: &[BvhNode],
    idx: u32,
    center: Vec3,
    radius_sq: f32,
    result: &mut Vec<u32>,
) {
    let node = &nodes[idx as usize];

    if !sphere_aabb_intersect(&node.aabb, center, radius_sq) {
        return;
    }

    if node.is_leaf() {
        result.push(node.object_index);
        return;
    }

    query_sphere_recursive(nodes, node.left, center, radius_sq, result);
    query_sphere_recursive(nodes, node.right, center, radius_sq, result);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_box(center: Vec3) -> Aabb {
        Aabb::new(center - Vec3::splat(0.5), center + Vec3::splat(0.5))
    }

    // ── Build tests ─────────────────────────────────────────────────────

    #[test]
    fn build_empty() {
        let bvh = Bvh::build(&[]);
        assert!(bvh.is_empty());
        assert_eq!(bvh.node_count(), 0);
        assert_eq!(bvh.leaf_count(), 0);
    }

    #[test]
    fn build_single_object() {
        let objects = [(1, unit_box(Vec3::ZERO))];
        let bvh = Bvh::build(&objects);
        assert_eq!(bvh.node_count(), 1);
        assert_eq!(bvh.leaf_count(), 1);
        assert!(bvh.nodes[0].is_leaf());
        assert_eq!(bvh.nodes[0].object_index, 1);
    }

    #[test]
    fn build_two_objects() {
        let objects = [
            (1, unit_box(Vec3::new(-2.0, 0.0, 0.0))),
            (2, unit_box(Vec3::new(2.0, 0.0, 0.0))),
        ];
        let bvh = Bvh::build(&objects);
        assert_eq!(bvh.node_count(), 3); // 1 internal + 2 leaves
        assert_eq!(bvh.leaf_count(), 2);
        assert!(!bvh.nodes[0].is_leaf()); // root is internal
    }

    #[test]
    fn build_many_objects() {
        let objects: Vec<(u32, Aabb)> = (0..20)
            .map(|i| (i, unit_box(Vec3::new(i as f32 * 3.0, 0.0, 0.0))))
            .collect();
        let bvh = Bvh::build(&objects);
        assert_eq!(bvh.leaf_count(), 20);
        // Internal nodes = leaves - 1 for a full binary tree.
        assert!(bvh.node_count() >= 20);
    }

    #[test]
    fn build_root_aabb_encloses_all() {
        let objects = [
            (0, Aabb::new(Vec3::new(-5.0, -1.0, -1.0), Vec3::new(-4.0, 1.0, 1.0))),
            (1, Aabb::new(Vec3::new(4.0, -1.0, -1.0), Vec3::new(5.0, 1.0, 1.0))),
        ];
        let bvh = Bvh::build(&objects);
        let root = &bvh.nodes[0];
        assert!(root.aabb.min.x <= -5.0);
        assert!(root.aabb.max.x >= 5.0);
    }

    // ── Ray query tests ─────────────────────────────────────────────────

    #[test]
    fn ray_hits_object() {
        let objects = [(42, unit_box(Vec3::new(0.0, 0.0, -5.0)))];
        let bvh = Bvh::build(&objects);

        let hits = bvh.query_ray(Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0), 100.0);
        assert_eq!(hits, vec![42]);
    }

    #[test]
    fn ray_misses_object() {
        let objects = [(1, unit_box(Vec3::new(10.0, 0.0, 0.0)))];
        let bvh = Bvh::build(&objects);

        let hits = bvh.query_ray(Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0), 100.0);
        assert!(hits.is_empty());
    }

    #[test]
    fn ray_max_t_limits_range() {
        let objects = [(1, unit_box(Vec3::new(0.0, 0.0, -50.0)))];
        let bvh = Bvh::build(&objects);

        let hits = bvh.query_ray(Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0), 10.0);
        assert!(hits.is_empty(), "object beyond max_t should not be hit");
    }

    #[test]
    fn ray_hits_multiple() {
        let objects = [
            (1, unit_box(Vec3::new(0.0, 0.0, -3.0))),
            (2, unit_box(Vec3::new(0.0, 0.0, -7.0))),
            (3, unit_box(Vec3::new(5.0, 0.0, 0.0))), // off to the side
        ];
        let bvh = Bvh::build(&objects);

        let mut hits = bvh.query_ray(Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0), 100.0);
        hits.sort();
        assert_eq!(hits, vec![1, 2]);
    }

    #[test]
    fn ray_empty_bvh() {
        let bvh = Bvh::build(&[]);
        let hits = bvh.query_ray(Vec3::ZERO, Vec3::Z, 100.0);
        assert!(hits.is_empty());
    }

    // ── AABB query tests ────────────────────────────────────────────────

    #[test]
    fn aabb_query_overlap() {
        let objects = [
            (1, unit_box(Vec3::ZERO)),
            (2, unit_box(Vec3::new(5.0, 0.0, 0.0))),
        ];
        let bvh = Bvh::build(&objects);

        let query = Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0));
        let hits = bvh.query_aabb(&query);
        assert_eq!(hits, vec![1]);
    }

    #[test]
    fn aabb_query_no_overlap() {
        let objects = [(1, unit_box(Vec3::new(10.0, 10.0, 10.0)))];
        let bvh = Bvh::build(&objects);

        let query = Aabb::new(Vec3::ZERO, Vec3::ONE);
        let hits = bvh.query_aabb(&query);
        assert!(hits.is_empty());
    }

    // ── Sphere query tests ──────────────────────────────────────────────

    #[test]
    fn sphere_query_overlap() {
        let objects = [
            (1, unit_box(Vec3::ZERO)),
            (2, unit_box(Vec3::new(10.0, 0.0, 0.0))),
        ];
        let bvh = Bvh::build(&objects);

        let hits = bvh.query_sphere(Vec3::ZERO, 1.0);
        assert_eq!(hits, vec![1]);
    }

    #[test]
    fn sphere_query_all() {
        let objects = [
            (1, unit_box(Vec3::ZERO)),
            (2, unit_box(Vec3::new(2.0, 0.0, 0.0))),
        ];
        let bvh = Bvh::build(&objects);

        let mut hits = bvh.query_sphere(Vec3::new(1.0, 0.0, 0.0), 5.0);
        hits.sort();
        assert_eq!(hits, vec![1, 2]);
    }

    // ── Refit tests ─────────────────────────────────────────────────────

    #[test]
    fn refit_updates_aabbs() {
        let objects = [
            (1, unit_box(Vec3::new(-2.0, 0.0, 0.0))),
            (2, unit_box(Vec3::new(2.0, 0.0, 0.0))),
        ];
        let mut bvh = Bvh::build(&objects);

        // Move object 1 far away.
        let new_objects = [
            (1, unit_box(Vec3::new(-100.0, 0.0, 0.0))),
            (2, unit_box(Vec3::new(2.0, 0.0, 0.0))),
        ];
        bvh.refit(&new_objects);

        // Root AABB should now extend to -100.
        assert!(bvh.nodes[0].aabb.min.x <= -100.0);

        // Query should find object 1 at new location.
        let hits = bvh.query_aabb(&Aabb::new(
            Vec3::new(-101.0, -1.0, -1.0),
            Vec3::new(-99.0, 1.0, 1.0),
        ));
        assert!(hits.contains(&1));
    }

    #[test]
    fn refit_empty_bvh() {
        let mut bvh = Bvh::build(&[]);
        bvh.refit(&[]);
        assert!(bvh.is_empty());
    }
}
