//! Bounding volume hierarchy over mesh triangles for accelerated nearest-triangle queries.
//!
//! Builds a binary BVH using midpoint splits along the longest AABB axis.
//! Supports fast nearest-point queries for unsigned distance computation
//! during mesh-to-SDF voxelization.

use glam::Vec3;

use crate::mesh::MeshData;

/// Result of a nearest-triangle query.
#[derive(Debug, Clone, Copy)]
pub struct NearestResult {
    /// Unsigned distance to the closest triangle.
    pub distance: f32,
    /// Index of the closest triangle.
    pub triangle_index: usize,
    /// Barycentric coordinates of the closest point on the triangle.
    pub barycentric: [f32; 3],
    /// The closest point on the triangle surface.
    pub closest_point: Vec3,
}

/// Axis-aligned bounding box for BVH nodes.
#[derive(Debug, Clone, Copy)]
struct BvhAabb {
    min: Vec3,
    max: Vec3,
}

/// BVH node -- either interior (two children) or leaf (triangle range).
enum BvhNode {
    Interior {
        bounds: BvhAabb,
        left: Box<BvhNode>,
        right: Box<BvhNode>,
    },
    Leaf {
        bounds: BvhAabb,
        /// Range start of triangle indices in the reordered index array.
        start: usize,
        /// Number of triangles in this leaf.
        count: usize,
    },
}

/// BVH over mesh triangles for accelerated queries.
pub struct TriangleBvh {
    root: BvhNode,
    /// Triangle indices reordered by BVH construction.
    tri_order: Vec<usize>,
    /// Cached triangle vertex positions for fast access.
    positions: Vec<[Vec3; 3]>,
}

impl TriangleBvh {
    /// Build a BVH from mesh data.
    pub fn build(mesh: &MeshData) -> Self {
        let tri_count = mesh.triangle_count();
        let positions: Vec<[Vec3; 3]> = (0..tri_count)
            .map(|i| mesh.triangle_positions(i))
            .collect();

        let mut tri_order: Vec<usize> = (0..tri_count).collect();

        let root = Self::build_recursive(&positions, &mut tri_order, 0, tri_count, 0);

        Self {
            root,
            tri_order,
            positions,
        }
    }

    /// Recursive BVH construction using midpoint split along the longest axis.
    fn build_recursive(
        positions: &[[Vec3; 3]],
        tri_order: &mut [usize],
        start: usize,
        count: usize,
        depth: usize,
    ) -> BvhNode {
        let bounds = compute_bounds(positions, &tri_order[start..start + count]);

        // Leaf condition: few triangles or max depth
        if count <= 4 || depth >= 32 {
            return BvhNode::Leaf {
                bounds,
                start,
                count,
            };
        }

        // Choose split axis: longest axis of bounding box
        let extent = bounds.max - bounds.min;
        let axis = if extent.x >= extent.y && extent.x >= extent.z {
            0
        } else if extent.y >= extent.z {
            1
        } else {
            2
        };

        // Split at midpoint along chosen axis
        let mid = (bounds.min[axis] + bounds.max[axis]) * 0.5;

        // Partition triangles by centroid
        let slice = &mut tri_order[start..start + count];
        let mut left_count = 0;
        for i in 0..slice.len() {
            let tri = &positions[slice[i]];
            let centroid_axis = (tri[0][axis] + tri[1][axis] + tri[2][axis]) / 3.0;
            if centroid_axis < mid {
                slice.swap(i, left_count);
                left_count += 1;
            }
        }

        // Fallback: split in half if partition degenerates
        if left_count == 0 || left_count == count {
            left_count = count / 2;
        }

        let left =
            Self::build_recursive(positions, tri_order, start, left_count, depth + 1);
        let right = Self::build_recursive(
            positions,
            tri_order,
            start + left_count,
            count - left_count,
            depth + 1,
        );

        BvhNode::Interior {
            bounds,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    /// Find the nearest triangle to a query point.
    pub fn nearest(&self, point: Vec3) -> NearestResult {
        let mut best = NearestResult {
            distance: f32::MAX,
            triangle_index: 0,
            barycentric: [1.0, 0.0, 0.0],
            closest_point: Vec3::ZERO,
        };
        self.nearest_recursive(&self.root, point, &mut best);
        best
    }

    fn nearest_recursive(&self, node: &BvhNode, point: Vec3, best: &mut NearestResult) {
        match node {
            BvhNode::Leaf { start, count, .. } => {
                for i in *start..*start + *count {
                    let tri_idx = self.tri_order[i];
                    let tri = &self.positions[tri_idx];
                    let (dist, bary, closest) =
                        point_triangle_distance(point, tri[0], tri[1], tri[2]);
                    if dist < best.distance {
                        best.distance = dist;
                        best.triangle_index = tri_idx;
                        best.barycentric = bary;
                        best.closest_point = closest;
                    }
                }
            }
            BvhNode::Interior { left, right, .. } => {
                let left_bounds = node_bounds(left);
                let right_bounds = node_bounds(right);

                // Early out if AABB is farther than current best
                let left_dist = aabb_distance(point, &left_bounds);
                let right_dist = aabb_distance(point, &right_bounds);

                // Visit closer child first for better pruning
                if left_dist < right_dist {
                    if left_dist < best.distance {
                        self.nearest_recursive(left, point, best);
                    }
                    if right_dist < best.distance {
                        self.nearest_recursive(right, point, best);
                    }
                } else {
                    if right_dist < best.distance {
                        self.nearest_recursive(right, point, best);
                    }
                    if left_dist < best.distance {
                        self.nearest_recursive(left, point, best);
                    }
                }
            }
        }
    }
}

/// Extract the bounds from a BVH node.
fn node_bounds(node: &BvhNode) -> BvhAabb {
    match node {
        BvhNode::Interior { bounds, .. } | BvhNode::Leaf { bounds, .. } => *bounds,
    }
}

/// Closest point on a triangle to a query point.
///
/// Returns `(distance, barycentric_coords, closest_point)`.
/// Uses the Voronoi region method (Ericson, "Real-Time Collision Detection").
pub fn point_triangle_distance(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> (f32, [f32; 3], Vec3) {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;

    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    // Region A (vertex A closest)
    if d1 <= 0.0 && d2 <= 0.0 {
        let dist = (p - a).length();
        return (dist, [1.0, 0.0, 0.0], a);
    }

    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    // Region B (vertex B closest)
    if d3 >= 0.0 && d4 <= d3 {
        let dist = (p - b).length();
        return (dist, [0.0, 1.0, 0.0], b);
    }

    // Region AB (edge AB closest)
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        let closest = a + v * ab;
        let dist = (p - closest).length();
        return (dist, [1.0 - v, v, 0.0], closest);
    }

    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    // Region C (vertex C closest)
    if d6 >= 0.0 && d5 <= d6 {
        let dist = (p - c).length();
        return (dist, [0.0, 0.0, 1.0], c);
    }

    // Region AC (edge AC closest)
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        let closest = a + w * ac;
        let dist = (p - closest).length();
        return (dist, [1.0 - w, 0.0, w], closest);
    }

    // Region BC (edge BC closest)
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        let closest = b + w * (c - b);
        let dist = (p - closest).length();
        return (dist, [0.0, 1.0 - w, w], closest);
    }

    // Region interior (point projects inside the triangle)
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    let closest = a + ab * v + ac * w;
    let dist = (p - closest).length();
    (dist, [1.0 - v - w, v, w], closest)
}

/// Distance from a point to an AABB (0 if inside).
fn aabb_distance(point: Vec3, aabb: &BvhAabb) -> f32 {
    let clamped = point.clamp(aabb.min, aabb.max);
    (point - clamped).length()
}

/// Compute the bounding box of the given triangles.
fn compute_bounds(positions: &[[Vec3; 3]], indices: &[usize]) -> BvhAabb {
    let mut min = Vec3::splat(f32::MAX);
    let mut max = Vec3::splat(f32::MIN);
    for &idx in indices {
        for v in &positions[idx] {
            min = min.min(*v);
            max = max.max(*v);
        }
    }
    BvhAabb { min, max }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::{ImportMaterial, MeshData};

    /// Helper: build a single-triangle mesh in the XY plane.
    fn single_triangle_mesh() -> MeshData {
        MeshData {
            positions: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            normals: vec![Vec3::Z, Vec3::Z, Vec3::Z],
            uvs: vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            indices: vec![0, 1, 2],
            material_indices: vec![0],
            materials: vec![ImportMaterial {
                name: "test".to_string(),
                base_color: [1.0, 0.0, 0.0],
                metallic: 0.0,
                roughness: 0.5,
                albedo_texture: None,
            }],
            bounds_min: Vec3::ZERO,
            bounds_max: Vec3::new(1.0, 1.0, 0.0),
        }
    }

    /// Helper: build a two-triangle mesh (a quad in the XY plane).
    fn two_triangle_mesh() -> MeshData {
        MeshData {
            positions: vec![
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(1.0, 1.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
            ],
            normals: vec![Vec3::Z; 4],
            uvs: Vec::new(),
            indices: vec![0, 1, 2, 0, 2, 3],
            material_indices: vec![0, 0],
            materials: vec![ImportMaterial {
                name: "test".to_string(),
                base_color: [1.0, 1.0, 1.0],
                metallic: 0.0,
                roughness: 0.5,
                albedo_texture: None,
            }],
            bounds_min: Vec3::ZERO,
            bounds_max: Vec3::new(1.0, 1.0, 0.0),
        }
    }

    #[test]
    fn build_bvh_single_triangle() {
        let mesh = single_triangle_mesh();
        let bvh = TriangleBvh::build(&mesh);
        assert_eq!(bvh.positions.len(), 1);
        assert_eq!(bvh.tri_order.len(), 1);
    }

    #[test]
    fn build_bvh_two_triangles() {
        let mesh = two_triangle_mesh();
        let bvh = TriangleBvh::build(&mesh);
        assert_eq!(bvh.positions.len(), 2);
        assert_eq!(bvh.tri_order.len(), 2);
    }

    #[test]
    fn nearest_single_triangle_above_center() {
        let mesh = single_triangle_mesh();
        let bvh = TriangleBvh::build(&mesh);

        // Point directly above the centroid of the triangle
        let centroid = Vec3::new(1.0 / 3.0, 1.0 / 3.0, 0.0);
        let query = centroid + Vec3::new(0.0, 0.0, 2.0);
        let result = bvh.nearest(query);

        assert!(
            (result.distance - 2.0).abs() < 1e-5,
            "expected distance ~2.0, got {}",
            result.distance
        );
        assert_eq!(result.triangle_index, 0);
        assert!(
            (result.closest_point - centroid).length() < 1e-5,
            "closest point should be centroid, got {:?}",
            result.closest_point
        );
    }

    #[test]
    fn nearest_returns_correct_barycentric() {
        let mesh = single_triangle_mesh();
        let bvh = TriangleBvh::build(&mesh);

        // Point above vertex A (0,0,0)
        let query = Vec3::new(0.0, 0.0, 1.0);
        let result = bvh.nearest(query);

        assert!(
            (result.distance - 1.0).abs() < 1e-5,
            "distance should be 1.0, got {}",
            result.distance
        );
        // Barycentric should be (1, 0, 0) for vertex A
        assert!(
            (result.barycentric[0] - 1.0).abs() < 1e-5,
            "barycentric[0] should be 1.0, got {}",
            result.barycentric[0]
        );
        assert!(
            result.barycentric[1].abs() < 1e-5,
            "barycentric[1] should be 0.0, got {}",
            result.barycentric[1]
        );
        assert!(
            result.barycentric[2].abs() < 1e-5,
            "barycentric[2] should be 0.0, got {}",
            result.barycentric[2]
        );
    }

    #[test]
    fn nearest_at_vertex_returns_zero_distance() {
        let mesh = single_triangle_mesh();
        let bvh = TriangleBvh::build(&mesh);

        // Query exactly at vertex B (1, 0, 0)
        let result = bvh.nearest(Vec3::new(1.0, 0.0, 0.0));
        assert!(
            result.distance < 1e-6,
            "distance at vertex should be ~0, got {}",
            result.distance
        );
    }

    #[test]
    fn point_triangle_distance_above_center() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(1.0, 0.0, 0.0);
        let c = Vec3::new(0.0, 1.0, 0.0);
        let p = Vec3::new(0.25, 0.25, 3.0);

        let (dist, bary, closest) = point_triangle_distance(p, a, b, c);
        assert!(
            (dist - 3.0).abs() < 1e-5,
            "expected distance ~3.0, got {dist}"
        );
        assert!(
            (closest.z).abs() < 1e-5,
            "closest point z should be 0, got {}",
            closest.z
        );
        // Barycentric should sum to 1
        let sum: f32 = bary.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "bary sum should be 1.0, got {sum}");
    }

    #[test]
    fn point_triangle_distance_at_edge() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(2.0, 0.0, 0.0);
        let c = Vec3::new(1.0, 2.0, 0.0);
        // Point off the AB edge
        let p = Vec3::new(1.0, -1.0, 0.0);

        let (dist, bary, closest) = point_triangle_distance(p, a, b, c);
        // Closest point should be on edge AB at (1, 0, 0)
        assert!(
            (closest - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-5,
            "closest should be (1,0,0), got {closest:?}"
        );
        assert!(
            (dist - 1.0).abs() < 1e-5,
            "distance should be 1.0, got {dist}"
        );
        // c-component of barycentric should be 0 (on edge AB)
        assert!(
            bary[2].abs() < 1e-5,
            "bary[2] should be 0 on edge AB, got {}",
            bary[2]
        );
    }

    #[test]
    fn point_triangle_distance_at_vertex() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(1.0, 0.0, 0.0);
        let c = Vec3::new(0.0, 1.0, 0.0);

        // Point nearest to vertex C
        let p = Vec3::new(-1.0, 2.0, 0.0);
        let (dist, bary, closest) = point_triangle_distance(p, a, b, c);

        let expected_dist = (p - c).length();
        assert!(
            (dist - expected_dist).abs() < 1e-5,
            "expected distance {expected_dist}, got {dist}"
        );
        assert!(
            (closest - c).length() < 1e-5,
            "closest should be vertex C, got {closest:?}"
        );
        assert!(
            (bary[2] - 1.0).abs() < 1e-5,
            "bary should be (0,0,1) for vertex C, got {bary:?}"
        );
    }

    #[test]
    fn nearest_with_two_triangles_picks_closer() {
        // Two triangles at different Z positions
        let mesh = MeshData {
            positions: vec![
                // Triangle 0 at z=0
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                // Triangle 1 at z=5
                Vec3::new(0.0, 0.0, 5.0),
                Vec3::new(1.0, 0.0, 5.0),
                Vec3::new(0.0, 1.0, 5.0),
            ],
            normals: vec![Vec3::Z; 6],
            uvs: Vec::new(),
            indices: vec![0, 1, 2, 3, 4, 5],
            material_indices: vec![0, 0],
            materials: vec![ImportMaterial {
                name: "test".to_string(),
                base_color: [1.0, 1.0, 1.0],
                metallic: 0.0,
                roughness: 0.5,
                albedo_texture: None,
            }],
            bounds_min: Vec3::ZERO,
            bounds_max: Vec3::new(1.0, 1.0, 5.0),
        };

        let bvh = TriangleBvh::build(&mesh);

        // Query point closer to triangle 0
        let result = bvh.nearest(Vec3::new(0.25, 0.25, 1.0));
        assert_eq!(result.triangle_index, 0);
        assert!(
            (result.distance - 1.0).abs() < 1e-4,
            "expected ~1.0, got {}",
            result.distance
        );

        // Query point closer to triangle 1
        let result2 = bvh.nearest(Vec3::new(0.25, 0.25, 4.0));
        assert_eq!(result2.triangle_index, 1);
        assert!(
            (result2.distance - 1.0).abs() < 1e-4,
            "expected ~1.0, got {}",
            result2.distance
        );
    }

    #[test]
    fn aabb_distance_inside_is_zero() {
        let aabb = BvhAabb {
            min: Vec3::ZERO,
            max: Vec3::ONE,
        };
        let dist = aabb_distance(Vec3::new(0.5, 0.5, 0.5), &aabb);
        assert!(dist.abs() < 1e-6, "distance inside should be 0, got {dist}");
    }

    #[test]
    fn aabb_distance_outside() {
        let aabb = BvhAabb {
            min: Vec3::ZERO,
            max: Vec3::ONE,
        };
        let dist = aabb_distance(Vec3::new(2.0, 0.5, 0.5), &aabb);
        assert!(
            (dist - 1.0).abs() < 1e-5,
            "distance should be 1.0, got {dist}"
        );
    }
}
