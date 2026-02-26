//! Entity placement data model for the RKIField editor.
//!
//! Provides grid snapping, surface snapping via CPU ray casting, placement requests,
//! and an asset browser for selecting assets to place. This is a pure data model
//! independent of the GUI framework.

#![allow(dead_code)]

use glam::{Quat, Vec3};

/// Grid snapping configuration for entity placement.
#[derive(Debug, Clone)]
pub struct GridSnap {
    /// Whether grid snapping is active.
    pub enabled: bool,
    /// Grid cell size in world units.
    pub grid_size: f32,
}

impl Default for GridSnap {
    fn default() -> Self {
        Self {
            enabled: false,
            grid_size: 1.0,
        }
    }
}

impl GridSnap {
    /// Create a new grid snap with the given cell size.
    pub fn new(grid_size: f32) -> Self {
        Self {
            enabled: true,
            grid_size,
        }
    }

    /// Snap a position to the nearest grid point.
    ///
    /// If snapping is disabled, returns the position unchanged.
    /// Each axis is independently rounded to the nearest multiple of `grid_size`.
    pub fn snap(&self, pos: Vec3) -> Vec3 {
        if !self.enabled || self.grid_size <= 0.0 {
            return pos;
        }
        Vec3::new(
            (pos.x / self.grid_size).round() * self.grid_size,
            (pos.y / self.grid_size).round() * self.grid_size,
            (pos.z / self.grid_size).round() * self.grid_size,
        )
    }
}

/// Result of a CPU-side surface snap ray cast.
#[derive(Debug, Clone)]
pub struct SurfaceSnapResult {
    /// Hit position in world space.
    pub position: Vec3,
    /// Surface normal at the hit point.
    pub normal: Vec3,
    /// Whether the ray actually hit a surface.
    pub hit: bool,
}

impl SurfaceSnapResult {
    /// Create a miss result.
    pub fn miss() -> Self {
        Self {
            position: Vec3::ZERO,
            normal: Vec3::Y,
            hit: false,
        }
    }
}

/// Simple CPU-side sphere trace along a ray for placement preview.
///
/// Steps along the ray in fixed increments of `step_mult`, evaluating the provided
/// SDF function at each point. Returns a hit when the SDF distance drops below
/// a threshold (0.01 units).
///
/// This is NOT the GPU ray marcher. It is a simple iterative march for CPU-side
/// placement preview with a caller-provided SDF function.
pub fn ray_cast_sdf<F>(
    ray_origin: Vec3,
    ray_dir: Vec3,
    max_dist: f32,
    step_mult: f32,
    sdf: F,
) -> SurfaceSnapResult
where
    F: Fn(Vec3) -> f32,
{
    let ray_dir = ray_dir.normalize();
    let hit_threshold = 0.01;
    let normal_epsilon = 0.001;
    let step = step_mult.max(0.001); // Prevent zero/negative steps

    let mut t = 0.0;
    while t < max_dist {
        let pos = ray_origin + ray_dir * t;
        let dist = sdf(pos);

        if dist < hit_threshold {
            // Estimate normal via central differences of the SDF.
            let nx = sdf(pos + Vec3::X * normal_epsilon) - sdf(pos - Vec3::X * normal_epsilon);
            let ny = sdf(pos + Vec3::Y * normal_epsilon) - sdf(pos - Vec3::Y * normal_epsilon);
            let nz = sdf(pos + Vec3::Z * normal_epsilon) - sdf(pos - Vec3::Z * normal_epsilon);
            let normal = Vec3::new(nx, ny, nz).normalize_or(Vec3::Y);

            return SurfaceSnapResult {
                position: pos,
                normal,
                hit: true,
            };
        }

        t += step;
    }

    SurfaceSnapResult::miss()
}

/// A ground plane SDF at y=0 (useful for testing and default placement).
pub fn ground_plane_sdf(pos: Vec3) -> f32 {
    pos.y
}

/// A sphere SDF centered at the origin with the given radius.
pub fn sphere_sdf(pos: Vec3, radius: f32) -> f32 {
    pos.length() - radius
}

/// A request to place an asset in the scene.
#[derive(Debug, Clone)]
pub struct PlacementRequest {
    /// Path to the asset file (.rkf).
    pub asset_path: String,
    /// World-space position for placement.
    pub position: Vec3,
    /// Rotation of the placed entity.
    pub rotation: Quat,
    /// Scale of the placed entity.
    pub scale: Vec3,
}

impl PlacementRequest {
    /// Create a new placement request with identity rotation and unit scale.
    pub fn new(asset_path: impl Into<String>, position: Vec3) -> Self {
        Self {
            asset_path: asset_path.into(),
            position,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

/// Queue of pending placement requests.
#[derive(Debug, Clone, Default)]
pub struct PlacementQueue {
    pending: Vec<PlacementRequest>,
}

impl PlacementQueue {
    /// Create an empty placement queue.
    pub fn new() -> Self {
        Self {
            pending: Vec::new(),
        }
    }

    /// Push a new placement request onto the queue.
    pub fn push(&mut self, request: PlacementRequest) {
        self.pending.push(request);
    }

    /// Drain all pending requests, returning them and leaving the queue empty.
    pub fn drain(&mut self) -> Vec<PlacementRequest> {
        std::mem::take(&mut self.pending)
    }

    /// Check whether the queue has no pending requests.
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Return the number of pending requests.
    pub fn len(&self) -> usize {
        self.pending.len()
    }
}

/// An entry in the asset browser.
#[derive(Debug, Clone)]
pub struct AssetEntry {
    /// Human-readable display name.
    pub name: String,
    /// Path to the asset file.
    pub path: String,
    /// Category for grouping (e.g., "Terrain", "Props", "Characters").
    pub category: String,
}

impl AssetEntry {
    /// Create a new asset entry.
    pub fn new(name: impl Into<String>, path: impl Into<String>, category: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            path: path.into(),
            category: category.into(),
        }
    }
}

/// Asset browser for selecting assets to place in the scene.
#[derive(Debug, Clone, Default)]
pub struct AssetBrowser {
    /// All available asset entries.
    pub entries: Vec<AssetEntry>,
    /// Currently selected entry index.
    pub selected_index: Option<usize>,
    /// Current filter string for searching.
    pub filter: String,
}

impl AssetBrowser {
    /// Create a new empty asset browser.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            selected_index: None,
            filter: String::new(),
        }
    }

    /// Add an asset entry to the browser.
    pub fn add_entry(&mut self, entry: AssetEntry) {
        self.entries.push(entry);
    }

    /// Select an entry by index. Clears selection if index is out of bounds.
    pub fn select(&mut self, index: usize) {
        if index < self.entries.len() {
            self.selected_index = Some(index);
        } else {
            self.selected_index = None;
        }
    }

    /// Filter entries by a case-insensitive substring match on name, path, or category.
    ///
    /// Returns references to matching entries.
    pub fn filter_entries(&self, query: &str) -> Vec<&AssetEntry> {
        if query.is_empty() {
            return self.entries.iter().collect();
        }
        let query_lower = query.to_lowercase();
        self.entries
            .iter()
            .filter(|e| {
                e.name.to_lowercase().contains(&query_lower)
                    || e.path.to_lowercase().contains(&query_lower)
                    || e.category.to_lowercase().contains(&query_lower)
            })
            .collect()
    }

    /// Return the currently selected entry, if any.
    pub fn selected(&self) -> Option<&AssetEntry> {
        self.selected_index.and_then(|i| self.entries.get(i))
    }
}

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

/// Create a v2 `rkf_core::scene::SceneObject` from a placement request.
///
/// The returned object has no ID yet (id = 0); assign a final ID by
/// passing the object to `Scene::add_object_full`. The object's
/// `position` is set from the given `position` parameter.
pub fn create_v2_object(
    name: &str,
    position: glam::Vec3,
    primitive: rkf_core::scene_node::SdfPrimitive,
    material_id: u16,
) -> rkf_core::scene::SceneObject {
    use rkf_core::{
        scene::SceneObject,
        scene_node::SceneNode,
    };
    use glam::Quat;

    let root_node = SceneNode::analytical(name, primitive, material_id);

    let mut obj = SceneObject {
        id: 0,
        name: name.to_string(),
        parent_id: None,
        position,
        rotation: Quat::IDENTITY,
        scale: glam::Vec3::ONE,
        root_node,
        aabb: rkf_core::aabb::Aabb::new(glam::Vec3::ZERO, glam::Vec3::ZERO),
    };
    obj.aabb = compute_object_local_aabb(&obj);
    obj
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-3;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    fn vec3_approx_eq(a: Vec3, b: Vec3) -> bool {
        approx_eq(a.x, b.x) && approx_eq(a.y, b.y) && approx_eq(a.z, b.z)
    }

    // --- GridSnap tests ---

    #[test]
    fn test_grid_snap_default() {
        let snap = GridSnap::default();
        assert!(!snap.enabled);
        assert!(approx_eq(snap.grid_size, 1.0));
    }

    #[test]
    fn test_grid_snap_enabled() {
        let snap = GridSnap::new(2.0);
        assert!(snap.enabled);
        assert!(approx_eq(snap.grid_size, 2.0));
    }

    #[test]
    fn test_grid_snap_rounds_to_nearest() {
        let snap = GridSnap::new(1.0);
        let result = snap.snap(Vec3::new(1.3, 2.7, -0.4));
        assert!(vec3_approx_eq(result, Vec3::new(1.0, 3.0, 0.0)),
            "expected (1, 3, 0), got {:?}", result);
    }

    #[test]
    fn test_grid_snap_larger_grid() {
        let snap = GridSnap::new(0.5);
        let result = snap.snap(Vec3::new(1.3, 2.1, -0.8));
        assert!(vec3_approx_eq(result, Vec3::new(1.5, 2.0, -1.0)),
            "expected (1.5, 2, -1), got {:?}", result);
    }

    #[test]
    fn test_grid_snap_disabled_passthrough() {
        let snap = GridSnap::default(); // disabled
        let pos = Vec3::new(1.3, 2.7, -0.4);
        assert!(vec3_approx_eq(snap.snap(pos), pos));
    }

    #[test]
    fn test_grid_snap_zero_grid_size_passthrough() {
        let mut snap = GridSnap::new(0.0);
        snap.enabled = true;
        let pos = Vec3::new(1.3, 2.7, -0.4);
        assert!(vec3_approx_eq(snap.snap(pos), pos));
    }

    #[test]
    fn test_grid_snap_exact_grid_point() {
        let snap = GridSnap::new(1.0);
        let result = snap.snap(Vec3::new(3.0, 5.0, -2.0));
        assert!(vec3_approx_eq(result, Vec3::new(3.0, 5.0, -2.0)));
    }

    // --- ray_cast_sdf tests ---

    #[test]
    fn test_ray_cast_ground_plane_hit() {
        // Ray from above, pointing down, should hit ground plane at y=0
        let result = ray_cast_sdf(
            Vec3::new(0.0, 5.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            100.0,
            0.1,
            ground_plane_sdf,
        );
        assert!(result.hit, "should hit the ground plane");
        assert!(approx_eq(result.position.y, 0.0),
            "hit y should be ~0: {}", result.position.y);
        // Normal should point up (Y+)
        assert!(result.normal.y > 0.9,
            "ground normal should point up: {:?}", result.normal);
    }

    #[test]
    fn test_ray_cast_ground_plane_miss() {
        // Ray pointing away from the ground plane (upward from above)
        let result = ray_cast_sdf(
            Vec3::new(0.0, 5.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            100.0,
            0.1,
            ground_plane_sdf,
        );
        assert!(!result.hit, "should miss when pointing away");
    }

    #[test]
    fn test_ray_cast_sphere_hit() {
        // Ray aimed at a unit sphere at origin
        let result = ray_cast_sdf(
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::new(0.0, 0.0, -1.0),
            100.0,
            0.05,
            |p| sphere_sdf(p, 1.0),
        );
        assert!(result.hit, "should hit the sphere");
        // Hit position should be near (0, 0, 1) on the sphere surface
        assert!(approx_eq(result.position.length(), 1.0),
            "hit should be on sphere surface: dist = {}", result.position.length());
    }

    #[test]
    fn test_ray_cast_sphere_miss() {
        // Ray that misses the sphere entirely
        let result = ray_cast_sdf(
            Vec3::new(5.0, 0.0, 5.0),
            Vec3::new(0.0, 0.0, -1.0),
            100.0,
            0.1,
            |p| sphere_sdf(p, 1.0),
        );
        assert!(!result.hit, "should miss the sphere when offset");
    }

    #[test]
    fn test_ray_cast_max_dist_exceeded() {
        // Ray that would hit but max_dist is too short
        let result = ray_cast_sdf(
            Vec3::new(0.0, 5.0, 0.0),
            Vec3::new(0.0, -1.0, 0.0),
            2.0, // max_dist too short to reach y=0
            0.1,
            ground_plane_sdf,
        );
        assert!(!result.hit, "should miss when max_dist too short");
    }

    // --- PlacementRequest tests ---

    #[test]
    fn test_placement_request_defaults() {
        let req = PlacementRequest::new("assets/tree.rkf", Vec3::new(1.0, 0.0, 2.0));
        assert_eq!(req.asset_path, "assets/tree.rkf");
        assert!(vec3_approx_eq(req.position, Vec3::new(1.0, 0.0, 2.0)));
        assert!(approx_eq(req.rotation.w, 1.0)); // identity quat
        assert!(vec3_approx_eq(req.scale, Vec3::ONE));
    }

    // --- PlacementQueue tests ---

    #[test]
    fn test_placement_queue_empty() {
        let queue = PlacementQueue::new();
        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);
    }

    #[test]
    fn test_placement_queue_push_and_drain() {
        let mut queue = PlacementQueue::new();
        queue.push(PlacementRequest::new("a.rkf", Vec3::ZERO));
        queue.push(PlacementRequest::new("b.rkf", Vec3::ONE));
        assert!(!queue.is_empty());
        assert_eq!(queue.len(), 2);

        let drained = queue.drain();
        assert_eq!(drained.len(), 2);
        assert_eq!(drained[0].asset_path, "a.rkf");
        assert_eq!(drained[1].asset_path, "b.rkf");

        // Queue should be empty after drain
        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);
    }

    #[test]
    fn test_placement_queue_drain_empty() {
        let mut queue = PlacementQueue::new();
        let drained = queue.drain();
        assert!(drained.is_empty());
    }

    // --- AssetEntry tests ---

    #[test]
    fn test_asset_entry_creation() {
        let entry = AssetEntry::new("Oak Tree", "assets/trees/oak.rkf", "Vegetation");
        assert_eq!(entry.name, "Oak Tree");
        assert_eq!(entry.path, "assets/trees/oak.rkf");
        assert_eq!(entry.category, "Vegetation");
    }

    // --- AssetBrowser tests ---

    fn sample_browser() -> AssetBrowser {
        let mut browser = AssetBrowser::new();
        browser.add_entry(AssetEntry::new("Oak Tree", "assets/trees/oak.rkf", "Vegetation"));
        browser.add_entry(AssetEntry::new("Pine Tree", "assets/trees/pine.rkf", "Vegetation"));
        browser.add_entry(AssetEntry::new("Rock", "assets/props/rock.rkf", "Props"));
        browser.add_entry(AssetEntry::new("Barrel", "assets/props/barrel.rkf", "Props"));
        browser.add_entry(AssetEntry::new("Guard", "assets/characters/guard.rkf", "Characters"));
        browser
    }

    #[test]
    fn test_asset_browser_empty() {
        let browser = AssetBrowser::new();
        assert!(browser.entries.is_empty());
        assert!(browser.selected_index.is_none());
        assert!(browser.filter.is_empty());
        assert!(browser.selected().is_none());
    }

    #[test]
    fn test_asset_browser_add_entry() {
        let mut browser = AssetBrowser::new();
        browser.add_entry(AssetEntry::new("Test", "test.rkf", "Test"));
        assert_eq!(browser.entries.len(), 1);
        assert_eq!(browser.entries[0].name, "Test");
    }

    #[test]
    fn test_asset_browser_select() {
        let mut browser = sample_browser();
        browser.select(2);
        assert_eq!(browser.selected_index, Some(2));
        let selected = browser.selected().unwrap();
        assert_eq!(selected.name, "Rock");
    }

    #[test]
    fn test_asset_browser_select_out_of_bounds() {
        let mut browser = sample_browser();
        browser.select(100);
        assert!(browser.selected_index.is_none());
        assert!(browser.selected().is_none());
    }

    #[test]
    fn test_asset_browser_filter_by_name() {
        let browser = sample_browser();
        let results = browser.filter_entries("tree");
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|e| e.name.to_lowercase().contains("tree")));
    }

    #[test]
    fn test_asset_browser_filter_case_insensitive() {
        let browser = sample_browser();
        let results = browser.filter_entries("ROCK");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "Rock");
    }

    #[test]
    fn test_asset_browser_filter_by_category() {
        let browser = sample_browser();
        let results = browser.filter_entries("props");
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|e| e.category == "Props"));
    }

    #[test]
    fn test_asset_browser_filter_by_path() {
        let browser = sample_browser();
        let results = browser.filter_entries("characters");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "Guard");
    }

    #[test]
    fn test_asset_browser_filter_empty_returns_all() {
        let browser = sample_browser();
        let results = browser.filter_entries("");
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_asset_browser_filter_no_match() {
        let browser = sample_browser();
        let results = browser.filter_entries("zzzzz");
        assert!(results.is_empty());
    }

    #[test]
    fn test_surface_snap_result_miss() {
        let miss = SurfaceSnapResult::miss();
        assert!(!miss.hit);
        assert!(vec3_approx_eq(miss.position, Vec3::ZERO));
    }

    // --- create_v2_object tests ---

    #[test]
    fn test_create_v2_object_fields() {
        use rkf_core::scene_node::{SdfPrimitive, SdfSource};

        let pos = Vec3::new(1.0, 2.0, 3.0);
        let prim = SdfPrimitive::Sphere { radius: 0.5 };
        let obj = super::create_v2_object("MySphere", pos, prim, 7);

        assert_eq!(obj.name, "MySphere");
        assert_eq!(obj.id, 0);
        assert!(vec3_approx_eq(obj.position, pos));
        assert!(approx_eq(obj.scale.x, 1.0));

        // Root node should be analytical with the given primitive
        match &obj.root_node.sdf_source {
            SdfSource::Analytical { primitive, material_id } => {
                assert!(matches!(primitive, SdfPrimitive::Sphere { radius } if (*radius - 0.5).abs() < EPS));
                assert_eq!(*material_id, 7);
            }
            _ => panic!("expected Analytical sdf_source"),
        }
    }

    #[test]
    fn test_create_v2_object_name_propagates_to_node() {
        use rkf_core::scene_node::SdfPrimitive;

        let obj = super::create_v2_object("BoxPrimitive", Vec3::ZERO, SdfPrimitive::Box { half_extents: Vec3::ONE }, 0);
        assert_eq!(obj.root_node.name, "BoxPrimitive");
    }

    #[test]
    fn test_create_v2_object_added_to_scene() {
        use rkf_core::{scene::Scene, scene_node::SdfPrimitive};

        let obj = super::create_v2_object("Capsule", Vec3::ZERO, SdfPrimitive::Capsule { radius: 0.2, half_height: 0.5 }, 1);

        let mut scene = Scene::new("test");
        let id = scene.add_object_full(obj);
        assert!(id > 0);
        assert_eq!(scene.object_count(), 1);
        assert_eq!(scene.objects[0].name, "Capsule");
    }
}
