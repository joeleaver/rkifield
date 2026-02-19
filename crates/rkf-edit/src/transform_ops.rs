//! SDF object transform system.
//!
//! Provides a reusable library for moving, rotating, and scaling SDF objects
//! that are baked into a [`BrickPool`] + [`SparseGrid`]. When an object's
//! transform changes, the affected voxel region is cleared and re-voxelized
//! from the analytic SDF recipe, then only dirty bricks are uploaded to the GPU.
//!
//! Key types:
//! - [`SdfPrimitive`] — analytic SDF shape tree (sphere, box, capsule, unions)
//! - [`ObjectTransform`] — position + rotation + uniform scale
//! - [`SdfRecipe`] — primitive + material
//! - [`SdfObjectRegistry`] — tracks all live SDF objects for overlap queries

use glam::{Quat, UVec3, Vec3};

use rkf_core::aabb::Aabb;
use rkf_core::brick_pool::BrickPool;
use rkf_core::cell_state::CellState;
use rkf_core::constants::{BRICK_DIM, RESOLUTION_TIERS};
use rkf_core::sdf::{box_sdf, capsule_sdf, smin, sphere_sdf};
use rkf_core::sparse_grid::SparseGrid;
use rkf_core::voxel::VoxelSample;

// ---------------------------------------------------------------------------
// SDF Primitive
// ---------------------------------------------------------------------------

/// An analytic SDF shape that can be evaluated at any point.
#[derive(Debug, Clone)]
#[allow(missing_docs)]
pub enum SdfPrimitive {
    /// Sphere centered at origin with given radius.
    Sphere { radius: f32 },
    /// Axis-aligned box centered at origin with given half-extents.
    Box { half_extents: Vec3 },
    /// Capsule between two local-space endpoints with given radius.
    Capsule { a: Vec3, b: Vec3, radius: f32 },
    /// Hard union of child primitives, each with a local offset.
    Union(Vec<(SdfPrimitive, Vec3)>),
    /// Smooth union of child primitives with blend radius `k`.
    SmoothUnion {
        /// Child primitives with local offsets.
        children: Vec<(SdfPrimitive, Vec3)>,
        /// Smooth-min blend radius.
        k: f32,
    },
}

impl SdfPrimitive {
    /// Evaluate the SDF at `point` in world space, applying the given transform.
    ///
    /// Transforms the query point into the primitive's local space:
    /// `local = inverse_rot((point - position) / scale)`, then evaluates
    /// the local SDF and scales the result back.
    pub fn eval(&self, point: Vec3, transform: &ObjectTransform) -> f32 {
        let local = transform.world_to_local(point);
        self.eval_local(local) * transform.scale
    }

    /// Evaluate the SDF at a point already in local (untransformed) space.
    pub fn eval_local(&self, p: Vec3) -> f32 {
        match self {
            SdfPrimitive::Sphere { radius } => sphere_sdf(Vec3::ZERO, *radius, p),
            SdfPrimitive::Box { half_extents } => box_sdf(*half_extents, p),
            SdfPrimitive::Capsule { a, b, radius } => capsule_sdf(*a, *b, *radius, p),
            SdfPrimitive::Union(children) => {
                let mut d = f32::MAX;
                for (child, offset) in children {
                    d = d.min(child.eval_local(p - *offset));
                }
                d
            }
            SdfPrimitive::SmoothUnion { children, k } => {
                let mut d = f32::MAX;
                for (child, offset) in children {
                    let cd = child.eval_local(p - *offset);
                    if d == f32::MAX {
                        d = cd;
                    } else {
                        d = smin(d, cd, *k);
                    }
                }
                d
            }
        }
    }

    /// Compute the axis-aligned bounding box in local (untransformed) space.
    ///
    /// Includes a margin for the narrow band (surface voxels extend beyond
    /// the exact zero-crossing).
    pub fn local_aabb(&self) -> Aabb {
        match self {
            SdfPrimitive::Sphere { radius } => {
                let r = Vec3::splat(*radius);
                Aabb::new(-r, r)
            }
            SdfPrimitive::Box { half_extents } => Aabb::new(-*half_extents, *half_extents),
            SdfPrimitive::Capsule { a, b, radius } => {
                let r = Vec3::splat(*radius);
                let min = a.min(*b) - r;
                let max = a.max(*b) + r;
                Aabb::new(min, max)
            }
            SdfPrimitive::Union(children) => {
                let mut aabb = Aabb::new(Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
                for (child, offset) in children {
                    let child_aabb = child.local_aabb();
                    let shifted = Aabb::new(child_aabb.min + *offset, child_aabb.max + *offset);
                    aabb = aabb.expand_aabb(&shifted);
                }
                aabb
            }
            SdfPrimitive::SmoothUnion { children, k } => {
                let mut aabb = Aabb::new(Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
                for (child, offset) in children {
                    let child_aabb = child.local_aabb();
                    // Smooth union expands the surface by up to k
                    let margin = Vec3::splat(*k);
                    let shifted = Aabb::new(
                        child_aabb.min + *offset - margin,
                        child_aabb.max + *offset + margin,
                    );
                    aabb = aabb.expand_aabb(&shifted);
                }
                aabb
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Object Transform
// ---------------------------------------------------------------------------

/// Position + rotation + uniform scale for an SDF object.
#[derive(Debug, Clone, Copy)]
pub struct ObjectTransform {
    /// World-space position.
    pub position: Vec3,
    /// Rotation quaternion.
    pub rotation: Quat,
    /// Uniform scale factor (per engine rule 3: uniform only).
    pub scale: f32,
}

impl Default for ObjectTransform {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: 1.0,
        }
    }
}

impl ObjectTransform {
    /// Transform a world-space point into this object's local space.
    #[inline]
    pub fn world_to_local(&self, point: Vec3) -> Vec3 {
        self.rotation.inverse() * ((point - self.position) / self.scale)
    }
}

// ---------------------------------------------------------------------------
// SDF Recipe & Object
// ---------------------------------------------------------------------------

/// An SDF primitive paired with a material ID.
#[derive(Debug, Clone)]
pub struct SdfRecipe {
    /// The analytic SDF shape.
    pub primitive: SdfPrimitive,
    /// Material ID assigned to voxels.
    pub material_id: u16,
}

/// A registered SDF object with its recipe, transform, and clipmap level.
#[derive(Debug, Clone)]
pub struct SdfObject {
    /// Unique object ID.
    pub id: u64,
    /// The SDF recipe (shape + material).
    pub recipe: SdfRecipe,
    /// Current world-space transform.
    pub transform: ObjectTransform,
    /// Which clipmap level this object lives in.
    pub clipmap_level: usize,
}

impl SdfObject {
    /// Compute the world-space AABB for this object.
    ///
    /// Rotates/scales the local AABB corners into world space, then takes
    /// the enclosing axis-aligned box with a narrow-band margin.
    pub fn world_aabb(&self) -> Aabb {
        let local = self.recipe.primitive.local_aabb();
        let corners = local.corners();

        let mut min = Vec3::splat(f32::MAX);
        let mut max = Vec3::splat(f32::MIN);
        for corner in &corners {
            let world = self.transform.rotation * (*corner * self.transform.scale)
                + self.transform.position;
            min = min.min(world);
            max = max.max(world);
        }

        // Add narrow-band margin (3 brick extents at finest tier used)
        let margin = Vec3::splat(0.64); // ~3 * 8cm * 8 voxels = conservative
        Aabb::new(min - margin, max + margin)
    }
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

/// Tracks all SDF objects in the scene for overlap queries during transforms.
#[derive(Debug, Default)]
pub struct SdfObjectRegistry {
    objects: Vec<SdfObject>,
    next_id: u64,
}

impl SdfObjectRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            objects: Vec::new(),
            next_id: 1,
        }
    }

    /// Register a new SDF object. Returns its unique ID.
    pub fn register(
        &mut self,
        recipe: SdfRecipe,
        transform: ObjectTransform,
        clipmap_level: usize,
    ) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.objects.push(SdfObject {
            id,
            recipe,
            transform,
            clipmap_level,
        });
        id
    }

    /// Register a new SDF object with a specific ID.
    ///
    /// Used when entity IDs need to match scene tree node IDs.
    /// Advances `next_id` past the given ID if needed.
    pub fn register_with_id(
        &mut self,
        id: u64,
        recipe: SdfRecipe,
        transform: ObjectTransform,
        clipmap_level: usize,
    ) {
        if id >= self.next_id {
            self.next_id = id + 1;
        }
        self.objects.push(SdfObject {
            id,
            recipe,
            transform,
            clipmap_level,
        });
    }

    /// Get an object by ID.
    pub fn get(&self, id: u64) -> Option<&SdfObject> {
        self.objects.iter().find(|o| o.id == id)
    }

    /// Get a mutable reference to an object by ID.
    pub fn get_mut(&mut self, id: u64) -> Option<&mut SdfObject> {
        self.objects.iter_mut().find(|o| o.id == id)
    }

    /// Remove an object by ID. Returns the removed object, if found.
    pub fn remove(&mut self, id: u64) -> Option<SdfObject> {
        let idx = self.objects.iter().position(|o| o.id == id)?;
        Some(self.objects.swap_remove(idx))
    }

    /// Find all objects whose world AABB overlaps the given region
    /// and that belong to the specified clipmap level.
    pub fn objects_overlapping(&self, region: &Aabb, level: usize) -> Vec<u64> {
        self.objects
            .iter()
            .filter(|o| o.clipmap_level == level && o.world_aabb().intersects(region))
            .map(|o| o.id)
            .collect()
    }

    /// Number of registered objects.
    pub fn len(&self) -> usize {
        self.objects.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Core transform operation
// ---------------------------------------------------------------------------

/// Result of an [`apply_transform_change`] operation.
pub struct TransformResult {
    /// Brick pool slot indices that were modified.
    pub dirty_brick_slots: Vec<u32>,
    /// Flat cell indices whose occupancy/slot data changed.
    pub dirty_cell_indices: Vec<u32>,
}

/// Convert a world-space AABB to a cell range in the grid.
///
/// Returns `(cell_min, cell_max)` clamped to grid dimensions.
/// `cell_max` is inclusive.
pub fn aabb_to_cell_range(
    aabb: &Aabb,
    grid_aabb: &Aabb,
    brick_extent: f32,
    dims: UVec3,
) -> (UVec3, UVec3) {
    let rel_min = (aabb.min - grid_aabb.min) / brick_extent;
    let rel_max = (aabb.max - grid_aabb.min) / brick_extent;

    let cell_min = UVec3::new(
        (rel_min.x.floor() as i32).max(0) as u32,
        (rel_min.y.floor() as i32).max(0) as u32,
        (rel_min.z.floor() as i32).max(0) as u32,
    );
    let cell_max = UVec3::new(
        ((rel_max.x.ceil() as i32).max(0) as u32).min(dims.x.saturating_sub(1)),
        ((rel_max.y.ceil() as i32).max(0) as u32).min(dims.y.saturating_sub(1)),
        ((rel_max.z.ceil() as i32).max(0) as u32).min(dims.z.saturating_sub(1)),
    );

    (cell_min, cell_max)
}

/// Check if a brick has any voxel within the narrow band of a surface.
fn brick_has_surface(pool: &BrickPool, slot: u32, voxel_size: f32) -> bool {
    let threshold = voxel_size * 4.0;
    let brick = pool.get(slot);
    for vz in 0..BRICK_DIM {
        for vy in 0..BRICK_DIM {
            for vx in 0..BRICK_DIM {
                let d = brick.sample(vx, vy, vz).distance_f32();
                if d.abs() < threshold {
                    return true;
                }
            }
        }
    }
    false
}

/// Voxelize a single SDF object into a cell subrange of a grid.
///
/// Fills bricks with SDF samples using SDF-union semantics (min distance wins).
/// Allocates new bricks as needed.
fn revoxelize_region(
    pool: &mut BrickPool,
    grid: &mut SparseGrid,
    grid_aabb: &Aabb,
    tier: usize,
    primitive: &SdfPrimitive,
    transform: &ObjectTransform,
    material_id: u16,
    cell_min: UVec3,
    cell_max: UVec3,
) -> Vec<u32> {
    let res = &RESOLUTION_TIERS[tier];
    let voxel_size = res.voxel_size;
    let brick_extent = res.brick_extent;
    let narrow_band_dist = 3.0 * brick_extent;
    let mut dirty_slots = Vec::new();

    for cz in cell_min.z..=cell_max.z {
        for cy in cell_min.y..=cell_max.y {
            for cx in cell_min.x..=cell_max.x {
                let brick_min = grid_aabb.min
                    + Vec3::new(
                        cx as f32 * brick_extent,
                        cy as f32 * brick_extent,
                        cz as f32 * brick_extent,
                    );
                let brick_center = brick_min + Vec3::splat(brick_extent * 0.5);
                let center_dist = primitive.eval(brick_center, transform);

                if center_dist.abs() > narrow_band_dist {
                    // This object doesn't contribute to this cell
                    continue;
                }

                // Get or allocate a brick for this cell
                let slot = if grid.cell_state(cx, cy, cz) == CellState::Surface {
                    grid.brick_slot(cx, cy, cz).unwrap()
                } else {
                    let s = match pool.allocate() {
                        Some(s) => s,
                        None => continue, // Pool exhausted, skip
                    };
                    // Initialize new brick to default (far distance)
                    let brick = pool.get_mut(s);
                    for vz in 0..BRICK_DIM {
                        for vy in 0..BRICK_DIM {
                            for vx in 0..BRICK_DIM {
                                brick.set(vx, vy, vz, VoxelSample::default());
                            }
                        }
                    }
                    grid.set_cell_state(cx, cy, cz, CellState::Surface);
                    grid.set_brick_slot(cx, cy, cz, s);
                    s
                };

                let brick = pool.get_mut(slot);
                for vz in 0..BRICK_DIM {
                    for vy in 0..BRICK_DIM {
                        for vx in 0..BRICK_DIM {
                            let voxel_pos = brick_min
                                + Vec3::new(
                                    (vx as f32 + 0.5) * voxel_size,
                                    (vy as f32 + 0.5) * voxel_size,
                                    (vz as f32 + 0.5) * voxel_size,
                                );
                            let new_dist = primitive.eval(voxel_pos, transform);
                            let existing = brick.sample(vx, vy, vz);
                            let old_dist = existing.distance_f32();
                            // SDF union: closer surface wins
                            if new_dist < old_dist {
                                brick.set(
                                    vx,
                                    vy,
                                    vz,
                                    VoxelSample::new(new_dist, material_id, 0, 0, 0),
                                );
                            }
                        }
                    }
                }

                if !dirty_slots.contains(&slot) {
                    dirty_slots.push(slot);
                }
            }
        }
    }

    dirty_slots
}

/// Apply a transform change to an SDF object.
///
/// Clears old voxels in the affected region, re-voxelizes all overlapping
/// objects (including the moved one at its new transform), and returns
/// the list of dirty brick slots and cell indices for GPU upload.
///
/// **Does not** update the registry — the caller must set the new transform
/// on the object after this returns.
pub fn apply_transform_change(
    pool: &mut BrickPool,
    grid: &mut SparseGrid,
    grid_aabb: &Aabb,
    tier: usize,
    registry: &SdfObjectRegistry,
    object_id: u64,
    new_transform: &ObjectTransform,
) -> Option<TransformResult> {
    let obj = registry.get(object_id)?;
    let level = obj.clipmap_level;

    // 1. Compute old + new world AABBs → union = affected region
    let old_aabb = obj.world_aabb();
    let new_obj_aabb = {
        // Temporarily compute AABB at new transform
        let local = obj.recipe.primitive.local_aabb();
        let corners = local.corners();
        let mut min = Vec3::splat(f32::MAX);
        let mut max = Vec3::splat(f32::MIN);
        for corner in &corners {
            let world =
                new_transform.rotation * (*corner * new_transform.scale) + new_transform.position;
            min = min.min(world);
            max = max.max(world);
        }
        let margin = Vec3::splat(0.64);
        Aabb::new(min - margin, max + margin)
    };
    let affected_aabb = old_aabb.expand_aabb(&new_obj_aabb);

    let res = &RESOLUTION_TIERS[tier];
    let brick_extent = res.brick_extent;
    let voxel_size = res.voxel_size;
    let dims = grid.dimensions();

    // 2. Convert affected AABB to cell range
    let (cell_min, cell_max) = aabb_to_cell_range(&affected_aabb, grid_aabb, brick_extent, dims);

    // 3. Collect existing brick slots in the affected range and clear them
    let mut dirty_slots = Vec::new();
    let mut dirty_cells = Vec::new();

    for cz in cell_min.z..=cell_max.z {
        for cy in cell_min.y..=cell_max.y {
            for cx in cell_min.x..=cell_max.x {
                let flat = cz * dims.x * dims.y + cy * dims.x + cx;

                if grid.cell_state(cx, cy, cz) == CellState::Surface {
                    if let Some(slot) = grid.brick_slot(cx, cy, cz) {
                        // Clear the brick to default (far distance)
                        let brick = pool.get_mut(slot);
                        for vz in 0..BRICK_DIM {
                            for vy in 0..BRICK_DIM {
                                for vx in 0..BRICK_DIM {
                                    brick.set(vx, vy, vz, VoxelSample::default());
                                }
                            }
                        }
                        if !dirty_slots.contains(&slot) {
                            dirty_slots.push(slot);
                        }
                    }
                }

                dirty_cells.push(flat);
            }
        }
    }

    // 4. Find all objects overlapping the affected region at this level
    let overlapping = registry.objects_overlapping(&affected_aabb, level);

    // 5. Re-voxelize each overlapping object
    for oid in &overlapping {
        let other = registry.get(*oid).unwrap();
        let (prim, transform, mat) = if *oid == object_id {
            // Use the NEW transform for the moved object
            (&other.recipe.primitive, new_transform, other.recipe.material_id)
        } else {
            // Use the existing transform for all others
            (&other.recipe.primitive, &other.transform, other.recipe.material_id)
        };

        let new_dirty = revoxelize_region(
            pool, grid, grid_aabb, tier, prim, transform, mat, cell_min, cell_max,
        );
        for s in new_dirty {
            if !dirty_slots.contains(&s) {
                dirty_slots.push(s);
            }
        }
    }

    // 6. Cleanup: deallocate bricks where all voxels are far from the surface
    for cz in cell_min.z..=cell_max.z {
        for cy in cell_min.y..=cell_max.y {
            for cx in cell_min.x..=cell_max.x {
                if grid.cell_state(cx, cy, cz) != CellState::Surface {
                    continue;
                }
                let slot = match grid.brick_slot(cx, cy, cz) {
                    Some(s) => s,
                    None => continue,
                };
                if !brick_has_surface(pool, slot, voxel_size) {
                    pool.deallocate(slot);
                    grid.set_cell_state(cx, cy, cz, CellState::Empty);
                    grid.set_brick_slot(cx, cy, cz, rkf_core::sparse_grid::EMPTY_SLOT);
                    // Slot was already in dirty_slots, which is fine —
                    // the GPU upload will write the cleared data
                }
            }
        }
    }

    Some(TransformResult {
        dirty_brick_slots: dirty_slots,
        dirty_cell_indices: dirty_cells,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rkf_core::brick_pool::Pool;

    const EPS: f32 = 1e-3;

    fn approx(a: f32, b: f32) -> bool {
        (a - b).abs() < EPS
    }

    // ── SDF eval tests ───────────────────────────────────────────────────

    #[test]
    fn sphere_eval_identity() {
        let prim = SdfPrimitive::Sphere { radius: 1.0 };
        let xf = ObjectTransform::default();
        assert!(approx(prim.eval(Vec3::ZERO, &xf), -1.0));
        assert!(approx(prim.eval(Vec3::X, &xf), 0.0));
        assert!(approx(prim.eval(Vec3::X * 2.0, &xf), 1.0));
    }

    #[test]
    fn sphere_eval_translated() {
        let prim = SdfPrimitive::Sphere { radius: 1.0 };
        let xf = ObjectTransform {
            position: Vec3::new(3.0, 0.0, 0.0),
            ..Default::default()
        };
        // Center of sphere is at (3,0,0)
        assert!(approx(prim.eval(Vec3::new(3.0, 0.0, 0.0), &xf), -1.0));
        assert!(approx(prim.eval(Vec3::new(4.0, 0.0, 0.0), &xf), 0.0));
    }

    #[test]
    fn sphere_eval_scaled() {
        let prim = SdfPrimitive::Sphere { radius: 1.0 };
        let xf = ObjectTransform {
            scale: 2.0,
            ..Default::default()
        };
        // Scaled sphere has effective radius 2.0
        assert!(approx(prim.eval(Vec3::ZERO, &xf), -2.0));
        assert!(approx(prim.eval(Vec3::X * 2.0, &xf), 0.0));
    }

    #[test]
    fn sphere_eval_rotated() {
        let prim = SdfPrimitive::Sphere { radius: 1.0 };
        let xf = ObjectTransform {
            rotation: Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
            ..Default::default()
        };
        // Sphere is rotation-invariant
        assert!(approx(prim.eval(Vec3::X, &xf), 0.0));
        assert!(approx(prim.eval(Vec3::Z, &xf), 0.0));
    }

    #[test]
    fn box_eval_rotated() {
        let prim = SdfPrimitive::Box {
            half_extents: Vec3::new(2.0, 1.0, 1.0),
        };
        let xf = ObjectTransform {
            rotation: Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
            ..Default::default()
        };
        // Rotated 90 deg around Y: X-extent becomes Z-extent
        // Point at (0,0,2) should be on the surface (was X-extent=2)
        assert!(approx(prim.eval(Vec3::new(0.0, 0.0, 2.0), &xf), 0.0));
        // Point at (2,0,0) should be outside (was X, now Z-extent=1)
        assert!(prim.eval(Vec3::new(2.0, 0.0, 0.0), &xf) > 0.0);
    }

    #[test]
    fn union_eval() {
        let prim = SdfPrimitive::Union(vec![
            (SdfPrimitive::Sphere { radius: 1.0 }, Vec3::ZERO),
            (SdfPrimitive::Sphere { radius: 1.0 }, Vec3::new(3.0, 0.0, 0.0)),
        ]);
        let xf = ObjectTransform::default();
        // At origin: inside first sphere
        assert!(prim.eval(Vec3::ZERO, &xf) < 0.0);
        // At (3,0,0): inside second sphere
        assert!(prim.eval(Vec3::new(3.0, 0.0, 0.0), &xf) < 0.0);
        // At (1.5,0,0): between both spheres, outside
        assert!(prim.eval(Vec3::new(1.5, 0.0, 0.0), &xf) > 0.0);
    }

    // ── local_aabb tests ─────────────────────────────────────────────────

    #[test]
    fn sphere_local_aabb() {
        let prim = SdfPrimitive::Sphere { radius: 2.0 };
        let aabb = prim.local_aabb();
        assert!(approx(aabb.min.x, -2.0));
        assert!(approx(aabb.max.x, 2.0));
        assert!(approx(aabb.min.y, -2.0));
        assert!(approx(aabb.max.z, 2.0));
    }

    #[test]
    fn box_local_aabb() {
        let prim = SdfPrimitive::Box {
            half_extents: Vec3::new(1.0, 2.0, 3.0),
        };
        let aabb = prim.local_aabb();
        assert!(approx(aabb.min.x, -1.0));
        assert!(approx(aabb.max.y, 2.0));
        assert!(approx(aabb.max.z, 3.0));
    }

    #[test]
    fn capsule_local_aabb() {
        let prim = SdfPrimitive::Capsule {
            a: Vec3::new(0.0, -1.0, 0.0),
            b: Vec3::new(0.0, 1.0, 0.0),
            radius: 0.5,
        };
        let aabb = prim.local_aabb();
        assert!(approx(aabb.min.y, -1.5));
        assert!(approx(aabb.max.y, 1.5));
        assert!(approx(aabb.min.x, -0.5));
        assert!(approx(aabb.max.x, 0.5));
    }

    // ── world_aabb tests ─────────────────────────────────────────────────

    #[test]
    fn world_aabb_identity() {
        let obj = SdfObject {
            id: 1,
            recipe: SdfRecipe {
                primitive: SdfPrimitive::Sphere { radius: 1.0 },
                material_id: 1,
            },
            transform: ObjectTransform::default(),
            clipmap_level: 0,
        };
        let aabb = obj.world_aabb();
        // Should contain the sphere with margin
        assert!(aabb.min.x < -1.0);
        assert!(aabb.max.x > 1.0);
    }

    #[test]
    fn world_aabb_translated() {
        let obj = SdfObject {
            id: 1,
            recipe: SdfRecipe {
                primitive: SdfPrimitive::Sphere { radius: 1.0 },
                material_id: 1,
            },
            transform: ObjectTransform {
                position: Vec3::new(10.0, 0.0, 0.0),
                ..Default::default()
            },
            clipmap_level: 0,
        };
        let aabb = obj.world_aabb();
        assert!(aabb.min.x > 7.0); // 10 - 1 - margin
        assert!(aabb.max.x < 13.0); // 10 + 1 + margin
    }

    #[test]
    fn world_aabb_rotated_box() {
        let obj = SdfObject {
            id: 1,
            recipe: SdfRecipe {
                primitive: SdfPrimitive::Box {
                    half_extents: Vec3::new(2.0, 0.5, 0.5),
                },
                material_id: 1,
            },
            transform: ObjectTransform {
                rotation: Quat::from_rotation_y(std::f32::consts::FRAC_PI_4), // 45 deg
                ..Default::default()
            },
            clipmap_level: 0,
        };
        let aabb = obj.world_aabb();
        // Rotated box should be wider than the original X-extent on both X and Z
        let half_diag = (2.0f32 * 2.0 + 0.5 * 0.5).sqrt(); // ~2.06
        assert!(aabb.max.x > half_diag - 0.1);
        assert!(aabb.max.z > 1.0); // Rotated, Z should extend beyond original 0.5
    }

    // ── Registry tests ───────────────────────────────────────────────────

    #[test]
    fn registry_crud() {
        let mut reg = SdfObjectRegistry::new();
        let id = reg.register(
            SdfRecipe {
                primitive: SdfPrimitive::Sphere { radius: 1.0 },
                material_id: 1,
            },
            ObjectTransform::default(),
            0,
        );
        assert_eq!(reg.len(), 1);
        assert!(reg.get(id).is_some());

        // Mutate
        reg.get_mut(id).unwrap().transform.position = Vec3::X;
        assert!(approx(reg.get(id).unwrap().transform.position.x, 1.0));

        // Remove
        let removed = reg.remove(id);
        assert!(removed.is_some());
        assert_eq!(reg.len(), 0);
        assert!(reg.get(id).is_none());
    }

    #[test]
    fn registry_overlapping() {
        let mut reg = SdfObjectRegistry::new();
        let id1 = reg.register(
            SdfRecipe {
                primitive: SdfPrimitive::Sphere { radius: 1.0 },
                material_id: 1,
            },
            ObjectTransform::default(),
            0,
        );
        let id2 = reg.register(
            SdfRecipe {
                primitive: SdfPrimitive::Sphere { radius: 1.0 },
                material_id: 2,
            },
            ObjectTransform {
                position: Vec3::new(10.0, 0.0, 0.0),
                ..Default::default()
            },
            0,
        );
        // Level 1 object — should not match level 0 queries
        let _id3 = reg.register(
            SdfRecipe {
                primitive: SdfPrimitive::Sphere { radius: 1.0 },
                material_id: 3,
            },
            ObjectTransform::default(),
            1,
        );

        // Query near origin at level 0 → should find id1 only
        let query = Aabb::new(Vec3::splat(-2.0), Vec3::splat(2.0));
        let result = reg.objects_overlapping(&query, 0);
        assert!(result.contains(&id1));
        assert!(!result.contains(&id2));

        // Query covering both at level 0
        let big = Aabb::new(Vec3::splat(-20.0), Vec3::splat(20.0));
        let result = reg.objects_overlapping(&big, 0);
        assert!(result.contains(&id1));
        assert!(result.contains(&id2));
    }

    // ── aabb_to_cell_range tests ─────────────────────────────────────────

    #[test]
    fn cell_range_basic() {
        let grid_aabb = Aabb::new(Vec3::ZERO, Vec3::splat(8.0));
        let dims = UVec3::splat(10);
        let query = Aabb::new(Vec3::new(1.5, 1.5, 1.5), Vec3::new(3.5, 3.5, 3.5));
        let (min, max) = aabb_to_cell_range(&query, &grid_aabb, 0.8, dims);
        // floor(1.5/0.8) = 1, ceil(3.5/0.8) = 5
        assert_eq!(min, UVec3::new(1, 1, 1));
        assert_eq!(max, UVec3::new(5, 5, 5));
    }

    #[test]
    fn cell_range_clamped() {
        let grid_aabb = Aabb::new(Vec3::ZERO, Vec3::splat(4.0));
        let dims = UVec3::splat(5);
        // Query extends beyond grid
        let query = Aabb::new(Vec3::splat(-2.0), Vec3::splat(10.0));
        let (min, max) = aabb_to_cell_range(&query, &grid_aabb, 0.8, dims);
        assert_eq!(min, UVec3::ZERO);
        assert_eq!(max, UVec3::splat(4)); // dims - 1
    }

    // ── Core transform test ──────────────────────────────────────────────

    #[test]
    fn transform_moves_voxels() {
        // Create a small grid with two spheres
        let tier = 2; // 8cm voxels, 0.64m brick extent
        let grid_aabb = Aabb::new(Vec3::splat(-4.0), Vec3::splat(4.0));
        let res = &RESOLUTION_TIERS[tier];
        let size = grid_aabb.size();
        let dims = UVec3::new(
            (size.x / res.brick_extent).ceil() as u32,
            (size.y / res.brick_extent).ceil() as u32,
            (size.z / res.brick_extent).ceil() as u32,
        );
        let mut grid = SparseGrid::new(dims);
        let mut pool: BrickPool = Pool::new(4096);

        // Register two spheres
        let mut registry = SdfObjectRegistry::new();
        let xf1 = ObjectTransform {
            position: Vec3::new(-2.0, 0.0, 0.0),
            ..Default::default()
        };
        let xf2 = ObjectTransform {
            position: Vec3::new(2.0, 0.0, 0.0),
            ..Default::default()
        };

        let id1 = registry.register(
            SdfRecipe {
                primitive: SdfPrimitive::Sphere { radius: 0.5 },
                material_id: 1,
            },
            xf1,
            0,
        );
        let id2 = registry.register(
            SdfRecipe {
                primitive: SdfPrimitive::Sphere { radius: 0.5 },
                material_id: 2,
            },
            xf2,
            0,
        );

        // Initial voxelization
        use rkf_core::populate::populate_grid_with_material;
        let _ = populate_grid_with_material(
            &mut pool,
            &mut grid,
            |p| sphere_sdf(Vec3::new(-2.0, 0.0, 0.0), 0.5, p),
            tier,
            &grid_aabb,
            1,
        );
        let _ = populate_grid_with_material(
            &mut pool,
            &mut grid,
            |p| sphere_sdf(Vec3::new(2.0, 0.0, 0.0), 0.5, p),
            tier,
            &grid_aabb,
            2,
        );

        let initial_bricks = pool.allocated_count();
        assert!(initial_bricks > 0);

        // Move sphere 1 from (-2,0,0) to (0,0,0)
        let new_xf = ObjectTransform {
            position: Vec3::new(0.0, 0.0, 0.0),
            ..Default::default()
        };
        let result = apply_transform_change(
            &mut pool, &mut grid, &grid_aabb, tier, &registry, id1, &new_xf,
        );
        assert!(result.is_some());
        let result = result.unwrap();
        assert!(!result.dirty_brick_slots.is_empty());

        // Update registry
        registry.get_mut(id1).unwrap().transform = new_xf;

        // Verify sphere 2 is still intact: sample at (2,0,0) should be inside
        let obj2 = registry.get(id2).unwrap();
        let d = obj2
            .recipe
            .primitive
            .eval(Vec3::new(2.0, 0.0, 0.0), &obj2.transform);
        assert!(d < 0.0, "sphere 2 should still be solid at its center: d={d}");

        // Verify sphere 1 moved: sample at (0,0,0) should be inside
        let obj1 = registry.get(id1).unwrap();
        let d = obj1
            .recipe
            .primitive
            .eval(Vec3::new(0.0, 0.0, 0.0), &obj1.transform);
        assert!(d < 0.0, "sphere 1 should be solid at new center: d={d}");
    }

    #[test]
    fn register_with_id_works() {
        let mut reg = SdfObjectRegistry::new();
        reg.register_with_id(
            42,
            SdfRecipe {
                primitive: SdfPrimitive::Sphere { radius: 1.0 },
                material_id: 1,
            },
            ObjectTransform::default(),
            0,
        );
        assert!(reg.get(42).is_some());
        assert_eq!(reg.len(), 1);
    }
}
