# Geometry-First Architecture — Implementation Plan

Reference: [GEOMETRY_FIRST_ARCHITECTURE.md](GEOMETRY_FIRST_ARCHITECTURE.md)

## Overview

This plan migrates the engine from SDF-distance-as-source-of-truth to geometry-first (occupancy + surface voxels). SDF distances become a derived cache computed from geometry.

The work is organized into 8 phases. Each phase produces a compiling, testable codebase. These are breaking changes — no backward compatibility with old formats or APIs is maintained.

---

## Phase 1: Core Data Types (Foundation)

New geometry-first types alongside existing ones. Nothing is deleted yet.

### 1.1 — BrickGeometry type

**File:** `crates/rkf-core/src/brick_geometry.rs` (NEW)

```rust
/// Compact geometry representation for one 8×8×8 brick.
pub struct BrickGeometry {
    /// 512-bit occupancy bitmask. Bit N = voxel N is solid.
    pub occupancy: [u64; 8],
    /// Data for voxels on the solid/empty boundary.
    pub surface_voxels: Vec<SurfaceVoxel>,
}

/// Per-surface-voxel data: color + material.
#[repr(C)]
pub struct SurfaceVoxel {
    /// Index within the 8×8×8 brick (0–511).
    pub index: u16,
    /// RGBA diffuse color. Alpha available for custom shader control.
    pub color: [u8; 4],
    /// Material table index for PBR properties.
    pub material_id: u8,
    /// Reserved for future use.
    pub _reserved: u8,
}
```

Implement:
- `BrickGeometry::new()`, `is_solid(x, y, z)`, `set_solid(x, y, z, solid)`
- `is_surface_voxel(x, y, z)` — solid with at least one empty 6-neighbor
- `rebuild_surface_list()` — scan occupancy, populate surface_voxels
- `voxel_index(x, y, z) -> u16` and `index_to_xyz(index) -> (u8, u8, u8)`
- `solid_count()`, `surface_count()`
- `is_fully_solid()`, `is_fully_empty()` — for INTERIOR_SLOT / EMPTY_SLOT classification
- Serialization: `to_bytes()` / `from_bytes()` (compact: occupancy + surface list)

Tests: occupancy bit manipulation, surface voxel identification, boundary detection, fully-solid/empty classification, roundtrip serialization.

### 1.2 — SdfCache type

**File:** `crates/rkf-core/src/sdf_cache.rs` (NEW)

```rust
/// Cached signed distance field for one 8×8×8 brick.
/// Derived from BrickGeometry. Not a source of truth.
pub struct SdfCache {
    /// f16 signed distance at each voxel center. 1024 bytes.
    pub distances: [u16; 512],  // f16 stored as u16 bits
}
```

Implement:
- `SdfCache::empty()` — all distances = +MAX
- `SdfCache::interior()` — all distances = -MAX
- `get_distance(x, y, z) -> f32`, `set_distance(x, y, z, d: f32)`
- `sample_trilinear(local_pos: Vec3) -> f32` — same math as current `sampling.rs`

Tests: distance get/set roundtrip, trilinear interpolation accuracy.

### 1.3 — GeometryPool and SdfCachePool

**File:** `crates/rkf-core/src/brick_pool.rs` (MODIFY)

Add type aliases:
```rust
pub type GeometryPool = Pool<BrickGeometry>;
pub type SdfCachePool = Pool<SdfCache>;
```

`Pool<T>` is already generic, so no structural changes needed. `BrickGeometry` needs `Clone` + `Default`. `SdfCache` needs `Pod`/`Zeroable` for GPU upload.

Tests: allocation/deallocation of new pool types.

### 1.4 — Surface voxel neighbor queries across bricks

**File:** `crates/rkf-core/src/brick_geometry.rs` (extend)

For surface identification at brick boundaries:
```rust
/// Context for cross-brick neighbor queries.
pub struct NeighborContext<'a> {
    pub center: &'a BrickGeometry,
    /// Neighbor bricks in ±x, ±y, ±z. None = EMPTY_SLOT (all empty).
    /// Some(None) = INTERIOR_SLOT (all solid).
    pub neighbors: [Option<Option<&'a BrickGeometry>>; 6],
}
```

- `is_surface_voxel_with_context(x, y, z, ctx)` — checks cross-brick boundaries
- `rebuild_surface_list_with_context(ctx)` — accurate surface list including boundary voxels

Tests: cross-brick surface detection, EMPTY_SLOT neighbor = empty, INTERIOR_SLOT neighbor = solid.

### 1.5 — Update lib.rs exports

**File:** `crates/rkf-core/src/lib.rs` (MODIFY)

Export new types: `BrickGeometry`, `SurfaceVoxel`, `SdfCache`, `GeometryPool`, `SdfCachePool`.

---

## Phase 2: SDF Computation from Geometry

The core algorithm: occupancy in, signed distances out.

### 2.1 — CPU Dijkstra distance computation

**File:** `crates/rkf-core/src/sdf_compute.rs` (NEW)

```rust
/// Compute SDF distances for all allocated bricks from their geometry.
pub fn compute_sdf_from_geometry(
    brick_map: &BrickMap,
    geometry_pool: &GeometryPool,
    sdf_pool: &mut SdfCachePool,
    // maps brick_map slot -> geometry_pool slot and sdf_pool slot
    slot_mapping: &[(u32, u32)],
) { ... }

/// Compute SDF distances for a subset of bricks (local edit region).
pub fn compute_sdf_region(
    brick_map: &BrickMap,
    geometry_pool: &GeometryPool,
    sdf_pool: &mut SdfCachePool,
    slot_mapping: &[(u32, u32)],
    region_min: UVec3,  // brick coordinates
    region_max: UVec3,
) { ... }
```

Algorithm (Dijkstra from surface voxels):
1. Identify all surface voxels across all bricks in the region (using `NeighborContext`)
2. Seed priority queue with surface voxels, initial distance = sub-voxel estimate to zero-crossing
3. Expand to 26-connected neighbors within allocated bricks
4. Distance = Euclidean distance between voxel centers
5. Sign: solid voxels negative, empty voxels positive
6. Write results to `SdfCache` bricks

Tests:
- Single brick with a half-solid plane: distances should increase linearly from surface
- Sphere occupancy: distances should approximate spherical SDF
- Cross-brick propagation: surface in one brick, distances propagate into neighbor brick
- Region-limited computation: only bricks in region are updated
- EMPTY_SLOT / INTERIOR_SLOT boundary handling

### 2.2 — GPU JFA distance computation

**File:** `crates/rkf-core/src/sdf_compute_gpu.rs` (NEW)
**Shader:** `crates/rkf-render/shaders/sdf_compute.wgsl` (NEW)

GPU-accelerated JFA for bulk SDF computation (full object, load-time, LOD generation):
1. Upload occupancy data to a 3D texture (1 bit per voxel, packed)
2. Initialize seed texture: surface voxels seeded with their position, others at infinity
3. Run log2(max_dim) JFA passes
4. Write f16 distances to SDF cache buffer
5. Readback to CPU `SdfCachePool` if needed

This is for large objects where CPU Dijkstra would be too slow. Not needed for per-stroke sculpting (that uses the CPU path on a small region).

Tests: GPU vs CPU distance computation should produce equivalent results (within f16 precision).

### 2.3 — Validation: geometry-to-SDF roundtrip

**File:** `crates/rkf-core/src/sdf_compute.rs` (extend tests)

Take existing VoxelSample data (current format), derive occupancy from sign, compute SDF from occupancy, compare distances. This validates the algorithm produces equivalent results to the current hand-maintained SDF.

---

## Phase 3: Voxelization (Geometry-First)

Rewrite `voxelize_sdf` and related functions to produce geometry first, then derive SDF.

### 3.1 — voxelize_to_geometry

**File:** `crates/rkf-core/src/voxelize_object.rs` (REWRITE)

```rust
/// Voxelize an SDF function into BrickGeometry + SdfCache.
pub fn voxelize_to_geometry<F>(
    sdf_fn: F,
    aabb: &Aabb,
    voxel_size: f32,
    geo_pool: &mut GeometryPool,
    sdf_pool: &mut SdfCachePool,
    map_alloc: &mut BrickMapAllocator,
) -> Option<VoxelizeResult>
where
    F: Fn(Vec3) -> (f32, u16, [u8; 4]),  // (distance, material_id, color)
```

Algorithm:
1. Compute brick grid dims from AABB
2. For each brick position: evaluate SDF at all 512 voxel centers
3. Set occupancy bit for voxels with negative distance
4. Identify surface voxels, store color + material_id
5. Classify fully-empty bricks as EMPTY_SLOT, fully-solid as INTERIOR_SLOT
6. Run `compute_sdf_from_geometry` to populate SDF cache
7. Return handles for both geometry and SDF pools

Tests: analytical sphere → geometry + SDF, verify occupancy is correct, verify computed distances are smooth and monotonic.

### 3.2 — Resampling via geometry

**File:** `crates/rkf-core/src/resample.rs` (NEW)

```rust
/// Resample geometry from old grid to new grid at a different voxel_size.
pub fn resample_geometry(
    old_map: &BrickMap,
    old_pool: &GeometryPool,
    old_voxel_size: f32,
    new_voxel_size: f32,
    new_geo_pool: &mut GeometryPool,
    new_sdf_pool: &mut SdfCachePool,
    new_map_alloc: &mut BrickMapAllocator,
) -> Option<ResampleResult> { ... }
```

Algorithm:
1. Compute new grid dims from old AABB and new voxel_size
2. For each new brick: check overlap with old allocated bricks
3. For each new voxel: map position to old grid, sample old occupancy
   - For downsampling: box filter — new voxel is solid if >50% of covered old voxels are solid
   - For upsampling: nearest-neighbor or trilinear of occupancy distance field
4. Transfer surface voxel colors from nearest old surface voxel
5. Classify fully-empty / fully-solid new bricks
6. Compute SDF for new grid

Tests:
- Resample sphere at 2x resolution: surface preserved, more detail
- Resample sphere at 0.5x resolution: surface preserved, coarser
- Material and color transfer accuracy
- Brick count changes appropriately with resolution

---

## Phase 4: Sculpt Pipeline (Rewrite)

Replace the entire sculpt pipeline with geometry-first operations.

### 4.1 — Geometry edit operations

**File:** `crates/rkf-edit/src/geometry_edit.rs` (NEW)

```rust
/// Apply a brush to geometry: modify occupancy + surface properties.
pub fn apply_geometry_edit(
    geo_pool: &mut GeometryPool,
    brick_map: &mut BrickMap,
    map_alloc: &mut BrickMapAllocator,
    op: &EditOp,
    voxel_size: f32,
    aabb: &Aabb,
) -> EditResult { ... }
```

For each voxel in the brush region:
- **Add**: set occupancy bit if brush SDF is negative at voxel position
- **Subtract**: clear occupancy bit if brush SDF is negative at voxel position
- **Paint**: modify color/material on existing surface voxels (no geometry change)
- **Smooth**: blur occupancy boundary (erode then dilate, or Gaussian on occupancy distance)
- **Flatten**: set occupancy to match a reference plane

After edit:
- Allocate new bricks if brush extends into EMPTY_SLOT / INTERIOR_SLOT regions
- Deallocate bricks that became fully empty or fully interior
- Rebuild surface voxel lists for affected bricks
- Return affected brick region for SDF recomputation

Tests: add sphere to empty grid, subtract sphere from solid grid, paint surface, smooth edge, flatten surface.

### 4.2 — Editor sculpt integration

**File:** `crates/rkf-editor/src/engine/sculpt.rs` (REWRITE)

Replace `apply_sculpt_displacement` and `apply_sculpt_edits` with:

1. Convert brush settings to `EditOp` (same as now)
2. Call `apply_geometry_edit` on the geometry pool
3. Call `compute_sdf_region` on the affected region
4. Upload changed bricks (geometry + SDF) to GPU
5. Undo: snapshot geometry bricks before edit, restore on undo

Remove:
- `apply_sculpt_displacement` (surface displacement on SDF — replaced by occupancy edit)
- `ensure_object_voxelized` (replaced by `convert_object_to_voxel` which produces geometry directly)
- `sample_dominant_material` (replaced by reading surface voxel color/material directly)

### 4.3 — Undo/redo for geometry

**File:** `crates/rkf-editor/src/editor_state/mod.rs` (MODIFY)

Sculpt undo snapshots change from `Vec<(u32, Brick)>` (slot, old VoxelSample data) to `Vec<(u32, BrickGeometry)>` (slot, old geometry). SDF cache is not snapshotted — it's recomputed after undo.

### 4.4 — Remove old sculpt code

Delete or gut:
- `crates/rkf-editor/src/engine/sdf_repair.rs` — replaced by `sdf_compute.rs`
- `crates/rkf-editor/src/engine/sdf_fmm.rs` — replaced by `sdf_compute.rs`
- `crates/rkf-editor/src/engine/brick_ops_repair.rs` — `prefill_new_brick_faces`, `ensure_sdf_consistency`, `mark_interior` all replaced by geometry-first operations
- `crates/rkf-edit/src/cpu_apply.rs` — replaced by `geometry_edit.rs`

Keep `brick_ops.rs` for `grow_brick_map` and `allocate_bricks` (still needed, but simplified).

---

## Phase 5: GPU Rendering Pipeline

Update shaders to read from geometry + SDF cache.

### 5.1 — GPU data layout

**File:** `crates/rkf-render/src/gpu_scene.rs` (MODIFY)

New GPU buffers:
- **Geometry buffer**: occupancy bitmasks (64 bytes per brick, packed)
- **Surface voxel buffer**: `SurfaceVoxel` data (variable per brick, packed contiguously)
- **SDF cache buffer**: f16 distances (1024 bytes per brick) — replaces current brick pool buffer for distance queries
- **Surface voxel index buffer**: per-brick offset + count into surface voxel buffer

Upload pipeline: geometry pool + SDF cache pool → GPU buffers each frame (or on change).

### 5.2 — Ray march shader update

**File:** `crates/rkf-render/shaders/ray_march.wgsl` (REWRITE)

`sample_voxelized()` changes:
- Read SDF distance from the SDF cache buffer (not the old brick pool)
- Occupancy check: read occupancy bitmask for hit detection confirmation
- Material + color: read from surface voxel buffer at hit point (nearest surface voxel or interpolated)

Normal computation: gradient of SDF cache (same finite-difference kernel, different data source).

The ray marching algorithm itself (sphere tracing) is unchanged — it still uses f16 distances for step sizing.

### 5.3 — Shading shader update

**Files:** `crates/rkf-render/shaders/shade.wgsl`, `shade_common.wgsl` (MODIFY)

- Material lookup: read `material_id` from surface voxel (u8, not u16)
- Albedo: read RGBA color from surface voxel directly (no material table lookup for color, no companion color pool)
- PBR properties: material table lookup for roughness, metallic, etc. (256-entry table, u8 index)
- Remove companion color pool bind group

### 5.4 — Remove companion color pool

**Files:** `crates/rkf-core/src/color_pool.rs` (DELETE or gut)
**Files:** All shader references to color pool bind group (REMOVE)

Per-voxel color now stored directly in `SurfaceVoxel.color`. No indirection needed.

### 5.5 — GpuObject struct update

**File:** `crates/rkf-render/src/gpu_object.rs` (MODIFY)

Add fields for geometry buffer offset, surface voxel buffer offset + count, SDF cache buffer offset. Remove old brick pool references.

---

## Phase 6: Asset Format & Import Pipeline

### 6.1 — .rkf v3 file format

**File:** `crates/rkf-core/src/asset_file.rs` (REWRITE)

```
[RkfFileHeader v3]
    magic: "RKF3"
    version: 3
    lod_count, material_count, aabb
    flags: has_sdf_cache (bit 0)

[Per LOD level]
    voxel_size, brick_dims
    brick_map: [u32] — EMPTY / INTERIOR / allocated_index
    geometry_data (LZ4 compressed):
        per allocated brick:
            occupancy: [u64; 8]
            surface_count: u16
            surface_voxels: [SurfaceVoxel; surface_count]
    sdf_cache (LZ4 compressed, OPTIONAL):
        per allocated brick:
            distances: [u16; 512]   // f16 bits
```

Implement:
- `save_object_v3()` — write geometry + optional SDF cache
- `load_object_v3()` — read geometry, optionally read SDF cache or compute on load
- `load_object_header_v3()` — read header + LOD table without brick data

### 6.2 — Import pipeline update

**File:** `crates/rkf-import/src/voxelize.rs` (REWRITE)

`voxelize_mesh` now produces `BrickGeometry` directly:
1. BVH nearest-triangle + winding number → occupancy (negative winding = solid)
2. Per-triangle material + vertex color → surface voxel color + material_id
3. Post-process: compute SDF from geometry

### 6.3 — LOD generation from geometry

**File:** `crates/rkf-import/src/lod.rs` (REWRITE)

`generate_lods` now works on geometry:
- For mesh imports: re-voxelize from mesh at each LOD size (same as before, but producing geometry)
- For voxel-only objects: use `resample_geometry` from Phase 3.3

### 6.4 — Scene file format update

**File:** `crates/rkf-core/src/scene_node.rs` (MODIFY)

`SdfSource::Voxelized` changes:
```rust
SdfSource::Voxelized {
    geometry_handle: BrickMapHandle,  // indexes geometry pool
    sdf_cache_handle: BrickMapHandle, // indexes SDF cache pool
    voxel_size: f32,
    aabb: Aabb,
}
```

Update scene serialization (`.rkscene` RON format) accordingly.

---

## Phase 7: Physics, Streaming, LOD, Editor

### 7.1 — Physics collision adapter

**File:** `crates/rkf-physics/src/sdf_collision.rs` (MODIFY)

Collision queries read from the CPU-side SDF cache (same interface — sample distance at a point). The SDF cache is kept up to date after geometry edits.

Add `query_occupancy(pos) -> bool` for fast inside/outside tests.

### 7.2 — Streaming system

**File:** `crates/rkf-runtime/src/object_streaming.rs` (MODIFY)

Load path:
1. Load geometry from .rkf v3
2. Load SDF cache if present, otherwise compute from geometry
3. Upload both to GPU

LOD transitions:
- Each LOD level has its own geometry + SDF cache handles
- Swapping LOD swaps both handles

### 7.3 — LOD manager

**File:** `crates/rkf-core/src/lod_manager.rs` (MODIFY)

Update `ObjectLodState` to track geometry + SDF handles per level. LOD selection algorithm unchanged (screen-space driven).

### 7.4 — Editor resolution control

**File:** `crates/rkf-editor/src/engine/brick_ops_repair.rs` (REWRITE → rename to `brick_ops_resample.rs`)

`process_resample` uses `resample_geometry` from Phase 3.3. The resolution slider and "Apply Resolution" button from the earlier work connect to this.

`process_revoxelize` (bake non-uniform scale) also uses `resample_geometry`.

### 7.5 — Convert to Voxel Object

**File:** `crates/rkf-editor/src/engine/sculpt.rs` (already exists, update)

`convert_object_to_voxel` uses `voxelize_to_geometry` to produce geometry + SDF from analytical primitive.

### 7.6 — MCP / automation updates

**File:** `crates/rkf-editor/src/automation/` (MODIFY)

- `spatial_query`: sample SDF cache (same external behavior)
- `sculpt_apply`: route through new geometry edit pipeline
- `entity_inspect`: report geometry info (brick count, surface voxel count, voxel_size)

---

## Phase 8: Cleanup & Documentation

### 8.1 — Delete old code

Remove entirely:
- `crates/rkf-core/src/voxel.rs` — `VoxelSample` type (replaced by `BrickGeometry` + `SdfCache`)
- `crates/rkf-core/src/brick.rs` — `Brick` type (was 512 × VoxelSample)
- `crates/rkf-core/src/sampling.rs` — old trilinear sampling (replaced by `SdfCache::sample_trilinear`)
- `crates/rkf-core/src/color_pool.rs` — companion color pool (replaced by per-voxel color)
- `crates/rkf-editor/src/engine/sdf_repair.rs` — old SDF repair
- `crates/rkf-editor/src/engine/sdf_fmm.rs` — old FMM repair
- `crates/rkf-editor/src/engine/eikonal.rs` — old eikonal solver (if unused)
- `crates/rkf-edit/src/cpu_apply.rs` — old CPU CSG on VoxelSample
- `crates/rkf-edit/shaders/csg_edit.wgsl` — old GPU CSG on VoxelSample
- Old `voxelize_sdf` function (replaced by `voxelize_to_geometry`)

### 8.2 — Update all documentation

Rewrite or update:
- `docs/v2/ARCHITECTURE.md` — update data model, render pipeline, voxel types, file format sections
- `docs/v2/GEOMETRY_FIRST_ARCHITECTURE.md` — finalize with implementation details
- `docs/architecture/01-core-data-structure.md` — VoxelSample → BrickGeometry + SdfCache
- `docs/architecture/02-material-system.md` — per-voxel RGBA color, 256-entry material table
- `docs/architecture/07-procedural-editing.md` — geometry-first sculpting
- `CLAUDE.md` — update Key Data Types, Coding Conventions, render pipeline, file formats

### 8.3 — Update CLAUDE.md

Key sections to change:
- **Key Data Types**: replace VoxelSample with BrickGeometry + SurfaceVoxel + SdfCache
- **Render Pipeline**: note SDF cache is derived from geometry
- **File Formats**: .rkf v3
- **Novel/Unusual Patterns**: geometry-first editing, SDF as cache
- **Coding Conventions**: "geometry is source of truth, SDF is derived"
- **Material System**: per-voxel RGBA, 256-entry PBR table

### 8.4 — Final test audit

- Run full workspace test suite: all tests must pass
- Remove or rewrite any tests that reference deleted types
- Verify no references to `VoxelSample`, `Brick`, `sampling::sample_brick_trilinear`, companion color pool remain in non-test code
- Visual validation via MCP: render comparison before/after migration

---

## Phase Dependencies

```
Phase 1 (Core Types)
  ↓
Phase 2 (SDF Computation)
  ↓
Phase 3 (Voxelization + Resampling)
  ↓
Phase 4 (Sculpt Pipeline)    Phase 5 (GPU Rendering)
  ↓                            ↓
Phase 6 (Asset Format + Import)
  ↓
Phase 7 (Physics, Streaming, Editor)
  ↓
Phase 8 (Cleanup + Docs)
```

Phases 4 and 5 can be worked in parallel once Phase 3 is complete.

## Files Summary

### New files
| File | Phase | Purpose |
|------|-------|---------|
| `rkf-core/src/brick_geometry.rs` | 1.1 | BrickGeometry, SurfaceVoxel types |
| `rkf-core/src/sdf_cache.rs` | 1.2 | SdfCache type |
| `rkf-core/src/sdf_compute.rs` | 2.1 | CPU Dijkstra SDF from geometry |
| `rkf-core/src/sdf_compute_gpu.rs` | 2.2 | GPU JFA SDF from geometry |
| `rkf-core/src/resample.rs` | 3.3 | Geometry resampling for resolution change / LOD |
| `rkf-edit/src/geometry_edit.rs` | 4.1 | Geometry-first edit operations |
| `rkf-render/shaders/sdf_compute.wgsl` | 2.2 | GPU JFA shader |

### Deleted files
| File | Phase | Replaced by |
|------|-------|-------------|
| `rkf-core/src/voxel.rs` | 8.1 | `brick_geometry.rs` + `sdf_cache.rs` |
| `rkf-core/src/brick.rs` | 8.1 | `brick_geometry.rs` + `sdf_cache.rs` |
| `rkf-core/src/sampling.rs` | 8.1 | `SdfCache::sample_trilinear` |
| `rkf-core/src/color_pool.rs` | 8.1 | `SurfaceVoxel.color` |
| `rkf-editor/src/engine/sdf_repair.rs` | 8.1 | `sdf_compute.rs` |
| `rkf-editor/src/engine/sdf_fmm.rs` | 8.1 | `sdf_compute.rs` |
| `rkf-editor/src/engine/eikonal.rs` | 8.1 | `sdf_compute.rs` |
| `rkf-edit/src/cpu_apply.rs` | 8.1 | `geometry_edit.rs` |
| `rkf-edit/shaders/csg_edit.wgsl` | 8.1 | geometry-first GPU edit |

### Major rewrites
| File | Phase | What changes |
|------|-------|-------------|
| `rkf-core/src/voxelize_object.rs` | 3.1 | Produce geometry, not VoxelSample |
| `rkf-core/src/asset_file.rs` | 6.1 | .rkf v3 format |
| `rkf-editor/src/engine/sculpt.rs` | 4.2 | Geometry-first sculpting |
| `rkf-editor/src/engine/brick_ops_repair.rs` | 7.4 | Rename, resample only |
| `rkf-import/src/voxelize.rs` | 6.3 | Mesh → geometry |
| `rkf-import/src/lod.rs` | 6.4 | LOD from geometry |
| `rkf-render/shaders/ray_march.wgsl` | 5.2 | Read SDF cache + surface voxels |
| `rkf-render/shaders/shade.wgsl` | 5.3 | Per-voxel color, u8 material |
| `rkf-render/shaders/sdf_common.wgsl` | 5.2 | New sample functions |
| `rkf-render/src/gpu_scene.rs` | 5.1 | New buffer layout |
