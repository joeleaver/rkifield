# Geometry-First Voxel Architecture

## Problem Statement

The current architecture treats the SDF distance field as the source of truth for voxel geometry. Every voxel stores a signed distance, and the surface is defined as the zero-crossing of that field. This creates cascading problems:

1. **Sculpting is fragile.** Edits must simultaneously modify geometry AND maintain a consistent distance field. Every sculpt operation needs SDF repair (JFA, FMM, BFS) to fix corrupted distances, and these repairs are themselves buggy.

2. **Resampling fails.** Changing an object's resolution requires resampling the distance field into a new grid. The narrow-band test (designed for clean analytical SDFs) breaks when sampling from discrete brick data — empty bricks return sentinel values that cause entire regions to be skipped.

3. **Two sources of truth.** Per-voxel color lives in a separate companion pool. Material properties come from a global table indexed by `material_id`. The actual color of a voxel is reconstructed from three different places at shade time.

4. **Wasted memory.** Every voxel in an allocated brick stores 8 bytes (distance + material + blend data), even though most voxels in a brick are either fully inside or fully outside the surface and carry no meaningful geometric information.

## Core Principle

**Geometry is the source of truth. SDF distances are derived data, computed from geometry and cached for performance.**

The surface is defined by which voxels are solid and which are empty. The SDF distance field is computed from this binary classification and stored as a cache — it accelerates ray marching and physics queries but is never the canonical representation of shape.

## Data Model

### Voxel Geometry (Source of Truth)

Per brick, geometry is stored as:

```
BrickGeometry:
    occupancy: [u64; 8]          // 512-bit bitmask — 1 = solid, 0 = empty
    surface_voxels: Vec<SurfaceVoxel>  // data for voxels on the solid/empty boundary
```

A **surface voxel** is a solid voxel that has at least one empty neighbor (6-connected), or an empty voxel that has at least one solid neighbor. These are the only voxels that carry material and color data:

```
SurfaceVoxel:                        // 8 bytes, naturally aligned
    index: u16                       // position within the 8×8×8 brick (0–511)
    color: [u8; 4]                   // RGBA — alpha available for custom shader control
    material_id: u8                  // indexes material table for PBR properties
    _reserved: u8                    // future use
```

**Memory per brick:**
- Occupancy bitmask: 64 bytes (fixed)
- Surface voxels: 8 bytes × count (typically 30–150 per surface brick)
- Total: ~300–1,260 bytes vs 4,096 bytes (current)

### Material Table (Shared)

Non-color PBR properties that don't vary per-voxel:

```
Material:
    roughness: f32
    metallic: f32
    emission_strength: f32
    emission_color: [f32; 3]
    subsurface: f32
    subsurface_color: [f32; 3]
    ior: f32
    // ... other PBR properties
```

256 materials (u8 index). Per-voxel color handles the variation that currently requires thousands of material slots or companion color pools.

### SDF Cache (Derived)

The full 8×8×8 distance field, computed from geometry:

```
BrickSdf:
    distances: [f16; 512]        // signed distance at each voxel center
```

- **1,024 bytes per brick** (f16 × 512)
- Computed from `BrickGeometry` by the SDF generation algorithm
- Stored on disk alongside geometry for fast loads
- Recomputed locally when geometry changes (edits, resampling)
- Can always be regenerated from geometry if missing or corrupt
- GPU ray marcher reads this for sphere tracing step sizes

### Brick-Level Sparsity

The brick map classifies each brick position:

- **EMPTY_SLOT** — all voxels in this region are empty (exterior). No data stored.
- **INTERIOR_SLOT** — all voxels in this region are solid (deep interior). No data stored.
- **Allocated** — surface passes through this brick. Stores `BrickGeometry` + `BrickSdf`.

### Total Per-Brick Memory

| Component | Size | Notes |
|-----------|------|-------|
| Occupancy bitmask | 64 B | Fixed |
| Surface voxels | ~240–1,200 B | Variable, typically 30–150 voxels × 8 B |
| SDF cache | 1,024 B | Fixed, f16 × 512 |
| **Total** | **~1.3–2.3 KB** | **vs 4 KB current** |

For objects where GPU memory is the bottleneck, the SDF cache could be computed on-GPU and never stored on CPU, reducing CPU memory to just the geometry (~200–700 B per brick).

## SDF Computation Algorithm

### Input
- `BrickGeometry` for all allocated bricks in an object
- Brick map (which positions are allocated, empty, or interior)

### Surface Identification
A voxel is a **surface voxel** if it is solid and has at least one empty neighbor, OR it is empty and has at least one solid neighbor. At brick boundaries:
- Neighbor in an EMPTY_SLOT brick → that neighbor is empty
- Neighbor in an INTERIOR_SLOT brick → that neighbor is solid
- Neighbor in an allocated brick → read its occupancy bitmask

### Distance Computation

**GPU path (primary, for bulk computation):**

Jump Flood Algorithm (JFA) on the object's voxel grid:
1. Initialize: surface voxels seeded with sub-voxel distance to zero-crossing, all others at infinity
2. Run log₂(max_dim) JFA passes — each pass propagates nearest-surface information
3. Write distances to `BrickSdf` for allocated bricks only
4. Sign from occupancy: solid voxels get negative distance, empty get positive

For a 80×80×80 object (10×10×10 bricks): 7 passes × 512K voxels. Sub-millisecond on modern GPUs.

**CPU path (for physics, spatial queries, small edits):**

Dijkstra from surface voxels, constrained to allocated bricks:
1. Seed all surface voxels with distance 0 into a priority queue
2. Expand to 26-connected neighbors within allocated bricks
3. Distance = Euclidean distance between voxel centers
4. Sign from occupancy

For local edits (sculpting), only recompute in the affected region + 1-brick margin.

### Cross-Brick Boundaries

SDF distances must be continuous across brick boundaries. The algorithm handles this by:
- Operating on the full object grid, not per-brick
- Surface identification checks neighbors across brick boundaries (via the brick map)
- JFA/Dijkstra naturally propagates across boundaries

## Operations

### Sculpting

1. Determine affected voxels from brush position + radius
2. Modify occupancy bitmasks (set solid for add, clear for subtract)
3. Update surface voxel lists (add/remove based on new neighbor relationships)
4. Allocate new bricks if the surface grows into previously empty/interior regions
5. Deallocate bricks that become fully empty or fully interior
6. Recompute SDF in the affected region (brush extent + 1 brick margin)
7. Upload changed bricks to GPU

Steps 1–5 are geometry operations — simple, deterministic, no SDF repair needed.
Step 6 is a well-defined computation on clean input data.

### Resampling (Resolution Change / LOD Generation)

1. For each brick in the new grid, check if it overlaps any allocated brick in the old grid
2. If yes, allocate the new brick
3. For each voxel in the new brick, sample the old geometry:
   - Map new voxel position to old grid coordinates
   - Read occupancy from old data (trilinear interpolation of the occupancy field, threshold at 0.5 for the new occupancy bit)
   - For surface voxels: interpolate color from old surface voxels, pick nearest material
4. Identify surface voxels in the new grid
5. Compute SDF for the new grid
6. Deallocate new bricks that ended up fully empty or fully interior

For **downsampling** (LOD generation): the occupancy interpolation naturally acts as a low-pass filter — a new voxel is solid only if >50% of the old voxels it covers were solid. Small features that fall below the new resolution are smoothly removed.

For **upsampling**: the interpolation produces a smoother version of the surface. The SDF recomputation then gives proper distances at the finer resolution.

### Import (Mesh → Voxel)

1. Evaluate mesh SDF (BVH + winding number) at each voxel center → occupancy (negative = solid)
2. Transfer per-triangle material and vertex colors to surface voxels
3. Compute SDF from occupancy
4. Generate LOD levels by downsampling geometry

### Physics / Collision

CPU-side SDF cache provides distance queries for:
- Rapier collision adapter (sample SDF at contact points)
- Character controller (capsule-vs-SDF)
- Particle collision

If the SDF cache is stale (geometry was edited), recompute before the physics step.

## File Format

The `.rkf` v3 format is the **current** asset format. It stores geometry-first data (occupancy + surface voxels) as the source of truth, with an optional SDF cache for fast loading.

> The v2 format (`asset_file.rs`, magic `"RKF2"`) stores raw VoxelSample data directly. It lacks secondary material and blend weight support. The loader still reads v2 files for backwards compatibility, but all new saves use v3.

```
.rkf v3 File Layout (CURRENT — see asset_file_v3.rs):

[RkfV3Header]                    128 bytes, fixed
    magic: [u8; 4] = "RKF3"
    version: u32 = 3
    lod_count: u32
    material_count: u32
    aabb_min: [f32; 3]
    aabb_max: [f32; 3]
    flags: u32                    bit 0 = has_sdf_cache
    analytical_type: u32
    analytical_params: [f32; 4]
    material_ids: [u8; 32]        up to 32 material IDs used by this object
    _reserved: [u8; 32]

[LodV3Entry × lod_count]         56 bytes each
    voxel_size: f32
    brick_count: u32
    brick_dims: [u32; 3]
    _pad: u32
    geometry_offset: u64
    geometry_compressed_size: u32
    geometry_uncompressed_size: u32
    sdf_offset: u64
    sdf_compressed_size: u32
    sdf_uncompressed_size: u32

[LOD data 0..N-1] (each LZ4 compressed separately):
    brick_map: [u32 × (dims.x × dims.y × dims.z)]
        values: EMPTY_SLOT (u32::MAX), INTERIOR_SLOT (u32::MAX-1), or local index (0..brick_count-1)

    geometry_data (per allocated brick):
        occupancy: [u64; 8]       64 bytes — 512-bit bitmask (bit N = voxel N is solid)
        surface_count: u16        number of surface voxels in this brick
        surface_voxels: [SurfaceVoxel; surface_count]  8 bytes each:
            index: u16            voxel position within brick (0–511)
            color: [u8; 4]        RGBA per-voxel color
            material_id: u8       primary material (index into material table)
            secondary_material_id: u8  secondary material for blending (0 = no blend)

    sdf_cache (optional, present when flags bit 0 set):
        per allocated brick:
            distances: [u16; 512]  1024 bytes — f16 signed distances
```

The SDF cache section is optional. If missing (flag cleared), the loader computes it from geometry on load. This allows stripping the cache for distribution (smaller files) while keeping it for fast iteration during development.

Blend weight (0–255, controlling primary vs secondary material interpolation) is stored in the `color[3]` byte (alpha channel) of SurfaceVoxel when material blending is active. On the GPU side, `Brick::from_geometry` converts this to `VoxelSample::from_geometry_data_blended`, encoding both materials and blend weight into the packed GPU format.

## Design Decisions

### Why per-voxel color instead of material table albedo?

Color varies per-voxel in practice (painted surfaces, imported textures, sculpted color). The current system handles this through a companion color pool with indirection — a separate brick-sized buffer indexed by a companion map. Storing RGBA directly on surface voxels:
- Eliminates the companion pool and its mapping overhead
- Makes painting a direct operation (set the color on the voxel)
- Reduces material table entries needed (no more "red stone" vs "blue stone" variants)

Non-color PBR properties (roughness, metallic, SSS) rarely vary per-voxel and are well-served by a 256-entry material table.

### Why u8 material_id (256) instead of u16 (65536)?

Per-voxel color absorbs the variation that previously required thousands of material slots. An object with "wood, metal, rust, paint" needs 4 materials, not 4000 color variants. 256 is generous for the remaining PBR-property-only materials.

### Why store SDF cache on disk?

Large objects (millions of voxels) take non-trivial time to compute SDF. A 500×500×500 voxel object (~4M surface voxels) would take ~50ms on GPU, acceptable for editing but noticeable when loading a complex scene with 50+ objects. Pre-computed cache eliminates this load-time cost.

The cache is optional (can be stripped), and can always be recomputed from geometry, so it adds no correctness risk.

### Why not compute SDF purely on GPU?

Physics and spatial queries need CPU-side distances. Computing on GPU and reading back adds latency and complexity. Maintaining a CPU-side SDF cache (updated on edit) keeps physics and MCP tools responsive without GPU round-trips.

For large bulk operations (full-object SDF recomputation on load), the GPU path is preferred. For local edits (sculpting), the CPU Dijkstra path on the affected region is fast enough and avoids GPU sync.
