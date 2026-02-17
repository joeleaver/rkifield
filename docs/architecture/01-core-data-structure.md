# Core Data Structure: Multi-Level Sparse Grid with Clipmap LOD

> **Status: DECIDED**

### Decision: Multi-Level Sparse Grid (NanoVDB-style)

**Chosen over:** SVO (pointer-chasing on GPU too slow), DAG (per-voxel material data destroys sharing), flat hash map (no hierarchy for skipping).

The spatial index is a multi-level sparse grid — index-based (no pointers), with hierarchical empty-space skipping and GPU-friendly flat memory layout. Dense bricks at the leaves store the actual voxel samples.

```
Per-LOD-Level Sparse Grid:
  Level 2 (root):    Coarse occupancy bitfield (~small, fits in cache)
                     Marks which coarse cells contain any data
  Level 1 (blocks):  Index array mapping occupied cells → brick pool slots
                     Each entry is a u32 offset into the brick pool
  Level 0 (voxels):  Implicit — within each brick (dense 3D grid)

Shared Brick Pool (GPU, fixed-size budget):
  ├── Contiguous array of Bricks
  ├── Free list for allocation/eviction
  └── LRU tracking for streaming
```

**Key properties:**
- No pointer chasing — all levels are flat arrays indexed by 3D coordinates
- Hierarchical skipping — Level 2 rejects large empty regions before Level 1 is consulted
- Cache-friendly — bricks are contiguous in memory, good for GPU wavefront coherence
- LOD-agnostic — the same grid structure works at every LOD level, only the world-space scale changes

### Decision: Clipmap LOD with Streaming

**Chosen over:** hierarchy-as-LOD (too few levels), per-object LOD (doesn't work for terrain/continuous worlds).

The world is rendered through concentric clipmap levels centered on the camera. Each level is a separate sparse grid at a different voxel resolution. All levels share the same GPU brick pool.

```
Clipmap Levels (example configuration):
  LOD 0:  2cm voxels,  128m radius  — near detail
  LOD 1:  4cm voxels,  256m radius
  LOD 2:  8cm voxels,  512m radius
  LOD 3: 16cm voxels, 1024m radius
  LOD 4: 32cm voxels, 2048m radius  — horizon
```

**Bricks are always the same sample count regardless of LOD.** At LOD 0, a brick covers a small region at high precision. At LOD 4, the same-sized brick covers a large region at low precision. This makes the ray marcher, allocator, and memory management code LOD-agnostic.

**Ray march LOD integration:** Rays start in LOD 0 near the camera and transition to coarser levels as they travel. Coarser voxels yield larger safe step sizes, so distant geometry is cheaper to march — a natural performance gradient.

```
Ray from camera:
  [LOD 0: small steps, full detail]──►[LOD 1: bigger steps]──►[LOD 2]──►...
```

### World Streaming

The world is divided into fixed-size **chunks** (e.g., 64m x 64m x 64m). Each chunk is stored on disk with pre-computed data at multiple LOD resolutions.

```
Streaming System:
  Each frame:
    1. Camera position → desired LOD per chunk (distance-based)
    2. Diff against currently loaded state
    3. Queue async loads for chunks that need higher LOD
    4. Queue evictions for chunks that moved to lower LOD
    5. Async I/O: disk → staging buffer → GPU brick pool
    6. Update spatial index to reference newly loaded bricks

  GPU Brick Pool (fixed budget, e.g., ~512MB):
    ├── Active bricks (currently rendering)
    ├── Warm bricks (recently used, kept for temporal coherence)
    └── Free slots (available for streaming)
    └── LRU eviction when pool is full
```

**On-disk format:** Chunks stored with brick data at each LOD level. Block compression (LZ4/zstd) for I/O efficiency. DAG compression considered as a future optimization for static environments.

### LOD Transitions

At boundaries between clipmap levels, a blend zone prevents visible popping:
- Overlap LOD levels by a small margin
- Trilinear interpolation between coarse and fine SDF evaluations in the transition band
- Initial implementation may skip blending (low internal resolution + upscaling may hide discontinuities)

### Decision: 8^3 Brick Resolution

**Chosen over:** 4^3 (too much index overhead, underutilizes cache), 16^3 (too coarse, wastes memory at thin surfaces).

Every brick in the pool is 8×8×8 = 512 voxel samples. At the standard resolution tier (2cm voxels), one brick covers 16cm × 16cm × 16cm. The brick size is constant — only the voxel spacing changes with resolution tier.

### Decision: Variable Resolution via Per-Resolution Grid Layers

**Chosen over:** Octree subdivision within grid (reintroduces pointer-chasing).

Multiple resolution tiers coexist, each with its own sparse grid. The ray marcher checks the finest available layer first at each step.

| Tier | Voxel Size | Brick Extent | Use Case |
|------|-----------|--------------|----------|
| 0 | 0.5cm | 4cm | Fine detail (faces, small props, inscriptions) |
| 1 | 2cm | 16cm | Standard geometry (default) |
| 2 | 8cm | 64cm | Large structures, terrain |
| 3 | 32cm | 256cm | Distant terrain, horizon |

- **Finest resolution: 0.5cm** — sufficient for recognizable stylized faces and small props
- Clipmap LOD sets the ceiling — distant objects never render at Tier 0 even if the data exists
- Content authors choose the appropriate tier per object/region
- Tiers 2-3 double as clipmap LOD levels for distance-based fallback

### Decision: Voxel Sample Layout — 8 Bytes Packed

**Chosen over:** 12-byte wide pack (fewer bricks per budget), 4-byte minimal (too constrained on material range and blend precision).

Each voxel sample is 2 × u32 = 8 bytes, tightly packed:

```
Word 0 (u32): [ f16 distance (16 bits) | u16 material_id (16 bits) ]
Word 1 (u32): [ u8 blend_weight (8 bits) | u8 secondary_id (8 bits) | u8 flags (8 bits) | u8 reserved (8 bits) ]
```

**WGSL representation:**
```wgsl
struct VoxelSample {
    word0: u32,  // f16 distance | u16 material_id
    word1: u32,  // u8 blend_weight | u8 secondary_id | u8 flags | u8 reserved
}
```

**Field details:**
- `distance` (f16): Signed distance to nearest surface. 10-bit mantissa gives ~0.001 precision, more than sufficient within a brick's local range.
- `material_id` (u16): Primary material. 65536 possible materials.
- `blend_weight` (u8): 0-255 mapped to 0.0-1.0. Blend factor between primary and secondary material.
- `secondary_id` (u8): Secondary material for blending. 256 materials — sufficient for local blend transitions.
- `flags` (u8): Bit flags:
  - Bit 0: `has_bone_data` — companion brick exists in bone data pool
  - Bit 1: `has_volumetric_data` — companion brick exists in volumetric pool
  - Bits 2-7: Reserved
- `reserved` (u8): Future use.

**Memory math:**
- 8 bytes/voxel × 512 voxels = **4KB per brick**
- 512MB brick pool = ~131,000 bricks
- At Tier 1 (2cm voxels): 131K bricks × 16cm³ per brick = coverage for a large visible scene

### Decision: Separate Companion Pools for Bone and Volumetric Data

**Chosen over:** Bloating the core sample (most voxels don't need bone or volumetric data — would waste memory everywhere).

Bone data and volumetric channels are stored in parallel brick pools. A core brick's `flags` field indicates whether companion bricks exist. Companion bricks share the same brick ID / spatial mapping.

**Bone Data Pool (animated objects only):**
```
Word 0 (u32): [ u8 bone_index_0 | u8 bone_index_1 | u8 bone_index_2 | u8 bone_index_3 ]
Word 1 (u32): [ u8 bone_weight_0 | u8 bone_weight_1 | u8 bone_weight_2 | u8 bone_weight_3 ]
```
- 8 bytes/voxel × 512 = 4KB per bone brick
- Up to 4 bone influences per voxel
- Weights are u8 normalized (0-255), must sum to 255
- Only allocated for bricks belonging to animated/skinned objects

**Volumetric Data Pool (fog/smoke/fire regions only):**
```
Word 0 (u32): [ f16 density | f16 emission_intensity ]
```
- 4 bytes/voxel × 512 = 2KB per volumetric brick
- `density`: extinction coefficient for fog/smoke
- `emission_intensity`: self-illumination scalar (fire, glow). Emission color comes from the material.
- Only allocated for bricks in volumetric media regions

**Total memory budget:**
| Pool | Per Brick | Budget | Capacity |
|------|-----------|--------|----------|
| Core geometry | 4KB | ~512MB | ~131K bricks |
| Bone data | 4KB | ~64MB | ~16K bricks |
| Volumetric | 2KB | ~64MB | ~32K bricks |
| **Total** | | **~640MB** | |

These budgets are configurable. The bone and volumetric pools can be much smaller on scenes that don't use those features.

### Decision: Narrow Band ±3 Bricks with Interior Marking

Bricks are only allocated within ±3 brick-widths of a surface. The spatial index marks each cell with a 2-bit state:

| State | Meaning | Ray Marcher Behavior |
|-------|---------|---------------------|
| `EMPTY` | Outside all geometry, no bricks | Skip through, large steps |
| `SURFACE` | Has bricks in the pool | Evaluate SDF, normal marching |
| `INTERIOR` | Inside solid geometry, no bricks | Skip to far side of cell |
| `VOLUMETRIC` | Has volumetric companion bricks | Accumulate density/emission |

This avoids filling solid interiors with bricks while maintaining correct inside/outside behavior for enclosed spaces (caves, rooms). Easily changeable — the state is just metadata in the existing spatial index.

### Decision: 8m Chunk Size

World is divided into 8m × 8m × 8m chunks for streaming. Each chunk has its own per-tier sparse grid.

- At Tier 1 (2cm voxels): 400 cells/axis — compact index
- At Tier 0 (0.5cm voxels): 1600 cells/axis — still manageable
- 8m is room-scale — good granularity for streaming decisions
- Chunk size is a constant, trivially adjustable later

### Decision: Fixed Brick Pool Allocation at Startup

Brick pools (core, bone, volumetric) are allocated once at engine startup with a configurable budget. No dynamic reallocation — avoids GPU stalls and buffer rebinding.

Default budgets (configurable per application):

| Pool | Default | Bricks |
|------|---------|--------|
| Core geometry | 512MB | ~131K |
| Bone data | 64MB | ~16K |
| Volumetric | 64MB | ~32K |

---

### Session 1 Summary: All Voxel Data Structure Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Spatial index | Multi-level sparse grid (NanoVDB-style) | GPU-friendly, no pointer chasing, hierarchical skipping |
| LOD strategy | Clipmap with distance-based levels | Natural for streaming worlds, smooth perf gradient |
| Brick resolution | 8^3 (512 voxels) | Industry standard, optimal cache utilization |
| Variable resolution | Per-resolution grid layers, 4 tiers (0.5cm–32cm) | Fine detail where needed, coarse elsewhere |
| Finest resolution | 0.5cm | Sufficient for stylized faces and small props |
| Voxel sample | 8 bytes (2 × u32), packed f16 distance + u16 material + u8 blend/secondary/flags | Memory-efficient, clean GPU alignment |
| Bone/volumetric data | Separate companion pools | Pay only for what you use |
| Narrow band | ±3 bricks from surface, INTERIOR marking in index | Memory efficient, geometrically correct |
| Chunk size | 8m³ | Room-scale streaming granularity |
| Pool allocation | Fixed at startup, configurable budget | Simple, predictable, no GPU stalls |
