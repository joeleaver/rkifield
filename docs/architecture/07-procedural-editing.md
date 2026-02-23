> **SUPERSEDED** by [v2 Architecture](../v2/ARCHITECTURE.md) — this document describes the v1 chunk-based engine.

# Procedural Editing

> **Status: DECIDED**

### Decision: CSG Operations via GPU Compute on Live Voxel Grid

All editing runs as GPU compute shaders operating directly on the brick pool. No CPU processing of voxel data.

**CSG operations:**
```wgsl
fn csg_union(existing: f32, shape: f32) -> f32 { return min(existing, shape); }
fn csg_subtract(existing: f32, shape: f32) -> f32 { return max(existing, -shape); }
fn csg_intersect(existing: f32, shape: f32) -> f32 { return max(existing, shape); }
fn csg_smooth_union(a: f32, b: f32, k: f32) -> f32 { return smooth_min(a, b, k); }
fn csg_smooth_subtract(a: f32, b: f32, k: f32) -> f32 { return -smooth_min(-a, b, k); }
```

**Edit pipeline:**
```
1. CPU: Compute edit AABB, determine affected bricks
2. GPU: Pre-edit — allocate new bricks for EMPTY cells that may gain surface
3. GPU: Apply edit — compute shader over affected bricks (1 workgroup = 8×8×8 per brick)
4. GPU: Post-edit — check for empty bricks to deallocate (reduction pass)
5. GPU/CPU: Capture delta for undo (async readback of pre-edit values)
6. CPU: Update spatial index for new/removed bricks
```

**Analytic SDF primitives:** Sphere, Box, Capsule, Cylinder, Torus, Plane, Custom. Primitives combine freely — a rounded box is `box(p) - radius`. The edit shader accepts any SDF evaluator.

### Decision: Sculpt Brush System

| Brush | Operation | Material Effect |
|-------|----------|-----------------|
| **Add** | Smooth union with brush shape | Paints brush's material onto new surface |
| **Subtract** | Smooth subtraction | No material change (removes surface) |
| **Smooth** | Weighted average of neighboring SDF | Optionally blends materials |
| **Flatten** | Pull SDF toward reference plane | Preserves material |
| **Paint** | No geometry change | Sets material_id on near-surface voxels |
| **Blend Paint** | No geometry change | Sets blend_weight and secondary_id |
| **Color Paint** | No geometry change | Writes to companion color pool |

**Brush parameters:**
```rust
struct Brush {
    brush_type: BrushType,
    shape: BrushShape,          // Sphere, Cube, Custom
    radius: f32,
    strength: f32,              // 0.0-1.0
    falloff: FalloffCurve,      // Linear, Smooth, Sharp
    material_id: u16,
    secondary_id: u8,
    blend_k: f32,               // smooth CSG blend radius
}
```

Falloff diminishes effect from brush center to edge. Applied per-voxel to modulate the CSG blend strength.

### Decision: Brick Allocation/Deallocation During Edits

**Pre-edit:** CPU identifies EMPTY grid cells overlapping the edit AABB. For cells that CSG union might fill: atomically claim bricks from GPU free list, initialize to +MAX_DISTANCE, update spatial index to SURFACE.

**Post-edit:** GPU reduction pass checks each affected brick's min(abs(distance)). If all voxels are outside narrow band: return brick to free list, update spatial index to EMPTY/INTERIOR.

### Decision: Undo/Redo — Delta Compression

**Chosen over:** Full snapshots (memory-hungry), operation replay (expensive for long histories).

```rust
struct EditDelta {
    operation_id: u64,
    affected_bricks: Vec<BrickDelta>,
    allocated_bricks: Vec<BrickAllocation>,
    deallocated_bricks: Vec<BrickAllocation>,
}

struct BrickDelta {
    brick_id: u32,
    changed_voxels: Vec<(u16, VoxelSample)>,  // (index, previous_value)
}
```

**Undo:** Restore previous voxel values, re-allocate/deallocate bricks, update spatial index.

Delta capture runs as a GPU compute shader copying pre-edit values to a staging buffer, read back async by CPU. By the time the next edit occurs, the previous delta is ready.

History depth: configurable (default 100 operations, 256MB budget). Oldest deltas discarded when budget exceeded.

### Decision: Editor vs Runtime Editing

Both use the same GPU compute shaders. Editor adds UI, undo, and tool management. Runtime uses predefined parameters.

**Editor:** Full brush toolkit, unlimited undo, async OK, visual previews/gizmos, any resolution tier.

**Runtime:** Limited operations (subtract for destruction, union for building), no undo, must complete within frame budget.

**Runtime performance budget:**

| Edit Size | Bricks | Time |
|-----------|--------|------|
| Small (projectile) | 5-10 | <0.5ms |
| Medium (grenade) | 20-50 | <1ms |
| Large (building collapse) | 100-500 | 1-5ms (can span frames) |

Large edits spread across multiple frames — the visual result progressively appears (crater expanding), which suits explosion effects.

### Decision: Persistent Edit Streaming

For world modification that persists across chunk load/unload cycles (player-dug tunnels, terraforming):

**Edit journal per chunk:** Each chunk has an append-only journal of edits applied to it, stored on disk alongside the chunk's base `.rkf` data.

```
Chunk directory on disk:
  chunk_x_y_z/
    base.rkf              — original authored chunk data
    edits.rkj             — edit journal (append-only)
```

**Journal format (`.rkj` — RKIField Journal):**
```
RKJ File Layout:
  Header:
    magic: u32 ("RKJ\0")
    version: u32
    entry_count: u32

  Entries (append-only):
    per entry:
      timestamp: u64
      edit_op: CompactEditOp    — compact operation description (see below)
```

**Chunk loading with edits:**
```
1. Load base.rkf → decompress bricks into brick pool
2. Read edits.rkj → replay each edit operation against the loaded bricks
3. Chunk is now up-to-date with all player modifications
```

**Journal compaction:** When a journal grows large (>1000 entries or >10MB), compact it by baking the edits into a new base `.rkf` and clearing the journal. This keeps load times bounded.

**Streaming integration:** The chunk streaming system loads `.rkj` alongside `.rkf`. Edit replay happens during the async load, before the chunk is made visible. New edits to loaded chunks are both applied to the GPU brick pool (immediate visual) and appended to the journal (persistence).

### Decision: Compact Edit Operation Format

A serialized edit operation for journals, networking, and replay. Designed for minimal size and full reproducibility.

```rust
/// Compact edit operation — 64 bytes fixed size
/// Suitable for disk journaling, network replication, and replay
struct CompactEditOp {
    // Identity (8 bytes)
    op_id: u32,                     // monotonic operation counter
    flags: u16,                     // bit 0: has_rotation, bit 1: has_secondary_material
                                    // bit 2: is_multi_frame, bits 3-15: reserved
    edit_type: u8,                  // CSG_UNION=0, CSG_SUBTRACT=1, CSG_INTERSECT=2,
                                    // SMOOTH_UNION=3, SMOOTH_SUBTRACT=4,
                                    // SMOOTH=5, FLATTEN=6, PAINT=7, BLEND_PAINT=8, COLOR_PAINT=9
    shape_type: u8,                 // SPHERE=0, BOX=1, CAPSULE=2, CYLINDER=3, TORUS=4, PLANE=5

    // Spatial (28 bytes)
    position: [f32; 3],             // world-space center
    rotation: [i16; 4],             // quaternion as normalized i16 (-32767..32767)
                                    // ignored if !has_rotation (axis-aligned)
    dimensions: [f16; 3],           // radius / half-extents (f16 sufficient for edit sizes)
    _pad_spatial: u16,

    // Parameters (12 bytes)
    strength: f16,                  // 0.0-1.0 brush strength
    blend_k: f16,                   // smooth CSG blend radius
    falloff: u8,                    // LINEAR=0, SMOOTH=1, SHARP=2
    material_id_lo: u8,             // primary material ID (low byte of u16)
    material_id_hi: u8,             // primary material ID (high byte of u16)
    secondary_id: u8,               // secondary material for blend paint
    color: [u8; 4],                 // RGBA for color paint (or unused)

    // Metadata (16 bytes)
    timestamp: u64,                 // frame number or wall-clock ms
    source_id: u32,                 // who made this edit (player ID, editor session, etc.)
    _reserved: u32,
}
```

**Properties of this format:**
- **Fixed 64 bytes** — no variable-length fields, no allocation, trivially indexable
- **Self-contained** — every field needed to reproduce the edit
- **Network-friendly** — 64 bytes per edit at 60fps = ~3.8 KB/s for continuous editing. Trivially fits in any network budget.
- **Deterministic** — replaying the same sequence of CompactEditOps against the same base data produces identical results
- **Versioned** — header version field allows format evolution

**Usage across systems:**

| System | How it uses CompactEditOp |
|--------|--------------------------|
| Undo/redo | Store alongside BrickDelta for operation replay |
| Edit journal (.rkj) | Append-only sequence on disk per chunk |
| Future networking | Serialize and transmit to other clients |
| Replay/recording | Record edit streams for playback |

### Session 8 Summary: All Procedural Editing Decisions

| Decision | Choice | Upgrade Path |
|----------|--------|--------------|
| Edit execution | GPU compute shaders on brick pool | — |
| CSG operations | Union, subtract, intersect + smooth variants | Custom SDF evaluators |
| Brush system | 7 brush types with configurable shape/radius/strength/falloff | — |
| Brick management | Pre-edit allocation, post-edit deallocation, free list | — |
| Undo/redo | Delta compression (changed voxels only) | — |
| Editor vs runtime | Same shaders, different wrapping and budgets | — |
| Edit persistence | Per-chunk append-only journal (.rkj), compaction | — |
| Edit format | CompactEditOp, 64 bytes fixed, deterministic | Network replication, replay |
