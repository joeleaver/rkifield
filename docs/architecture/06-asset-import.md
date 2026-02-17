# Asset Import Pipeline

> **Status: DECIDED**

### Decision: Offline Conversion to Binary `.rkf` Format

All mesh assets are converted offline by a CLI tool (`rkf-convert`) to the engine's native binary format (`.rkf` — RKIField). No runtime mesh processing.

**Supported source formats:**
- glTF 2.0 / GLB (primary — modern, well-specified, PBR-native)
- FBX (via russimp — widespread in game pipelines)
- OBJ (simple meshes, no animation)

**CLI tool:**
```
rkf-convert input.glb -o output.rkf [options]

Options:
  --tier <0-3>          Force resolution tier (default: auto)
  --tier-face <0-3>     Tier for face/head region
  --palette <file>      Material palette to map into
  --segments <file>     Manual segment override config
  --lod-levels <N>      Number of LOD tiers to generate (default: 3)
  --compression <lz4|zstd>  Brick compression (default: lz4)
  --no-color            Skip per-voxel color baking
  --no-skeleton         Strip animation data
```

### Decision: Mesh-to-SDF Algorithm — BVH Distance + Generalized Winding Number

**Phase 1 — Unsigned distance:** Build BVH over mesh triangles. For each voxel in the narrow band, query BVH for nearest triangle distance. Cost: O(voxels × log(triangles)).

**Phase 2 — Sign determination:** Generalized winding number (Barill et al., 2018). Computes winding number at each point; > 0.5 = inside. Robust with non-watertight meshes, non-manifold geometry, and intersecting surfaces — all common in real-world game assets.

**Chosen over:** Ray casting for sign (fragile with open meshes), normal-based sign (unreliable at edges).

**Narrow band optimization:** Only compute accurate distances within ±3 bricks of the surface. Distant voxels get ±MAX_DISTANCE via flood fill.

### Decision: Material Transfer — Per-Voxel Color + Uniform PBR per Material

For each surface voxel during conversion:
```
1. Find nearest triangle (BVH query)
2. Compute barycentric coords of closest point on triangle
3. Interpolate UV from triangle vertices
4. Source material index → engine material ID
5. If source has albedo texture:
   - Sample at UV → store in companion color pool
   - Set has_color_data flag
6. PBR properties (roughness, metallic): from material's base values (uniform per material)
```

Per-voxel color captures albedo texture detail. PBR variation (roughness/metallic textures) uses uniform values per material. Procedural noise compensates for lost micro-detail.

**Upgrade path — PBR blending:** Use primary/secondary material fields and blend_weight to represent continuous PBR variation. Primary = one PBR endpoint, secondary = other, weight interpolates. Clean reuse of existing blend system for gradients (clean-to-rusty, dry-to-wet).

### Decision: Skeleton Transfer — Automatic Segmentation with Manual Override

**Step 1 — Import skeleton:** Bone hierarchy, bind-pose transforms, per-vertex weights, animation clips from source file.

**Step 2 — Automatic segmentation:**
```
For each vertex: dominant_bone = bone with highest weight
For each bone: segment_triangles = triangles where all 3 vertices share dominant_bone
Joint regions = triangles where vertices have different dominant bones
```

Produces natural body-part segments: torso (spine bone), upper arm (upper arm bone), etc.

**Step 3 — Per-segment voxelization:** Isolate segment triangles, voxelize into bricks, tag with segment ID and bone index.

**Step 4 — Joint region extraction:** Bounding volume around joint triangles, voxelize both segments, store bone weights in companion bone pool.

**Step 5 — Blend shape conversion:** For each morph target, voxelize the deformed mesh, subtract base SDF, store as sparse delta bricks.

**Manual override:** Optional config file mapping bones to segments for cases where automatic analysis produces suboptimal splits.

### Decision: Resolution Selection — Automatic with Override

**Default heuristic:**
```
average_edge_length = mean triangle edge length
target_voxel_size = average_edge_length / 3
snap to nearest tier:
  < 1cm → Tier 0 (0.5cm)
  < 4cm → Tier 1 (2cm)
  < 16cm → Tier 2 (8cm)
  else → Tier 3 (32cm)
```

Artist override via CLI flags (`--tier`, `--tier-face`). Multi-tier per asset supported (face at Tier 0, body at Tier 1).

**LOD pre-computation:** During import, downsample from the target tier to all coarser tiers. Each tier stored in the `.rkf` file for clipmap streaming.

### Decision: `.rkf` Binary File Format

```
RKF File Layout:

  Header (64 bytes):
    magic: u32 ("RKF\0"), version: u32, flags: u32
    bounding_box: AABB (6 × f32)
    tier_count: u32, material_count: u32
    segment_count: u32, joint_count: u32, blend_shape_count: u32

  Material Table:
    array of Material structs (96 bytes each)
    Source-to-engine material ID mapping

  Per Tier:
    tier_header: tier_index, voxel_size, brick_count, grid_dimensions
    Spatial Index: occupancy bitfield + block index (compressed)
    Brick Data: array of bricks, LZ4 compressed (4KB/brick raw)
    Color Data: companion color bricks, LZ4 compressed (2KB/brick raw, if has_color)

  Skeleton (if animated):
    bone_count, bone_hierarchy (parent, name, bind_transform per bone)
    animation_clips (keyframes per bone per clip)

  Segments (if animated):
    per segment: bone_index, brick_range, bounding_box

  Joints (if animated):
    per joint: adjacent_segments, blend_radius, brick_range, bone_data_range

  Companion Bone Data (if animated): LZ4 compressed (4KB/brick raw)

  Blend Shapes (if any):
    per shape: name, affected_region AABB, delta_brick_count, compressed delta bricks
```

**Compression:** LZ4 default (fast decompression). Zstd optional (better ratio, slower decompress). SDF brick data compresses well — smooth distance fields with high local coherence.

### Conversion Pipeline Stages

```
1. LOAD:     Parse source file → mesh, materials, skeleton, animations
2. ANALYZE:  Mesh stats, determine tier, build BVH
3. SEGMENT:  If animated: decompose into segments and joints
4. VOXELIZE: Per-segment or whole mesh:
             a. Unsigned distances (BVH queries)
             b. Sign (generalized winding number)
             c. Material assignment + texture sampling
             d. Build sparse brick structure
5. LOD:      Downsample for each additional tier
6. COMPRESS: LZ4/zstd brick data
7. WRITE:    Serialize to .rkf
```

**Upgrade path — Runtime conversion:** Async import for editor workflows. Show low-res preview immediately, refine in background.

**Upgrade path — GPU-accelerated voxelization:** Move distance computation and sign determination to compute shaders. Massive speedup for high-resolution assets.

### Session 7 Summary: All Asset Import Decisions

| Decision | Choice | Upgrade Path |
|----------|--------|--------------|
| Conversion | Offline CLI tool (`rkf-convert`) | Runtime async; GPU-accelerated |
| SDF algorithm | BVH distance + generalized winding number | — |
| Material transfer | Per-voxel color (albedo texture) + uniform PBR per material | PBR blending via material blend fields |
| Skeleton transfer | Automatic segmentation from bone weights + manual override | — |
| Resolution | Automatic from mesh stats, artist override, multi-tier per asset | — |
| LOD | Pre-computed at import, all tiers in file | — |
| File format | `.rkf` binary, LZ4 compressed | Zstd option |
| Source formats | glTF/GLB, FBX, OBJ | — |
