# RKIField v2 Architecture — Object-Centric SDF Engine

## Motivation

The v1 architecture uses a world-space chunk grid as the fundamental data structure. All SDF geometry is voxelized into bricks, organized into chunks, and the ray marcher samples the brick pool via texture fetches.

This works well for static worlds but creates severe problems for interactive editing:

1. **Transform cost**: Moving, rotating, or scaling an object requires re-voxelizing all affected bricks. Every editor interaction triggers expensive GPU work.
2. **The removal problem**: Bricks store `min(all SDFs)` — the combined field. Moving an object requires reconstructing the field without it, which means re-evaluating every other object that touches those bricks.
3. **Chunk boundaries**: Objects that span chunks must update multiple chunks, each with their own octree. Scaling an object into a new chunk is especially painful.
4. **Loss of identity**: Once voxelized, objects lose their identity. The brick pool doesn't know which object contributed which distances.

## Core Principle

**Objects own their SDF in local space. Transforms are applied at ray march time, not at voxelization time.**

```
v1:  Object → voxelize into world-space chunks → ray march samples chunks
v2:  Object (local-space SDF) → ray march transforms ray into object space → evaluate
```

Transforms become a matrix update. Moving an object across the world is instant — nothing gets re-voxelized. Cross-chunk boundaries aren't a concept because objects aren't in chunks.

## Scene Hierarchy

The scene is a tree of nodes, similar to Unity's transform hierarchy.

### SceneNode

Every entity in the scene is a `SceneNode`:

```
SceneNode:
    name: String
    local_transform: Transform      // position, rotation, uniform scale relative to parent
    sdf_source: SdfSource           // how this node contributes geometry
    blend_mode: BlendMode           // how this node combines with siblings
    children: Vec<SceneNode>
    metadata: NodeMetadata          // visibility, lock, editor state, etc.
```

### SDF Sources

Each node's geometry comes from one of:

- **Analytical** — A primitive (sphere, box, capsule, torus, etc.) or a CSG expression tree. Evaluated as math during ray marching. Zero memory cost. Infinite resolution.
- **Voxelized** — Local-space brick data from an imported mesh or sculpted shape. Stored as the node's own compact brick structure. Evaluated via texture sampling during ray marching.
- **None** — Pure transform node (like Unity's empty GameObject). Used for grouping, as skeleton joints, or as hierarchy anchors.

### Blend Modes

Each node specifies how it combines with its siblings within the same parent:

- **SmoothUnion(radius)** — Smooth-min blend, the default. Creates organic joins between sibling geometry.
- **Union** — Hard union (min). Sharp intersection lines.
- **Subtract** — Removes this node's volume from the combined sibling field. Used for holes, cavities, carving.
- **Intersect** — Only the overlapping volume survives.

**Blending is scoped to the tree.** Nodes in different root objects never blend with each other. A character's arm blends with its torso (same tree) but not with a nearby wall (different tree). This is almost always the desired behavior — accidental blending between unrelated objects is a bug, not a feature.

## Object Hierarchy and Coordinate Systems

### Scene Structure

```
Scene
  ├── Object A (root) ─── WorldPosition
  │     ├── Node A1 ─── local Transform relative to A
  │     │     ├── Node A1a ─── local Transform relative to A1
  │     │     └── Node A1b
  │     └── Node A2
  ├── Object B (root) ─── WorldPosition
  │     └── ...
  ├── Terrain (root) ─── WorldPosition, TerrainNode container
  │     ├── Tile (0,0) ─── fixed grid offset
  │     ├── Tile (1,0)
  │     └── ...
  └── Light / Camera / etc. (root) ─── WorldPosition
```

**Root objects** sit directly in the scene. They carry a `WorldPosition` for precision-safe world placement.

**Child nodes** carry a local `Transform` (Vec3 position, Quat rotation, f32 uniform scale) relative to their parent. They have no concept of world space — they exist entirely within their parent's coordinate system.

**Trees are fully independent.** No cross-tree references, no shared coordinate spaces, no blending between trees. Each tree is a self-contained unit.

### Coordinate Spaces

Five coordinate spaces, from outermost to innermost:

| Space | Type | Lives On | Purpose |
|-------|------|----------|---------|
| **World** | `WorldPosition { chunk: IVec3, local: Vec3 }` | CPU | Scene authoring, streaming, physics broadphase. Infinite range, no precision loss. |
| **Camera-relative** | `Vec3` (f32) | GPU | All GPU-side computation. Computed by subtracting camera `WorldPosition` on CPU. Effective range ±km around camera. |
| **Node local** | `Vec3` (f32) | GPU | Per-node SDF evaluation. Reached by applying the node's inverse world transform to camera-relative position. |
| **Brick coordinate** | `IVec3` | GPU | Which brick in the node's brick map. `floor(local_pos / brick_world_size)` |
| **In-brick** | `Vec3` (f32, 0-8 range) | GPU | Position within a single 8×8×8 brick. `fract(local_pos / brick_world_size) * 8.0` |

### Transform Types

**WorldPosition** (root objects only):
```
WorldPosition { chunk: IVec3, local: Vec3 }
// chunk: 8m × 8m × 8m grid cell
// local: position within the chunk (0..8m per axis)
// Effectively ±17 billion meters range
```

**Transform** (child nodes):
```
Transform {
    position: Vec3,         // offset from parent, in parent's local space
    rotation: Quat,         // rotation relative to parent
    scale: f32,             // UNIFORM scale only — non-uniform breaks SDF distances
}
```

**Uniform scale is a hard constraint.** Non-uniform scale distorts SDF distances differently along each axis, breaking ray marching. This rule carries forward from v1. Objects that need stretching must be re-voxelized at the new proportions, not scaled non-uniformly.

### Transform Propagation (CPU, Per-Frame)

Each frame, the hierarchy is flattened in a single depth-first pass:

```
// Step 1: Compute camera-relative root position (f64-safe via WorldPosition math)
root.camera_relative_pos = root.world_pos.to_f64() - camera.world_pos.to_f64()  // as Vec3 f32

// Step 2: Build root's camera-relative transform
root.world_transform = Mat4::from_scale_rotation_translation(
    root.scale, root.rotation, root.camera_relative_pos
)

// Step 3: Propagate through children
for each child node (depth-first):
    child.world_transform = parent.world_transform * child.local_transform_matrix()
    child.inverse_world = inverse(child.world_transform)
    child.accumulated_scale = parent.accumulated_scale * child.scale
```

**Key detail:** The root's position is converted from WorldPosition to camera-relative f32 using f64 subtraction on CPU. This is where the precision guarantee lives — the GPU never sees absolute world positions, only camera-relative offsets. Same principle as v1, applied per-object instead of per-chunk.

The flat arrays of `inverse_world` transforms and `accumulated_scale` values are uploaded to a GPU storage buffer once per frame.

### Scale and SDF Distance Correction

When a node has non-unit accumulated scale, the SDF distances returned in local space must be corrected to world space:

```
world_distance = local_distance * node.accumulated_scale
```

**Why:** A node at scale 2.0 means its local coordinate space is half the size of world space. A distance of 1.0 in local space = 2.0 in world space. The ray marcher must account for this to maintain correct step sizes.

The `accumulated_scale` is the product of all uniform scales from root to node, pre-computed during hierarchy flattening. For most nodes this is 1.0 and can be skipped.

### GPU Transform Flow

The full coordinate transform chain for ray marching:

```
1. Ray position is in camera-relative f32                          [GPU, per-pixel]
2. Multiply by node.inverse_world → node local space               [GPU, per-pixel per-node]
3. Evaluate SDF in local space:
   a. Analytical: evaluate primitive function(local_pos, params)
   b. Voxelized:  local_pos → brick_coord → brick_map → brick_pool sample
4. Multiply SDF distance by node.accumulated_scale → world distance [GPU, per-pixel per-node]
5. Apply blend mode with sibling results                            [GPU, per-pixel per-node]
```

Steps 2-5 repeat for each geometry node in the object tree. The inverse transforms and scales are read from the GPU storage buffer — no per-step matrix computation.

## Tree Evaluation (Ray Marching)

When a ray needs to evaluate an object tree:

```
evaluate_tree(camera_rel_pos, tree) -> (distance, material):
    dist = MAX_DIST
    mat = 0
    for each node in tree.flattened_nodes:
        if node.sdf_source == None: continue
        local_pos = node.inverse_world * camera_rel_pos
        (node_dist, node_mat) = evaluate_node(node, local_pos)
        node_dist *= node.accumulated_scale  // correct to world-space distance
        (dist, mat) = apply_blend(dist, mat, node_dist, node_mat, node.blend_mode)
    return (dist, mat)
```

For voxelized nodes, `evaluate_node` does a texture sample into the node's local brick data. For analytical nodes, it evaluates the SDF function. Both return distance and material in node-local space.

### Per-Node Cost

A tree with N geometry nodes costs N SDF evaluations per ray step when the ray is inside the tree's AABB. This is more than v1's single texture fetch, but:

- Only pixels whose rays enter the tree's AABB pay this cost
- Tile-based culling limits which trees are evaluated per pixel
- Far trees can LOD down to fewer nodes or a single merged SDF
- Most trees are small (a prop might be 1-5 nodes)
- The savings from eliminating rebaking, re-voxelization, and chunk management offset this

## Spatial Acceleration

Without a world-space voxel grid, the ray marcher needs acceleration structures to avoid evaluating every object for every ray.

### Coarse Acceleration Field

A low-resolution world-space distance field (32cm or 64cm voxels) that stores the minimum distance to any surface across all objects. Contains only scalar distances — no materials, no colors.

**Purpose:** Empty-space skipping. The ray marcher takes big steps through empty space using this field, then switches to per-object evaluation near surfaces.

**Update cost:** When an object moves, only the cells near the old and new positions need updating. This is orders of magnitude cheaper than re-voxelizing full brick data. For analytical objects, updating is trivial — evaluate the SDF at each affected cell. For voxelized objects, use the object's AABB as a conservative bound.

### BVH Over Objects

A bounding volume hierarchy over root-level object AABBs. Used for:

- Determining which objects a ray might intersect
- Spatial queries (picking, physics broadphase)
- Streaming decisions

Refitting a BVH node when an object moves is cheap — update one AABB and propagate up. No rebuild needed unless the topology changes significantly.

### Tile-Based Object Culling

The same concept as tiled light culling (already implemented in v1):

```
Pass 1 (compute): For each screen tile, determine which object AABBs overlap the tile's frustum.
                   Write per-tile object lists to a buffer.
Pass 2 (ray march): Each pixel reads its tile's object list.
                     Only evaluate objects in the list.
```

A typical tile might have 3-5 relevant objects. This keeps per-pixel evaluation cost manageable regardless of total scene object count.

### Ray March Pipeline

```
1. Step through coarse acceleration field (big jumps through empty space)
2. When distance is small, look up tile's object list
3. For each nearby object:
     a. Check object's AABB (cheap reject)
     b. Transform ray position into object's local space
     c. Evaluate SDF (texture sample for voxelized, math for analytical)
     d. Blend per object's blend mode
4. Take min distance across all objects
5. If hit: record material, normal, object ID → G-buffer
```

## Animation

Animation in v2 is transform hierarchy animation — the same mechanism as the editor's transform gizmo.

### Skeletal Animation

A skeleton IS a transform hierarchy:

- Each bone is a node with local-space SDF data (voxelized segment or analytical shape)
- Animating = updating bone nodes' local transforms from keyframe data
- Joint blending = smooth-min between sibling bone nodes during ray march evaluation
- No rebaking step. No joint bricks. No Lipschitz mitigation for baked zones.

The v1 "segmented rigid body parts + joint rebaking" system is entirely replaced. Moving a character's arm is the same operation as moving any object — update a local transform.

### Blend Shapes

Blend shape deltas modify a node's local SDF data. Since nodes own their data in local space, blend shapes are straightforward local modifications.

### Performance

A 14-bone character = 14 SDF evaluations per ray step for pixels inside the character's AABB. In exchange:

- No per-frame rebake compute pass
- No joint brick allocation / management
- No Lipschitz safety multiplier overhead during marching
- Transform updates are trivial (write 14 matrices)

## Opinionated Container Types

Most scene nodes are freeform — arbitrary transforms, any SDF source, any hierarchy structure. But some use cases benefit from constrained containers that enforce rules on their children for performance and consistency.

### Terrain

The first and most important container type. A Terrain node manages a regular grid of tiles.

```
TerrainNode:
    tile_size: f32                          // e.g., 64m per tile
    resolution: f32                         // voxel size within tiles, e.g., 2cm
    blend_radius: f32                       // seam blending between adjacent tiles
    lod_tiers: Vec<LodTier>                // distance → resolution mapping
    generation_fn: Option<ProceduralSdf>    // for infinite procedural terrain
    tiles: HashMap<IVec2, TileNode>         // grid-indexed tile children
```

**Constraints enforced by the container:**
- All tiles are the same size
- All tiles use the same base resolution
- Tiles are placed on a regular grid (no arbitrary positioning)
- Blend radius between tiles is consistent

**Why these constraints matter:**
- Predictable memory usage per tile
- Simple spatial queries — grid coordinate lookup, not BVH traversal
- Guaranteed seamless blending between adjacent tiles
- Easy streaming with uniform cost per tile
- Consistent LOD transitions

**Terrain evaluation:**

```
evaluate_terrain(world_pos, terrain) -> (distance, material):
    // Grid lookup — O(1), no BVH needed
    tile_coord = floor((world_pos - terrain.origin) / terrain.tile_size)

    dist = MAX_DIST
    mat = 0
    // Evaluate current tile + neighbors (for blend margin)
    for each loaded tile near tile_coord:
        local_pos = world_pos - tile.world_offset
        if far_outside_tile_bounds(local_pos, blend_margin): continue
        (tile_dist, tile_mat) = evaluate_tile(tile, local_pos)
        (dist, mat) = smooth_min_blend(dist, mat, tile_dist, tile_mat, blend_radius)
    return (dist, mat)
```

At most 4-9 tiles evaluated per step (current + adjacent), and most are rejected by the bounds check. Fast and predictable.

**Procedural + sculpted hybrid:**
- Tiles start as procedural (analytical noise SDF, zero memory cost)
- Only when the user sculpts a tile does it get voxelized and stored
- Unsculpted tiles are regenerated on demand — infinite world with near-zero storage
- Sculpt edits are saved per-tile; procedural tiles are disposable

**Streaming:**
- Load/unload tiles by distance from camera
- Same budget/watermark/LRU logic as v1, but per-tile within the terrain
- Each tile has known, uniform memory cost — budget management is simple

**Multiple terrains:**
- A terrain is just a node in the scene tree
- Surface terrain, underground cave system, floating islands — each is a separate TerrainNode
- They don't blend with each other (separate trees)

### Future Container Types (not for initial implementation)

- **Foliage** — Instances small SDF objects (trees, grass, rocks) across a distribution pattern. Enforces LOD, culling, and instancing rules.
- **Destruction** — Manages debris fragments from a breakable object. Enforces physics simulation and cleanup rules.

## GPU Data Layout

### Brick Pool (Single 3D Texture Atlas)

A single large 3D texture containing all voxel data for all objects. Fixed-size 8×8×8 brick slots, managed by a simple free-list allocator. This is the v1 brick pool stripped of chunk association, octrees, and spatial indexing — just a GPU memory allocator.

The brick pool is resolution-agnostic. A brick is always 8×8×8 = 512 voxel samples regardless of world-space size. An object at 0.5cm resolution stores 4cm³ per brick. An object at 8cm resolution stores 64cm³ per brick. Same slot in the atlas.

### Per-Object Brick Maps (Storage Buffer)

Each voxelized object has a brick map — a small 3D array that maps local brick coordinates to pool slots. All brick maps are packed into a single storage buffer. Each object's metadata stores its offset and dimensions into this buffer.

```
evaluate_voxelized(object, local_pos):
    brick_world_size = object.voxel_size * 8.0
    brick_coord = floor(local_pos / brick_world_size)

    // Look up brick slot from per-object brick map
    map_index = object.brick_map_offset + flatten(brick_coord, object.brick_dims)
    brick_slot = brick_maps[map_index]
    if brick_slot == EMPTY: return MAX_DIST

    // Sample within the brick
    local_in_brick = fract(local_pos / brick_world_size) * 8.0
    atlas_pos = brick_slot_to_atlas(brick_slot) + local_in_brick
    return textureSample(brick_pool, atlas_pos)
```

For small objects (32³ voxels), the brick map is 4×4×4 = 64 entries — tiny. For terrain tiles, the map is larger but sparse — only surface-adjacent bricks are allocated. Interior and exterior air bricks are `EMPTY` and cost nothing.

### Object Metadata (Storage Buffer)

One storage buffer containing per-object structs, indexed by object ID:

```
GpuObject:
    inverse_world_transform: mat4       // pre-computed each frame
    aabb_min: vec3                       // local-space AABB
    aabb_max: vec3
    brick_map_offset: u32               // index into brick maps buffer
    brick_map_dims: uvec3               // dimensions of this object's brick map
    voxel_size: f32                     // world-space size of one voxel
    material_id: u32                    // for analytical objects
    sdf_type: u32                       // analytical primitive type, or VOXELIZED
    sdf_params: vec4                    // analytical primitive parameters
    blend_mode: u32                     // SmoothUnion | Union | Subtract | Intersect
    blend_radius: f32                   // for SmoothUnion
    lod_level: u32                      // current active LOD
```

### Total GPU Bindings for All Geometry

- **Brick pool** — one 3D texture (all voxel data)
- **Brick maps** — one storage buffer (all objects' brick maps packed)
- **Object metadata** — one storage buffer (all objects' GPU structs)
- **Companion pools** — color data, same structure as brick pool (one 3D texture + one map buffer)

Minimal binding count. No per-object textures. No bindless requirement.

## Level of Detail

### Per-Object LOD

LOD is a per-object property, not a global world-space tier. Each object chooses its resolution based on screen coverage, not just distance.

```
ObjectLod:
    levels: Vec<LodLevel>               // pre-computed, finest to coarsest
    analytical_bound: AnalyticalSdf      // extreme distance fallback (always available)
    importance_bias: f32                 // artist override — keep detailed longer

LodLevel:
    voxel_size: f32
    brick_map: BrickMapHandle
    brick_count: u32                     // memory cost of this level
```

### LOD Selection

Screen-space driven — pick the LOD level where one voxel is approximately one pixel:

```
select_lod(object, camera_pos, viewport_height):
    distance = length(object.world_pos - camera_pos)
    for level in object.lod_levels:  // finest to coarsest
        voxel_screen_pixels = (level.voxel_size / distance) * viewport_height
        if voxel_screen_pixels < 1.0:
            return level  // sub-pixel voxels = sufficient detail
    return analytical_bound  // extreme distance, zero bricks
```

### LOD Tiers

```
LOD 0:  0.5cm voxels   (full detail, near camera)
LOD 1:  2cm voxels     (medium, moderate distance)
LOD 2:  8cm voxels     (coarse, far)
LOD 3:  analytical     (bounding shape, extreme distance — zero bricks)
```

The analytical tier doesn't exist in v1. At extreme distance, an object's voxel data is fully evicted and replaced by a simple SDF equation (sphere, box, capsule). Memory cost drops to zero. For large scenes with many distant objects, this is a major memory win.

### Advantages Over v1 Clipmap LOD

- **Per-object priority**: A character's face stays high-res while a featureless wall at the same distance drops to coarse. Artist-controllable via `importance_bias`.
- **No global seams**: v1 clipmap tier boundaries are visible lines in the world. Per-object LOD transitions are independent — one object changing LOD doesn't affect its neighbors.
- **Budget-driven**: Instead of fixed distance thresholds, the system can optimize globally — "I have N brick slots, allocate them by screen coverage and importance." Low-priority objects get pushed coarser under memory pressure.
- **Analytical fallback**: Objects can fully evict their voxel data at extreme distance. v1 can't do this — the chunk grid has no concept of "this region is just a sphere now."
- **Analytical objects are free**: They're already mathematical. Resolution is infinite at every distance. No LOD management needed.

### Terrain LOD

Handled internally by the TerrainNode container. The terrain defines distance-based LOD tiers for its tiles. Near tiles at full resolution, far tiles at coarser voxel sizes. All tiles at a given tier have the same resolution (enforced by the container), so transitions are consistent. Tile LOD files follow the same layered format as regular objects.

## Asset File Format (.rkf v2)

### Layered LOD Storage

Each LOD level is stored as an independent, self-contained section in the file. Coarsest first.

```
.rkf v2 file layout:

┌─────────────────────────────────────────┐
│ Header                                  │
│   magic, version                        │
│   object AABB (local space)             │
│   analytical bound (primitive type +    │
│     params for extreme-distance LOD)    │
│   material references                   │
│   LOD count                             │
├─────────────────────────────────────────┤
│ LOD Table                               │
│   per-level: voxel_size, offset,        │
│     compressed_size, brick_count        │
├─────────────────────────────────────────┤
│ LOD 2 data  (8cm, coarsest)             │
│   brick map + LZ4 compressed bricks    │
│   + companion data (color, etc.)        │
├─────────────────────────────────────────┤
│ LOD 1 data  (2cm)                       │
│   brick map + LZ4 compressed bricks    │
│   + companion data                      │
├─────────────────────────────────────────┤
│ LOD 0 data  (0.5cm, finest)             │
│   brick map + LZ4 compressed bricks    │
│   + companion data                      │
└─────────────────────────────────────────┘
```

### Design Principles

**Coarsest first.** A single small read (header + LOD table + coarsest level) makes the object visible immediately. Finer levels stream in progressively. You never read past the LOD you need.

**Each LOD is fully independent.** Not deltas, not refinements — a complete, standalone representation. LOD 2 alone gives you a functional object at 8cm. No dependency on other levels being loaded. This means:

- Load any single LOD and the object works
- Evict fine LODs under memory pressure — object degrades gracefully, doesn't disappear
- No decompression chain (don't need LOD 1 to decode LOD 0)

**Each LOD is independently compressed.** LZ4 per level. No need to decompress fine data to access coarse data.

**LODs are pre-computed at import time.** The mesh-to-SDF converter generates all LOD levels during the (already expensive) voxelization step. Generating coarser LODs from fine data is a cheap downsample. Disk is cheap; loading should be "decompress and upload," no GPU processing required.

### Streaming Integration

```
Extreme distance:  load header only → analytical bound, zero bricks
Far:               load header + LOD 2 → coarse bricks, few KB
Medium:            stream LOD 1 → swap brick map, release LOD 2 bricks
Near:              stream LOD 0 → full detail
Camera pulls away: evict fine LOD, fall back to coarser (already loaded or cheap reload)
Memory pressure:   don't promote to finer LODs — objects degrade gracefully
```

Objects appear at analytical quality almost instantly, then progressively sharpen as data streams in. No stalls waiting for full-detail data before the object is visible.

### Terrain Tile Files

Same layered format. Since the TerrainNode enforces uniform tile dimensions, all tile files at a given LOD tier have identical brick map dimensions and predictable memory cost. Budget management is trivial.

Procedural terrain tiles have no file — they're generated at the requested resolution on demand. Only sculpted tiles need storage.

## Streaming

Streaming becomes per-object rather than per-chunk, integrated with the LOD system.

### Object Streaming

- Each object with voxelized data has a known memory cost per LOD level
- Streaming system decides both *whether* to load an object and *at what LOD*
- BVH structure is always in memory (lightweight metadata); actual SDF data is streamed
- Analytical objects are always "loaded" — zero streaming cost

### Terrain Tile Streaming

- Handled by the TerrainNode container
- Tiles loaded/unloaded by distance within the terrain's coordinate system
- Procedural tiles generated on demand, sculpted tiles loaded from disk
- LOD selection per-tile based on distance from camera

### Budget Management

Same principles as v1 (memory budget, watermark-based eviction, LRU tracking) but with more control:

- Budget system knows the cost of each LOD level per object
- Under pressure: demote objects to coarser LODs before evicting entirely
- Priority objects (importance_bias) resist demotion
- Graceful degradation: the scene never "pops" objects in and out — it adjusts quality smoothly

## Materials

The material system is largely unchanged from v1:

- Global material table (up to 65536 materials, indexed by u16)
- PBR properties + SSS + procedural noise per material
- Material blending between primary and secondary IDs with blend weight

### Per-Node Materials

Each node carries a material assignment:
- Analytical nodes: material ID directly on the node
- Voxelized nodes: per-voxel material IDs in the brick data (same as v1)
- Material blending happens within the node's local evaluation

### Per-Voxel Color

Color bricks remain as companion data for voxelized nodes. Stored in the node's local space alongside the SDF brick data.

## Project Structure and Game Data Model

### Hierarchy

```
Project (.rkproject)
  ├── Main Camera + Environment System
  ├── Game Manager (persistent state, scene management, save/load)
  ├── Material Table (global, shared across all scenes)
  ├── Settings (quality, input, audio)
  │
  ├── Scene: "persistent" (always loaded — player, companions)
  ├── Scene: "level_01" (gameplay scene, loadable)
  ├── Scene: "level_01_caves" (additive scene, loadable)
  └── ...
```

**Project** is the top-level container. It exists from startup to shutdown and owns everything that survives scene transitions: the camera, the game manager, the material table, and global settings.

**Scenes** are the unit of content — loadable, unloadable, swappable. Each scene contains an object hierarchy (tree of SceneNodes) and metadata. Multiple scenes can be loaded simultaneously.

### Project File (.rkproject)

```
// RON format
Project(
    name: "My Game",
    startup_scenes: ["persistent.rkscene", "main_menu.rkscene"],
    material_library: "materials.ron",
    default_environment: "outdoor_sunny.rkenv",
    settings: ProjectSettings( ... ),
)
```

### Scene File (.rkscene)

```
// RON format
Scene(
    name: "Level 01",
    objects: [ /* scene tree of SceneNodes */ ],
    camera_points: {
        "spawn": CameraPoint(position: ..., rotation: ...),
        "cutscene_01": CameraRig(path: ..., duration: ...),
        "overlook": CameraPoint(position: ..., rotation: ...),
    },
    environment_zones: [
        EnvironmentZone(bounds: ..., profile: "cave.rkenv", blend_speed: 1.5),
    ],
    point_lights: [ ... ],
    spot_lights: [ ... ],
)
```

Scenes do NOT own the environment. They define **environment zones** (trigger volumes that tell the camera to blend to a profile) and **camera points** (named positions/rigs that the game manager can activate).

### Scene Management

Scenes can be loaded in two modes:

**Additive:** Load alongside existing scenes. Used for streaming open worlds, layered interiors, etc.

**Swap:** Unload all non-persistent scenes, then load the new scene. Used for level transitions, menu screens, etc.

```
Project startup:
  1. Load persistent scene (player, companions)
  2. Load startup gameplay scene(s) from project config

Scene transition (enter dungeon):
  1. game_manager.load_scene("dungeon.rkscene", Swap)
  2. Unloads current gameplay scene(s)
  3. Loads dungeon scene
  4. Persistent scene remains untouched
  5. Player character still exists, game manager repositions to spawn point

Additive load (streaming open world):
  1. game_manager.load_scene("zone_02.rkscene", Additive)
  2. New scene loads alongside existing ones
  3. Persistent scene untouched
```

### Persistent Scene

A scene flagged as always-loaded. The scene manager never unloads it during normal gameplay. Contains entities that travel with the player between gameplay scenes:

- **Player character** — full object tree (skeleton, equipped items, visual attachments)
- **Companion/follower entities** — party members, pets
- **Player-attached effects** — held torch, buff aura, status particles

The persistent scene is just a regular scene following all normal rules (hierarchy, transforms, animation, physics, rendering). The only special property is that it's never unloaded. No special-cased "persistent entity" system needed.

**What does NOT go in the persistent scene:**
- Game state (inventory, flags, progress) — that's the game manager's state store
- Camera — project-level
- UI — separate system, not part of the 3D scene
- Environment — camera-owned profiles
- Music/audio state — game manager

### Game Manager

The persistent API for state management, scene control, and serialization. Exists at the project level from startup to shutdown.

```
GameManager:
    // Scene management
    load_scene(path, mode: Additive | Swap) -> SceneHandle
    unload_scene(handle)
    loaded_scenes() -> Vec<SceneHandle>

    // Persistent state (survives scene transitions and save/load)
    state: HashMap<String, GameValue>
    set_state(key, value)
    get_state(key) -> Option<GameValue>

    // Camera control
    set_camera_rig(scene, camera_point_name)  // move main camera to a scene's camera point
    release_camera()                           // return to player control

    // Environment control
    blend_environment(profile_path, duration)
    set_env_override(property, value, duration)
    clear_env_override(property, duration)

    // Serialization
    save(slot) -> Result<SaveFile>
    load(slot) -> Result<()>
```

**GameValue** is a simple tagged union: `Bool | Int | Float | String | Vec<GameValue>`. Sufficient for game flags, counters, inventory, quest state, etc.

### Save Game Format (.rksave)

```
// RON format
SaveGame(
    timestamp: ...,
    loaded_scenes: ["persistent.rkscene", "level_01.rkscene"],
    camera: CameraState(position: ..., rotation: ..., active_rig: None),
    active_environment: "outdoor_sunny.rkenv",
    environment_overrides: { ... },
    game_state: {
        "player_health": Float(85.0),
        "has_key_red": Bool(true),
        "quest_01_stage": Int(3),
        "inventory": Vec([...]),
    },
    entity_overrides: {
        // Objects modified from their scene defaults
        "level_01::door_03": EntityOverride(destroyed: true),
        "level_01::chest_07": EntityOverride(transform: ..., state: "opened"),
    },
)
```

The save captures which scenes are loaded, the camera, environment, game state variables, and any entities that have been modified from their scene defaults (destroyed, moved, state changed). Loading a save restores all of this — reload the scenes, apply overrides, restore camera and environment.

## Main Camera and Environment System

### Main Camera

The main camera is project-level — it exists outside of any scene and persists across all scene transitions. It is the single camera that drives rendering.

```
MainCamera:
    position: WorldPosition              // precision-safe world placement
    rotation: Quat
    fov: f32
    near: f32
    far: f32
    // Environment
    active_env: EnvironmentProfile
    target_env: Option<EnvironmentProfile>
    env_blend_t: f32
    env_blend_speed: f32
    env_overrides: EnvironmentOverrides
```

In the editor, the editor's free-fly camera overrides the main camera for viewport rendering. At runtime, the game manager controls the main camera (player control, camera rigs, cutscenes).

### Environment Profiles (.rkenv)

Standalone assets that define atmospheric and environmental rendering parameters. Authored independently from scenes, reusable across the project.

```
// RON format
EnvironmentProfile(
    // Sky
    sky_type: Procedural,              // Procedural | HDRI(path)
    sun_direction: Vec3(0.4, -0.7, 0.3),
    sun_color: Vec3(1.0, 0.95, 0.85),
    sun_intensity: 5.0,

    // Fog
    fog_color: Vec3(0.7, 0.75, 0.85),
    fog_density: 0.002,
    fog_height_falloff: 0.1,

    // Ambient
    ambient_color: Vec3(0.15, 0.18, 0.25),
    ambient_intensity: 0.3,

    // Atmosphere
    scattering_coefficients: Vec3(5.8e-6, 13.5e-6, 33.1e-6),
    mie_coefficient: 21e-6,

    // Volumetrics
    god_ray_intensity: 0.4,
    cloud_density: 0.3,
    cloud_shadow_opacity: 0.5,

    // Post-process hints
    exposure_bias: 0.0,
    color_tint: Vec3(1.0, 1.0, 1.0),
)
```

**Profiles are blendable.** Every field can be lerped. Transitioning from "outdoor sunny" to "cave" smoothly interpolates all parameters over the specified duration.

### Environment Blending

The camera blends between profiles for smooth transitions:

```
camera.blend_to("cave.rkenv", duration: 2.0)
// Over 2 seconds: all environment params lerp from active to target
// On completion: active = target, target = None
```

### Programmatic Overrides

Scripts and gameplay code can override individual environment properties without authoring a new profile. Overrides layer on top of the current (or blending) profile.

```
// Shift sun direction over 5 minutes (time-of-day)
game_manager.set_env_override("sun_direction", Vec3(0.1, -0.9, 0.2), duration: 300.0)

// Flash fog red for a damage effect
game_manager.set_env_override("fog_color", Vec3(0.8, 0.1, 0.1), duration: 0.3)

// Dim ambient for a horror sequence
game_manager.set_env_override("ambient_intensity", 0.05, duration: 2.0)

// Blend back to profile value
game_manager.clear_env_override("fog_color", duration: 1.0)
```

**Resolution order each frame:**

```
1. base = active_env
2. if target_env: base = lerp(active_env, target_env, blend_t)
3. for each override: base.property = lerp(base.property, override.value, override.t)
4. Final resolved environment → passed to renderer
```

Overrides take priority over profile blending. Clearing an override smoothly blends the property back to whatever the base profile says.

### Scene Environment Zones

Scenes don't own the environment, but they can define **environment zones** — trigger volumes that request a profile change when the camera enters them.

```
EnvironmentZone:
    bounds: Aabb                        // trigger volume in scene space
    profile: String                     // path to .rkenv asset
    blend_speed: f32                    // seconds to transition
    priority: u32                       // higher priority wins when overlapping
```

When the main camera enters a zone, it's equivalent to calling `camera.blend_to(profile, blend_speed)`. When the camera leaves all zones, it blends back to the scene's default environment (or the project default).

### Directional Light

The sun/moon is part of the environment profile, not a scene object. Sun direction, color, and intensity drive shadows, god rays, sky color, and atmospheric scattering — these are fundamentally atmospheric properties.

**Scene-level lights** (point lights, spot lights) remain scene objects in the object hierarchy.

**Time-of-day** is built on top of overrides: a gameplay system animates `sun_direction`, `sun_color`, `sun_intensity`, `ambient_color`, etc. via `set_env_override()` driven by a clock value. The engine provides the mechanism (profiles + overrides + blending). The game layer provides the policy (clock speed, day/night cycle).

## What Changes from v1

### Removed

- World-space chunk grid
- Global brick pool (replaced by per-object local brick storage)
- Chunk octrees
- Joint rebaking system
- Brick merging across objects
- Chunk-based streaming
- Global spatial index for voxel data

### Retained

- Material table and blending system
- PBR shading pipeline
- GI (voxel cone tracing — adapted to work with the coarse acceleration field)
- Volumetric effects
- Post-processing pipeline
- Particle system
- Physics (Rapier integration, SDF collision adapter)
- Editor UI framework (rinch)
- MCP integration
- WGSL compute shader pipeline

### New

- Project structure (.rkproject) with multi-scene management
- Game Manager (persistent state, scene control, save/load)
- Main camera at project level with environment ownership
- Environment profiles (.rkenv) with blending and programmatic overrides
- Persistent scene (always-loaded, player/companions)
- Additive and swap scene loading
- Scene hierarchy with transform propagation
- Per-object local SDF storage (analytical + voxelized)
- Tree-scoped blend modes
- BVH over objects
- Tile-based object culling
- Coarse acceleration field for empty-space skipping
- TerrainNode container type
- Per-object screen-space LOD with analytical fallback
- Layered .rkf v2 file format (coarsest-first, independent LODs)
- Object-based streaming integrated with LOD

## Resolved Decisions

### GI: Voxel Cone Tracing (Retained)

**Decision:** Keep the clipmap radiance volume and voxel cone tracing. No probes (not even as a future path). The radiance volume was always decoupled from the geometry representation — it's a separate, coarser data structure that can survive the v2 transition intact.

**What changes:** Only the injection pass. Instead of sampling the brick pool to find surfaces, injection queries the coarse acceleration field for quick empty-space rejection, then evaluates nearby objects via BVH for surface-adjacent voxels.

```
v1 injection: sample brick pool (texture fetch) → inject if near surface
v2 injection: sample coarse accel field (texture fetch, fast reject) → BVH + object eval → inject
```

**What doesn't change:** Cone tracing reads from the radiance volume. It doesn't care how the volume was populated. The clipmap structure, mip generation, cone sampling — all identical to v1.

**Why not probes:** Irradiance probes (DDGI-style) are a fundamentally different GI model with different tradeoffs (lower spatial resolution, temporal lag, light leaking through thin geometry). Voxel cone tracing through a clipmap gives higher spatial fidelity and is already implemented and validated. Replacing it with probes would be a downgrade, not an upgrade.

### Shadow Rays and Secondary Rays

**Decision:** All secondary rays (shadows, AO, GI injection) use the coarse acceleration field as their primary acceleration structure, with BVH + object evaluation near surfaces.

The coarse acceleration field is the backbone of v2 ray traversal — not just for primary rays. It provides empty-space skipping for every ray type:

- **Primary rays** — coarse field for big steps, tile-based culling for per-pixel object lists near surfaces
- **Shadow rays** — coarse field marching, BVH lookup near surfaces. Only need distance (not material), so cheaper than primary rays.
- **AO rays** — short-range, coarse field usually sufficient. Fall back to object evaluation only for very close surfaces.
- **GI injection** — coarse field rejects empty voxels. Only surface-adjacent voxels do full object evaluation.

This makes the coarse acceleration field one of the most critical data structures in v2. Getting its resolution, update strategy, and memory budget right is essential.

### Object-to-Object CSG: Destructive Edit Only

**Decision:** No live cross-tree CSG. Blending and boolean operations are scoped to within a tree. Cross-tree operations are destructive edits that modify the target's voxel data.

**Within a tree:** Use blend modes (`Subtract`, `Intersect`, `SmoothUnion`, `Union`). A window frame subtracting from a wall = siblings with the frame set to `Subtract`. Move the frame, the hole follows. This is the hierarchy working as designed.

**Across trees:** Destructive edit. The CSG shape is a *tool*, not a persistent relationship. Carving a tunnel through terrain = sculpt operation that modifies the terrain tiles' voxel data. The tool shape can be deleted afterward — the modification is permanent in the target's data.

**Why no live cross-tree CSG:**
- Trees stay independent — no cross-tree dependency graphs
- No per-frame evaluation cost from maintaining live CSG links
- No ambiguity about deletion ("does the hole disappear if I delete the carving object?")
- Consistent with established editor UX (Unreal landscape tools, voxel editors, v1 brush system)
- Procedural terrain tiles get voxelized on first edit (already in the design)

The v1 CSG brush system (Phase 16) carries forward — it just targets a specific object's local voxel data instead of world-space chunks.

### GPU Data Layout: Single Brick Pool + Per-Object Brick Maps

**Decision:** One shared brick pool (3D texture atlas) with per-object brick maps (storage buffer). Object metadata in a third storage buffer. Three bindings total for all geometry.

The brick pool is resolution-agnostic — a brick is always 8×8×8 voxels regardless of world-space size. Different objects with different `voxel_size` values coexist in the same pool. No octrees, no chunk association — just a flat free-list allocator.

Per-object brick maps replace per-chunk octrees. A brick map is a flat 3D array (no tree traversal during ray marching). Small objects have tiny maps (64 entries for a 32³ object). Large objects have sparse maps (only surface-adjacent bricks allocated).

Analytical objects skip all of this — zero bricks, zero brick map, just a primitive type and parameters in the object metadata buffer.

See **GPU Data Layout** section above for full details.

### Physics: Object Identity + BVH Broadphase

**Decision:** The SDF collision adapter evaluates objects individually via BVH lookup instead of sampling a global brick pool. Same pattern as ray marching. Physics gets simpler and gains object identity.

**What improves:**

- **Object identity in collisions.** Every collision knows which object was hit. Enables per-object physics materials (friction, bounciness), collision callbacks with identity ("player touched lava"), collision layers/masks, trigger volumes — standard game engine capabilities that were awkward with a merged SDF field.
- **Moving physics objects are free.** Transform update is instant. No re-voxelization.
- **Broadphase is the BVH.** The same object BVH used by the ray marcher provides spatial queries for collision. No extra structure needed.
- **Destruction.** Fragments become new objects in the scene tree with their own SDF data, physics bodies, materials, and LOD. Object identity means the system knows what broke.

**Collision evaluation:**

```
evaluate_collision(world_pos) -> (distance, object_id):
    query BVH for objects near world_pos
    for each nearby object:
        local_pos = object.inverse_world * world_pos
        dist = evaluate_sdf(object, local_pos)
        if dist < closest:
            closest = dist
            hit_object = object
    return (closest, hit_object)
```

**Performance:** Physics needs contact evaluation at tens of points per frame, not millions of pixels. The cost of BVH query + multi-object evaluation at each contact point is negligible.

**Character controller:** Same capsule-vs-SDF iterative slide, but with object identity at every contact. Enables surface-dependent gameplay (footstep sounds, friction variation, material responses).

**Terrain collision:** Routes through the TerrainNode's grid lookup for relevant tiles at contact positions. Fast and predictable.
