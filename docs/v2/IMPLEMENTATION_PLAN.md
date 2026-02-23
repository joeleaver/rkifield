# RKIField v2 Implementation Plan: Object-Centric SDF Engine

## Context

The v1 engine uses a world-space chunk grid as the fundamental data structure. All SDF geometry is voxelized into bricks organized into chunks. This creates severe problems for interactive editing: every transform requires re-voxelization, objects lose identity once merged into chunks, chunk boundaries cause complexity when objects span them, and the removal problem (reconstructing a field without one object's contribution) is expensive.

The v2 architecture (fully designed in `ARCHITECTURE.md`, 993 lines) replaces this with an object-centric model where objects own their SDF in local space and transforms are applied at ray march time. This eliminates re-voxelization on transforms, gives objects persistent identity, and simplifies animation (skeleton = transform hierarchy, no rebaking).

**This is a greenfield rewrite. No backwards compatibility code. Old code paths are deleted, not wrapped.**

## Codebase Scope

- **13 crates**, ~67K lines Rust, ~5.8K lines WGSL, ~1,520 tests
- **Critical path:** rkf-core → rkf-render → rkf-runtime → rkf-editor/testbed
- **Engine will be non-functional during Phases 0-4.** First pixels return in Phase 5.

## Dependency Graph

```
Phase 0  (Cleanup — delete v1 chunk code, stub binaries)
    │
Phase 1  (Core types — SceneNode, Transform, hierarchy)
    │
Phase 2  (Brick maps — per-object brick storage, voxelization)
    │
Phase 3  (GPU layout — object metadata buffer, new GpuScene bind group)
    │
Phase 4  (BVH — spatial acceleration over objects)
    │
Phase 5  (Ray marcher — FIRST PIXELS)
    │
Phase 6  (Tile culling + coarse acceleration field)
    │          \
Phase 7  (Shading)    Phase 8  (GI injection)
    │          /
Phase 9  (Volumetrics + post-processing verification)
    │
    ├── Phase 10 (Animation — transform hierarchy) ──┐
    ├── Phase 11 (Editing + LOD)                     ├─ parallel
    ├── Phase 12 (Streaming + .rkf v2 format)        │
    └── Phase 13 (Project + environment + game mgr)  ┘
    │
Phase 14 (Editor + MCP)
    │
Phase 15 (Terrain + physics + particles + polish)
```

---

## Phase 0: Scaffolding and v1 Cleanup

**Goal:** Delete all v1 chunk-centric code. Mark v1 docs as superseded. Stub binaries to compile. The engine will NOT render after this phase.

### v2-0.1 — Mark v1 docs as superseded, create v2 implementation plan
- Add `> **SUPERSEDED** by [v2 Architecture](v2/ARCHITECTURE.md)` header to `docs/ARCHITECTURE.md`
- Add same header to all `docs/architecture/01-*.md` through `docs/architecture/12-*.md`
- Copy this plan to `docs/v2/IMPLEMENTATION_PLAN.md`
- Update `CLAUDE.md` architecture reference section to point to v2 docs

### v2-0.2 — Delete chunk system from rkf-core
- **DELETE:** `chunk.rs`, `sparse_grid.rs`, `clipmap.rs`, `populate.rs`
- **DELETE:** any `cell_state` module if separate
- Update `lib.rs`: remove all deleted module declarations and re-exports
- Delete all tests referencing deleted types
- **Expected:** downstream crates will not compile — intentional

### v2-0.3 — Delete chunk-based GPU scene from rkf-render
- **DELETE:** `gpu_scene.rs` (chunk-based occupancy/slot upload)
- **DELETE:** `clipmap_gpu.rs` (ClipmapGpuData, GpuClipmapLevel)
- Stub `ray_march.rs` to empty struct with `todo!()` constructor
- Stub `radiance_inject.rs` similarly
- Update `lib.rs`

### v2-0.4 — Delete chunk-based streaming from rkf-runtime
- **DELETE:** `streaming.rs`, `async_io.rs`, `lru_eviction.rs`, `streaming_budget.rs`, `procgen.rs`
- Remove `ChunkRef` and `AnimatedCharacter` from `components.rs`
- Stub `frame.rs` (references deleted GpuScene)
- Update `lib.rs`
- **DELETE:** `tests/streaming_world.rs` and any chunk-referencing test files

### v2-0.5 — Delete joint rebaking from rkf-animation
- **DELETE:** `rebake.rs`, `segment.rs`
- **DELETE:** `shaders/joint_rebake.wgsl`
- Keep: `skeleton.rs`, `clip.rs`, `player.rs`, `blend_shape.rs`, `character.rs`
- Update `lib.rs`

### v2-0.6 — Delete per-chunk journal from rkf-edit
- **DELETE:** `journal.rs`
- Keep: `types.rs`, `pipeline.rs`, `brush.rs`, `undo.rs`, `edit_op.rs`, `transform_ops.rs`
- Update `lib.rs`

### v2-0.7 — Stub testbed, editor, and game to compile
- Gut `rkf-testbed/src/main.rs` to minimal window + empty render loop
- Gut `rkf-editor/src/main.rs` and `engine_viewport.rs` to stubs
- Gut `rkf-game/src/main.rs` to stub
- Goal: `cargo build --workspace` compiles with zero errors

### v2-0.8 — Verify clean compile, run surviving tests
- `cargo build --workspace` — must succeed
- `cargo test --workspace` — must pass (reduced count expected)
- `cargo clippy --workspace` — must pass

**Verification:** Clean compile, all surviving tests pass. ~4,500 lines of Rust and ~270 lines of WGSL deleted.

**Docs updated:** v1 docs marked superseded, v2 implementation plan committed.

---

## Phase 1: Scene Hierarchy and Transform Propagation

**Goal:** Implement the core v2 data model: `SceneNode` tree, `SdfSource`, `BlendMode`, transform flattening. This is the foundation everything builds on.

### v2-1.1 — SceneNode, SdfSource, BlendMode, SdfPrimitive types
- Create `rkf-core/src/scene_node.rs`
- `SceneNode { name, local_transform, sdf_source, blend_mode, children, metadata }`
- `SdfSource` enum: `None | Analytical { primitive, material_id } | Voxelized { brick_map_handle, voxel_size, aabb }`
- `BlendMode` enum: `SmoothUnion(f32) | Union | Subtract | Intersect`
- `SdfPrimitive` enum: `Sphere | Box | Capsule | Torus | Cylinder | Plane` with parameters
- `Transform { position: Vec3, rotation: Quat, scale: f32 }` (local, uniform scale only)
- Tests: construction, tree building, defaults, display

### v2-1.2 — Scene and SceneObject types
- Create `rkf-core/src/scene.rs`
- `Scene { name, root_objects: Vec<SceneObject> }`
- `SceneObject { id: u32, world_position: WorldPosition, rotation: Quat, scale: f32, root_node: SceneNode, aabb: Aabb }`
- Methods: `add_object()`, `remove_object()`, `find_by_name()`, `find_by_id()`
- Object IDs: monotonically increasing u32
- Tests: add/remove, name/ID lookup, AABB computation

### v2-1.3 — Transform flattening system
- Create `rkf-core/src/transform_flatten.rs`
- `FlatNode { inverse_world: Mat4, accumulated_scale: f32, sdf_source_ref, blend_mode, node_index }`
- `flatten_object(object, camera_pos) -> Vec<FlatNode>`
- Root position via f64 subtraction (WorldPosition::relative_to)
- Depth-first traversal, `accumulated_scale = parent.scale * node.scale`
- Tests: identity, nested transforms, scale accumulation, inverse correctness, camera-relative precision

### v2-1.4 — Integrate with rkf-runtime
- Rewrite `rkf-runtime/src/components.rs` for v2 types
- Rewrite `rkf-runtime/src/transform_system.rs` to call `flatten_object`
- Rewrite `rkf-runtime/src/scene.rs` to hold `Vec<Scene>` (not hecs World for geometry)
- Tests: flattening in frame context

**Verification:** All unit tests pass. Scene hierarchy builds, flattens, and produces correct camera-relative transforms.

---

## Phase 2: Brick Pool Decoupling and Per-Object Brick Maps

**Goal:** Introduce per-object brick maps. Adapt the brick pool to work without chunks. Build object-level voxelization.

### v2-2.1 — BrickMap and BrickMapAllocator types
- Create `rkf-core/src/brick_map.rs`
- `BrickMap { dims: UVec3, entries: Vec<u32> }` — flat 3D array, EMPTY_SLOT for empty
- `BrickMapHandle { offset: u32, dims: UVec3 }` — view into packed allocator buffer
- `BrickMapAllocator` — packs multiple BrickMaps contiguously, free-list for deallocation
- Tests: allocate/deallocate, entry get/set, multiple maps, bounds checking

### v2-2.2 — Adapt Pool<T> for bulk operations
- `brick_pool.rs` is already chunk-agnostic — keep as-is
- Add `allocate_range(count) -> Option<Vec<u32>>` for bulk allocation
- Add `deallocate_range(slots: &[u32])` for bulk deallocation
- Tests: bulk alloc/dealloc, free-list integrity

### v2-2.3 — Per-object voxelization
- Create `rkf-core/src/voxelize_object.rs`
- `voxelize_sdf(sdf_fn, aabb, voxel_size, pool, map_alloc) -> (BrickMapHandle, u32)`
- Replaces old `populate_grid` — writes to brick map + pool
- Narrow band optimization: only allocate bricks near the surface
- Tests: sphere, box, empty object, brick count validation, narrow band correctness

### v2-2.4 — Update rkf-core exports
- Export new modules: `scene_node`, `scene`, `transform_flatten`, `brick_map`, `voxelize_object`
- `cargo build -p rkf-core && cargo test -p rkf-core`

**Verification:** Voxelization of analytical SDFs produces correct brick maps. Brick pool allocation works in bulk.

---

## Phase 3: GPU Data Layout

**Goal:** Build the new GPU scene representation: object metadata buffer, packed brick maps buffer, new bind group. This replaces the deleted GpuScene.

### v2-3.1 — GpuObject struct and metadata buffer
- Create `rkf-render/src/gpu_object.rs`
- `GpuObject` (256 bytes, bytemuck Pod): inverse_world (mat4), aabb_min/max (vec4), brick_map_offset, brick_map_dims (uvec3), voxel_size, material_id, sdf_type, blend_mode, blend_radius, sdf_params (vec4), accumulated_scale, lod_level, object_id, padding
- `ObjectMetadataBuffer`: wgpu storage buffer management
- Tests: size_of == 256, Pod roundtrip, field alignment

### v2-3.2 — GPU brick maps buffer
- Create `rkf-render/src/gpu_brick_maps.rs`
- Uploads packed brick map entries (u32 array) to single storage buffer
- Methods: `upload()`, `update_region()`
- Tests: upload and verify

### v2-3.3 — New GpuScene v2
- Create `rkf-render/src/gpu_scene.rs` (replacing the deleted v1 version)
- Bind group layout:
  - 0: brick pool (storage, read) — array of VoxelSample
  - 1: brick maps (storage, read) — array of u32
  - 2: object metadata (storage, read) — array of GpuObject
  - 3: camera uniforms (uniform)
  - 4: scene uniforms (uniform) — num_objects, max_steps, max_distance
- `GpuSceneV2::new()`, `upload()`, `update_objects()`, `update_camera()`
- Tests: bind group creation, upload with test data

### v2-3.4 — Camera uniforms verification
- `camera.rs` is largely retained — verify it compiles with new GpuScene
- No structural changes expected

**Verification:** GPU buffers upload. Bind group creation succeeds. `cargo build -p rkf-render` compiles.

---

## Phase 4: BVH Over Objects

**Goal:** CPU-side BVH over root object AABBs + GPU-uploadable representation.

### v2-4.1 — BVH data structure
- Create `rkf-core/src/bvh.rs`
- `BvhNode { aabb, left, right, object_index }` (leaf if left==right==INVALID)
- `Bvh { nodes: Vec<BvhNode> }`
- `build(objects: &[(u32, Aabb)])` — SAH top-down build
- `refit(objects)` — bottom-up AABB refit (for when objects move)
- `query_ray(origin, dir, max_t) -> Vec<u32>`
- `query_aabb(aabb) -> Vec<u32>`
- `query_sphere(center, radius) -> Vec<u32>`
- Tests: build 0/1/many objects, ray intersection, refit, AABB query

### v2-4.2 — GPU BVH buffer
- Create `rkf-render/src/gpu_bvh.rs`
- `GpuBvhNode` (32 bytes Pod): aabb_min(vec3) + left(u32), aabb_max(vec3) + right_or_object(u32)
- Upload to storage buffer, add to GpuScene bind group as binding 5
- Tests: upload, node count

### v2-4.3 — Integrate BVH with scene management
- Rebuild BVH when objects added/removed
- Refit each frame when transforms change
- Tests: dynamic add/remove/move

**Verification:** BVH correctly accelerates spatial queries. All tests pass.

---

## Phase 5: Object-Centric Ray Marcher — FIRST PIXELS

**Goal:** New ray march shader that evaluates objects via BVH + brick maps. The engine renders again.

### v2-5.1 — WGSL shared definitions
- Create `rkf-render/shaders/sdf_common.wgsl`
- Struct definitions: GpuObject, BvhNode, VoxelSample, CameraUniforms, SceneUniforms
- SDF primitive functions: `sdf_sphere()`, `sdf_box()`, `sdf_capsule()`, etc.
- Blend mode: `apply_blend(dist_a, mat_a, dist_b, mat_b, mode, radius)`
- Brick map sampling: `sample_brick_map(object, local_pos) -> (f32, u32)`

### v2-5.2 — Object evaluation function
- Create `rkf-render/shaders/evaluate_object.wgsl`
- `evaluate_object(camera_rel_pos, object_index) -> (f32, u32)`
- Transform to local space, evaluate SDF source, scale correction
- Voxelized: brick map lookup + trilinear sample
- Analytical: primitive function with sdf_params

### v2-5.3 — New ray march shader
- **Rewrite** `rkf-render/shaders/ray_march.wgsl`
- Bindings: group 0 = GpuScene v2 (pool, maps, objects, camera, scene, BVH), group 1 = G-buffer
- Algorithm: generate ray → BVH traversal for candidates → per-step evaluate candidate objects → min distance → on hit write G-buffer
- G-buffer output: position+distance, normal+blend, packed material IDs, motion vectors
- Write object_id to G-buffer for per-object identity

### v2-5.4 — RayMarchPass Rust update
- **Rewrite** `rkf-render/src/ray_march.rs`
- Constructor takes GpuSceneV2 + GBuffer + GpuBvh
- Dispatch: one thread per pixel, 8x8 workgroups

### v2-5.5 — Minimal testbed with analytical objects
- **Rewrite** `rkf-testbed/src/main.rs`
- Scene: 3-5 analytical objects (spheres, boxes, capsules)
- Build BVH, flatten transforms, upload to GPU
- Ray march + debug normals visualization + blit
- Goal: **see objects on screen**

### v2-5.6 — Add voxelized object to testbed
- Voxelize a sphere into brick map using `voxelize_sdf()`
- Upload brick pool + brick map
- Mix analytical and voxelized objects
- Verify both render correctly

**Verification:** Visual — testbed shows objects. Both analytical and voxelized visible. Normals debug mode shows correct surfaces. **This is the "first pixels" milestone.**

---

## Phase 6: Tile-Based Object Culling and Coarse Acceleration Field

**Goal:** Per-tile object lists (like v1 tiled light culling) and coarse distance field for empty-space skipping.

### v2-6.1 — Tile-based object culling pass
- **Rewrite** `rkf-render/src/tile_cull.rs` for objects (was lights in v1)
- **Rewrite** `rkf-render/shaders/tile_cull.wgsl` — project object AABBs, test tile frustum overlap
- Output: per-tile object count + index list (storage buffer)

### v2-6.2 — Integrate tile culling into ray marcher
- Ray march reads per-tile object list
- Only evaluate objects in tile's list (typically 3-5 instead of all)

### v2-6.3 — Coarse acceleration field
- Create `rkf-render/src/coarse_field.rs`
- 3D texture (R16Float), 32cm voxel resolution
- CPU update: for dirty cells, evaluate min distance to nearby objects via BVH
- GPU upload: standard 3D texture

### v2-6.4 — Integrate coarse field into ray marcher
- First steps through coarse field (big jumps through empty space)
- Switch to per-tile object evaluation when close to surfaces

### v2-6.5 — Coarse field update pipeline
- Compute pass for dirty region updates
- Triggered when objects move/add/remove

**Verification:** Visual — scene renders correctly with culling. Performance improvement measurable.

---

## Phase 7: Shading Adaptation

**Goal:** PBR shading reads from new GPU layout. Shadow and AO rays use coarse field + BVH.

### v2-7.1 — Adapt shade.wgsl bindings
- **Rewrite** scene data bindings in `rkf-render/shaders/shade.wgsl`
- Replace brick pool/occupancy/slot references with new GpuObject/brick_maps/BVH
- Keep all PBR math identical (Cook-Torrance GGX, F_Schlick, G_Smith)

### v2-7.2 — SDF shadow ray using coarse field + BVH
- New WGSL function: `shadow_ray_v2(origin, dir, max_t) -> f32`
- Coarse field for empty-space skipping, BVH + object eval near surfaces
- Shared by shade.wgsl and later radiance_inject.wgsl

### v2-7.3 — AO ray adaptation
- Short-range AO rays use coarse field
- Near-surface samples use object evaluation

### v2-7.4 — ShadingPass Rust update
- Update `rkf-render/src/shading.rs` bind groups

### v2-7.5 — Testbed with full shading
- Multiple materials, directional + point lights
- Verify PBR shading, shadows, AO
- MCP screenshot for validation

**Verification:** Visual — PBR shading correct, shadows cast between objects, AO visible, materials distinct.

---

## Phase 8: GI Injection Adaptation

**Goal:** Radiance volume injection uses coarse field + BVH instead of brick pool sampling. Cone tracing unchanged.

### v2-8.1 — Adapt radiance_inject.wgsl
- **Rewrite** `rkf-render/shaders/radiance_inject.wgsl`
- Sample coarse field to reject empty voxels
- Surface-adjacent voxels: BVH query, evaluate objects, get material, compute lighting, inject

### v2-8.2 — RadianceInjectPass Rust update
- Update `rkf-render/src/radiance_inject.rs` bind groups
- Radiance volume structure and mip gen passes unchanged

### v2-8.3 — Verify GI in testbed
- Emissive objects, colored surfaces, GI-only debug mode
- MCP screenshot

**Verification:** Visual — color bleeding between objects. Radiance volume populated correctly.

---

## Phase 9: Volumetrics and Post-Processing Verification

**Goal:** Adapt volumetric effects, verify post-processing pipeline works unchanged.

### v2-9.1 — Volumetric shadow map adaptation
- Adapt `rkf-render/shaders/vol_shadow.wgsl` — coarse field for marching

### v2-9.2 — Volumetric march adaptation
- Adapt `rkf-render/shaders/vol_march.wgsl` — SDF queries use coarse field

### v2-9.3 — Cloud shadow adaptation
- Adapt `rkf-render/shaders/cloud_shadow.wgsl` — minimal changes (clouds are analytical)

### v2-9.4 — Post-processing verification
- Post-processing passes (bloom, DoF, motion blur, tone map, color grade, cosmetics, sharpen, upscale) do NOT read scene SDF data — they should work unchanged
- Verify all compile and produce correct output

### v2-9.5 — Full pipeline testbed
- All render passes enabled
- Full pipeline: ray march → shade → GI → volumetrics → post → blit
- Measure frame time
- MCP screenshot

**Verification:** Visual — complete pipeline operational. Volumetrics, bloom, tone mapping all working.

---

## Phase 10: Animation via Transform Hierarchy

**Goal:** Skeletal animation as pure transform updates. No rebaking. Skeleton = node tree.

### v2-10.1 — Adapt animation types
- Adapt `rkf-animation/src/skeleton.rs` — bones map to node indices in SceneNode tree
- Adapt `rkf-animation/src/clip.rs` — keyframes update node local transforms
- Adapt `rkf-animation/src/player.rs` — playback writes to SceneNode transforms

### v2-10.2 — Character as SceneNode tree
- Adapt `rkf-animation/src/character.rs`
- `AnimatedCharacter` owns a SceneObject where each bone is a child node with voxelized SDF
- `update_pose(clip, time)` writes bone local transforms from keyframe interpolation
- Joint blending: siblings with SmoothUnion — natural from tree evaluation

### v2-10.3 — Blend shape adaptation
- Adapt `rkf-animation/src/blend_shape.rs`
- Blend shapes modify a node's local brick data

### v2-10.4 — Animated character testbed
- 14-bone humanoid as SceneNode tree
- Play walk animation (just updating transforms)
- Verify smooth-min joint blending during ray march
- MCP screenshot of T-pose and walk pose

**Verification:** Visual — animated character with smooth joints. Walk animation plays. No rebake pass.

---

## Phase 11: Per-Object Editing and LOD

**Goal:** CSG in object-local space. Per-object LOD with analytical fallback.

### v2-11.1 — Adapt CSG pipeline for object-local editing
- Adapt `rkf-edit/src/pipeline.rs` — CSG targets object's brick map, not world chunks
- Adapt `rkf-edit/src/edit_op.rs` — EditOp references object_id + local coordinates
- Adapt `rkf-edit/shaders/csg_edit.wgsl` — reads/writes object's brick map region

### v2-11.2 — Adapt brush system
- Adapt `rkf-edit/src/brush.rs` — brushes in object-local space
- Transform brush position from world to object-local before applying

### v2-11.3 — Per-object undo/redo
- Adapt `rkf-edit/src/undo.rs` — undo stack per object, not per chunk

### v2-11.4 — LOD system
- Create `rkf-core/src/lod.rs`
- `ObjectLod { levels, analytical_bound, importance_bias }`
- `select_lod(object, camera_distance, viewport_height) -> LodSelection`
- Screen-space driven: pick LOD where one voxel ≈ one pixel
- Tests: selection at various distances, analytical fallback

### v2-11.5 — LOD integration
- Per-frame LOD selection per object
- Swap brick map when LOD changes
- Deallocate evicted LOD bricks

**Verification:** CSG edits work on objects in local space. Undo/redo functional. LOD transitions smooth.

---

## Phase 12: Streaming and Asset Format v2

**Goal:** Per-object streaming with layered LOD .rkf v2 files.

### v2-12.1 — .rkf v2 file format
- Create `rkf-core/src/asset_file.rs` (replaces old chunk.rs I/O)
- Header: magic "RKF2", version, AABB, analytical bound, material refs, LOD count
- LOD table: per-level voxel_size, offset, compressed_size, brick_count
- LOD data: coarsest first, each independently LZ4 compressed, self-contained
- `save_object()`, `load_object_header()`, `load_object_lod()`
- Tests: roundtrip, partial LOD loading, header-only read

### v2-12.2 — Per-object streaming system
- Create `rkf-runtime/src/object_streaming.rs`
- States: Unloaded → Loading → Loaded(lod) → Upgrading → Evicting
- LOD-aware: coarsest first, progressively load finer
- Priority: screen_coverage * importance_bias

### v2-12.3 — Async I/O for object loading
- **Rewrite** `rkf-runtime/src/async_io.rs`
- Background thread pool loads .rkf v2 files
- Staging: decompress → allocate bricks → upload to GPU
- Per-frame budget caps

### v2-12.4 — LRU eviction
- **Rewrite** `rkf-runtime/src/lru_eviction.rs`
- Per-object (not per-chunk) tracking
- LOD-aware: demote before evict

### v2-12.5 — Import pipeline update
- Adapt `rkf-import/src/voxelize.rs` — output BrickMap instead of grid
- Adapt `rkf-import/src/lod.rs` — generate all LOD levels
- Adapt `rkf-convert/src/main.rs` — write .rkf v2

**Verification:** Objects load from .rkf v2. Progressive LOD loading works. Import → save → load → render roundtrip validated.

---

## Phase 13: Project Structure, Environment, Game Manager

**Goal:** The game-facing data model: project/scene management, environment profiles, game state.

### v2-13.1 — Project and Scene files
- Create `rkf-runtime/src/project.rs` — .rkproject (RON)
- Create `rkf-runtime/src/scene_file.rs` — .rkscene v2 (RON)
- Scene files contain object hierarchies, camera points, environment zones, lights
- No chunk manifests

### v2-13.2 — Multi-scene management
- Create `rkf-runtime/src/scene_manager.rs`
- `load_scene(path, Additive | Swap)`, `unload_scene(handle)`
- Persistent scene never unloaded

### v2-13.3 — Environment profiles (.rkenv)
- Create `rkf-runtime/src/environment.rs`
- `EnvironmentProfile`: sky, fog, ambient, atmosphere, volumetrics, post hints
- Blending: `lerp_profiles(a, b, t)`
- Programmatic overrides: per-property animated values
- Resolution: base → blend target → overrides

### v2-13.4 — Main camera with environment ownership
- Create `rkf-runtime/src/main_camera.rs`
- `MainCamera { position, rotation, fov, active_env, target_env, env_blend_t, env_overrides }`
- Environment zone triggers

### v2-13.5 — Game Manager
- Create `rkf-runtime/src/game_manager.rs`
- Scene management, state store (HashMap<String, GameValue>), camera control, environment control
- `GameValue`: Bool | Int | Float | String | Vec<GameValue>

### v2-13.6 — Save/Load (.rksave)
- Create `rkf-runtime/src/save_system.rs`
- Captures: loaded scenes, camera, environment, game state, entity overrides

**Verification:** Project loads from .rkproject. Multiple scenes load additively. Environment blends. Save/load roundtrip.

---

## Phase 14: Editor and MCP Adaptation

**Goal:** Editor uses v2 scene hierarchy. MCP tools updated for object-centric model.

### v2-14.1 — Editor scene tree
- **Rewrite** `rkf-editor/src/scene_tree.rs` — SceneNode hierarchy display
- **Rewrite** `rkf-editor/src/properties.rs` — node properties (transform, SDF source, blend mode)
- **Rewrite** `rkf-editor/src/editor_state.rs` — track selected node in hierarchy

### v2-14.2 — Gizmo and transform
- Adapt `rkf-editor/src/gizmo.rs` — manipulates node local transforms (instant, no re-voxelization)
- Enforce uniform scale in gizmo

### v2-14.3 — Object placement
- Adapt `rkf-editor/src/placement.rs` — place analytical or voxelized objects as root SceneObjects

### v2-14.4 — Sculpt and paint tools
- Adapt `rkf-editor/src/sculpt.rs` — operates on selected object's brick map in local space
- Adapt `rkf-editor/src/paint.rs` — paint material on selected object

### v2-14.5 — Editor engine viewport
- **Rewrite** `rkf-editor/src/engine_viewport.rs` — build v2 scene, flatten, upload, render

### v2-14.6 — MCP tool updates
- Update `rkf-core/src/automation.rs` — v2 AutomationApi trait
  - `scene_graph()` returns v2 hierarchy
  - Remove chunk-related methods
  - Add: `object_spawn()`, `object_despawn()`, `node_set_transform()`, `node_set_sdf_source()`
  - Add: `environment_get()`, `environment_blend()`, `env_override()`
- Update `rkf-editor/src/automation.rs` — implement updated trait
- Update `rkf-mcp/src/tools/` — new tools, remove deprecated chunk tools

### v2-14.7 — Scene I/O
- **Rewrite** `rkf-editor/src/scene_io.rs` — save/load .rkscene v2

**Verification:** Editor renders v2 scene. Gizmo transforms objects instantly. Sculpt works in local space. MCP connects, screenshots work, scene_graph returns hierarchy.

---

## Phase 15: Terrain, Physics, Particles, and Polish

**Goal:** TerrainNode container, adapt remaining systems, full integration testing.

### v2-15.1 — TerrainNode container type
- Create `rkf-core/src/terrain.rs`
- `TerrainNode { tile_size, resolution, blend_radius, lod_tiers, generation_fn, tiles }`
- Grid-based spatial query (O(1) tile lookup)
- Tile evaluation: current + neighbors for seam blending
- Procedural + sculpted hybrid

### v2-15.2 — Terrain rendering integration
- Special evaluation path in ray march shader for terrain
- `evaluate_terrain()` in WGSL: grid lookup, current + neighbor tiles, smooth-min blend
- Terrain tiles bypass BVH (grid lookup faster)

### v2-15.3 — Terrain streaming
- Tile load/unload by distance within terrain coordinates
- Procedural tiles generated on demand
- Per-tile LOD based on camera distance

### v2-15.4 — Physics adaptation
- Adapt `rkf-physics/src/sdf_collision.rs` — BVH + object-local evaluation
- Object identity in collisions
- Character controller: capsule-vs-SDF with per-object collision
- Terrain collision via grid lookup

### v2-15.5 — Particle system adaptation
- Adapt `rkf-particles/` — collision queries use BVH + object eval
- Render backends unchanged

### v2-15.6 — Performance optimization
- Profile with realistic scenes (100+ objects, terrain, animated characters)
- Optimize: BVH traversal, tile evaluation, coarse field lookup

### v2-15.7 — Full integration test suite
- Rewrite tests across all crates for v2 types
- Target 80%+ coverage on library crates
- Visual regression via testbed MCP screenshots
- Stress test: 200+ objects, terrain, characters, all effects

### v2-15.8 — Documentation finalization
- Update v2 architecture docs with implementation details
- Update CLAUDE.md with v2 patterns and conventions
- Verify all v1 docs marked as superseded

**Verification:** TerrainNode with seamless tiles. Physics with object identity. All crates compile and test. Full pipeline achieves target frame time. MCP fully operational.

---

## Summary

| Phase | Tasks | Milestone | Engine State |
|-------|-------|-----------|--------------|
| 0 | 8 | Clean compile, v1 deleted | Non-functional |
| 1 | 4 | Scene hierarchy | Non-functional |
| 2 | 4 | Per-object bricks | Non-functional |
| 3 | 4 | GPU layout | Non-functional |
| 4 | 3 | BVH acceleration | Non-functional |
| 5 | 6 | **First pixels** | Renders (debug only) |
| 6 | 5 | Culling + accel field | Renders (faster) |
| 7 | 5 | PBR shading | Renders (lit) |
| 8 | 3 | GI | Renders (GI) |
| 9 | 5 | Full pipeline | Fully rendering |
| 10 | 4 | Animation | Animated |
| 11 | 5 | Editing + LOD | Editable |
| 12 | 5 | Streaming + assets | Streamable |
| 13 | 6 | Project + environment | Game-ready data model |
| 14 | 7 | Editor + MCP | Editor functional |
| 15 | 8 | Terrain + polish | **Complete** |

**Total: 82 tasks across 16 phases.** Critical path to first pixels: Phases 0-5 (29 tasks).

## Critical Files

| File | Action | Why |
|------|--------|-----|
| `rkf-render/shaders/ray_march.wgsl` | Complete rewrite | Core of v2 — object-centric SDF evaluation |
| `rkf-render/src/gpu_scene.rs` | Complete rewrite | New bind group layout for object metadata |
| `rkf-render/shaders/shade.wgsl` | Rewrite SDF queries | Shadow/AO rays use coarse field + BVH |
| `rkf-render/shaders/radiance_inject.wgsl` | Rewrite SDF sampling | GI injection via coarse field + object eval |
| `rkf-core/src/lib.rs` | Continuous updates | Every new/deleted module flows through here |
| `docs/v2/ARCHITECTURE.md` | Reference (read-only) | Authoritative spec for all design decisions |
