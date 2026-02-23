> **SUPERSEDED** by [v2 Architecture](../v2/ARCHITECTURE.md) — this document describes the v1 chunk-based engine.

# Engine Architecture

> **Status: DECIDED**

### Decision: Coordinate System — Right-Handed, Y-Up, Meters

| Property | Choice | Rationale |
|----------|--------|-----------|
| Handedness | Right-handed | wgpu/WebGPU convention, glam default, Rapier default |
| Up axis | +Y | wgpu convention, Rust game ecosystem standard |
| Forward | -Z | Standard for right-handed Y-up (camera looks down -Z) |
| Units | 1 unit = 1 meter | Voxel sizes defined in cm/m, Rapier expects meters |
| World origin | (0, 0, 0) = scene center, sea level | Y=0 is ground plane / sea level |
| Clip space Z | [0, 1] | wgpu standard (not [-1, 1] like OpenGL) |

**Chunk addressing:**
```
Chunk at integer coordinates (cx, cy, cz)
World position of chunk corner = (cx * CHUNK_SIZE, cy * CHUNK_SIZE, cz * CHUNK_SIZE)
CHUNK_SIZE = 8.0 meters (from Session 1)

Chunk (0, 0, 0) spans [0..8, 0..8, 0..8] meters
Chunk (-1, 0, 0) spans [-8..0, 0..8, 0..8] meters
```

Vertical chunks exist (cy can be negative for underground, positive for sky). The clipmap LOD system is radial from the camera — axis-agnostic.

### Decision: Floating Point Precision — WorldPosition + Camera-Relative Rendering

Large worlds cause float32 precision loss. At 50km from origin, f32 precision (~5mm) approaches our finest voxel size (5mm). For an SDF engine this is catastrophic: imprecise distances cause ray march overshooting, normal jitter, and material flickering.

**Solution: Two-layer coordinate system.** The GPU never sees absolute world positions.

```rust
/// Replaces raw Vec3 for anything that exists in the world
#[derive(Clone, Copy)]
struct WorldPosition {
    chunk: IVec3,       // i32 — chunk grid coordinates
    local: Vec3,        // f32 — 0.0..CHUNK_SIZE offset within chunk
}

impl WorldPosition {
    fn normalize(&mut self) {
        // If local drifts outside 0..CHUNK_SIZE, adjust chunk coords
        while self.local.x >= CHUNK_SIZE { self.local.x -= CHUNK_SIZE; self.chunk.x += 1; }
        while self.local.x < 0.0 { self.local.x += CHUNK_SIZE; self.chunk.x -= 1; }
        // same for y, z
    }

    fn distance_f64(&self, other: &WorldPosition) -> f64 {
        let dx = (self.chunk.x - other.chunk.x) as f64 * CHUNK_SIZE as f64
               + (self.local.x - other.local.x) as f64;
        let dy = (self.chunk.y - other.chunk.y) as f64 * CHUNK_SIZE as f64
               + (self.local.y - other.local.y) as f64;
        let dz = (self.chunk.z - other.chunk.z) as f64 * CHUNK_SIZE as f64
               + (self.local.z - other.local.z) as f64;
        (dx*dx + dy*dy + dz*dz).sqrt()
    }

    fn relative_to(&self, origin: &WorldPosition) -> Vec3 {
        // f64 subtract, cast to f32 — the core precision-preserving operation
        let dx = (self.chunk.x - origin.chunk.x) as f64 * CHUNK_SIZE as f64
               + (self.local.x - origin.local.x) as f64;
        let dy = (self.chunk.y - origin.chunk.y) as f64 * CHUNK_SIZE as f64
               + (self.local.y - origin.local.y) as f64;
        let dz = (self.chunk.z - origin.chunk.z) as f64 * CHUNK_SIZE as f64
               + (self.local.z - origin.local.z) as f64;
        Vec3::new(dx as f32, dy as f32, dz as f32)
    }
}
```

**Per-system precision strategy:**

| System | Coordinate Space | How |
|--------|-----------------|-----|
| Ray marching | Chunk-local (0–8m) | Transform ray into chunk space on entry |
| Entity transforms | Camera-relative f32 | f64 subtract on CPU, f32 to GPU |
| Lights, fog volumes | Camera-relative f32 | Same f64 subtract |
| Physics (Rapier) | Camera-relative f32 | Active region always near player |
| Streaming distance | f64 | `WorldPosition::distance_f64()` |
| Voxel data | Chunk-local | Already stored per-chunk |

**Ray marching in chunk-local space:**
```wgsl
// When ray enters chunk (cx, cy, cz):
// chunk_offset is precomputed on CPU as camera-relative position of chunk origin
let local_ray_origin = ray_origin - chunk_offset;
// local_ray_origin is now 0-8m range — full f32 precision
// All SDF evaluation, trilinear interpolation, normal computation in local space
```

**Entity transform upload:**
```rust
fn camera_relative_transform(entity: &WorldPosition, camera: &WorldPosition) -> Mat4 {
    let offset = entity.relative_to(camera); // f64 math internally, f32 result
    Mat4::from_translation(offset) * entity.rotation_scale_matrix()
}
```

**Effective world range:** i32 chunk coordinates × 8m = ±17 billion meters (~0.1 AU). Sub-millimeter precision everywhere within.

### Decision: Application Architecture — Hybrid ECS + Global GPU Systems

**Chosen over:** Pure ECS (GPU resources don't benefit from archetype storage), pure custom (loses flexibility for scene management), full scene graph (deep trees unnecessary for SDF engine).

A thin ECS (hecs) manages the scene — transforms, hierarchy, visibility, metadata. GPU-heavy resources (brick pool, radiance volume, light buffers) are engine-global systems, not ECS components.

```
ECS World:
  Entity = ID + components:
    Transform { position: WorldPosition, rotation: Quat, scale: f32 }
    WorldTransform { matrix: Mat4 }     — camera-relative, computed each frame
    ChunkRef { chunk_id }               — links to streaming chunk
    SdfObject { brick_range, tier }     — links to brick pool region
    AnimatedCharacter { skeleton_id, segment_ranges, joint_ranges }
    Light { kind, color, intensity, range, cookie_id }
    FogVolume { brick_range, settings }
    Camera { projection, exposure }
    ParticleEmitter { emitter_settings }
    RigidBody { rapier_handle }
    EditorMetadata { name, tags, locked }
    Parent { entity, bone_index }

Engine-Global (NOT in ECS):
    BrickPool         — shared GPU buffer, central coordinator
    SpatialIndex      — per-LOD sparse grids
    MaterialTable     — storage buffer
    RadianceVolume    — GI clipmap
    LightBuffer       — packed for tiled culling
    VolumetricShadow  — 3D texture
    CloudShadow       — 2D texture
    ParticleBuffer    — GPU particle storage
    PostProcessState  — history buffers, bloom chain
    UpscaleState      — DLSS context or custom history
    RapierContext     — physics world
```

**Uniform scale only** — non-uniform scale distorts SDF distances. If an artist needs a stretched object, it must be re-voxelized at the new proportions.

### Decision: Scene Graph — Shallow Hierarchy

**Chosen over:** Flat (no hierarchy — can't attach weapon to hand), full scene tree (unnecessary depth, complex propagation).

Max depth of 2-3. Parent-child links for attachment (weapon → hand bone, particle emitter → entity). No deep trees.

```rust
struct Parent {
    entity: Entity,
    bone_index: Option<u8>,  // if attached to an animated bone
}
```

**Transform update pass (each frame):**
```
1. For all root entities (no Parent): WorldTransform = camera_relative(Transform)
2. For all children: WorldTransform = parent.WorldTransform * local_transform()
3. If bone_index set: WorldTransform = parent.WorldTransform * bone_matrices[bone_index] * local_transform()
```

Two iterations (roots then children) covers depth-2. A third pass handles depth-3 if ever needed.

### Decision: Resource Management — Three-Tier Lifecycle

| Tier | Lifetime | Examples | Management |
|------|----------|----------|------------|
| **Permanent** | App lifetime | Brick pool, spatial index, radiance volume, G-buffer, material table, noise textures | Allocated at startup, sized by quality preset |
| **Per-scene** | Scene load/unload | Cloud shadow map, weather textures, LUT textures, fluid sim grids, cookie textures | Created on scene load, released on unload |
| **Transient** | Per-frame | Staging buffers, readback buffers, temp compute outputs | Ring buffer or per-frame allocator |

**Brick pool orchestration — central BrickPoolManager:**

```rust
struct BrickPoolManager {
    pool: GpuBuffer,                        // the actual brick data on GPU
    free_list: Vec<u32>,                    // CPU-side free brick indices
    lru: LruCache<BrickId, u32>,            // brick → pool slot, for eviction
    pending_uploads: VecDeque<BrickUpload>, // async staging → pool copies
    pending_evictions: Vec<u32>,            // slots to reclaim this frame

    // Sub-pools with reserved ranges (guaranteed available)
    animation_reserved: Range<u32>,         // for joint rebaking workspace
    editing_reserved: Range<u32>,           // for edit scratch space
    // Remainder: streaming + static
}
```

Coordinates streaming (async disk → staging → pool), animation (joint rebaking writes), editing (CSG writes), and eviction (LRU when pool full).

**Asset handles — generational typed indices:**

```rust
struct Handle<T> {
    index: u32,
    generation: u32,
    _marker: PhantomData<T>,
}
```

Generational indices catch use-after-free. Typed handles prevent mixing up asset types at compile time.

### Decision: Frame Scheduling — Static Pass Order

**Chosen over:** Dynamic render graph (over-engineered for a fixed pipeline). Each pass is a self-contained function. Upgrading to a dynamic graph later is straightforward — same functions, wrapped in graph nodes.

**Frame execution:**

```rust
fn execute_frame(ctx: &mut FrameContext) {
    // Animation (if any characters dirty)
    pass_animation_rebake(ctx);

    // Particle simulation
    pass_particle_simulate(ctx);

    // Primary ray march
    pass_ray_march(ctx);

    // Independent passes (no conflicts, submit together)
    pass_tile_light_cull(ctx);
    pass_radiance_inject(ctx);
    pass_volumetric_shadow_map(ctx);
    pass_cloud_shadow_map(ctx);

    // Depends on radiance inject
    pass_radiance_mip_gen(ctx);

    // Depends on tile lights + radiance mips + G-buffer
    pass_shade(ctx);

    // Depends on vol shadow map + cloud shadow
    pass_volumetric_march(ctx);    // includes volumetric particles
    pass_volumetric_upscale(ctx);

    // Composite volumetrics onto shaded HDR
    pass_volumetric_composite(ctx);

    // Pre-upscale post-processing
    pass_bloom_extract_and_blur(ctx);
    if ctx.settings.dof_enabled { pass_depth_of_field(ctx); }
    if ctx.settings.motion_blur_enabled { pass_motion_blur(ctx); }

    // Upscale
    pass_upscale(ctx);  // dispatches to DLSS or custom internally
    if ctx.upscale_backend == Custom { pass_edge_sharpen(ctx); }

    // Post-upscale
    pass_bloom_composite(ctx);
    pass_tone_map(ctx);
    if ctx.settings.color_grade_lut.is_some() { pass_color_grade(ctx); }
    if ctx.settings.vignette_enabled { pass_vignette(ctx); }

    // Screen-space particles (depth-tested overlay)
    pass_screen_particles(ctx);

    // SDF micro-object rendering handled in ray march pass

    // Present (to editor viewport texture or swapchain)
    pass_present(ctx);
}
```

**Dependency graph:**
```
  ┌─ Animation Rebake ──┐
  ├─ Particle Simulate ──┤
  │                       ▼
  │  1. Ray March ──────────────────────────────┐
  │     → G-buffer                               │
  │                                              ▼
  │  2. Tile Light Cull ───────────────► 5. Shade ──► 7. Vol Composite ──► 8. Pre-PP
  │     → tile light lists               ▲  ▲                                  │
  │                                      │  │                                  ▼
  │  3. Radiance Inject ─────────────────┘  │                            9. Upscale
  │  4. Radiance Mip Gen ──────────────────┘                                  │
  │                                                                           ▼
  │  6a. Vol Shadow Map ─┐                                              10. Sharpen
  │  6b. Cloud Shadow ───┤                                                    │
  │  6c. Vol March ──────┤→ 6d. Vol Upscale ──► 7. Vol Composite        11. Post-PP
  │                                                                           │
  └───────────────────────────────────────────────────────────────────── 12. Present
```

### Decision: CPU-GPU Synchronization — Double-Buffered Uniforms

```
Frame N:
  CPU: [Input] [Update ECS] [Cull] [Build commands] [Submit] [Begin N+1]
  GPU:                                                [Execute passes 1-12]
```

Two sets of uniform buffers. CPU writes frame N+1 while GPU executes frame N. wgpu manages synchronization via `queue.submit()`.

**Async operations (don't block the frame):**

| Operation | Pipeline |
|-----------|----------|
| Chunk streaming | I/O thread → decompress → staging buffer → `queue.write_buffer` → brick pool |
| Edit delta readback | GPU → staging → `buffer.map_async` → CPU undo stack |
| Auto-exposure | GPU luminance reduction → readback → next frame's exposure |

```rust
struct AsyncWork {
    chunk_loads: Vec<ChunkLoadTask>,      // disk I/O futures
    brick_uploads: Vec<StagingUpload>,    // staging → pool copies
    readbacks: Vec<ReadbackTask>,         // GPU → CPU async maps
}
```

### Decision: Scene Manager — `.rkscene` Format with Additive Loading

A scene is a self-contained world definition: terrain chunks, placed entities, lights, environment settings, spawn points.

**Scene file format — `.rkscene` (RON serialization):**

```rust
struct Scene {
    // Metadata
    name: String,
    version: u32,

    // World bounds
    world_bounds: AABB,
    chunk_manifest: Vec<ChunkRef>,

    // Placed entities
    entities: Vec<EntityDef>,

    // Environment
    sun: DirectionalLight,
    ambient: AmbientSettings,
    fog: FogSettings,
    clouds: CloudSettings,
    atmosphere: AtmosphereSettings,
    post_process: PostProcessSettings,

    // Gameplay
    spawn_points: Vec<SpawnPoint>,
    triggers: Vec<TriggerVolume>,
}

struct ChunkRef {
    coords: IVec3,
    rkf_path: PathBuf,          // relative path to .rkf file
    has_journal: bool,           // edits.rkj exists
}

struct EntityDef {
    kind: EntityKind,            // StaticObject, AnimatedCharacter, Light, FogVolume, Emitter, ...
    transform: Transform,
    parent: Option<EntityRef>,
    components: Vec<ComponentDef>,
}
```

**Scene lifecycle:**
```
Load:
  1. Parse .rkscene → Scene struct
  2. Register chunks with streaming system (don't load yet)
  3. Spawn entities into ECS
  4. Apply environment settings
  5. Streaming system begins loading chunks near camera

Unload:
  1. Despawn all entities
  2. Flush streaming system (evict all chunks)
  3. Release per-scene GPU resources
  4. Reset environment to defaults

Save (editor):
  1. Serialize current ECS state → Scene struct
  2. Write .rkscene file
  3. Chunk .rkf files saved independently
```

**Additive loading (interiors/dungeons):**
Load secondary scenes additively — merge chunks (non-overlapping coords), spawn entities tagged with `scene_id` for later removal, blend environment settings (interior overrides outdoor in a volume).

### Decision: Asset Streaming and Management

**Asset registry — central handle-based manager:**

```rust
struct AssetRegistry {
    chunks: SlotMap<ChunkHandle, ChunkAsset>,
    characters: SlotMap<CharacterHandle, CharacterAsset>,
    textures: SlotMap<TextureHandle, TextureAsset>,

    pending_loads: Vec<LoadTask>,
    load_thread_pool: ThreadPool,
}

struct ChunkAsset {
    state: AssetState,              // Unloaded, Loading, Loaded, Error
    coords: IVec3,
    rkf_path: PathBuf,
    brick_slots: Vec<u32>,
    has_journal: bool,
    journal_applied: bool,
}

enum AssetState {
    Unloaded,
    Loading { progress: f32 },
    Loaded,
    Error(String),
}
```

**Async loading pipeline:**
```
1. Request: streaming system or entity spawn requests an asset
2. I/O thread: read .rkf from disk, LZ4 decompress bricks (CPU thread pool)
3. Staging: write decompressed bricks to wgpu staging buffer
4. Upload: queue.write_buffer or staging → brick pool copy (GPU)
5. Activate: update spatial index, mark asset as Loaded
6. If .rkj exists: replay edit journal against loaded bricks (GPU compute)
```

Steps 2-3 on background threads. Steps 4-6 on main thread during the frame's upload window.

**Reference counting and eviction:** Assets track referencing entities. Refcount zero + outside streaming radius = eviction-eligible after a grace period (prevents thrashing at stream boundaries).

**Memory budget enforcement:**
```rust
struct StreamingBudget {
    max_brick_pool_mb: u32,         // e.g., 512MB
    max_pending_uploads_mb: u32,    // staging buffer cap, e.g., 64MB
    max_io_bandwidth_mb_s: f32,     // throttle disk I/O, e.g., 200 MB/s
    eviction_headroom_pct: f32,     // start evicting at 90% full
}
```

### Decision: Editor Architecture — rinch UI, Engine as Library

The runtime engine is a Rust library crate. The editor and game are binary crates that depend on it.

**Crate structure:**
```
rkifield/
  crates/
    rkf-core/        — voxel data structures, brick pool, spatial index, WorldPosition
    rkf-render/      — ray march, shading, volumetrics, upscale, post-process
    rkf-animation/   — skeletal animation, blend shapes
    rkf-edit/        — CSG, brushes, undo/redo, edit journal
    rkf-import/      — mesh-to-SDF conversion (used by rkf-convert CLI)
    rkf-physics/     — Rapier integration, SDF collision adapter, character controller
    rkf-particles/   — particle system, emitters, GPU simulation
    rkf-runtime/     — frame scheduling, ECS integration, streaming, asset management
    rkf-editor/      — editor UI, gizmos, tools (binary)
    rkf-convert/     — CLI asset converter (binary)
    rkf-game/        — example game / playground (binary)
```

**Editor as superset of runtime:**
```
┌──────────────────────────────────────────┐
│  Editor Application (rkf-editor)          │
│  ┌────────────────────────────────────┐  │
│  │  Runtime Engine (rkf-runtime)       │  │
│  │  - Frame pipeline                   │  │
│  │  - Brick pool                       │  │
│  │  - Streaming, animation, physics    │  │
│  └────────────────────────────────────┘  │
│  + Editor UI (rinch, wgpu backend)        │
│  + Gizmos (wireframe overlay pass)        │
│  + Undo/Redo manager                      │
│  + Asset browser                          │
│  + Property inspector                     │
│  + Sculpt/paint tools                     │
│  + Scene serialization                    │
└──────────────────────────────────────────┘
```

**rinch integration with wgpu:**
Engine renders to an offscreen texture (not the swapchain). rinch composites the viewport texture + UI panels onto the swapchain. Input routing: clicks in the viewport go to the engine camera/tools; clicks on panels go to rinch.

```rust
struct EditorViewport {
    render_texture: wgpu::Texture,
    render_texture_view: wgpu::TextureView,
    size: UVec2,                        // tracks panel resize
    camera: EditorCamera,
    input_focused: bool,                // true when mouse is over viewport
}
```

**Editor modes:**

| Mode | Primary Action | Tools Active |
|------|----------------|--------------|
| **Navigate** | Orbit/fly camera | — |
| **Select** | Pick entities, multi-select | Transform gizmo |
| **Place** | Instantiate from asset browser | Snap, grid align |
| **Sculpt** | Modify voxel terrain | CSG brushes |
| **Paint** | Material/color painting | Paint brushes |
| **Light** | Place and tune lights | Light gizmos, cookie preview |
| **Animate** | Preview animations, pose | Timeline, bone selection |
| **Environment** | Tune fog, clouds, atmosphere, post-process | Live parameter sliders |

**Editor camera:**
```rust
struct EditorCamera {
    mode: CameraMode,
    target: Vec3,           // orbit center (camera-relative)
    distance: f32,          // orbit radius
    yaw: f32,
    pitch: f32,
    fly_speed: f32,
    position: WorldPosition, // full-precision world position
}

enum CameraMode {
    Orbit,      // rotate around target, scroll to zoom
    Fly,        // WASD + mouse look (hold right-click)
    Follow,     // track selected entity
}
```

**Gizmo rendering:**
A dedicated wireframe overlay pass after post-processing draws transform gizmos, selection outlines, light shapes, volume wireframes, grid plane, and bone visualizations. Uses line rasterization — the one place we use traditional rendering (debug visualization, not geometry).

**Unified undo stack:**

```rust
enum UndoAction {
    TransformEntity { entity: Entity, prev: Transform },
    SpawnEntity { entity: Entity },
    DespawnEntity { serialized: EntityDef },
    VoxelEdit { delta: EditDelta },
    PropertyChange { entity: Entity, component: TypeId, prev: Box<dyn Any> },
    EnvironmentChange { prev: EnvironmentSnapshot },
}
```

**Debug visualizations (editor-only):**
Normals, material IDs, LOD levels, brick occupancy, radiance volume slices, physics colliders, particle bounds — toggled via editor view menu.

### Decision: Configuration — Preset-Based with Per-System Overrides

```rust
struct EngineConfig {
    display_resolution: [u32; 2],
    render_scale: f32,              // 0.25 to 1.0
    vsync: bool,
    quality_preset: QualityPreset,

    // Per-system overrides
    ray_march: RayMarchSettings,
    shading: ShadingSettings,
    volumetrics: VolumetricSettings,
    upscale: UpscaleSettings,
    post_process: PostProcessSettings,
    clouds: CloudSettings,
    fog: FogSettings,
    gi: GiSettings,
    particles: ParticleSettings,
    physics: PhysicsSettings,
    streaming: StreamingBudget,
}

enum QualityPreset {
    Low,      // 1/4 res, 16 vol steps, GI off, clouds off
    Medium,   // 1/3 res, 32 vol steps, GI 32³, procedural clouds
    High,     // 1/2 res, 48 vol steps, GI 64³, full clouds
    Ultra,    // 3/4 res, 64 vol steps, GI 128³, full clouds + brick clouds
    Custom,   // all manual
}
```

Presets set sensible defaults. Individual settings override at runtime. Serializable (RON) for user config files.

**Hot-reload targets:** material table, WGSL shader source (recompile pipelines), quality settings, post-process LUTs, color grading.

### Session 10 Summary: All Engine Architecture Decisions

| Decision | Choice | Notes |
|----------|--------|-------|
| Coordinate system | Right-handed, Y-up, meters, -Z forward | wgpu convention |
| Chunk addressing | IVec3 × 8m | Vertical chunks supported |
| Float precision | WorldPosition (IVec3 chunk + Vec3 local) | Camera-relative f32 on GPU, f64 math on CPU |
| World range | ±17 billion meters (i32 chunks × 8m) | ~0.1 AU |
| App architecture | Hybrid ECS (hecs) + global GPU systems | ECS for scene, globals for rendering |
| Scene graph | Shallow hierarchy (depth 2-3) | Uniform scale only |
| Resource lifecycle | 3-tier: permanent, per-scene, transient | Ring buffer for transient |
| Brick pool | Central BrickPoolManager with reserved sub-pools | Animation + editing reserved |
| Asset handles | Generational typed indices | Compile-time type safety |
| Frame scheduling | Static pass order, self-contained pass functions | Upgrade path: dynamic render graph |
| CPU-GPU sync | Double-buffered uniforms, async streaming/readback | — |
| Scene format | `.rkscene` (RON), additive loading | — |
| Asset streaming | Handle-based registry, async I/O thread pool | Budget-enforced with LRU eviction |
| Crate structure | 11 crates (core, render, animation, edit, import, physics, particles, runtime, editor, convert, game) | Engine is a library |
| Editor UI | rinch (wgpu backend) | Engine renders to offscreen texture |
| Editor modes | 8 modes (navigate, select, place, sculpt, paint, light, animate, environment) | — |
| Gizmos | Wireframe overlay pass (line rasterization) | — |
| Undo | Unified stack with typed actions including voxel deltas | — |
| Config | Preset-based with per-system overrides, RON serializable | Hot-reload for materials, shaders, LUTs |
