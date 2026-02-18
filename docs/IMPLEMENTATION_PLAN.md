# RKIField — Implementation Plan

Detailed, step-by-step plan for implementing the RKIField SDF engine. Each phase produces a testable milestone. Phases are sequential — each builds on the previous — but tasks within a phase can often be parallelized.

Reference: [ARCHITECTURE.md](./ARCHITECTURE.md) for all design decisions.

---

## Phase 0: Project Scaffolding

**Milestone:** Cargo workspace compiles, all crates exist (empty), CI passes.

### Tasks

0.1. **Create Cargo workspace**
  - `rkifield/Cargo.toml` with `[workspace]` members
  - Workspace-level dependencies (shared versions)

0.2. **Create crate stubs (lib crates)**
  - `crates/rkf-core/` — voxel data structures, brick pool, spatial index, WorldPosition
  - `crates/rkf-render/` — ray march, shading, volumetrics, upscale, post-process
  - `crates/rkf-animation/` — skeletal animation, blend shapes
  - `crates/rkf-edit/` — CSG, brushes, undo/redo, edit journal
  - `crates/rkf-import/` — mesh-to-SDF conversion library
  - `crates/rkf-physics/` — Rapier integration, SDF collision, character controller
  - `crates/rkf-particles/` — particle system, emitters, GPU simulation
  - `crates/rkf-runtime/` — frame scheduling, ECS, streaming, asset management

0.3. **Create crate stubs (binary crates)**
  - `crates/rkf-editor/` — editor application
  - `crates/rkf-convert/` — CLI asset converter
  - `crates/rkf-game/` — example game / playground

0.4. **Add workspace dependencies**
  - `wgpu`, `winit`, `glam`, `bytemuck` (or `encase`), `hecs`, `rapier3d`, `log`, `env_logger`
  - `ron`, `serde`, `serde_derive` for serialization
  - `lz4_flex` for compression
  - Dev dependencies: standard test framework

0.5. **Create test harness binary**
  - `crates/rkf-testbed/` — permanent binary for visual testing and minimal repro
  - Opens a window, runs the render pipeline, displays results
  - Used heavily in Phases 1-11, remains useful for isolated testing and debugging

0.6. **CI configuration**
  - `cargo check --workspace`, `cargo test --workspace`, `cargo clippy --workspace`
  - Run on push (GitHub Actions or similar)

0.7. **Progress tracking script**
  - `scripts/progress.sh` — parses `git log` for `phase-N: N.X` commit prefixes
  - Cross-references against task list in `IMPLEMENTATION_PLAN.md`
  - Outputs per-phase completion status (e.g., `Phase 2: 6/8 tasks`)
  - This is the sole source of truth for progress — no manual checklists

### Done when
- `cargo build --workspace` succeeds
- `cargo test --workspace` succeeds (no tests yet, but compiles)
- All 13 crates exist with basic `lib.rs` / `main.rs`
- `scripts/progress.sh` runs and reports 0/N tasks for all phases

---

## Phase 1: MCP Foundation

**Milestone:** `rkf-mcp` binary connects to testbed via IPC, agents can take screenshots and query basic engine state.

**Crate:** `rkf-mcp`

### Tasks

1.1. **Create `rkf-mcp` crate stub**
  - Binary crate in `crates/rkf-mcp/`
  - Dependencies: `serde`, `serde_json`, `tokio`, `jsonrpsee` (or similar JSON-RPC crate)
  - Basic `main.rs` with argument parsing (`--mode editor|debug`, `--connect <socket_path>`)

1.2. **AutomationApi trait**
  - Define in `rkf-runtime` (or `rkf-core` initially): `trait AutomationApi: Send + Sync`
  - Observation methods: `screenshot()`, `scene_graph()`, `entity_inspect()`, `render_stats()`, `read_log()`, `camera_state()`
  - Mutation methods: `entity_spawn()`, `entity_despawn()`, `entity_set_component()`, `material_set()`, `brush_apply()`, `scene_load()`, `scene_save()`, `camera_set()`, `execute_command()`
  - Stub implementation returning `Err("not yet implemented")` for each method
  - Unit tests: trait is object-safe, stub compiles

1.3. **Tool discovery registry**
  - `ToolDefinition` struct: name, description, category, parameters (JSON Schema-compatible), return type, mode (Editor/Debug/Both)
  - `ToolRegistry`: `register()`, `list_tools(mode)`, `call(name, params)`
  - `ToolHandler` trait: `fn call(&self, api: &dyn AutomationApi, params: Value) -> Result<Value>`
  - Unit tests: register a tool, list it, call it

1.4. **MCP protocol layer (JSON-RPC 2.0)**
  - Implement MCP `initialize`, `tools/list`, `tools/call` handlers
  - `tools/list` generated dynamically from `ToolRegistry`
  - `tools/call` dispatches to registered `ToolHandler`
  - Unit tests: JSON-RPC request/response roundtrip

1.5. **IPC communication**
  - Engine side: open Unix socket listener at `/tmp/rkifield-{pid}.sock`
  - `rkf-mcp` side: connect to socket, send/receive JSON-RPC messages
  - Fallback: localhost TCP on configurable port
  - Integration test: connect, send `tools/list`, receive response

1.6. **Built-in observation tools (stubs)**
  - Register stub tools: `screenshot`, `scene_graph`, `entity_inspect`, `render_stats`, `log_read`, `camera_get`
  - Each calls the corresponding `AutomationApi` method
  - Returns placeholder data (tools become functional as engine features are built in later phases)

1.7. **Testbed MCP integration**
  - `rkf-testbed` starts with `--mcp` flag → opens IPC listener
  - Implements `AutomationApi` with real data as available (initially just stubs)
  - Agent can connect, list tools, call stubs

1.8. **Mode filtering**
  - `--mode editor`: expose all tools (observation + mutation)
  - `--mode debug`: expose only observation tools
  - `list_tools()` filters by mode
  - Test: verify debug mode hides mutation tools

### Done when
- `rkf-mcp` binary compiles and connects to testbed via Unix socket
- `tools/list` returns registered tools with full metadata
- `tools/call` dispatches to tool handlers and returns results
- Mode filtering correctly hides mutation tools in debug mode
- Tool discovery is extensible (adding a tool = implement trait + register)

---

## Phase 2: Core Data Types

**Milestone:** All foundational types defined, tested, and documented. No GPU code yet.

**Crate:** `rkf-core`

### Tasks

2.1. **WorldPosition type**
  - `WorldPosition { chunk: IVec3, local: Vec3 }`
  - `normalize()` — re-centers local into 0..CHUNK_SIZE
  - `relative_to(&self, origin: &WorldPosition) -> Vec3` — f64 subtraction, f32 result
  - `distance_f64(&self, other: &WorldPosition) -> f64`
  - Arithmetic: `translate`, `from_world_f64`
  - `Serde` derive for serialization
  - Unit tests: normalization, precision at large distances, relative_to symmetry

2.2. **Constants and configuration**
  - `CHUNK_SIZE: f32 = 8.0`
  - `BRICK_DIM: u32 = 8` (8×8×8 = 512 voxels)
  - Resolution tiers: `Tier { voxel_size: f32, brick_extent: f32 }`
  - Tier table: `[0.5cm, 2cm, 8cm, 32cm]`
  - `MAX_MATERIALS: u32 = 65536`

2.3. **VoxelSample type**
  - `VoxelSample { word0: u32, word1: u32 }` — 8 bytes packed
  - Accessor methods: `distance() -> f16`, `material_id() -> u16`, `blend_weight() -> u8`, `secondary_id() -> u8`, `flags() -> u8`
  - Constructor: `new(distance: f32, material_id: u16, ...)`
  - `bytemuck::Pod` derive for GPU upload
  - Unit tests: pack/unpack roundtrip, edge cases (max distance, zero blend)

2.4. **Brick type**
  - `Brick { voxels: [VoxelSample; 512] }` — 4KB
  - Index helper: `index(x, y, z) -> usize` for 8×8×8
  - `sample(x, y, z) -> VoxelSample`
  - `set(x, y, z, sample: VoxelSample)`
  - `bytemuck::Pod` derive

2.5. **Companion brick types**
  - `BoneBrick { data: [BoneVoxel; 512] }` — 4KB (bone indices + weights)
  - `VolumetricBrick { data: [VolumetricVoxel; 512] }` — 2KB (density + emission)
  - `ColorBrick { data: [ColorVoxel; 512] }` — 2KB (RGBI)
  - `bytemuck::Pod` for all

2.6. **Material struct**
  - `Material` — 96 bytes, matching ARCHITECTURE.md exactly
  - All PBR + SSS + noise fields
  - `bytemuck::Pod` derive
  - Default material (gray, roughness 0.5, non-metallic)

2.7. **CellState enum**
  - `EMPTY = 0`, `SURFACE = 1`, `INTERIOR = 2`, `VOLUMETRIC = 3`
  - Stored as 2 bits in occupancy bitfields

2.8. **AABB and BoundingBox utilities**
  - `AABB { min: Vec3, max: Vec3 }`
  - Intersection, containment, expand, center, size
  - WorldPosition-aware AABB variant for chunk-scale bounds

### Done when
- All types compile and serialize correctly
- `cargo test -p rkf-core` passes with comprehensive unit tests
- Types match the architecture document byte-for-byte

---

## Phase 3: Brick Pool + Spatial Index

**Milestone:** Can allocate bricks, fill with SDF data, and query the spatial index. CPU-only.

**Crate:** `rkf-core`

### Tasks

3.1. **BrickPool (CPU-side)**
  - `BrickPool { bricks: Vec<Brick>, free_list: Vec<u32>, capacity: u32 }`
  - `allocate() -> Option<u32>` — pop from free list
  - `deallocate(slot: u32)` — push to free list
  - `get(slot: u32) -> &Brick`, `get_mut(slot: u32) -> &mut Brick`
  - Companion pools: `BonePool`, `VolumetricPool`, `ColorPool` (same pattern)
  - Initialize with configurable capacity

3.2. **SparseGrid (single LOD)**
  - Level 2: occupancy bitfield — `Vec<u32>` bitfield, one bit per coarse cell
  - Level 1: block index — `Vec<u32>` mapping occupied cells → brick pool slots
  - `dimensions: UVec3` — grid size in bricks
  - `cell_state(x, y, z) -> CellState` — read 2-bit state from Level 2
  - `set_cell_state(x, y, z, state: CellState)`
  - `brick_slot(x, y, z) -> Option<u32>` — Level 1 lookup
  - `set_brick_slot(x, y, z, slot: u32)`

3.3. **SDF generation utilities (for testing)**
  - `fn sphere_sdf(center: Vec3, radius: f32, point: Vec3) -> f32`
  - `fn box_sdf(half_extents: Vec3, point: Vec3) -> f32`
  - `fn voxelize_sdf(sdf_fn, tier, aabb) -> (SparseGrid, Vec<Brick>)` — voxelize an analytic SDF into bricks
  - Fill narrow band (±3 bricks), mark INTERIOR, leave rest EMPTY

3.4. **BrickPool integration with SparseGrid**
  - `fn populate_grid(pool: &mut BrickPool, grid: &mut SparseGrid, sdf_fn, tier)` — allocate bricks, fill voxels, update grid
  - Test: voxelize a sphere, verify brick count matches expected narrow band

3.5. **Trilinear sampling (CPU reference implementation)**
  - `fn sample_brick_trilinear(brick: &Brick, local_pos: Vec3) -> f32` — interpolate distance
  - `fn sample_brick_nearest_material(brick: &Brick, local_pos: Vec3) -> u16` — nearest material ID
  - Tests: sample at voxel centers equals stored value, interpolation is smooth

### Done when
- Can voxelize analytic SDFs into the brick pool + sparse grid
- Trilinear sampling produces smooth distance values
- All CPU-side data structures match architecture byte layouts
- Ready for GPU upload

---

## Phase 4: First Pixels

**Milestone:** A window opens showing a ray-marched SDF sphere. Ugly, unlit, single color — but pixels on screen.

**Crates:** `rkf-render`, `rkf-testbed`

### Tasks

4.1. **wgpu device initialization**
  - Request adapter, device, queue
  - Configure surface for window presentation
  - Create `RenderContext` holding device, queue, surface config

4.2. **winit window + event loop**
  - Open window (1920×1080 or configurable)
  - Handle resize, close, basic input
  - Frame loop: poll events → update → render → present

4.3. **GPU buffer upload**
  - Upload `BrickPool.bricks` as `storage` buffer
  - Upload `SparseGrid` (Level 2 bitfield + Level 1 block index) as `storage` buffers
  - Upload camera uniforms as `uniform` buffer
  - Create bind group layout + bind groups

4.4. **Camera system (basic)**
  - `Camera { position: Vec3, forward: Vec3, right: Vec3, up: Vec3 }`
  - `CameraUniforms` struct (bytemuck) for GPU upload
  - Pinhole projection (no jitter yet)
  - Keyboard/mouse fly camera for testing

4.5. **Ray march compute shader (basic)**
  - WGSL compute shader: `@workgroup_size(8, 8, 1)`
  - One thread per pixel at internal resolution (960×540)
  - Generate ray from camera uniforms
  - Simple sphere tracing: step by SDF distance, no DDA yet
  - On hit: write white pixel; on miss: write black
  - Output to `texture_storage_2d<rgba8unorm, write>`

4.6. **Blit to screen**
  - Full-screen render pass (or compute copy) from output texture to swapchain
  - Handle resolution mismatch (internal → display)

4.7. **Test scene: single sphere**
  - Use Phase 3's `voxelize_sdf` to create a sphere
  - Upload to GPU
  - Position camera looking at sphere

### Done when
- Window opens, white sphere visible on black background
- Can fly camera around the sphere
- Frame time displayed in title bar
- Screenshot matches expected output

---

## Phase 5: Multi-Level DDA

**Milestone:** Efficient traversal through sparse scenes. Multiple objects render correctly. Empty space is skipped.

**Crate:** `rkf-render`

### Tasks

5.1. **DDA traversal in WGSL**
  - Implement 3D DDA (Amanatides & Woo) for Level 2 grid
  - Step through coarse cells, check occupancy bits
  - On occupied cell: DDA through Level 1 within that cell
  - On occupied brick: enter sphere tracing (Phase 4 shader)

5.2. **Cell state handling in shader**
  - `EMPTY`: DDA skip (large step to cell boundary)
  - `SURFACE`: sphere trace within brick bounds
  - `INTERIOR`: skip to far side of cell (ray exits interior)
  - `VOLUMETRIC`: flag for later (skip for now)

5.3. **Brick bounds tracking**
  - When DDA enters a brick, compute local ray origin/direction within brick
  - Track when ray exits brick bounds → return to DDA
  - Handle brick-to-brick transitions

5.4. **LOD level transitions (stub)**
  - For now: single LOD level (LOD 0 everywhere)
  - Add `current_lod` tracking to ray state
  - Placeholder for clipmap switching (Phase 13)

5.5. **Test scene: multiple objects**
  - Voxelize sphere + box + capsule at different positions
  - Verify all render correctly
  - Profile: compare step counts with vs without DDA

### Done when
- Multiple SDF objects render with correct occlusion
- Frame time is measurably better than naive sphere tracing
- Step count per ray is logged and reasonable (most rays < 50 steps)

---

## Phase 6: G-Buffer + Basic Shading

**Milestone:** Lit scene with materials, normals, and a single directional light. Looks like a real render.

**Crate:** `rkf-render`

### Tasks

6.1. **G-buffer textures**
  - Target 0: `Rgba32Float` — position.xyz + hit_distance
  - Target 1: `Rgba16Float` — normal.xyz + blend_weight
  - Target 2: `Rg16Uint` — material_id + secondary_id_and_flags
  - Target 3: `Rg16Float` — motion_vector.xy (zeros for now)
  - All at internal resolution (960×540)

6.2. **Ray march → G-buffer output**
  - Modify ray march shader to write G-buffer instead of color
  - Normal computation: central differences (6 SDF samples)
  - Material ID from nearest-neighbor brick sampling
  - Sky pixels: hit_distance = MAX_FLOAT

6.3. **Material table GPU buffer**
  - Upload `Material` array as `storage` buffer
  - Create 4-5 test materials: stone, metal, wood, emissive, skin

6.4. **Shading compute shader**
  - Read G-buffer, look up material from table
  - Implement Cook-Torrance GGX BRDF:
    - GGX normal distribution (D term)
    - Schlick Fresnel (F term)
    - Smith geometry (G term)
    - Lambertian diffuse
  - Single directional light (sun) with hardcoded direction/color
  - Output: HDR color texture (`Rgba16Float`)

6.5. **Tone mapping (minimal)**
  - Simple Reinhard or ACES tone map (compute shader)
  - HDR → LDR for display
  - Will be replaced by full post-process stack in Phase 10

6.6. **Test scene: materials showcase**
  - Objects with different materials (rough stone, shiny metal, emissive)
  - Sun casting light from above-left
  - Verify PBR looks correct (metal reflects, rough surfaces diffuse)

### Done when
- Scene is lit with correct PBR shading
- Different materials are visually distinct
- Normals are smooth (no visible staircase artifacts)
- Frame time is reasonable at 960×540

---

## Phase 7: Full Shading

**Milestone:** Multiple lights, shadows, AO, SSS. Scene looks production-quality under direct lighting.

**Crate:** `rkf-render`

### Phase 6 Verification Results

Debug visualization (modes 0-5 via MCP `debug_mode` tool) confirms Phase 6 G-buffer
and shading pipeline are correct:

- **Normals:** Correct — smooth gradients on curved surfaces, sharp edges on box
- **World positions:** Correct — color gradients match known object positions
- **Material IDs:** Correct — three distinct flat colors, no inter-object bleeding
- **Diffuse shading:** Correct — Lambertian gradient visible; kd=0 for metallic surfaces
- **Specular shading:** Near-zero on all objects (known limitation, see 7.0 below)
- **Emission:** Correct — emissive capsule bright after ACES tone mapping
- **Tone mapping:** Correct — ACES curve produces expected output

The flat/muted appearance of the Phase 6 test scene is entirely caused by the broken
specular term. The metal box (metallic=1.0, kd=0) shows only ambient because specular
is misdirected. The stone sphere's diffuse shading is correct. No G-buffer data issues.

### Tasks

7.0. **Fix view direction in shade.wgsl** _(Phase 6 prerequisite)_
  - Currently `normalize(-world_pos)` — incorrect, assumes camera at origin
  - Add camera position to shading uniforms (bind group 2 or extend scene uniforms)
  - Change to `normalize(camera_pos - world_pos)`
  - This unblocks specular highlights, and is required for correct shadows/AO/SSS

7.1. **SDF soft shadows**
  - Implement `soft_shadow()` in shade shader (ARCHITECTURE.md §Shading)
  - Penumbra from `k * d / t` ratio
  - MAX_SHADOW_STEPS = 64
  - Apply to directional light first

7.2. **SDF ambient occlusion**
  - Implement `sdf_ao()` — 6 samples along normal (ARCHITECTURE.md §Shading)
  - Exponentially decaying weights
  - Apply as multiplier on ambient/indirect term

7.3. **SSS (subsurface scattering)**
  - Implement `sss_contribution()` — SDF thickness probe (ARCHITECTURE.md §Shading)
  - Only evaluate for materials with `subsurface > 0`
  - Add test material: skin, wax, leaf

7.4. **Light buffer + tiled culling**
  - `Light` struct (64 bytes, GPU-aligned) matching architecture
  - Upload light array as storage buffer
  - Tile light culling compute shader:
    - 16×16 pixel tiles
    - Per-tile min/max depth from G-buffer
    - Test each light's bounding sphere against tile frustum
    - Write per-tile light index lists

7.5. **Point light evaluation**
  - Distance attenuation (inverse square, clamped at range)
  - SDF soft shadow for top-K shadow casters per tile
  - Add to shade pass

7.6. **Spot light evaluation**
  - Inner/outer angle falloff
  - Cookie texture sampling (texture_2d_array)
  - Create test cookie (window pattern, circular gradient)
  - Add to shade pass

7.7. **Shadow budget system**
  - Configurable K (shadow-casting lights per tile)
  - Rank lights by contribution (intensity / distance²)
  - Top K get SDF shadow rays, rest are unshadowed

7.8. **Test scene: lighting showcase**
  - Room with 20+ lights (mix of point, spot, directional)
  - Spot lights with cookies casting patterns
  - Soft shadows from nearby and distant occluders
  - SSS material (candle wax, character skin)
  - Verify no light leaking, correct falloff

### Done when
- Multiple light types render correctly
- Shadows are soft and physically plausible
- AO darkens crevices
- SSS shows translucency on thin geometry
- Tiled culling handles 50+ lights without frame drop

---

## Phase 8: Global Illumination

**Milestone:** Indirect lighting via voxel cone tracing. Color bleeding between surfaces. Emissive materials cast light.

**Crate:** `rkf-render`

### Tasks

8.1. **Radiance volume textures**
  - 4 levels of `texture_3d<rgba16float>` (128³ each)
  - Level 0: ~4cm voxels, ~5m radius
  - Levels 1-3: progressively coarser
  - Total: ~64MB VRAM

8.2. **Radiance injection compute shader**
  - Dispatch over occupied geometry bricks
  - Per surface voxel: compute position + normal (SDF gradient)
  - Evaluate direct illumination from sun + top N lights
  - Simplified shadow (fewer steps)
  - Emissive materials self-inject
  - Write radiance + opacity to Level 0

8.3. **Radiance mip generation**
  - 3 compute dispatches: downsample Level 0 → 1 → 2 → 3
  - Radiance: average of 8 children
  - Opacity: max of 8 children (conservative)

8.4. **Voxel cone tracing in shade pass**
  - Implement `trace_gi_cone()` — step through radiance volume, accumulate radiance × transmittance
  - 6 cones in cosine-weighted hemisphere around surface normal
  - Automatic mip selection from cone radius
  - Hardware trilinear filtering on 3D textures

8.5. **Integrate GI into shading**
  - `indirect = compute_indirect(pos, normal)`
  - Add to final color: `direct + indirect * material.albedo`
  - Energy conservation between direct and indirect terms

8.6. **Radiance volume camera-relative offset**
  - Volume follows camera (clipmap behavior)
  - Re-inject when camera moves significantly
  - Rolling update: only re-inject changed slices

8.7. **Test scene: GI showcase**
  - Cornell box (colored walls, white ceiling/floor)
  - Verify color bleeding (red wall → red tint on white floor)
  - Emissive object illuminating surroundings
  - Compare with/without GI

### Done when
- Color bleeding visible between surfaces
- Emissive materials cast visible indirect light
- No significant light leaking through walls
- GI cost is < 3ms at 960×540

---

## Phase 9: Upscaling

**Milestone:** Engine renders at low internal resolution and upscales to display resolution with temporal accumulation.

**Crate:** `rkf-render`

### Tasks

9.1. **Sub-pixel jitter**
  - Halton 2,3 sequence (16-frame cycle)
  - Apply jitter offset to camera ray generation
  - Pass jitter as uniform alongside camera data

9.2. **Motion vectors in G-buffer**
  - Per-pixel: reproject hit position through previous frame's VP matrix
  - Compute screen-space motion vector (current pixel - reprojected pixel)
  - Write to G-buffer target 3 (Rg16Float)
  - Store previous frame's VP matrix

9.3. **History buffers**
  - `history_color: Rgba16Float` at display resolution
  - `history_metadata: Rg32Uint` at display resolution (packed depth + material)
  - Ping-pong: current frame writes to history for next frame

9.4. **Custom temporal upscaler**
  - Compute shader at display resolution
  - Bilinear sample current frame from internal resolution
  - Reproject into history using motion vectors
  - Multi-signal rejection:
    - Material ID mismatch → reject
    - Depth discontinuity → reject
    - Normal discontinuity → reject
    - Neighborhood color clipping (3×3 AABB in YCoCg) → clamp
    - Motion magnitude → reduce trust
  - Blend: `mix(current, clipped_history, blend_factor)`
  - Write result to output + history

9.5. **Edge-aware sharpening**
  - 5×5 cross kernel weighted by material + depth similarity
  - Unsharp mask: `center + (center - blur) * strength`
  - Material ID boundaries create hard sharpening edges

9.6. **DLSS integration**
  - Add `dlss_wgpu` dependency
  - Detect DLSS availability at startup
  - Provide DLSS inputs: color, depth, motion vectors, exposure, jitter
  - Fall back to custom if unavailable

9.7. **Upscale backend selection**
  - `UpscaleBackend::DLSS` vs `UpscaleBackend::Custom`
  - Auto-select at startup, overridable in config
  - Both produce same output format (display-res HDR)

9.8. **Resolution configuration**
  - Quality modes: Ultra (1/1), Quality (3/4), Balanced (1/2), Performance (1/3), Potato (1/4)
  - Runtime-switchable
  - Resize all internal-resolution textures on change

### Done when
- Render at 1/2 resolution, upscale to 1080p with minimal quality loss
- Temporal accumulation shows effective super-sampling over 16 frames
- Material edges are sharp (no haloing)
- Moving camera shows no significant ghosting
- DLSS works on NVIDIA hardware (if available for testing)

---

## Phase 10: Post-Processing

**Milestone:** Full post-processing chain. Image looks cinematic.

**Crate:** `rkf-render`

### Tasks

10.1. **Post-process stack architecture**
  - `PostProcessStack { pre_upscale: Vec<Pass>, post_upscale: Vec<Pass> }`
  - Ping-pong buffer management
  - Enable/disable passes at runtime
  - Pass trait: `fn execute(&self, ctx: &PostProcessContext)`

10.2. **Bloom (pre-upscale)**
  - Extract: threshold HDR → bloom buffer
  - Downsample chain: 4 mip levels, 13-tap tent filter
  - Blur: separable Gaussian per mip (5-9 taps)
  - Store for composite after upscale

10.3. **Depth of field (pre-upscale)**
  - Circle of confusion from depth buffer + focus settings
  - Two-layer gather: near field + far field
  - Composite near over sharp

10.4. **Motion blur (pre-upscale)**
  - Per-pixel directional blur along motion vector
  - Sample count proportional to motion magnitude (max 16)

10.5. **Bloom composite (post-upscale)**
  - Bilinear upsample bloom from internal res
  - Additive blend onto upscaled HDR

10.6. **Auto-exposure**
  - HDR luminance histogram (compute reduction)
  - Async readback → next frame's exposure
  - Configurable min/max EV, adaptation speed

10.7. **Tone mapping (post-upscale)**
  - ACES fitted curve
  - AgX curve (better highlight preservation)
  - Selectable at runtime
  - Apply exposure before tone map

10.8. **Color grading (post-upscale)**
  - Load 3D LUT from image file (32³ or 64³)
  - Upload as `texture_3d`
  - Apply as final color transform

10.9. **Optional cosmetics (post-upscale)**
  - Vignette: radial darkening
  - Film grain: luminance-weighted noise
  - Chromatic aberration: radial RGB offset
  - All off by default, configurable

### Done when
- Full post-process chain runs
- Bloom creates soft glow on bright surfaces
- Tone mapping produces pleasing HDR → LDR conversion
- DoF and motion blur work when enabled
- Can load and apply color grading LUT

---

## Phase 11: Volumetrics

**Milestone:** Fog, god rays, clouds. Atmospheric scene rendering.

**Crate:** `rkf-render`

### Tasks

11.1. **Volumetric shadow map**
  - 3D texture (256³) covering camera-centered volume
  - Compute shader: per texel, march toward sun, accumulate transmittance
  - Update once per frame

11.2. **Volumetric march shader**
  - Half internal resolution (480×270)
  - Fixed-step front-to-back compositing
  - Jittered step offset (interleaved gradient noise)
  - Sample density from all sources:
    - Brick-backed local fog volumes
    - Analytic height fog
    - Analytic distance fog
    - Ambient dust (for god rays)

11.3. **In-scattering computation**
  - Sun visibility from volumetric shadow map
  - Henyey-Greenstein phase function (per-volume g parameter)
  - Secondary lights from radiance volume opacity
  - Emission from volumetric brick pool

11.4. **Analytic fog**
  - Height fog: exponential falloff above base height
  - Distance fog: exponential with camera distance
  - Configurable `FogSettings` struct

11.5. **Local fog volumes**
  - Read from volumetric companion brick pool
  - Integrate with ECS (FogVolume component → brick range)
  - Test: place fog volume in cave, verify correct lighting

11.6. **Temporal reprojection for volumetrics**
  - Reproject previous frame's volumetric buffer
  - Validate (depth consistency, disocclusion)
  - Blend: 90% history, 10% current (valid pixels only)

11.7. **Bilateral upscale**
  - 480×270 → 960×540 (or current internal res)
  - Edge-aware: weighted by depth similarity to full-res depth
  - Prevent fog bleeding across depth discontinuities

11.8. **Volumetric compositing**
  - `final = shaded_color * vol_transmittance + vol_color`
  - Insert into frame pipeline after shading, before post-process

11.9. **God rays**
  - Verify god rays emerge from ambient dust + volumetric shadow map
  - Tune dust density (0.001-0.01 range)
  - Test: sunlight through gaps in geometry → visible light shafts

11.10. **Procedural clouds (high-altitude)**
  - FBM noise + weather map texture
  - Height gradient shaping
  - Wind scrolling over time
  - Sample in volumetric march
  - Cloud settings struct

11.11. **Cloud shadow map**
  - 2D texture (1024²)
  - Project cloud density downward from sun direction
  - Sample in shading pass for terrain/outdoor surfaces

11.12. **Brick-backed clouds (low-altitude)**
  - Use volumetric companion pool for dense, terrain-interacting clouds
  - Fog banks, valley mist, mountain wrapping
  - Both brick-backed and procedural feed same volumetric march

### Done when
- Height fog fades objects into distance
- God rays visible through gaps in geometry
- Procedural clouds drift across sky
- Cloud shadows move across terrain
- Local fog volumes light correctly
- Volumetrics are temporally stable (no flickering)

---

## Phase 12: ECS and Frame Scheduling

**Milestone:** Engine has proper entity management and a structured frame loop. Transition from ad-hoc testbed rendering to a real frame pipeline.

**Crate:** `rkf-runtime`

### Tasks

12.1. **ECS setup (hecs)**
  - Create ECS `World`
  - Define component types: `Transform`, `WorldTransform`, `Parent`, `ChunkRef`, `SdfObject`, `Light`, `FogVolume`, `Camera`, `EditorMetadata`
  - Basic spawn/despawn/query helpers

12.2. **Transform hierarchy update**
  - System: compute `WorldTransform` from `Transform` + `Parent` chain
  - Two-pass: roots first, then children
  - Bone attachment: if `Parent.bone_index` set, multiply by bone matrix
  - Camera-relative: all `WorldTransform` matrices are relative to camera `WorldPosition`

12.3. **Frame scheduling**
  - Implement `execute_frame(ctx)` with static pass ordering (ARCHITECTURE.md §Engine Architecture)
  - All passes from Phases 4-11 wired into the frame function
  - Conditional execution (skip disabled passes based on settings)

12.4. **Double-buffered uniforms**
  - Two sets of camera/light/settings uniform buffers
  - CPU writes frame N+1 while GPU executes frame N
  - Swap after queue.submit()

12.5. **Configuration system**
  - `EngineConfig` struct with all per-system settings
  - `QualityPreset` enum (Low/Medium/High/Ultra/Custom)
  - Load from RON file, save to RON file
  - Runtime modification via mutable config reference

12.6. **Test: structured frame loop**
  - Port testbed scene to ECS (entities with components instead of hardcoded data)
  - Verify all existing rendering still works through the frame scheduler
  - Lights, fog volumes, camera as proper ECS entities

### Done when
- ECS manages all entities
- Frame scheduling runs all passes in correct order
- Configuration loads from file and affects rendering
- No regression from testbed rendering quality

---

## Phase 13: Clipmap LOD

**Milestone:** Multiple LOD levels render simultaneously. Distant geometry uses coarser voxels with larger step sizes.

**Crates:** `rkf-core`, `rkf-render`

### Tasks

13.1. **Multi-LOD sparse grids**
  - Multiple `SparseGrid` instances (one per LOD level)
  - Clipmap configuration: voxel sizes `[2cm, 4cm, 8cm, 16cm, 32cm]`, radii `[128m, 256m, 512m, 1024m, 2048m]`
  - GPU upload: all LOD grids as storage buffers
  - LOD selection uniform: per-level radius + voxel size

13.2. **Ray march LOD transitions**
  - When ray's `t` crosses a clipmap boundary: switch to next LOD level's grid
  - Coarser voxels yield larger safe step sizes (natural performance gradient)
  - Handle boundary: clamp/blend at transition zone (or hard switch initially)

13.3. **Variable resolution tier support**
  - Ray marcher checks finest available tier at each step
  - Tier 0 (0.5cm) only present where content authors placed fine detail
  - Fallback to next coarser tier if finest is absent

13.4. **LOD transition blending (optional)**
  - Overlap LOD levels by small margin
  - Trilinear interpolation between coarse and fine SDF in transition band
  - Can defer — low internal resolution + upscaling may hide discontinuities

13.5. **Test: multi-LOD scene**
  - Generate terrain at multiple LOD levels
  - Camera at center: near terrain is detailed, distant terrain is coarse
  - Fly camera: verify no popping at LOD boundaries
  - Profile: confirm distant geometry is cheaper (fewer steps)

### Done when
- Multiple LOD levels render simultaneously
- Distant geometry uses coarser grids and fewer steps
- No visible seams at LOD boundaries (or acceptable with hard switch)
- Frame time improves vs single-LOD for large scenes

---

## Phase 14: Streaming and Asset Management

**Milestone:** Chunks load and unload as the camera moves. Large worlds work within a fixed memory budget.

**Crates:** `rkf-runtime`, `rkf-core`

### Tasks

14.1. **Chunk data structure**
  - `Chunk { coords: IVec3, grids: PerTier<SparseGrid>, brick_ranges: Vec<Range<u32>> }`
  - On-disk format: `.rkf` chunks with per-tier brick data
  - Chunk loading: decompress → allocate bricks → populate grid → activate

14.2. **Streaming system**
  - Camera `WorldPosition` → desired LOD per chunk (distance-based)
  - Diff against currently loaded state
  - Priority queue: load nearest first, evict farthest first
  - Spatial index update on load/evict

14.3. **Async I/O pipeline**
  - Thread pool reads `.rkf` from disk, decompresses (LZ4)
  - Staging buffer: decompressed bricks queued for GPU upload
  - Main thread: `queue.write_buffer` from staging → brick pool (during frame's upload window)
  - Journal replay: if `.rkj` exists, replay edits after brick upload (GPU compute)

14.4. **LRU eviction**
  - Track brick usage (last-accessed frame)
  - When pool exceeds budget threshold: evict oldest bricks
  - Grace period for recently unloaded chunks (prevents thrashing at stream boundaries)

14.5. **Streaming budget enforcement**
  - `StreamingBudget` struct: max pool MB, max staging MB, max I/O bandwidth
  - Throttle loads when staging buffer is full
  - Monitor and log budget utilization

14.6. **Asset registry**
  - `AssetRegistry` with `SlotMap` per asset type
  - Generational `Handle<T>` for type-safe references
  - `AssetState` tracking: Unloaded → Loading → Loaded → Error
  - Reference counting for eviction eligibility

14.7. **Procedural chunk generator (for testing)**
  - Generate `.rkf` chunks with procedural terrain SDF (perlin noise heightmap)
  - Write to temp directory
  - Configurable world size (e.g., 10×10 grid, 20×20 grid)

14.8. **Test: streaming world**
  - Generate a 20×20 grid of procedural chunks
  - Fly camera through world
  - Verify chunks load/unload without hitching
  - Monitor: memory stays within budget, no brick leaks
  - Verify: LOD transitions work with streamed data

### Done when
- Chunks stream in/out based on camera distance
- No visible hitching during streaming
- Memory budget is respected (LRU eviction works)
- Can fly continuously through a large world
- Asset registry tracks all loaded resources

---

## Phase 15: Extended Materials

**Milestone:** Material blending, per-voxel color, procedural noise variation. Rich material presentation.

**Crate:** `rkf-core`, `rkf-render`

### Tasks

15.1. **Material blending in shader**
  - When `blend_weight > 0`: interpolate all properties between primary and secondary
  - Linear lerp matching ARCHITECTURE.md
  - Test: transition zone between stone and dirt materials

15.2. **Per-voxel color (companion color pool)**
  - GPU buffer for `ColorBrick` data
  - Flag check in shading: if `has_color_data`, sample color pool
  - Color application: `final_albedo = material.albedo * voxel_color` (multiply mode)
  - Test: imported object with baked color vs material-only

15.3. **Procedural noise in shading**
  - 3D simplex/gradient noise (WGSL implementation or precomputed 3D texture)
  - Per-material: `noise_scale`, `noise_strength`, `noise_channels`
  - Apply to albedo, roughness, normal perturbation based on channel bits
  - Test: stone material with noise vs without — visible micro-detail

### Done when
- Material transitions blend smoothly
- Per-voxel color modulates material albedo
- Procedural noise adds visible surface variation
- No visual artifacts at blend boundaries

---

## Phase 16: Procedural Editing

**Milestone:** Real-time sculpting and painting of the voxel world. Undo/redo works. Edits persist.

**Crate:** `rkf-edit`

### Tasks

16.1. **CSG operations (GPU compute)**
  - Compute shader: iterate over affected bricks, apply CSG per-voxel
  - `csg_union`, `csg_subtract`, `csg_intersect`
  - `csg_smooth_union`, `csg_smooth_subtract`
  - Input: edit parameters as uniform, affected brick range

16.2. **Analytic SDF primitives**
  - WGSL: `sphere_sdf`, `box_sdf`, `capsule_sdf`, `cylinder_sdf`, `torus_sdf`, `plane_sdf`
  - Transform primitive by edit position/rotation
  - Edit shader accepts primitive type + dimensions

16.3. **Edit pipeline**
  - CPU: compute edit AABB, determine affected bricks
  - GPU: pre-edit allocate new bricks for empty cells that may gain surface
  - GPU: apply edit (1 workgroup per brick)
  - GPU: post-edit check for empty bricks (reduction, deallocate)
  - CPU: update spatial index

16.4. **Brush system**
  - 7 brush types: Add, Subtract, Smooth, Flatten, Paint, Blend Paint, Color Paint
  - `Brush` struct: type, shape, radius, strength, falloff, material_id, blend_k
  - Falloff curves: Linear, Smooth, Sharp
  - Apply via edit pipeline with brush-specific CSG logic

16.5. **Brick allocation/deallocation during edits**
  - Pre-edit: claim bricks from free list for empty cells overlapping edit
  - Post-edit: GPU reduction checks `min(abs(distance))` per brick, return empty bricks
  - Handle companion pool allocation (color, volumetric) for paint operations

16.6. **Delta undo/redo**
  - Pre-edit: GPU compute copies affected voxel values to staging buffer
  - Async readback: staging → CPU `EditDelta` struct
  - Undo: restore previous voxel values, re-allocate/deallocate bricks
  - Redo: re-apply the edit operation
  - History stack: configurable depth (default 100, 256MB budget)

16.7. **Edit journal (.rkj)**
  - `CompactEditOp` — 64 bytes fixed size (ARCHITECTURE.md §Procedural Editing)
  - `.rkj` file: header + append-only entries
  - Write: append operation on each edit
  - Read: replay operations on chunk load
  - Journal compaction: when entries > 1000, bake into new base .rkf

16.8. **Test: sculpting session**
  - Load terrain chunk
  - Add material (smooth union sphere)
  - Subtract crater (smooth subtract)
  - Paint material on surface
  - Undo 3 times, redo 2 times
  - Save, reload, verify edits persist via journal

### Done when
- All 7 brush types work
- CSG operations produce clean geometry
- Undo/redo is reliable
- Edit journal persists and replays correctly
- Edit performance: small edits < 1ms

---

## Phase 17: Skeletal Animation

**Milestone:** Animated character with smooth joint blending renders correctly. Facial blend shapes work.

**Crate:** `rkf-animation`

### Tasks

17.1. **Skeleton data structures**
  - `Skeleton { bones: Vec<Bone>, hierarchy: Vec<i32> }` (parent indices)
  - `Bone { name, bind_transform, inverse_bind }`
  - `AnimationClip { name, duration, channels: Vec<BoneChannel> }`
  - `BoneChannel { bone_index, keyframes: Vec<Keyframe> }` (position, rotation, scale)

17.2. **Bone matrix evaluation**
  - Keyframe interpolation (lerp position/scale, slerp rotation)
  - Hierarchical: multiply parent × local for each bone
  - Output: `bone_matrices: [Mat4; MAX_BONES]` — upload as uniform buffer

17.3. **Segment data**
  - `Segment { bone_index: u32, brick_range: Range<u32>, aabb: AABB }`
  - Rest-pose SDF stored in brick pool
  - Spatial index entries: `RIGID_SEGMENT` type with bone index

17.4. **Segment transforms**
  - Per frame: update spatial index bounding boxes for each segment
  - Ray marcher: on entering `RIGID_SEGMENT`, transform ray by inverse bone matrix
  - Evaluate rest-pose SDF in local space

17.5. **Joint rebaking (GPU compute)**
  - Per joint: small brick volume (~10-20cm radius)
  - Compute shader: for each voxel in joint region
    - Transform to segment A local space → evaluate SDF_A
    - Transform to segment B local space → evaluate SDF_B
    - `smooth_min(SDF_A, SDF_B, blend_radius)`
    - Material from closer segment
  - Write to joint bricks in brick pool

17.6. **Smooth-min blending**
  - WGSL implementation matching ARCHITECTURE.md
  - Per-joint configurable `k` (small for mechanical, large for organic)

17.7. **Lipschitz mitigation**
  - Joint bricks flagged in spatial index
  - Ray marcher uses 0.8× step multiplier in flagged regions

17.8. **Blend shapes (facial animation)**
  - `BlendShape { name, weight, brick_offset, brick_count, aabb }`
  - Delta-SDF stored as sparse bricks
  - Apply during head segment rebaking: `final = base + Σ(weight_i × delta_i)`
  - Only active (non-zero weight) shapes evaluated

17.9. **Animation playback**
  - `AnimationPlayer` component: current clip, time, speed, looping
  - Update system: advance time → evaluate keyframes → upload bone matrices
  - Blend shape weight animation (same keyframe system)

17.10. **Test: animated character**
  - Create test character with ~20 bones (humanoid)
  - Procedurally generate segment SDFs (cylinders + spheres)
  - Play walk animation
  - Verify smooth joint blending (no gaps, no seams)
  - Test facial blend shapes (smile, blink)

### Done when
- Character animates with smooth joint transitions
- No visible seams between segments
- Facial blend shapes produce visible expressions
- Animation cost < 2ms per character
- Multiple characters render simultaneously

---

## Phase 18: Static Mesh Import

**Milestone:** Convert a static glTF model to `.rkf`, load it into the engine, render it with correct materials and per-voxel color.

**Crates:** `rkf-import`, `rkf-convert`, `rkf-core`

### Tasks

18.1. **rkf-convert CLI scaffold**
  - clap-based argument parsing
  - Input file, output file, options (tier, compression, etc.)
  - Progress reporting

18.2. **Mesh loading (glTF)**
  - `gltf` crate: load mesh, materials
  - Extract: vertex positions, normals, UVs, indices
  - Extract: material properties → map to engine `Material`

18.3. **BVH construction**
  - Build bounding volume hierarchy over mesh triangles
  - Nearest-triangle query: `fn nearest(point) -> (distance, triangle_index, barycentric)`
  - Use for SDF computation and material transfer

18.4. **Unsigned distance computation**
  - For each voxel in narrow band: BVH nearest-triangle distance
  - Narrow band: ±3 bricks from surface (estimated from mesh bounds)
  - Parallelize across voxels (rayon)

18.5. **Sign determination (generalized winding number)**
  - Barill et al. 2018 algorithm
  - Compute winding number at each voxel center
  - > 0.5 = inside (negative distance), else outside (positive)
  - Robust with non-watertight meshes

18.6. **Material transfer**
  - Per surface voxel: find nearest triangle (BVH)
  - Interpolate UV from barycentric coordinates
  - Sample albedo texture at UV → companion color pool
  - Map source material → engine material ID
  - PBR properties: uniform per material (roughness, metallic from material base values)

18.7. **Resolution auto-selection**
  - Heuristic: `target_voxel = avg_edge_length / 3`
  - Snap to nearest tier
  - CLI override: `--tier`

18.8. **LOD pre-computation**
  - Downsample from target tier to all coarser tiers
  - Store each tier in the `.rkf` file

18.9. **`.rkf` binary format (write)**
  - Header (64 bytes): magic, version, flags, bounds, counts
  - Material table
  - Per-tier: spatial index + LZ4-compressed brick data + color data
  - Write with `std::io::BufWriter`

18.10. **`.rkf` binary format (read) — in rkf-core**
  - Loader: parse header, read material table, decompress brick data per tier
  - Return structured data ready for GPU upload
  - Validate version, magic, checksums

18.11. **FBX and OBJ support**
  - `russimp` crate for FBX loading
  - OBJ via `gltf` or `tobj` crate
  - Map to same intermediate representation as glTF path

18.12. **Test: static import pipeline**
  - Convert a static glTF model (e.g., environment prop)
  - `rkf-convert model.glb -o model.rkf`
  - Load model.rkf in engine
  - Verify: geometry shape, materials, per-voxel color from textures

### Done when
- Static glTF/FBX/OBJ models convert to `.rkf` successfully
- Converted models render with correct geometry and materials
- Per-voxel color captures albedo texture detail
- Conversion time is reasonable (< 30s for moderate models)
- `.rkf` files load quickly (< 1s)

---

## Phase 19: Animated Mesh Import

**Milestone:** Convert an animated glTF model to `.rkf` with skeleton segmentation, joint regions, and blend shapes.

**Crates:** `rkf-import`, `rkf-convert`

### Tasks

19.1. **Skeleton extraction**
  - Extract bone hierarchy, bind-pose transforms, per-vertex weights from glTF/FBX
  - Extract animation clips (keyframes per bone per clip)
  - Map to engine `Skeleton`, `AnimationClip` structures

19.2. **Automatic skeleton segmentation**
  - Per vertex: dominant bone (highest weight)
  - Per triangle: segment = bone where all 3 vertices agree
  - Joint regions: triangles with mixed dominant bones
  - Handle edge cases: shared bones, chain of small bones

19.3. **Per-segment voxelization**
  - Isolate segment triangles
  - Voxelize each segment independently into separate brick ranges
  - Tag with segment ID and bone index

19.4. **Joint region extraction**
  - Bounding volume around joint triangles
  - Voxelize both overlapping segments in joint region
  - Store bone weights in companion bone pool
  - Set blend_radius per joint (from bone distance heuristic)

19.5. **Blend shape conversion**
  - Per morph target: voxelize deformed mesh
  - Subtract base SDF → delta brick data
  - Store as sparse delta bricks covering only affected region

19.6. **Manual segment override**
  - Optional config file (`--segments <file>`) mapping bones to segments
  - For cases where automatic analysis produces suboptimal splits

19.7. **`.rkf` format extension for animation**
  - Write skeleton, segments, joints, blend shapes to `.rkf`
  - Companion bone data (LZ4 compressed)
  - Animation clips serialized as keyframe arrays

19.8. **Test: animated import pipeline**
  - Convert animated glTF character (e.g., humanoid with walk cycle)
  - `rkf-convert character.glb -o character.rkf`
  - Load in engine, play animation
  - Verify: correct segmentation, smooth joints, facial blend shapes

### Done when
- Animated models convert with correct segmentation
- Skeleton hierarchy and animations preserved
- Joint regions blend smoothly after import
- Blend shapes produce correct deltas
- Round-trip: import → engine → animation playback works

---

## Phase 20: Particles

**Milestone:** Particle effects: sparks, debris, rain. Three render backends working.

**Crate:** `rkf-particles`

### Tasks

20.1. **GPU particle buffer**
  - `Particle` struct (48 bytes) matching ARCHITECTURE.md
  - Storage buffer on GPU (capacity: configurable, default 100K)
  - Atomic counter for live particle count

20.2. **Particle simulation compute shader**
  - Spawn: emitters append new particles (atomic increment)
  - Integrate: `position += velocity * dt`, apply gravity, wind, drag
  - Collide: if flagged, evaluate SDF at position, push out along gradient
  - Age: `lifetime -= dt`, mark dead if ≤ 0
  - Stream compaction: prefix sum to remove dead particles, compact buffer

20.3. **Emitter system**
  - `ParticleEmitter` component (ECS)
  - Emitter shapes: Point, Sphere, Box, Cone
  - Rate (continuous) or burst (one-shot)
  - Per-particle randomized ranges for lifetime, speed, size, color, emission

20.4. **Particle binning**
  - 3D grid binning (compute shader)
  - Each cell: list of particles within
  - Used by volumetric march to efficiently iterate nearby particles

20.5. **Volumetric particles**
  - In volumetric march shader: iterate binned particles near each step
  - Accumulate density + emission kernel per particle
  - Gaussian falloff, age-modulated
  - Test: spark shower, embers from fire

20.6. **SDF micro-objects**
  - Register particle positions as temporary spatial index entries
  - Ray marcher evaluates analytic SDF (sphere/capsule) per micro-object
  - Full shading, shadow casting
  - Budget: ~1000 simultaneous
  - Test: debris from explosion

20.7. **Screen-space particles**
  - Post-process overlay pass
  - Project particle positions to screen
  - Depth test against G-buffer
  - Draw oriented streaks (rain) or points (snow) with alpha blend
  - Test: rain, snowfall

### Done when
- All three render backends produce correct visual output
- Particle simulation runs at 100K particles without frame drop
- SDF collision pushes particles out of geometry
- Emitters spawn and kill particles correctly

---

## Phase 21: Physics

**Milestone:** Character controller walks on SDF terrain. Rigid bodies fall and collide. Destruction spawns debris.

**Crate:** `rkf-physics`

### Tasks

21.1. **Rapier world setup**
  - Create `RapierContext` with gravity, timestep config
  - Fixed timestep (60Hz) with accumulator
  - Interpolated transforms for rendering

21.2. **SDF collision adapter**
  - `generate_sdf_contacts(body, sdf) -> Vec<ContactPoint>`
  - Sample SDF at body's contact points
  - Negative distance → contact (position, normal from gradient, penetration depth)
  - Register as custom collision handler in Rapier pipeline

21.3. **Rigid body component**
  - `RigidBody` ECS component holding Rapier handle
  - Sync: Rapier transforms → ECS `Transform` each frame
  - Standard colliders (box, sphere, capsule) for object-vs-object

21.4. **Character controller**
  - `SdfCharacterController` struct (ARCHITECTURE.md §Physics)
  - Capsule-vs-SDF collision with multi-point sampling
  - Iterative slide (up to 4 iterations for corners)
  - Ground detection (SDF sample below feet)
  - Slope limiting (steep slopes become walls)
  - Step climbing (small ledges auto-mounted)

21.5. **Destruction → debris**
  - `DestructionEvent { aabb, volume, material_id }` from edit system
  - Debris spawner: create SDF micro-object particles with outward velocity
  - Physics rigid bodies for larger debris pieces

21.6. **Test: physics playground**
  - Terrain with hills and caves
  - Character controller: walk, jump, slide on slopes
  - Drop rigid body boxes/spheres onto terrain
  - Explode a crater, verify debris spawns and falls

### Done when
- Character walks on terrain without clipping through
- Rigid bodies rest on surfaces correctly
- Objects stack without jitter
- Destruction creates convincing debris
- Physics runs at 60Hz without frame drops

---

## Phase 22: Editor

**Milestone:** Full editor with rinch UI. Scene authoring workflow: place, sculpt, paint, light, save/load.

**Crate:** `rkf-editor`

### Tasks

22.1. **rinch UI setup**
  - Initialize rinch with wgpu backend
  - Basic window with menu bar + panels
  - Engine renders to offscreen texture
  - rinch displays viewport texture as image widget

22.2. **Input routing**
  - Detect mouse over viewport vs UI panels
  - Viewport focused: input goes to editor camera/tools
  - Panel focused: input goes to rinch

22.3. **Editor camera**
  - Orbit mode: rotate around target, scroll to zoom
  - Fly mode: WASD + mouse look (hold right-click)
  - Follow mode: track selected entity
  - Smooth transitions between modes
  - Camera stores `WorldPosition` for precision

22.4. **Scene hierarchy panel**
  - Tree view of all entities in ECS
  - Expand/collapse groups
  - Click to select, Ctrl+click for multi-select
  - Drag to reparent (within depth limit)
  - Right-click context menu: rename, duplicate, delete

22.5. **Property inspector panel**
  - Shows components of selected entity
  - Editable fields: position, rotation, scale, material, light params, etc.
  - Changes apply immediately (live preview)
  - Generates undo actions for each change

22.6. **Transform gizmo**
  - Translate: axis arrows + planes
  - Rotate: axis rings
  - Scale: axis handles (uniform only — warn if non-uniform attempted)
  - Toggle mode with keyboard (W/E/R)
  - Render as gizmo overlay

22.7. **Entity placement**
  - Asset browser panel: list available `.rkf` assets
  - Drag from browser to viewport → place entity
  - Snap to grid (configurable grid size)
  - Surface snap: place on SDF surface (ray cast to find hit)

22.8. **Sculpt mode integration**
  - Connect rkf-edit brushes to editor UI
  - Brush toolbar: select type, shape, radius, strength, material
  - Mouse drag in viewport → stream edit operations
  - Real-time preview of brush footprint (wireframe sphere)

22.9. **Paint mode**
  - Material paint: select material from palette → paint onto surface
  - Color paint: color picker → paint per-voxel color
  - Blend paint: two materials + blend gradient
  - Same brush radius/strength controls as sculpt

22.10. **Light editing**
  - Place lights (point, spot, directional)
  - Visual gizmos: point radius sphere, spot cone, direction arrow
  - Cookie texture assignment
  - Real-time lighting update as parameters change

22.11. **Environment editing**
  - Panel with fog, cloud, atmosphere, post-process settings
  - Sliders/color pickers for all parameters
  - Live preview — changes apply immediately
  - Save as part of `.rkscene`

22.12. **Animation preview**
  - Select animated entity → show animation controls
  - Play/pause/scrub timeline
  - Bone visualization (wireframe skeleton overlay)
  - Blend shape weight sliders

22.13. **Gizmo overlay rendering**
  - Line rasterization pass (after post-process)
  - Draw: transform gizmo, selection outlines, light shapes, volume wireframes, grid plane
  - Depth-tested where appropriate (gizmo in front of geometry)

22.14. **Debug visualizations**
  - View menu toggles: normals, material IDs, LOD levels, brick occupancy, radiance volume slices
  - Override shader output to visualize selected data
  - FPS / frame time overlay

22.15. **Unified undo stack**
  - `UndoStack<UndoAction>` with push/undo/redo
  - Action types: transform, spawn, despawn, voxel edit, property change, environment change
  - Ctrl+Z / Ctrl+Y keyboard shortcuts

22.16. **Scene save/load**
  - Save: serialize ECS → `.rkscene` (RON)
  - Load: parse `.rkscene` → spawn entities, apply settings
  - Recent files list
  - Unsaved changes warning on close

### Done when
- Can create a scene from scratch: place objects, sculpt terrain, paint materials, add lights
- Scene saves and loads correctly
- Undo/redo works for all operations
- Editor is responsive and usable
- All 8 editor modes function

---

## Phase 23: Integration Testing and Polish

**Milestone:** Engine is stable, performant, and all systems work together correctly.

**Crates:** all

### Tasks

23.1. **rkf-game example**
  - Create a showcase scene demonstrating all features:
    - Terrain with varying materials and per-voxel color
    - Animated characters with facial expressions
    - Multiple light types including spot with cookies
    - Volumetric fog, god rays, clouds
    - Particle effects (fire, debris, rain)
    - Physics objects
    - Real-time editing (destruction)
  - Controllable character with character controller

23.2. **Performance profiling**
  - Integrate puffin or tracy
  - GPU timing queries per pass
  - Identify bottlenecks per phase:
    - Ray march: step count, brick reads
    - Shading: light count, shadow ray budget
    - GI: cone trace cost
    - Volumetrics: step count
    - Upscale: history rejection rate
  - Establish performance targets per pass at target resolution

23.3. **GPU memory audit**
  - Verify all pools stay within budget
  - Check for leaks (bricks allocated but never freed)
  - Profile peak vs average utilization
  - Tune default budgets

23.4. **Quality preset tuning**
  - Define Low/Medium/High/Ultra presets with specific per-system values
  - Test each preset: visual quality vs performance
  - Ensure Low runs on modest hardware

23.5. **Stress testing**
  - Large world: 100+ chunks loaded simultaneously
  - Many lights: 100+ point lights
  - Many characters: 20+ animated characters
  - Heavy editing: rapid sculpting with large brushes
  - Rapid camera movement: streaming keeps up

23.6. **Bug fixes and edge cases**
  - Chunk boundary artifacts
  - LOD transition seams
  - Memory exhaustion handling (graceful degradation, not crash)
  - Window resize handling
  - Fullscreen toggle
  - Multi-monitor support

23.7. **Shader hot-reload**
  - Watch WGSL files for changes
  - Recompile pipelines on change
  - Report errors without crashing

23.8. **Material hot-reload**
  - Watch material definition files
  - Re-upload material table on change

### Done when
- Demo scene runs at stable frame rate
- No visual artifacts under normal use
- All systems interact correctly
- Performance profiling data collected and optimized against
- Quality presets provide meaningful trade-offs

---

## Dependency Graph

```
Phase 0 ── 1 ── 2 ── 3 ── 4 ── 5 ── 6 ── 7 ── 8
                                           │     │
                                           9 ── 10
                                           │
                                          11
                                           │
                                          12 ── 13 ── 14 ─────────────────────────┐
                                                       │                           │
                                         ┌─────────────┼──────────┐               │
                                         │             │          │               │
                                       Ph 15        Ph 16      Ph 17            │
                                         │             │          │               │
                                         │             │     Ph 18 ── Ph 19      │
                                         │             │          │               │
                                         │          Ph 20      Ph 21            │
                                         │             │          │               │
                                         └─────────────┴──────────┴── Phase 22 ── 23
```

**Critical path:** 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 11 → 12 → 13 → 14 → 22 → 23

**Phase dependencies:**

| Phase | Depends on |
|-------|-----------|
| 0 (Scaffolding) | — |
| 1 (MCP Foundation) | 0 |
| 2-11 | Sequential (rendering pipeline) |
| 12 (ECS) | 11 |
| 13 (LOD) | 12 |
| 14 (Streaming) | 13 |
| 15 (Materials ext.) | 14 |
| 16 (Editing) | 14 |
| 17 (Animation) | 14 |
| 18 (Static import) | 14 |
| 19 (Animated import) | 17, 18 |
| 20 (Particles) | 11, 14 |
| 21 (Physics) | 14, 16 (for destruction) |
| 22 (Editor) | 14, 15, 16, 17, 18 (needs all systems) |
| 23 (Polish) | All |

**Parallelizable after Phase 14:** Phases 15, 16, 17, 18 can be developed concurrently (different crates, minimal overlap). Phase 19 waits for 17 + 18. Phase 20 can start alongside 15-18. Phase 21 waits for 16 (destruction events).

---

## Phase-by-Feature Matrix

Shows which architecture decisions are implemented in each phase.

| Feature | Phase |
|---------|-------|
| MCP server + tool discovery | 1 |
| AutomationApi trait | 1 |
| IPC communication | 1 |
| WorldPosition + precision | 2, 12 |
| Brick pool + sparse grid | 3 |
| Ray march (basic) | 4 |
| Multi-level DDA | 5 |
| G-buffer | 6 |
| PBR shading | 6 |
| SDF shadows | 7 |
| SDF AO | 7 |
| SSS | 7 |
| Point/spot lights + tiled culling | 7 |
| Cookie textures | 7 |
| Voxel cone tracing GI | 8 |
| Temporal upscaling + DLSS | 9 |
| Post-processing stack | 10 |
| Volumetric fog + god rays | 11 |
| Procedural clouds | 11 |
| Brick-backed clouds | 11 |
| Cloud shadows | 11 |
| ECS + scene graph | 12 |
| Frame scheduling | 12 |
| Configuration + presets | 12 |
| Clipmap LOD | 13 |
| LOD transitions | 13 |
| Chunk streaming | 14 |
| Asset registry | 14 |
| LRU eviction + memory budget | 14 |
| Material blending + per-voxel color | 15 |
| Procedural noise variation | 15 |
| CSG editing + brushes | 16 |
| Undo/redo + edit journal | 16 |
| CompactEditOp format | 16 |
| Skeletal animation (segments + joints) | 17 |
| Blend shapes (facial) | 17 |
| Mesh-to-SDF conversion (static) | 18 |
| .rkf format (read + write) | 18 |
| rkf-convert CLI | 18 |
| Skeleton segmentation + animated import | 19 |
| Blend shape conversion | 19 |
| Particle system (3 backends) | 20 |
| Rapier physics + SDF collision | 21 |
| Character controller | 21 |
| Editor (rinch) | 22 |
| Editor tools (sculpt, paint, place, light) | 22 |
| Scene save/load (.rkscene) | 22 |
| MCP screenshot tool (functional) | 4 |
| MCP render_stats tool (functional) | 6 |
| MCP scene_graph + entity tools (functional) | 12 |
| MCP material_edit tool (functional) | 15 |
| MCP brush_apply tool (functional) | 16 |
| MCP asset_status tool (functional) | 14 |
| MCP scene_load/save tools (functional) | 22 |
| MCP full editor tool set | 22 |
