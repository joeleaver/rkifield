# RKIField — SDF Graphics Engine

## What This Is

A real-time graphics engine where **signed distance fields are the only geometry**. No meshes, no triangles, no UV mapping. Material data lives in the volume. Everything renders via compute shader ray marching through voxel-backed distance fields.

This is a novel engine — not a wrapper around an existing renderer. Many subsystems (skeletal animation via segmented joint rebaking, SDF soft shadows, voxel cone tracing from native data) have no direct precedent in shipping engines.

## Critical Rules

1. **No meshes, ever.** Every geometric primitive is an SDF stored in voxel bricks. If you're tempted to add a mesh path, you're solving the wrong problem.
2. **No textures on surfaces.** Materials are volumetric — PBR properties come from a global material table indexed by `material_id` (u8, 256 max). Per-voxel RGBA color is stored in `SurfaceVoxel.color`, not UV-mapped textures.
3. **Non-uniform scale uses conservative correction.** Per-axis `Vec3` scale is supported. SDF distances are multiplied by `min(sx, sy, sz)` to keep ray marching safe (never overshoots). For voxelized objects, the editor offers a "Re-voxelize" button that resamples the brick volume at the current stretched dimensions and resets scale to `(1,1,1)`, eliminating the extra march-step overhead.
4. **All rendering is compute shaders.** No rasterization pipeline except editor gizmo wireframes.
5. **WorldPosition everywhere.** Never use raw `Vec3` for world-space positions. Always `WorldPosition { chunk: IVec3, local: Vec3 }` to avoid float precision loss. GPU receives camera-relative f32 only.
6. **Geometry pools are the source of truth.** CPU-side `GeometryPool` (BrickGeometry) and `SdfCachePool` hold authoritative voxel data. The GPU brick pool holds derived VoxelSample data (via `Brick::from_geometry`). The `BrickPoolManager` coordinates streaming, animation, and editing.
7. **The architecture docs are the spec. Follow them exactly.** Never substitute simpler alternatives, skip features, or deviate from the documented designs. If the architecture says "multi-level DDA + sphere tracing," implement multi-level DDA + sphere tracing — not a simplified single-level version. When in doubt, read the relevant architecture doc. The decisions were made deliberately.
8. **Test-driven development.** Write tests before implementation. Every new type, algorithm, and pipeline pass gets tests first. Red → green → refactor. Target 80%+ coverage on library crates. GPU code that can't be unit-tested gets visual regression tests in `rkf-testbed`.
9. **Commit after every step.** Each completed task within a phase gets its own atomic commit. Don't batch multiple tasks into one commit. Commit messages reference the phase and task number (e.g., `phase-1: 1.1 — implement WorldPosition type`). This creates a clean, reviewable history that maps directly to the implementation plan.
10. **Ask questions, don't assume.** When requirements are ambiguous, a design decision has multiple valid paths, or you're unsure what the user wants — stop and ask. A 30-second question saves hours of rework. Never silently choose a simpler alternative, skip a feature, or reinterpret a requirement. If the architecture docs don't cover something, ask. If a task feels underspecified, ask. Assumptions are bugs.
11. **Nothing is verified until the user sees it.** Tests passing and code compiling does not mean a feature is done. Visual and runtime correctness must be confirmed by the user seeing a screenshot or live testbed and signing off. Never claim a rendering change is "verified" based on tests alone — the user must visually approve it.
12. **Agents must use the MCP.** This engine is designed to be operated by LLMs and AI agents. When developing or testing, always connect to the engine via `rkf-mcp` and use MCP tools for screenshots, scene inspection, entity manipulation, and validation. **If the MCP server doesn't exist, is disconnected, or doesn't work — fixing it is the top priority.** No other work proceeds until agent tooling is functional. Every new feature must ship with corresponding MCP tools.

## Tech Stack

| Component | Choice | Notes |
|-----------|--------|-------|
| Language | **Rust** | Entire codebase |
| GPU API | **wgpu** | WebGPU via wgpu crate |
| Shaders | **WGSL** | Compute-only (no vertex/fragment) |
| Windowing | **winit** | |
| Math | **glam** | f32 vectors, quaternions, matrices |
| ECS | **hecs** | Thin — scene management only, not GPU resources |
| Physics | **Rapier** | Dynamics engine; we write the SDF collision adapter |
| Upscaling | **dlss_wgpu** | NVIDIA preferred; custom temporal fallback for all GPUs |
| Editor UI | **rinch** | wgpu backend, composites over engine viewport |
| Serialization | **RON** | Scene files (`.rkscene`), config, asset metadata |
| Compression | **lz4_flex** | Brick data in `.rkf` files |

## Crate Structure

```
rkifield/
  crates/
    rkf-core/        — WorldPosition, brick pool, spatial index, voxel types, constants
    rkf-render/      — ray march, shading, GI, volumetrics, upscale, post-process
    rkf-animation/   — segmented joint rebaking, blend shape delta-SDFs
    rkf-edit/        — GPU CSG operations, sculpt brushes, undo/redo, edit journal
    rkf-import/      — mesh-to-SDF conversion library (BVH + winding number)
    rkf-physics/     — Rapier integration, SDF collision adapter, character controller
    rkf-particles/   — GPU particle sim, 3 render backends
    rkf-runtime/     — frame scheduling, ECS glue, streaming, asset management
    rkf-mcp/         — MCP server binary (tool discovery, automation API bridge)
    rkf-editor/      — rinch UI, gizmos, tools, scene serialization (binary)
    rkf-convert/     — offline mesh-to-.rkf CLI (binary)
    rkf-game/        — example game / playground (binary)
    rkf-testbed/     — permanent visual testing harness (binary)
```

**Dependency flow:** `core` ← `render` ← `runtime` ← `editor`/`game`. Feature crates (`animation`, `edit`, `import`, `physics`, `particles`) depend on `core` and are consumed by `runtime`.

## Build Commands

```bash
cargo build --workspace          # build everything
cargo test --workspace           # run all tests
cargo clippy --workspace         # lint
cargo run -p rkf-testbed         # visual test harness
cargo run -p rkf-editor          # editor
cargo run -p rkf-convert -- input.glb -o output.rkf  # asset conversion
```

## Key Data Types

```rust
// World-space position — ALWAYS use this, never raw Vec3
struct WorldPosition { chunk: IVec3, local: Vec3 }

// Scene hierarchy — objects own SDF in local space
SceneObject { id: u32, world_position: WorldPosition, rotation: Quat, scale: Vec3, root_node: SceneNode, aabb: Aabb }
SceneNode { name, local_transform: Transform, sdf_source: SdfSource, blend_mode: BlendMode, children: Vec<SceneNode> }
SdfSource::Analytical { primitive, material_id } | SdfSource::Voxelized { brick_map_handle, voxel_size, aabb }

// Per-object brick map — flat 3D array mapping brick coords to pool slots
BrickMap { dims: UVec3, entries: Vec<u32> }  // EMPTY_SLOT = u32::MAX

// Geometry-first data model (source of truth):
// BrickGeometry { occupancy: [u64; 8], surface_voxels: Vec<SurfaceVoxel> }
//   occupancy: 512-bit bitmask — bit N = voxel N is solid
//   SurfaceVoxel { index: u16, color: [u8; 4], material_id: u8, _reserved: u8 }
// SdfCache { distances: [u16; 512] }  // f16 distances, DERIVED from geometry

// GPU format (bridge pattern — Brick::from_geometry converts geometry→GPU):
// VoxelSample { word0: u32, word1: u32 }  // 8 bytes
//   word0: lower 16 = f16 distance (from SdfCache), bits 16-23 = u8 material_id
//   word1: RGBA8 per-voxel color (from SurfaceVoxel.color)
// Brick: 8×8×8 = 512 VoxelSamples = 4KB (GPU upload format)

// Material: 96 bytes — PBR + SSS + procedural noise
// Max 256 materials (u8 index), stored in GPU storage buffer
```

## Render Pipeline (all compute)

```
1. Update transforms → flatten SceneNode trees → upload GpuObject metadata
2. BVH refit → upload GPU BVH nodes
3. Tile-based object culling → per-tile object lists
4. Particle Simulate
5. Ray March → BVH traversal → per-object SDF evaluation → G-buffer (internal res)
6. Tile Light Cull → per-tile light lists
7. Radiance Inject → GI volume L0 (coarse field + BVH for SDF queries)
8. Radiance Mip Gen → GI volume L1-L3
9. Shade (PBR + shadows + AO + SSS + GI + atmosphere)
10. Volumetric Shadow Map + Cloud Shadow Map
11. Volumetric March (half-res) → bilateral upscale
12. Volumetric Composite
13. Pre-upscale post-process (bloom, DoF, motion blur)
14. Upscale (spatial bilinear + sharpen)
15. Post-upscale post-process (tone map, color grade)
16. Screen-space particles
17. Present
```

## Novel/Unusual Patterns — Don't Get These Wrong

**Object-centric SDF.** Each object owns its SDF in local space (brick map or analytical primitive). Transforms are applied at ray march time — moving an object never triggers re-voxelization. Objects have persistent identity throughout their lifecycle.

**BVH-accelerated ray marching.** A CPU-side BVH over object AABBs is uploaded to GPU. The ray marcher traverses the BVH to find candidate objects, then evaluates each in object-local space. Tile-based culling produces per-tile object lists for further acceleration.

**Skeletal animation** uses a SceneNode tree where bones are child nodes with voxelized SDF. Animation updates bone local transforms (no rebaking). Smooth-min blending between sibling nodes produces natural joints during ray marching.

**SDF collision** for physics: Rapier handles dynamics, but world collision iterates scene objects via BVH, evaluating SDF in object-local space. Character controller is custom (capsule-vs-SDF, iterative slide with 14 sample points).

**Three particle backends:** Volumetric density splats (glowing), SDF micro-objects (solid), screen-space overlay (weather). NOT traditional billboards. Collision uses per-object BVH queries.

**Geometry-first architecture.** Occupancy bitmask (`[u64; 8]` = 512 bits per brick) is the source of truth for shape. SDF distances are a derived cache computed from geometry via CPU Dijkstra propagation. Per-voxel color and material live in `SurfaceVoxel` (only boundary voxels carry data). A bridge pattern (`Brick::from_geometry`) converts geometry-first data to the GPU's VoxelSample format for ray marching.

**Per-object editing.** Sculpt operations modify occupancy bits and surface voxel properties in local space. After edit, SDF distances are recomputed for the affected region. Undo/redo snapshots geometry bricks, not SDF. Brush positions transform from world to object-local before applying.

**Floating point precision** solved via WorldPosition (IVec3 chunk + Vec3 local). CPU does f64 subtraction, GPU only sees camera-relative f32. Effective range ±17 billion meters.

**Per-object streaming.** Objects load from `.rkf` v2/v3 files with multi-LOD LZ4 compression (coarsest first). v3 stores geometry-first data (occupancy + surface voxels + optional SDF cache). Streaming priority = screen_coverage × importance_bias. LRU eviction with watermark-based policy demotes LOD before full eviction.

**MCP-native engine.** The engine is designed to be operated by AI agents. `rkf-mcp` is a standalone MCP server binary that connects to any running engine process via IPC. Tools self-register via a discovery system — add new tools by implementing a trait, not modifying the server. Every feature ships with MCP tools. If the MCP is broken, that's priority zero.

## MCP Integration

The engine is agent-native. `rkf-mcp` is the MCP server that bridges Claude Code (or any MCP client) to any running engine process (editor, testbed, game).

### Architecture

```
Claude Code ←stdio→ rkf-mcp ←IPC (Unix socket)→ rkf-editor / rkf-testbed / rkf-game
```

- **rkf-mcp** (`crates/rkf-mcp/`) — standalone binary, runs over **stdio** for Claude Code. Connects to engine processes over IPC (Unix domain sockets at `/tmp/rkifield-{pid}.sock`).
- **AutomationApi** trait (`rkf-core/src/automation.rs`) — the engine's control surface. Defines all observation and mutation methods. Implemented by each binary (editor, testbed, game).
- **BridgeAutomationApi** (`rkf-mcp/src/bridge.rs`) — IPC proxy that forwards `AutomationApi` calls from `rkf-mcp` to the engine over the socket.
- **EditorAutomationApi** (`rkf-editor/src/automation.rs`) — editor's implementation backed by `SharedState` (render loop data) and `EditorState` (scene tree, tools, etc.).
- **ToolRegistry** (`rkf-mcp/src/registry.rs`) — self-describing tool system. Tools register with full metadata (name, description, params, return type, mode). MCP `tools/list` is generated dynamically from the registry.

### Auto-Discovery & Auto-Connect

On startup, `rkf-mcp` automatically:
1. Scans `/tmp/rkifield-*.sock` for running engine sockets
2. If exactly one socket found and connectable → auto-connects (no manual `connect` call needed)
3. Reads `/tmp/rkifield-{pid}.json` discovery metadata for engine type/name
4. If multiple sockets → logs the list, user picks via `connect` tool

The editor writes discovery metadata on startup:
```json
// /tmp/rkifield-{pid}.json
{ "type": "editor", "pid": 12345, "socket": "/tmp/rkifield-12345.sock", "name": "RKIField Editor", "version": "0.1.0" }
```
Cleaned up on exit (close, Escape, File > Quit).

### Available MCP Tools

**Observation (both Editor and Debug modes):**

| Tool | Description |
|------|-------------|
| `screenshot` | Capture viewport as PNG (returns image content block) |
| `camera_get` | Current camera position, orientation, FOV |
| `camera_set` | Teleport camera to position + yaw/pitch |
| `scene_graph` | List all entities with hierarchy and transforms |
| `entity_inspect` | Read all components of a specific entity |
| `render_stats` | Frame time, pass timings, brick pool usage, memory |
| `brick_pool_stats` | Brick pool capacity, allocated, free list |
| `spatial_query` | Sample SDF distance and material at a world position |
| `asset_status` | Loaded chunks, pending uploads, total bricks |
| `log_read` | Recent engine log entries (500-entry ring buffer) |
| `debug_mode` | Set shading visualization (0=normal, 1=normals, 2=positions, 3=material IDs, 4=diffuse, 5=specular) |

**Meta (both modes):**

| Tool | Description |
|------|-------------|
| `connect` | Connect to engine via IPC socket (auto-discovers if no arg) |
| `disconnect` | Revert to stub API |
| `status` | Connection state, available engines, server version |

**Mutation (forwarded over IPC bridge):**

| Tool | Description |
|------|-------------|
| `entity_spawn` | Spawn entity with components |
| `entity_despawn` | Remove entity by ID |
| `entity_set_component` | Add/replace component on entity |
| `material_set` | Update material table entry |
| `brush_apply` | Apply CSG/sculpt brush operation |
| `scene_load` | Load `.rkscene` file |
| `scene_save` | Save current scene to file |
| `quality_preset` | Switch render quality preset |

### Key Implementation Details

- **SharedState** (`Arc<Mutex<SharedState>>`) — render loop writes camera pos, frame pixels, pool stats, frame time each frame. MCP reads it for observation tools. Also holds a 500-entry log ring buffer (`push_log()`).
- **EditorState** (`Arc<Mutex<EditorState>>`) — scene tree, tool states, undo stack. MCP reads it for `scene_graph` and `entity_inspect`.
- **Swappable API slot** — `Arc<RwLock<Arc<dyn AutomationApi>>>`. Meta tools (connect/disconnect) swap the inner Arc. Tool dispatch clones the Arc and releases the lock before calling, preventing deadlocks.
- **Tool modes** — `ToolMode::Editor` (full access), `ToolMode::Debug` (observation only), `ToolMode::Both`. Mutation tools are `Editor`-only.
- **Screenshot** returns `ContentBlock::Image` (base64 PNG), not text. Bridge uses `call_tool_raw` (not `call_tool`) to handle this.
- **camera_set and debug_mode** route through `execute_command()` string-based dispatch for IPC simplicity.
- **All logging to stderr** in rkf-mcp — stdout is the MCP protocol channel.

### Adding New MCP Tools

1. Create a handler struct implementing `ToolHandler` trait in `rkf-mcp/src/tools/`
2. Register it in the appropriate `register_*_tools()` function with a `ToolDefinition`
3. Add the corresponding `AutomationApi` method in `rkf-core/src/automation.rs` if it touches engine state
4. Implement the method in `EditorAutomationApi` (and testbed/game equivalents)
5. Forward the method in `BridgeAutomationApi` for IPC bridge support

### Configuration

MCP server is configured in `.mcp.json` at the project root:
```json
{
  "mcpServers": {
    "rkf-mcp": {
      "command": "cargo",
      "args": ["run", "-p", "rkf-mcp", "--"],
      "cwd": "/home/joe/dev/rkifield"
    }
  }
}
```

## File Formats

| Format | Extension | Purpose |
|--------|-----------|---------|
| RKIField Asset v2/v3 | `.rkf` | Per-object multi-LOD voxel data, LZ4 compressed; v3 = geometry-first (occupancy + surface voxels + optional SDF cache) |
| Scene v2 | `.rkscene` | RON-serialized scene (object hierarchy, cameras, lights, environment) |
| Project | `.rkproject` | RON-serialized project descriptor (scene list, asset paths, quality) |
| Environment | `.rkenv` | RON-serialized environment profile (sky, fog, ambient, volumetrics) |
| Save | `.rksave` | RON-serialized game state snapshot (scenes, camera, state, overrides) |
| Config | `.ron` | Engine configuration, quality presets |

## Architecture Reference

**v2 (current):** Object-centric SDF engine — the authoritative design:

- [v2 Architecture](docs/v2/ARCHITECTURE.md) — object-centric scene hierarchy, per-object SDF, BVH acceleration, transform-at-march-time
- [v2 Implementation Plan](docs/v2/IMPLEMENTATION_PLAN.md) — 16 phases, 82 tasks, dependency graph
- [Geometry-First Architecture](docs/v2/GEOMETRY_FIRST_ARCHITECTURE.md) — occupancy bitmask source of truth, SDF as derived cache
- [Geometry-First Implementation Plan](docs/v2/GEOMETRY_FIRST_IMPLEMENTATION_PLAN.md) — 8 phases, geometry-first migration

**v1 (superseded):** Chunk-based engine — retained for reference only:

- [Architecture Hub](docs/ARCHITECTURE.md) — overview, dependencies, crate map
- [Core Data Structure](docs/architecture/01-core-data-structure.md) — sparse grid, clipmap, brick pool
- [Material System](docs/architecture/02-material-system.md) — material table, blending, color pool
- [Rendering Pipeline](docs/architecture/03-rendering-pipeline.md) — ray march, shading, GI, atmosphere
- [Skeletal Animation](docs/architecture/04-skeletal-animation.md) — segmented rebaking, blend shapes
- [Volumetric Effects](docs/architecture/05-volumetric-effects.md) — fog, clouds, god rays, fluid sim
- [Asset Import](docs/architecture/06-asset-import.md) — mesh-to-SDF, `.rkf` format
- [Procedural Editing](docs/architecture/07-procedural-editing.md) — CSG, brushes, undo, edit journal
- [Upscaling & Post-Processing](docs/architecture/08-upscaling-post-processing.md) — DLSS, temporal, post-FX
- [Particle System](docs/architecture/09-particle-system.md) — three backends, GPU simulation
- [Physics](docs/architecture/10-physics.md) — Rapier adapter, character controller
- [Engine Architecture](docs/architecture/11-engine-architecture.md) — ECS, streaming, editor, config
- [MCP Integration](docs/architecture/12-mcp-integration.md) — agent-native engine, tool discovery, automation API
- [v1 Implementation Plan](docs/IMPLEMENTATION_PLAN.md) — 24 phases, ~160 tasks

## Coding Conventions

- **Shader code is WGSL** — not GLSL, not HLSL. All GPU work is `@compute` dispatches.
- **GPU structs use `bytemuck` or `encase`** for Rust ↔ GPU layout. Match WGSL alignment rules.
- **Geometry is source of truth, SDF is derived.** Occupancy bitmask + surface voxels are authoritative. SDF distances are computed from geometry via CPU Dijkstra and cached. Never edit SDF directly — edit geometry, then recompute SDF.
- **Brick pool access is centralized** through `BrickPoolManager`. Never write to the brick pool buffer directly. The GPU brick pool holds VoxelSample data converted from geometry via `Brick::from_geometry()`.
- **All world positions flow through WorldPosition.** The only f32 Vec3 positions on the GPU are camera-relative or object-local.
- **BVH is the spatial acceleration structure.** CPU-side BVH over object AABBs is uploaded to GPU. Objects not in the BVH won't be ray marched.
- **Per-object brick maps.** Each voxelized object owns a `BrickMap` — a flat 3D array mapping brick coordinates to pool slots. `BrickMapAllocator` packs multiple maps contiguously.
- **Error handling:** `anyhow` for CLI tools, typed errors for library crates. GPU errors are wgpu device errors — log and recover, don't panic.
- **Testing is TDD.** Write the test first, watch it fail, then implement. Unit tests for math/data types in `rkf-core`. Visual regression via `rkf-testbed` screenshots. GPU tests use wgpu in headless mode where possible. Every public function in library crates has at least one test.
- **No `unsafe`** unless required for FFI (DLSS) or proven-necessary GPU buffer mapping. Document every `unsafe` block.

## Implementation Progress

**Source of truth: git history.** Commit messages use `v2-N.M` prefixes (e.g., `v2-12.1: .rkf v2 file format`). Progress is derived from commits, not a manually-maintained checklist.

**Check progress:**
```bash
# Which v2 phases have commits?
git log --oneline | grep -oP 'v2-\d+' | sort -t- -k2 -n -u

# Which tasks are done in a phase?
git log --oneline --grep='v2-12'

# Full commit history
git log --oneline --grep='v2-'
```

**All 16 phases (0–15) of the v2 implementation plan are implemented.** 1,680+ tests, 0 failures.

See [v2 Implementation Plan](docs/v2/IMPLEMENTATION_PLAN.md) for the full 16-phase, 82-task plan.
