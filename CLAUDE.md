# RKIField — SDF Graphics Engine

## What This Is

A real-time graphics engine where **signed distance fields are the only geometry**. No meshes, no triangles, no UV mapping. Material data lives in the volume. Everything renders via compute shader ray marching through voxel-backed distance fields.

This is a novel engine — not a wrapper around an existing renderer. Many subsystems (skeletal animation via segmented joint rebaking, SDF soft shadows, voxel cone tracing from native data) have no direct precedent in shipping engines.

## Critical Rules

1. **No meshes, ever.** Every geometric primitive is an SDF stored in voxel bricks. If you're tempted to add a mesh path, you're solving the wrong problem.
2. **No textures on surfaces.** Materials are volumetric — PBR properties come from a global material table indexed by `material_id`. Per-voxel color comes from companion color bricks, not UV-mapped textures.
3. **Uniform scale only.** Non-uniform scale distorts SDF distances and breaks ray marching. Objects that need stretching must be re-voxelized.
4. **All rendering is compute shaders.** No rasterization pipeline except editor gizmo wireframes.
5. **WorldPosition everywhere.** Never use raw `Vec3` for world-space positions. Always `WorldPosition { chunk: IVec3, local: Vec3 }` to avoid float precision loss. GPU receives camera-relative f32 only.
6. **Brick pool is the central resource.** All voxel data lives in a single GPU brick pool with companion pools (bone, volumetric, color). The `BrickPoolManager` coordinates streaming, animation, and editing.
7. **The architecture docs are the spec. Follow them exactly.** Never substitute simpler alternatives, skip features, or deviate from the documented designs. If the architecture says "multi-level DDA + sphere tracing," implement multi-level DDA + sphere tracing — not a simplified single-level version. When in doubt, read the relevant architecture doc. The decisions were made deliberately.
8. **Test-driven development.** Write tests before implementation. Every new type, algorithm, and pipeline pass gets tests first. Red → green → refactor. Target 80%+ coverage on library crates. GPU code that can't be unit-tested gets visual regression tests in `rkf-testbed`.
9. **Commit after every step.** Each completed task within a phase gets its own atomic commit. Don't batch multiple tasks into one commit. Commit messages reference the phase and task number (e.g., `phase-1: 1.1 — implement WorldPosition type`). This creates a clean, reviewable history that maps directly to the implementation plan.
10. **Ask questions, don't assume.** When requirements are ambiguous, a design decision has multiple valid paths, or you're unsure what the user wants — stop and ask. A 30-second question saves hours of rework. Never silently choose a simpler alternative, skip a feature, or reinterpret a requirement. If the architecture docs don't cover something, ask. If a task feels underspecified, ask. Assumptions are bugs.
11. **Agents must use the MCP.** This engine is designed to be operated by LLMs and AI agents. When developing or testing, always connect to the engine via `rkf-mcp` and use MCP tools for screenshots, scene inspection, entity manipulation, and validation. **If the MCP server doesn't exist, is disconnected, or doesn't work — fixing it is the top priority.** No other work proceeds until agent tooling is functional. Every new feature must ship with corresponding MCP tools.

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

// Voxel sample — 8 bytes, tightly packed
// Word 0: f16 distance | u16 material_id
// Word 1: u8 blend_weight | u8 secondary_id | u8 flags | u8 reserved

// Brick: 8×8×8 = 512 voxel samples = 4KB
// Resolution tiers: 0.5cm, 2cm, 8cm, 32cm voxel sizes
// Chunk size: 8m × 8m × 8m

// Material: 96 bytes — PBR + SSS + procedural noise
// Max 65536 materials (u16), stored in GPU storage buffer
```

## Render Pipeline (all compute)

```
1. Animation Rebake (joints + blend shapes)
2. Particle Simulate
3. Ray March → G-buffer (internal res, e.g. 960×540)
4. Tile Light Cull → per-tile light lists
5. Radiance Inject → GI volume L0
6. Radiance Mip Gen → GI volume L1-L3
7. Shade (PBR + shadows + AO + SSS + GI + atmosphere)
8. Volumetric Shadow Map + Cloud Shadow Map
9. Volumetric March (half-res) → bilateral upscale
10. Volumetric Composite
11. Pre-upscale post-process (bloom, DoF, motion blur)
12. Upscale (DLSS or custom temporal)
13. Post-upscale post-process (tone map, color grade)
14. Screen-space particles
15. Present
```

## Novel/Unusual Patterns — Don't Get These Wrong

**Skeletal animation** uses segmented rigid body parts + joint rebaking (smooth-min blend in a compute shader). Bones are NOT evaluated during ray marching. Joint bricks use 0.8× conservative step multiplier for Lipschitz safety.

**SDF collision** for physics: Rapier handles dynamics, but world collision is a custom adapter that evaluates the SDF at contact sample points. Character controller is custom (capsule-vs-SDF, iterative slide).

**Three particle backends:** Volumetric density splats (glowing), SDF micro-objects (solid), screen-space overlay (weather). NOT traditional billboards.

**Edit persistence** uses per-chunk append-only journals (`.rkj`) with 64-byte `CompactEditOp` entries. Replayed on chunk load. Compacted when large.

**Floating point precision** solved via WorldPosition (IVec3 chunk + Vec3 local). CPU does f64 subtraction, GPU only sees camera-relative f32. Effective range ±17 billion meters.

**MCP-native engine.** The engine is designed to be operated by AI agents. `rkf-mcp` is a standalone MCP server binary that connects to any running engine process via IPC. Tools self-register via a discovery system — add new tools by implementing a trait, not modifying the server. Every feature ships with MCP tools. If the MCP is broken, that's priority zero.

## File Formats

| Format | Extension | Purpose |
|--------|-----------|---------|
| RKIField Asset | `.rkf` | Binary voxel data (bricks + skeleton + materials), LZ4 compressed |
| Edit Journal | `.rkj` | Append-only edit history per chunk, 64-byte entries |
| Scene | `.rkscene` | RON-serialized scene definition (entities, environment, chunk manifest) |
| Config | `.ron` | Engine configuration, quality presets |

## Architecture Reference

Full design docs with all decisions, code snippets, data layouts, and upgrade paths:

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
- [Implementation Plan](docs/IMPLEMENTATION_PLAN.md) — 24 phases, ~160 tasks, dependency graph

## Coding Conventions

- **Shader code is WGSL** — not GLSL, not HLSL. All GPU work is `@compute` dispatches.
- **GPU structs use `bytemuck` or `encase`** for Rust ↔ GPU layout. Match WGSL alignment rules.
- **Brick pool access is centralized** through `BrickPoolManager`. Never write to the brick pool buffer directly.
- **All world positions flow through WorldPosition.** The only f32 Vec3 positions on the GPU are camera-relative or chunk-local.
- **Spatial index is the source of truth** for what exists in the world. If it's not in the index, the ray marcher won't see it.
- **Error handling:** `anyhow` for CLI tools, typed errors for library crates. GPU errors are wgpu device errors — log and recover, don't panic.
- **Testing is TDD.** Write the test first, watch it fail, then implement. Unit tests for math/data types in `rkf-core`. Visual regression via `rkf-testbed` screenshots. GPU tests use wgpu in headless mode where possible. Every public function in library crates has at least one test.
- **No `unsafe`** unless required for FFI (DLSS) or proven-necessary GPU buffer mapping. Document every `unsafe` block.

## Implementation Progress

**Source of truth: git history.** Rule 9 mandates commit messages like `phase-1: 1.1 — implement WorldPosition type`. Progress is derived from commits, not a manually-maintained checklist.

**Check progress:**
```bash
# Which phases have commits?
git log --oneline | grep -oP 'phase-\d+' | sort -t- -k2 -n -u

# Which tasks are done in a phase?
git log --oneline --grep='phase-2:'

# Full progress summary (all phases)
scripts/progress.sh
```

**`scripts/progress.sh`** — generates progress from git log. Created during Phase 0 scaffolding. Parses `phase-N: N.X` commit prefixes against `IMPLEMENTATION_PLAN.md` task list and prints completion status per phase.

**Never manually maintain a progress checklist.** If git says it's done, it's done. If there's no commit, it's not done. This cannot get out of sync.

See [IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) for the full 24-phase plan with ~160 tasks.
