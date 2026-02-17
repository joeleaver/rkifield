# RKIField

A real-time graphics engine where **signed distance fields are the only geometry**.

No meshes. No triangles. No UV mapping. Material data lives in the volume. Everything renders via compute shader ray marching through voxel-backed distance fields.

---

## What Makes This Different

Traditional engines rasterize triangle meshes. RKIField doesn't have meshes at all. Every surface, every object, every piece of terrain is a signed distance field stored in a sparse voxel brick pool. The renderer is a pure compute shader pipeline — ray marching through multi-level DDA traversal with sphere tracing at the brick level.

This unlocks things that are hard or impossible with meshes:

- **Real-time sculpting** — CSG operations modify the SDF directly on the GPU. Carve, add, smooth, paint. No remeshing, no retopology.
- **Volumetric materials** — Materials are intrinsic to the field, not projected onto surfaces. Blend between stone and dirt at the voxel level. Per-voxel color without UV maps.
- **Native volumetric effects** — Fog, clouds, smoke, and fire use the same data structures and marching infrastructure as solid geometry. Not bolted on — built in.
- **SDF-native physics** — The world itself is queryable as a distance field. Character controllers and rigid bodies collide against the actual SDF, not a simplified proxy mesh.
- **Skeletal animation without meshes** — Characters are segmented into rigid SDF parts with smooth-min blending at joints. A novel approach with no mesh deformation pipeline.

## Agent-Native Engine

RKIField is designed to be operated by LLMs and AI agents as a first-class use case.

The engine exposes its full functionality through the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) via a standalone `rkf-mcp` server binary. Agents can connect and autonomously:

- Take screenshots and inspect render buffers
- Query the scene graph and entity components
- Spawn, transform, and despawn entities
- Edit materials and apply brush operations
- Load and save scenes
- Monitor performance and asset loading

Two modes: **Editor MCP** (full scene authoring) and **Debug MCP** (read-only observation for testing and CI).

## Tech Stack

| | |
|---|---|
| **Language** | Rust |
| **GPU API** | wgpu (WebGPU) |
| **Shaders** | WGSL compute-only |
| **Physics** | Rapier + custom SDF collision adapter |
| **ECS** | hecs |
| **Editor UI** | rinch |
| **Upscaling** | DLSS (preferred) + custom temporal fallback |

## Architecture

The engine is organized as a Cargo workspace with 13 crates:

```
crates/
  rkf-core/        Voxel data structures, brick pool, spatial index, WorldPosition
  rkf-render/      Ray march, shading, GI, volumetrics, upscale, post-process
  rkf-animation/   Segmented joint rebaking, blend shape delta-SDFs
  rkf-edit/        GPU CSG operations, sculpt brushes, undo/redo, edit journal
  rkf-import/      Mesh-to-SDF conversion (BVH + generalized winding number)
  rkf-physics/     Rapier integration, SDF collision adapter, character controller
  rkf-particles/   GPU particle sim, 3 render backends
  rkf-runtime/     Frame scheduling, ECS, streaming, asset management
  rkf-mcp/         MCP server (tool discovery, automation API)
  rkf-editor/      Editor UI, gizmos, tools, scene serialization
  rkf-convert/     Offline mesh-to-.rkf CLI converter
  rkf-game/        Example game / playground
  rkf-testbed/     Visual testing harness
```

## Render Pipeline

All compute shaders. No rasterization (except editor gizmo wireframes).

```
 1. Animation Rebake (joints + blend shapes)
 2. Particle Simulate
 3. Ray March --> G-buffer (internal res)
 4. Tile Light Cull
 5. Radiance Inject --> GI volume
 6. Radiance Mip Gen
 7. Shade (PBR + shadows + AO + SSS + GI + atmosphere)
 8. Volumetric Shadow Map + Cloud Shadow Map
 9. Volumetric March (half-res) --> bilateral upscale
10. Volumetric Composite
11. Pre-upscale post-process (bloom, DoF, motion blur)
12. Upscale (DLSS or custom temporal)
13. Post-upscale post-process (tone map, color grade)
14. Screen-space particles
15. Present
```

## Documentation

Comprehensive architecture docs cover every design decision:

| Topic | Document |
|-------|----------|
| Architecture Hub | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| Voxel Data Structure | [01 - Core Data Structure](docs/architecture/01-core-data-structure.md) |
| Material System | [02 - Material System](docs/architecture/02-material-system.md) |
| Rendering Pipeline | [03 - Rendering Pipeline](docs/architecture/03-rendering-pipeline.md) |
| Skeletal Animation | [04 - Skeletal Animation](docs/architecture/04-skeletal-animation.md) |
| Volumetric Effects | [05 - Volumetric Effects](docs/architecture/05-volumetric-effects.md) |
| Asset Import | [06 - Asset Import](docs/architecture/06-asset-import.md) |
| Procedural Editing | [07 - Procedural Editing](docs/architecture/07-procedural-editing.md) |
| Upscaling & Post-Processing | [08 - Upscaling & Post-Processing](docs/architecture/08-upscaling-post-processing.md) |
| Particle System | [09 - Particle System](docs/architecture/09-particle-system.md) |
| Physics | [10 - Physics](docs/architecture/10-physics.md) |
| Engine Architecture | [11 - Engine Architecture](docs/architecture/11-engine-architecture.md) |
| MCP Integration | [12 - MCP Integration](docs/architecture/12-mcp-integration.md) |
| Implementation Plan | [24-phase build plan](docs/IMPLEMENTATION_PLAN.md) |

## Building

```bash
cargo build --workspace          # build everything
cargo test --workspace           # run all tests
cargo clippy --workspace         # lint
cargo run -p rkf-testbed         # visual test harness
cargo run -p rkf-editor          # editor
cargo run -p rkf-convert -- input.glb -o output.rkf
```

## Status

Architecture design is complete. Implementation follows a [24-phase plan](docs/IMPLEMENTATION_PLAN.md) with ~160 tasks, starting from workspace scaffolding through to a full editor and example game.

## License

TBD
