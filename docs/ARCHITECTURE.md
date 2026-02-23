> **SUPERSEDED** by [v2 Architecture](v2/ARCHITECTURE.md) — this document describes the v1 chunk-based engine.

# RKIField — SDF Graphics Engine Architecture

## Overview

RKIField is a real-time graphics engine that uses signed distance fields (SDFs) as its sole geometric representation. It replaces traditional mesh-based rendering with volumetric ray marching through voxel-backed distance fields. Material data is stored directly in the volume — no textures, UV mapping, or projection. The engine targets low internal resolution with upscaling to final output.

**Tech stack:** Rust + wgpu (WebGPU) + WGSL compute shaders

---

## Design Principles

1. **No meshes, ever** — All geometry is represented as signed distance fields.
2. **Materials are volumetric** — Material identity is intrinsic to the field, not mapped onto surfaces.
3. **Render low, upscale high** — Internal rendering at ~1/4 resolution, with temporal and spatial upscaling to final output.
4. **Voxels as the universal format** — All SDFs are backed by voxel grids. Procedural SDFs are voxelized before rendering. This unifies storage, editing, animation, and import.
5. **Volumetric effects are first-class** — Fog, clouds, smoke, fire, and subsurface scattering use the same data structures and marching infrastructure as geometry.

---

## Architecture Documents

| # | Topic | Document |
|---|-------|----------|
| 1 | **Voxel Data Structure** — Sparse grid, clipmap LOD, 8³ bricks, 4 resolution tiers, 8-byte voxel, companion pools | [01-core-data-structure.md](./architecture/01-core-data-structure.md) |
| 2 | **Material System** — Global table (96B/material, u16 indexed), PBR+SSS+noise, linear blend, companion color pool | [02-material-system.md](./architecture/02-material-system.md) |
| 3–4 | **Rendering Pipeline** — Compute-only ray march, multi-level DDA + sphere tracing, 4-target G-buffer, Cook-Torrance shading, tiled lighting, voxel cone tracing GI, Bruneton atmosphere | [03-rendering-pipeline.md](./architecture/03-rendering-pipeline.md) |
| 5 | **Skeletal Animation** — Segmented + joint rebaking (novel), blend shape delta-SDFs, 0.8× conservative step | [04-skeletal-animation.md](./architecture/04-skeletal-animation.md) |
| 6 | **Volumetric Effects** — Half-res pass, vol shadow map, brick-backed fog, dual cloud system, GPU fluid sim, HG phase | [05-volumetric-effects.md](./architecture/05-volumetric-effects.md) |
| 7 | **Asset Import** — Offline `rkf-convert` CLI, BVH+winding number SDF, auto segmentation, `.rkf` binary format | [06-asset-import.md](./architecture/06-asset-import.md) |
| 8 | **Procedural Editing** — GPU compute CSG, 7 brush types, delta undo/redo, edit journal `.rkj`, 64-byte edit format | [07-procedural-editing.md](./architecture/07-procedural-editing.md) |
| 9 | **Upscaling & Post-Processing** — DLSS preferred + custom temporal fallback, material ID rejection, split post-process stack | [08-upscaling-post-processing.md](./architecture/08-upscaling-post-processing.md) |
| 10a | **Particle System** — Three render backends (volumetric/SDF micro/screen-space), GPU compute simulation | [09-particle-system.md](./architecture/09-particle-system.md) |
| 10b | **Physics** — Rapier + SDF collision adapter, custom character controller, lazy destruction sync | [10-physics.md](./architecture/10-physics.md) |
| 10 | **Engine Architecture** — Hybrid ECS, WorldPosition precision, static frame graph, rinch editor, 13-crate workspace | [11-engine-architecture.md](./architecture/11-engine-architecture.md) |
| 12 | **MCP Integration** — Agent-native engine, `rkf-mcp` binary, AutomationApi trait, tool discovery registry, editor/debug modes | [12-mcp-integration.md](./architecture/12-mcp-integration.md) |

All sections are **DECIDED**. See [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) for the detailed 24-phase build plan.

---

## Platform and Dependencies

### Confirmed

| Component | Choice |
|-----------|--------|
| Language | Rust |
| GPU API | wgpu (WebGPU) |
| Shader Language | WGSL |
| Windowing | winit |
| Math | glam |
| Physics | Rapier |
| DLSS | dlss_wgpu |
| Editor UI | rinch |
| ECS | hecs |
| Serialization | RON |

### To Evaluate

| Component | Options |
|-----------|---------|
| Shader struct layout | encase, bytemuck |
| Asset loading (glTF) | gltf crate |
| Asset loading (FBX) | russimp |
| Image I/O | image crate |
| Profiling | puffin, tracy |
| Compression | lz4_flex, zstd |

---

## Crate Structure

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
    rkf-mcp/         — MCP server binary (agent automation interface)
    rkf-testbed/     — permanent visual testing binary
```

---

## Planning Roadmap

All 10 planning sessions have been completed. Every section has been designed interactively and documented.

### Next Steps

Architecture design is complete. Implementation can begin. See [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) for the detailed 24-phase build plan.

**Recommended implementation order:**

1. **rkf-mcp** — MCP server foundation, AutomationApi trait, tool discovery registry
2. **rkf-core** — WorldPosition, brick pool, spatial index, voxel data structures
3. **rkf-render** (ray march only) — basic ray marching with hardcoded test SDF
4. **rkf-render** (shading) — PBR shading, SDF shadows, AO
5. **rkf-core** (materials) — material table, per-voxel material fields
6. **rkf-render** (GI) — radiance volume, voxel cone tracing
7. **rkf-render** (upscale) — DLSS + custom temporal, post-process stack
8. **rkf-render** (volumetrics) — volumetric pass, fog, clouds
9. **rkf-animation** — skeletal animation, joint rebaking, blend shapes
10. **rkf-edit** — CSG operations, brushes, undo/redo
11. **rkf-import** / **rkf-convert** — mesh-to-SDF conversion CLI
12. **rkf-particles** — particle system, emitters
13. **rkf-physics** — Rapier integration, character controller
14. **rkf-runtime** — frame scheduling, ECS, streaming, asset management
15. **rkf-editor** — rinch UI, tools, gizmos
16. **rkf-game** — example playground
