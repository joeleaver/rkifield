> **SUPERSEDED** by [v2 Architecture](../v2/ARCHITECTURE.md) — this document describes the v1 chunk-based engine.

# Particle System

> **Status: DECIDED**

### Decision: Three Render Backends, Shared Simulation

Many traditional particle effects (smoke, fire, steam, fog wisps) are already handled by the volumetric system (see [Volumetric Effects](./05-volumetric-effects.md)). The particle system covers the remaining effects: debris, sparks, rain, snow, dust motes, splatter, muzzle flash, magical effects.

**Shared GPU particle simulation:**
All particles share one GPU compute simulation pass regardless of render backend. A persistent particle buffer holds all live particles. Emitters spawn particles with a type tag that determines which render backend draws them.

```rust
/// GPU particle — 48 bytes
struct Particle {
    position: [f32; 3],         // world space
    lifetime: f32,              // remaining (seconds)
    velocity: [f32; 3],         // world space
    max_lifetime: f32,          // initial (for age ratio)
    color_emission: [f16; 4],   // RGB color + emission intensity
    size: f16,                  // radius (world units)
    render_type: u8,            // VOLUMETRIC=0, SDF_MICRO=1, SCREEN=2
    flags: u8,                  // bit 0: gravity, bit 1: collision, bit 2: fade_out
    _pad: [u8; 2],
}
```

**Simulation (GPU compute, early in frame):**
```
1. Spawn: emitters append new particles via atomic counter
2. Integrate: position += velocity * dt, apply gravity/wind/drag
3. Collide (if flagged): evaluate SDF at position, push out if negative
4. Age: lifetime -= dt, kill if <= 0
5. Compact: stream compaction to remove dead particles
```

**Particle budget:** 10K–100K particles total. Configurable per quality preset.

### Decision: Volumetric Particles (Density Splats)

Each volumetric particle is a small density + emission kernel evaluated analytically during the volumetric march. Not stored in the brick pool — the volumetric march shader reads the particle buffer and accumulates contributions from nearby particles at each step.

```wgsl
fn particle_density_at(pos: vec3f, particle: Particle) -> vec2f {
    let d = distance(pos, particle.position);
    if d > particle.size { return vec2f(0.0); }
    let falloff = 1.0 - smoothstep(0.0, particle.size, d);
    let age = 1.0 - particle.lifetime / particle.max_lifetime;
    let density = falloff * (1.0 - age);  // fade with age
    let emission = falloff * particle.color_emission.w;
    return vec2f(density, emission);
}
```

**Acceleration:** Particles are binned into a 3D grid (same as tile culling). Each volumetric march step only iterates particles in the local grid cell. Keeps per-step cost bounded.

**Good for:** sparks, embers, dust motes, muzzle flash glow, magical effects, fireflies — anything that glows or scatters light.

### Decision: SDF Micro-Objects (Solid Particles)

Small SDF primitives (spheres, capsules) registered as temporary entries in the spatial index. The ray marcher hits them like any other geometry — full shading, shadows, GI contribution.

```rust
struct SdfMicroObject {
    shape: MicroShape,          // Sphere or Capsule
    position: [f32; 3],
    radius: f32,
    material_id: u16,
    particle_index: u32,        // back-reference to particle buffer
}
```

Registered in a dedicated region of the spatial index, updated each frame from the particle buffer. The ray marcher evaluates these analytically (no bricks needed — simple SDF primitives).

**Budget:** ~1000 simultaneous SDF micro-objects. Each is a single analytic SDF evaluation during ray march, so cost scales with screen coverage not particle count.

**Good for:** debris, shrapnel, hail, pebbles, shell casings — anything that needs to look solid and interact with lighting.

### Decision: Screen-Space Particles (Post-Process Overlay)

A lightweight pass after post-processing that composites oriented streaks or dots onto the final image. Uses the depth buffer for occlusion but doesn't interact with the SDF pipeline.

```wgsl
// Per screen-space particle:
//   Project position to screen
//   Depth test against G-buffer depth
//   Draw oriented streak (rain) or point (snow) with alpha blend
```

**Good for:** rain streaks, snowfall, lens dust, ambient floating particles. Cheap — bypasses the entire SDF pipeline.

### Emitter System

```rust
struct ParticleEmitter {
    shape: EmitterShape,            // Point, Sphere, Box, Cone
    rate: f32,                      // particles per second
    burst_count: u32,               // one-shot spawn count (0 = continuous)
    render_type: ParticleRenderType,

    // Per-particle randomized ranges
    lifetime: RangeF32,
    speed: RangeF32,
    size: RangeF32,
    color: ColorRange,
    emission: RangeF32,

    // Physics
    gravity_scale: f32,
    drag: f32,
    collision: bool,                // SDF collision check

    material_id: u16,               // for SDF_MICRO type
}
```

Emitters attach to ECS entities (via component). Destroyed when entity is removed.

### Session 10a Summary: Particle System Decisions

| Decision | Choice | Notes |
|----------|--------|-------|
| Simulation | GPU compute, shared buffer | 10K–100K budget |
| Volumetric particles | Density splats in vol march | Sparks, embers, dust, glow |
| Solid particles | SDF micro-objects in spatial index | Debris, shrapnel (~1K budget) |
| Screen-space particles | Post-process overlay | Rain, snow, lens effects |
| Emitter system | Attached to ECS entities | Rate or burst spawn |
| Particle collision | SDF evaluation at position | Optional per-particle flag |
