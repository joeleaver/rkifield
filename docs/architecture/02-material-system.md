# Material System

> **Status: DECIDED**

### Decision: Global Material Table on GPU

Materials are stored in a single `array<Material>` storage buffer, indexed directly by the voxel's `material_id` (u16). The shading pass reads material properties after the ray march resolves a surface hit.

- Maximum 65536 materials (u16 primary ID), 256 for secondary blend target (u8)
- Table size at 96 bytes/material × 65536 = 6MB — negligible
- Read once per pixel in shading pass, not per voxel during marching
- Full f32 precision for all properties — no packing needed in the table

```wgsl
@group(0) @binding(0)
var<storage, read> material_table: array<Material>;
```

### Decision: Material Properties — Full PBR + Subsurface + Procedural Noise

Each material carries PBR parameters, translucency/SSS controls, and procedural noise configuration. Unused fields default to zero/neutral and cost nothing at runtime (shader branches on them).

```rust
struct Material {                          // 96 bytes (padded for GPU alignment)
    // PBR Baseline
    albedo: [f32; 3],                      // base color (linear RGB)
    roughness: f32,                        // 0.0 = mirror, 1.0 = fully rough
    metallic: f32,                         // 0.0 = dielectric, 1.0 = metal
    emission_color: [f32; 3],              // emissive color (linear RGB)
    emission_strength: f32,                // emissive intensity (HDR, can exceed 1.0)

    // Subsurface and Translucency
    subsurface: f32,                       // 0.0 = none, 1.0 = full SSS
    subsurface_color: [f32; 3],            // color of scattered light (skin, wax, leaves)
    opacity: f32,                          // 1.0 = solid, 0.0 = fully transparent
    ior: f32,                              // index of refraction (glass ~1.5, water ~1.33)

    // Procedural Variation (evaluated at shade time from world position)
    noise_scale: f32,                      // spatial frequency of noise
    noise_strength: f32,                   // amplitude of noise perturbation
    noise_channels: u32,                   // bitfield: bit 0 = albedo, bit 1 = roughness,
                                           //           bit 2 = normal perturbation

    // Padding to 96 bytes for GPU alignment
    _padding: [f32; 2],
}
```

**Procedural noise:** 3D gradient noise (simplex or similar) evaluated at the surface hit position, scaled by `noise_scale`, applied with `noise_strength` amplitude to the selected channels. Normal perturbation uses the noise gradient as a bump offset on the SDF-derived normal. This gives surfaces visual micro-roughness and color variation without additional voxel data — prevents the "flat material" look common in voxel engines.

### Decision: Material Blending — Linear Interpolation

When `blend_weight > 0`, the shader linearly interpolates all properties between primary and secondary materials:

```wgsl
fn resolve_material(primary_id: u16, secondary_id: u8, weight: f32) -> ResolvedMaterial {
    let a = material_table[primary_id];
    let b = material_table[secondary_id];
    return ResolvedMaterial {
        albedo: mix(a.albedo, b.albedo, weight),
        roughness: mix(a.roughness, b.roughness, weight),
        metallic: mix(a.metallic, b.metallic, weight),
        // ... all fields
    };
}
```

Linear lerp for all properties including metallic. With u8 blend_weight (256 steps), transitions happen over a thin spatial band where any lerp artifacts are barely visible. If metallic transitions become a visual issue, a one-line change to threshold at 0.5 fixes it.

### Decision: Per-Voxel Color via Companion Color Pool

For imported textured assets and hand-painted objects, a companion color pool stores per-voxel RGB that modulates the material's albedo:

```
final_albedo = material.albedo * voxel_color
```

This gives material PBR properties (roughness, metallic, SSS) plus per-voxel color variation. Objects that don't need per-voxel color (procedural geometry, material-painted props) skip the companion brick entirely.

**Color Data Pool (imported/painted objects only):**
```
Word 0 (u32): [ u8 red | u8 green | u8 blue | u8 intensity ]
```
- 4 bytes/voxel × 512 = 2KB per color brick
- Flag bit `has_color_data` in core voxel sample indicates companion brick exists
- Budget: ~32MB → ~16K color bricks (configurable)

**Color application modes:**
- **Multiply** (default): `final_albedo = material.albedo * voxel_color` — tints the material
- **Replace**: `final_albedo = voxel_color` — ignores material albedo, uses per-voxel color directly
- Mode selection could be per-material or per-flag bit (TBD during shading design)

### Session 2 Summary: All Material System Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Table structure | Storage buffer, indexed by u16 material_id | Simple, direct, 6MB max |
| Properties | Full PBR + SSS/translucency + procedural noise (96 bytes) | Generous — table is tiny, enables rich shading |
| Blending | Linear lerp on all properties | Simple, thin transition bands hide artifacts |
| Per-voxel color | Companion color pool (2KB/brick, flag-gated) | Follows companion pattern, pay-for-what-you-use |
| Procedural variation | Per-material noise params, 3D gradient noise at shade time | Prevents flat-material look, no extra voxel data |
