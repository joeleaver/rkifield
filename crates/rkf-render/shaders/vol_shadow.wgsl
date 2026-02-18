// Volumetric shadow map compute shader — Phase 11 task 11.1.
//
// Fills a 3D R16Float texture with per-texel transmittance values.
// For each voxel in the shadow volume, marches from that position
// toward the sun and accumulates transmittance via density sampling.
//
// The volume is camera-centered, covering a configurable world-space AABB.
// Transmittance = 1.0 (fully lit) → 0.0 (fully shadowed).
//
// SDF density sampling is a placeholder for Phase 11.1 — the march
// loop structure is fully wired; actual SDF/fog density will be
// connected in Phase 12+.

// ---------- Params ----------

struct VolShadowParams {
    // Volume bounds (world-space AABB centered on camera)
    volume_min: vec3<f32>,
    _pad0: f32,
    volume_max: vec3<f32>,
    _pad1: f32,
    // Sun direction (normalized, pointing toward sun)
    sun_dir: vec3<f32>,
    _pad2: f32,
    // Volume dimensions
    dim_x: u32,
    dim_y: u32,
    dim_z: u32,
    // March settings
    max_steps: u32,    // offset 60 in Rust struct
    step_size: f32,    // offset 64 in Rust struct
    extinction_coeff: f32, // offset 68
    _pad3: u32,
    _pad4: u32,
}

// ---------- Bindings ----------

@group(0) @binding(0) var<uniform> params: VolShadowParams;
@group(0) @binding(1) var shadow_map: texture_storage_3d<r16float, write>;

// ---------- Density sampling ----------

// Simplified density sampling — will be connected to real SDF/fog data.
// Phase 12+ will wire this to actual SDF brick pool for opaque geometry
// and to a volumetric density field for participating media.
//
// Returns density at world position `pos` in [0, ∞).
// 0.0 = empty space (fully transparent).
fn sample_density(pos: vec3<f32>) -> f32 {
    // Placeholder: no density (fully transparent atmosphere).
    // Remove this when real SDF brick pool sampling is wired in.
    let _ = pos;
    return 0.0;
}

// ---------- Entry point ----------

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    // Bounds check — guard against over-dispatch
    if id.x >= params.dim_x || id.y >= params.dim_y || id.z >= params.dim_z {
        return;
    }

    // Compute world-space position for this texel.
    // Texel (0,0,0) maps to volume_min; texel (dim-1,dim-1,dim-1) maps to volume_max.
    let dim = vec3<f32>(f32(params.dim_x), f32(params.dim_y), f32(params.dim_z));
    let uv = (vec3<f32>(id) + vec3<f32>(0.5)) / dim;
    let world_pos = params.volume_min + uv * (params.volume_max - params.volume_min);

    // March from world_pos toward the sun along sun_dir.
    // Accumulate transmittance: T *= exp(-density * extinction * step_size).
    var transmittance: f32 = 1.0;
    var pos: vec3<f32> = world_pos;

    for (var step = 0u; step < params.max_steps; step++) {
        pos += params.sun_dir * params.step_size;

        let density = sample_density(pos);
        if density > 0.0 {
            transmittance *= exp(-density * params.extinction_coeff * params.step_size);
        }

        // Early-out: fully shadowed
        if transmittance < 0.01 {
            transmittance = 0.0;
            break;
        }
    }

    // Write transmittance to shadow map (r channel only).
    // textureStore expects vec4<f32> even for r16float.
    let texel = vec3<i32>(id);
    textureStore(shadow_map, texel, vec4<f32>(transmittance, 0.0, 0.0, 0.0));
}
