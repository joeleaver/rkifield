// Volumetric shadow map compute shader — v2 object-centric.
//
// Fills a 3D R32Float texture with per-texel transmittance values.
// Marches from each voxel toward the sun; if the march enters a region
// near SDF geometry (detected via the coarse acceleration field),
// transmittance drops. Used by the volumetric march pass for sun visibility.
//
// v2 adaptation: replaces v1 chunk-based SDF sampling with coarse field
// lookup. The coarse field stores conservative unsigned distance to the
// nearest object AABB — sufficient for soft volumetric shadow approximation.

// ---------- Params ----------

struct VolShadowParams {
    volume_min: vec3<f32>,
    _pad0: f32,
    volume_max: vec3<f32>,
    _pad1: f32,
    sun_dir: vec3<f32>,
    _pad2: f32,
    dim_x: u32,
    dim_y: u32,
    dim_z: u32,
    max_steps: u32,
    step_size: f32,
    extinction_coeff: f32,
    _pad3: u32,
    _pad4: u32,
}

struct CoarseFieldInfo {
    origin_cam_rel: vec4<f32>,
    dims: vec4<u32>,
    voxel_size: f32,
    inv_voxel_size: f32,
    _cf_pad0: f32,
    _cf_pad1: f32,
}

// ---------- Bindings ----------

// Group 0: vol shadow data
@group(0) @binding(0) var<uniform> params: VolShadowParams;
@group(0) @binding(1) var shadow_map: texture_storage_3d<r32float, write>;

// Group 1: coarse acceleration field (same layout as ray_march group 3)
@group(1) @binding(0) var coarse_field: texture_3d<f32>;
@group(1) @binding(1) var coarse_sampler: sampler;
@group(1) @binding(2) var<uniform> coarse_info: CoarseFieldInfo;

// ---------- Coarse Field Sampling ----------

/// Sample the coarse field at a world-space position.
/// Returns the conservative unsigned distance to the nearest object AABB.
fn sample_coarse_at_world(world_pos: vec3<f32>) -> f32 {
    // Derive camera position from volume center (volume is camera-centered).
    let camera_pos = (params.volume_min + params.volume_max) * 0.5;
    let cam_rel = world_pos - camera_pos;

    // Convert camera-relative position to coarse field UVW.
    let field_pos = cam_rel - coarse_info.origin_cam_rel.xyz;
    let uvw = field_pos * coarse_info.inv_voxel_size / vec3<f32>(coarse_info.dims.xyz);

    if any(uvw < vec3<f32>(0.0)) || any(uvw > vec3<f32>(1.0)) {
        return 999.0; // Outside field — no geometry.
    }

    return textureSampleLevel(coarse_field, coarse_sampler, uvw, 0.0).r;
}

/// Check if world position `pos` is near SDF geometry using the coarse field.
/// Returns a density (1.0 = solid, 0.0 = empty) with soft edges.
fn sample_sdf_density(pos: vec3<f32>) -> f32 {
    let dist = sample_coarse_at_world(pos);

    // Inside an object AABB (dist ≈ 0): high density.
    // The coarse field stores unsigned distance to AABB surfaces,
    // so dist ≈ 0 means we're on or inside an AABB boundary.
    let band = coarse_info.voxel_size * 2.0;
    if dist < 0.001 {
        return 1.0;
    }
    if dist < band {
        return 1.0 - dist / band;
    }
    return 0.0;
}

// ---------- Entry point ----------

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= params.dim_x || id.y >= params.dim_y || id.z >= params.dim_z {
        return;
    }

    let dim = vec3<f32>(f32(params.dim_x), f32(params.dim_y), f32(params.dim_z));
    let uv = (vec3<f32>(id) + vec3<f32>(0.5)) / dim;
    let world_pos = params.volume_min + uv * (params.volume_max - params.volume_min);

    var transmittance: f32 = 1.0;
    var pos: vec3<f32> = world_pos;

    for (var step = 0u; step < params.max_steps; step++) {
        pos += params.sun_dir * params.step_size;

        let density = sample_sdf_density(pos);
        if density > 0.0 {
            transmittance *= exp(-density * params.extinction_coeff * params.step_size);
        }

        if transmittance < 0.01 {
            transmittance = 0.0;
            break;
        }
    }

    let texel = vec3<i32>(id);
    textureStore(shadow_map, texel, vec4<f32>(transmittance, 0.0, 0.0, 0.0));
}
