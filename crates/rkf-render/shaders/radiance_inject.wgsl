// Radiance injection compute shader — Phase 8 GI
//
// Dispatched over the Level 0 radiance volume (128³). For each texel:
// 1. Compute world position from volume uniforms
// 2. Sample SDF — classify as surface / interior / exterior
// 3. Surface voxels: compute normal via SDF gradient, evaluate direct lighting
// 4. Write radiance (RGB) + opacity (A) to the 3D storage texture
//
// Workgroup size: 4×4×4 → dispatch 32×32×32 for 128³ volume.

// ---------- Structs (must match Rust/shade.wgsl layouts exactly) ----------

struct VoxelSample {
    word0: u32, // lower 16 = f16 distance, upper 16 = u16 material_id
    word1: u32, // byte0 = blend_weight, byte1 = secondary_id, byte2 = flags, byte3 = reserved
}

struct SceneUniforms {
    grid_dims:    vec4<u32>,   // xyz = dimensions, w = unused
    grid_origin:  vec4<f32>,   // xyz = origin, w = brick_extent
    params:       vec4<f32>,   // x = voxel_size, yzw = unused
}

struct Material {
    albedo_r: f32, albedo_g: f32, albedo_b: f32, roughness: f32,
    metallic: f32, emission_r: f32, emission_g: f32, emission_b: f32,
    emission_strength: f32,
    subsurface: f32, subsurface_r: f32, subsurface_g: f32, subsurface_b: f32,
    opacity: f32, ior: f32,
    noise_scale: f32, noise_strength: f32, noise_channels: u32,
    _pad0: f32, _pad1: f32, _pad2: f32, _pad3: f32, _pad4: f32, _pad5: f32,
}

struct Light {
    light_type: u32,
    pos_x: f32, pos_y: f32, pos_z: f32,
    dir_x: f32, dir_y: f32, dir_z: f32,
    color_r: f32, color_g: f32, color_b: f32,
    intensity: f32, range: f32,
    inner_angle: f32, outer_angle: f32,
    cookie_index: i32, shadow_caster: u32,
}

struct InjectUniforms {
    num_lights: u32,
    max_shadow_lights: u32,
    _pad0: u32,
    _pad1: u32,
}

struct RadianceVolumeUniforms {
    center:      vec4<f32>,
    voxel_sizes: vec4<f32>,
    inv_extents: vec4<f32>,
    params:      vec4<u32>,   // x = dim, y = num_levels
}

// ---------- Bindings ----------

// Group 0: Scene SDF data (same layout as GpuScene)
@group(0) @binding(0) var<storage, read> brick_pool: array<VoxelSample>;
@group(0) @binding(1) var<storage, read> occupancy:  array<u32>;
@group(0) @binding(2) var<storage, read> slots:      array<u32>;
// binding 3 = camera uniforms (part of bind group layout, unused here)
@group(0) @binding(4) var<uniform> scene: SceneUniforms;

// Group 1: Material table
@group(1) @binding(0) var<storage, read> materials: array<Material>;

// Group 2: Lights + inject uniforms
@group(2) @binding(0) var<storage, read> lights: array<Light>;
@group(2) @binding(1) var<uniform> inject: InjectUniforms;

// Group 3: Radiance volume Level 0 write + volume uniforms
@group(3) @binding(0) var radiance_out: texture_storage_3d<rgba16float, write>;
@group(3) @binding(1) var<uniform> vol: RadianceVolumeUniforms;

// ---------- Constants ----------

const PI: f32 = 3.14159265359;
const EMPTY_SLOT: u32 = 0xFFFFFFFFu;
const CELL_EMPTY: u32    = 0u;
const CELL_SURFACE: u32  = 1u;
const CELL_INTERIOR: u32 = 2u;

const INJECT_SHADOW_STEPS: u32 = 16u;
const INJECT_SHADOW_MAX_DIST: f32 = 20.0;

// ---------- SDF Sampling (same as shade.wgsl) ----------

fn extract_distance(word0: u32) -> f32 {
    return unpack2x16float(word0).x;
}

fn extract_material_id(word0: u32) -> u32 {
    return (word0 >> 16u) & 0xFFFFu;
}

fn sdf_grid_dims() -> vec3<u32>  { return scene.grid_dims.xyz; }
fn sdf_grid_origin() -> vec3<f32> { return scene.grid_origin.xyz; }
fn sdf_brick_extent() -> f32     { return scene.grid_origin.w; }
fn sdf_voxel_size() -> f32       { return scene.params.x; }

fn sdf_world_to_cell(pos: vec3<f32>) -> vec3<i32> {
    let local = pos - sdf_grid_origin();
    return vec3<i32>(floor(local / sdf_brick_extent()));
}

fn sdf_cell_in_bounds(cell: vec3<i32>) -> bool {
    return all(cell >= vec3<i32>(0)) && all(vec3<u32>(cell) < sdf_grid_dims());
}

fn sdf_get_cell_state(flat: u32) -> u32 {
    let word_idx = flat / 16u;
    let bit_offset = (flat % 16u) * 2u;
    return (occupancy[word_idx] >> bit_offset) & 3u;
}

fn sdf_cell_flat_index(cell: vec3<u32>) -> u32 {
    let d = sdf_grid_dims();
    return cell.x + cell.y * d.x + cell.z * d.x * d.y;
}

fn sdf_sample_brick(pos: vec3<f32>, brick_min: vec3<f32>, slot: u32) -> f32 {
    let vs = sdf_voxel_size();
    let brick_local = (pos - brick_min) / vs - vec3<f32>(0.5);
    let f = clamp(brick_local, vec3<f32>(0.0), vec3<f32>(6.9999));
    let i0 = vec3<u32>(floor(f));
    let i1 = min(i0 + vec3<u32>(1u), vec3<u32>(7u));
    let t = f - floor(f);

    let base = slot * 512u;
    let c000 = extract_distance(brick_pool[base + i0.x + i0.y*8u + i0.z*64u].word0);
    let c100 = extract_distance(brick_pool[base + i1.x + i0.y*8u + i0.z*64u].word0);
    let c010 = extract_distance(brick_pool[base + i0.x + i1.y*8u + i0.z*64u].word0);
    let c110 = extract_distance(brick_pool[base + i1.x + i1.y*8u + i0.z*64u].word0);
    let c001 = extract_distance(brick_pool[base + i0.x + i0.y*8u + i1.z*64u].word0);
    let c101 = extract_distance(brick_pool[base + i1.x + i0.y*8u + i1.z*64u].word0);
    let c011 = extract_distance(brick_pool[base + i0.x + i1.y*8u + i1.z*64u].word0);
    let c111 = extract_distance(brick_pool[base + i1.x + i1.y*8u + i1.z*64u].word0);

    let c00 = mix(c000, c100, t.x);
    let c10 = mix(c010, c110, t.x);
    let c01 = mix(c001, c101, t.x);
    let c11 = mix(c011, c111, t.x);
    let c0 = mix(c00, c10, t.y);
    let c1 = mix(c01, c11, t.y);
    return mix(c0, c1, t.z);
}

fn sample_sdf(pos: vec3<f32>) -> f32 {
    let cell_i = sdf_world_to_cell(pos);
    let be = sdf_brick_extent();

    if !sdf_cell_in_bounds(cell_i) {
        return be;
    }

    let cell = vec3<u32>(cell_i);
    let flat = sdf_cell_flat_index(cell);
    let state = sdf_get_cell_state(flat);

    if state == CELL_EMPTY {
        return be * 0.5;
    }
    if state == CELL_INTERIOR {
        return -be * 0.5;
    }
    if state == CELL_SURFACE {
        let slot = slots[flat];
        if slot == EMPTY_SLOT {
            return be * 0.5;
        }
        let brick_min = sdf_grid_origin() + vec3<f32>(cell) * be;
        return sdf_sample_brick(pos, brick_min, slot);
    }
    return be * 0.5;
}

// ---------- Material ID Sampling ----------

fn sample_material_id(pos: vec3<f32>) -> u32 {
    let cell_i = sdf_world_to_cell(pos);
    if !sdf_cell_in_bounds(cell_i) {
        return 0u;
    }
    let cell = vec3<u32>(cell_i);
    let flat = sdf_cell_flat_index(cell);
    let state = sdf_get_cell_state(flat);
    if state != CELL_SURFACE {
        return 0u;
    }
    let slot = slots[flat];
    if slot == EMPTY_SLOT {
        return 0u;
    }
    let brick_min = sdf_grid_origin() + vec3<f32>(cell) * sdf_brick_extent();
    let brick_local = (pos - brick_min) / sdf_voxel_size();
    let voxel = clamp(vec3<u32>(floor(brick_local)), vec3<u32>(0u), vec3<u32>(7u));
    let voxel_idx = voxel.x + voxel.y * 8u + voxel.z * 64u;
    let idx = slot * 512u + voxel_idx;
    return extract_material_id(brick_pool[idx].word0);
}

// ---------- SDF Normal via Central Differences ----------

fn sdf_normal(pos: vec3<f32>) -> vec3<f32> {
    let eps = sdf_voxel_size() * 0.5;
    let nx = sample_sdf(pos + vec3<f32>(eps, 0.0, 0.0)) - sample_sdf(pos - vec3<f32>(eps, 0.0, 0.0));
    let ny = sample_sdf(pos + vec3<f32>(0.0, eps, 0.0)) - sample_sdf(pos - vec3<f32>(0.0, eps, 0.0));
    let nz = sample_sdf(pos + vec3<f32>(0.0, 0.0, eps)) - sample_sdf(pos - vec3<f32>(0.0, 0.0, eps));
    return normalize(vec3<f32>(nx, ny, nz));
}

// ---------- Simplified Shadow (hard, 16 steps) ----------

fn simple_shadow(origin: vec3<f32>, dir: vec3<f32>, max_dist: f32) -> f32 {
    var t = 0.01;
    for (var i = 0u; i < INJECT_SHADOW_STEPS; i++) {
        let d = sample_sdf(origin + dir * t);
        if d < 0.002 {
            return 0.0;
        }
        t += max(d, 0.01);
        if t > max_dist {
            break;
        }
    }
    return 1.0;
}

// ---------- Light Attenuation ----------

fn distance_attenuation(dist: f32, range: f32) -> f32 {
    let d2 = dist * dist;
    let r2 = range * range;
    let factor = d2 / r2;
    let w = clamp(1.0 - factor, 0.0, 1.0);
    return (w * w) / max(d2, 0.0001);
}

// ---------- Entry Point ----------

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dim = vol.params.x;
    if gid.x >= dim || gid.y >= dim || gid.z >= dim {
        return;
    }

    // Compute world position of this texel centre
    let voxel_size = vol.voxel_sizes.x; // Level 0
    let half_extent = voxel_size * f32(dim) * 0.5;
    let pos = vol.center.xyz
        + (vec3<f32>(gid) + 0.5) * voxel_size
        - vec3<f32>(half_extent);

    // Sample SDF
    let d = sample_sdf(pos);
    let threshold = voxel_size * 3.0;

    // Deep interior: opaque, blocks light
    if d < -threshold {
        textureStore(radiance_out, vec3<i32>(gid), vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }
    // Exterior: transparent
    if d > threshold {
        textureStore(radiance_out, vec3<i32>(gid), vec4<f32>(0.0, 0.0, 0.0, 0.0));
        return;
    }

    // --- Near surface: compute direct lighting ---

    let normal = sdf_normal(pos);

    // Concavity probe: step along normal, check if we hit another surface quickly.
    // At clean surfaces, probe_d ≈ probe_step (moving away from surface into open space).
    // At min() junctions/fillets, the normal points ~45° outward and quickly encounters
    // the adjacent surface → probe_d << probe_step → concavity detected → attenuate.
    // Two probes at different distances for broader coverage of the fillet zone.
    let vs = sdf_voxel_size();
    let probe_d1 = sample_sdf(pos + normal * vs * 2.0);
    let probe_d2 = sample_sdf(pos + normal * vs * 5.0);
    let c1 = smoothstep(0.0, vs * 1.0, probe_d1);
    let c2 = smoothstep(0.0, vs * 2.0, probe_d2);
    let concavity = min(c1, c2);

    let mat_id = sample_material_id(pos);
    let mat = materials[mat_id];
    let albedo = vec3<f32>(mat.albedo_r, mat.albedo_g, mat.albedo_b);

    // Emissive self-injection
    let emission = vec3<f32>(mat.emission_r, mat.emission_g, mat.emission_b) * mat.emission_strength;

    var radiance = vec3<f32>(0.0);
    var shadow_count = 0u;
    let max_shadow = inject.max_shadow_lights;
    let num_lights_val = inject.num_lights;

    // Evaluate direct lighting (lambertian diffuse — no specular for GI bounce)
    for (var i = 0u; i < num_lights_val; i++) {
        let light = lights[i];
        let light_color = vec3<f32>(light.color_r, light.color_g, light.color_b);

        var light_dir: vec3<f32>;
        var atten = 1.0;

        if light.light_type == 0u {
            // Directional
            light_dir = normalize(vec3<f32>(light.dir_x, light.dir_y, light.dir_z));
        } else if light.light_type == 1u {
            // Point
            let light_pos = vec3<f32>(light.pos_x, light.pos_y, light.pos_z);
            let to_light = light_pos - pos;
            let dist = length(to_light);
            light_dir = to_light / max(dist, 0.0001);
            atten = distance_attenuation(dist, light.range);
        } else {
            // Spot
            let light_pos = vec3<f32>(light.pos_x, light.pos_y, light.pos_z);
            let spot_dir = normalize(vec3<f32>(light.dir_x, light.dir_y, light.dir_z));
            let to_light = light_pos - pos;
            let dist = length(to_light);
            light_dir = to_light / max(dist, 0.0001);
            let cos_angle = dot(-light_dir, spot_dir);
            let cos_outer = cos(light.outer_angle);
            let cos_inner = cos(light.inner_angle);
            let spot = clamp((cos_angle - cos_outer) / max(cos_inner - cos_outer, 0.0001), 0.0, 1.0);
            atten = spot * distance_attenuation(dist, light.range);
        }

        if atten < 0.001 {
            continue;
        }

        let n_dot_l = max(dot(normal, light_dir), 0.0);
        if n_dot_l <= 0.0 {
            continue;
        }

        // Simplified shadow (only for first max_shadow shadow casters)
        var shadow = 1.0;
        if light.shadow_caster == 1u && shadow_count < max_shadow {
            let shadow_origin = pos + normal * 0.02;
            shadow = simple_shadow(shadow_origin, light_dir, INJECT_SHADOW_MAX_DIST);
            shadow_count += 1u;
        }

        // Lambertian diffuse (no specular for GI bounce)
        radiance += albedo / PI * light_color * light.intensity * atten * n_dot_l * shadow;
    }

    // Add emission
    radiance += emission;

    // Opacity: smooth falloff at surface band boundary.
    let opacity = clamp(1.0 - abs(d) / threshold, 0.0, 1.0);

    // Apply concavity attenuation: darken radiance at min() junctions to prevent bright halos.
    // Opacity stays full so junction voxels still occlude (dark + opaque = correct AO).
    textureStore(radiance_out, vec3<i32>(gid), vec4<f32>(radiance * concavity, opacity));
}
