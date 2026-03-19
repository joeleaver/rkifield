// Radiance injection compute shader — v2 object-centric (Phase 8).
//
// Dispatched over the Level 0 radiance volume (128³). For each texel:
// 1. Compute world position from volume uniforms
// 2. Sample SDF via coarse field + BVH — classify as surface / interior / exterior
// 3. Surface voxels: compute normal via SDF gradient, evaluate direct lighting
// 4. Write radiance (RGB) + opacity (A) to the 3D storage texture
//
// Workgroup size: 4×4×4 → dispatch 32×32×32 for 128³ volume.

// ---------- Structs (must match Rust/shade.wgsl layouts exactly) ----------

struct VoxelSample {
    word0: u32, // lower 16 = f16 distance, bits 16-21 = material_id (6-bit) | bits 22-27 = secondary_material_id (6-bit) | bits 28-31 = reserved
    word1: u32,
}

struct GpuObject {
    inverse_world: mat4x4<f32>,
    aabb_min: vec4<f32>,
    aabb_max: vec4<f32>,
    brick_map_offset: u32,
    brick_map_dims_x: u32,
    brick_map_dims_y: u32,
    brick_map_dims_z: u32,
    voxel_size: f32,
    material_id: u32,
    sdf_type: u32,
    blend_mode: u32,
    blend_radius: f32,
    sdf_param_0: f32,
    sdf_param_1: f32,
    sdf_param_2: f32,
    sdf_param_3: f32,
    accumulated_scale_x: f32,
    accumulated_scale_y: f32,
    accumulated_scale_z: f32,
    lod_level: u32,
    object_id: u32,
    primitive_type: u32,
    geometry_aabb_min_x: f32, geometry_aabb_min_y: f32, geometry_aabb_min_z: f32,
    geometry_aabb_max_x: f32, geometry_aabb_max_y: f32, geometry_aabb_max_z: f32,
    _pad6: f32, _pad7: f32,
    _pad8: f32, _pad9: f32, _pad10: f32, _pad11: f32,
    _pad12: f32, _pad13: f32, _pad14: f32, _pad15: f32,
    _pad16: f32, _pad17: f32, _pad18: f32, _pad19: f32,
    _pad20: f32,
}

struct BvhNode {
    aabb_min_x: f32,
    aabb_min_y: f32,
    aabb_min_z: f32,
    left: u32,
    aabb_max_x: f32,
    aabb_max_y: f32,
    aabb_max_z: f32,
    right_or_object: u32,
}

struct SceneUniformsV2 {
    num_objects: u32,
    max_steps: u32,
    max_distance: f32,
    hit_threshold: f32,
}

struct Material {
    albedo_r: f32, albedo_g: f32, albedo_b: f32, roughness: f32,
    metallic: f32, emission_r: f32, emission_g: f32, emission_b: f32,
    emission_strength: f32,
    subsurface: f32, subsurface_r: f32, subsurface_g: f32, subsurface_b: f32,
    opacity: f32, ior: f32,
    noise_scale: f32, noise_strength: f32, noise_channels: u32,
    shader_id: u32, _pad1: f32, _pad2: f32, _pad3: f32, _pad4: f32, _pad5: f32,
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

struct CoarseFieldInfo {
    origin_cam_rel: vec4<f32>,
    dims: vec4<u32>,
    voxel_size: f32,
    inv_voxel_size: f32,
    _cf_pad0: f32,
    _cf_pad1: f32,
}

// ---------- Bindings ----------

// Group 0: v2 Scene data (same layout as ray march / shade group)
@group(0) @binding(0) var<storage, read> brick_pool: array<VoxelSample>;
@group(0) @binding(1) var<storage, read> brick_maps: array<u32>;
@group(0) @binding(2) var<storage, read> objects: array<GpuObject>;
// binding 3 = camera uniforms (unused here)
@group(0) @binding(4) var<uniform> v2_scene: SceneUniformsV2;
@group(0) @binding(5) var<storage, read> bvh_nodes: array<BvhNode>;

// Group 1: Material table
@group(1) @binding(0) var<storage, read> materials: array<Material>;

// Group 2: Lights + inject uniforms
@group(2) @binding(0) var<storage, read> lights: array<Light>;
@group(2) @binding(1) var<uniform> inject: InjectUniforms;

// Group 3: Radiance volume Level 0 write + volume uniforms
@group(3) @binding(0) var radiance_out: texture_storage_3d<rgba16float, write>;
@group(3) @binding(1) var<uniform> vol: RadianceVolumeUniforms;

// Group 4: Coarse acceleration field
@group(4) @binding(0) var coarse_field: texture_3d<f32>;
@group(4) @binding(1) var coarse_sampler: sampler;
@group(4) @binding(2) var<uniform> coarse_info: CoarseFieldInfo;

// ---------- Constants ----------

const PI: f32 = 3.14159265359;
const MAX_FLOAT: f32 = 3.402823e+38;
const EMPTY_SLOT: u32 = 0xFFFFFFFFu;
const INTERIOR_SLOT: u32 = 0xFFFFFFFEu;
const BVH_INVALID: u32 = 0xFFFFFFFFu;
const BVH_STACK_SIZE: u32 = 32u;

const SDF_TYPE_NONE: u32       = 0u;
const SDF_TYPE_ANALYTICAL: u32 = 1u;
const SDF_TYPE_VOXELIZED: u32  = 2u;

const PRIM_SPHERE: u32   = 0u;
const PRIM_BOX: u32      = 1u;
const PRIM_CAPSULE: u32  = 2u;
const PRIM_TORUS: u32    = 3u;
const PRIM_CYLINDER: u32 = 4u;
const PRIM_PLANE: u32    = 5u;

const INJECT_SHADOW_STEPS: u32 = 16u;
const INJECT_SHADOW_MAX_DIST: f32 = 20.0;
const COARSE_NEAR_THRESHOLD: f32 = 0.5;

// ---------- VoxelSample Helpers ----------

fn extract_distance(word0: u32) -> f32 {
    return unpack2x16float(word0).x;
}

fn extract_material_id(word0: u32) -> u32 {
    return (word0 >> 16u) & 0x3Fu;
}

// ---------- SDF Primitives ----------

fn sdf_sphere(p: vec3<f32>, radius: f32) -> f32 {
    return length(p) - radius;
}

fn sdf_box(p: vec3<f32>, half_extents: vec3<f32>) -> f32 {
    let q = abs(p) - half_extents;
    return length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

fn sdf_capsule(p: vec3<f32>, radius: f32, half_height: f32) -> f32 {
    let q = vec3<f32>(p.x, max(abs(p.y) - half_height, 0.0), p.z);
    return length(q) - radius;
}

fn sdf_torus(p: vec3<f32>, major_radius: f32, minor_radius: f32) -> f32 {
    let q = vec2<f32>(length(p.xz) - major_radius, p.y);
    return length(q) - minor_radius;
}

fn sdf_cylinder(p: vec3<f32>, radius: f32, half_height: f32) -> f32 {
    let d = vec2<f32>(length(p.xz) - radius, abs(p.y) - half_height);
    return min(max(d.x, d.y), 0.0) + length(max(d, vec2<f32>(0.0)));
}

fn sdf_plane(p: vec3<f32>, normal: vec3<f32>, dist: f32) -> f32 {
    return dot(p, normal) + dist;
}

// ---------- Object Evaluation ----------

fn evaluate_analytical(local_pos: vec3<f32>, obj: GpuObject) -> f32 {
    switch obj.primitive_type {
        case PRIM_SPHERE: { return sdf_sphere(local_pos, obj.sdf_param_0); }
        case PRIM_BOX: { return sdf_box(local_pos, vec3<f32>(obj.sdf_param_0, obj.sdf_param_1, obj.sdf_param_2)); }
        case PRIM_CAPSULE: { return sdf_capsule(local_pos, obj.sdf_param_0, obj.sdf_param_1); }
        case PRIM_TORUS: { return sdf_torus(local_pos, obj.sdf_param_0, obj.sdf_param_1); }
        case PRIM_CYLINDER: { return sdf_cylinder(local_pos, obj.sdf_param_0, obj.sdf_param_1); }
        case PRIM_PLANE: { return sdf_plane(local_pos, normalize(vec3<f32>(obj.sdf_param_0, obj.sdf_param_1, obj.sdf_param_2)), 0.0); }
        default: { return MAX_FLOAT; }
    }
}

fn sample_voxel_at(obj_offset: u32, vc: vec3<i32>, dims: vec3<u32>,
                    total_voxels: vec3<i32>, vs: f32) -> f32 {
    let c = clamp(vc, vec3<i32>(0), total_voxels - vec3<i32>(1));
    let brick = vec3<u32>(c / vec3<i32>(8));
    let local = vec3<u32>(c % vec3<i32>(8));
    let flat_brick = brick.x + brick.y * dims.x + brick.z * dims.x * dims.y;
    let slot = brick_maps[obj_offset + flat_brick];
    if slot == EMPTY_SLOT {
        return vs * 8.0;
    }
    if slot == INTERIOR_SLOT {
        return -(vs * 2.0);
    }
    let idx = slot * 512u + local.x + local.y * 8u + local.z * 64u;
    return extract_distance(brick_pool[idx].word0);
}

fn sample_voxelized(local_pos: vec3<f32>, obj: GpuObject) -> f32 {
    let vs = obj.voxel_size;
    let brick_extent = vs * 8.0;
    let dims = vec3<u32>(obj.brick_map_dims_x, obj.brick_map_dims_y, obj.brick_map_dims_z);
    let grid_size = vec3<f32>(dims) * brick_extent;
    let grid_pos = local_pos + grid_size * 0.5;
    let clamped = clamp(grid_pos, vec3<f32>(vs * 0.01), grid_size - vec3<f32>(vs * 0.01));
    let outside_dist = length(grid_pos - clamped);
    if outside_dist > brick_extent * 2.0 {
        return outside_dist;
    }
    // Geometry AABB early-out: skip empty expanded brick-map region.
    let geom_min = vec3<f32>(obj.geometry_aabb_min_x, obj.geometry_aabb_min_y, obj.geometry_aabb_min_z);
    let geom_max = vec3<f32>(obj.geometry_aabb_max_x, obj.geometry_aabb_max_y, obj.geometry_aabb_max_z);
    if geom_max.x > geom_min.x {
        let geom_closest = clamp(local_pos, geom_min, geom_max);
        let geom_dist = length(local_pos - geom_closest);
        if geom_dist > brick_extent {
            return geom_dist + outside_dist;
        }
    }
    let voxel_coord = clamped / vs - vec3<f32>(0.5);
    let v0 = vec3<i32>(floor(voxel_coord));
    let t = voxel_coord - vec3<f32>(v0);
    let total_voxels = vec3<i32>(dims) * 8;
    let c000 = sample_voxel_at(obj.brick_map_offset, v0 + vec3<i32>(0, 0, 0), dims, total_voxels, vs);
    let c100 = sample_voxel_at(obj.brick_map_offset, v0 + vec3<i32>(1, 0, 0), dims, total_voxels, vs);
    let c010 = sample_voxel_at(obj.brick_map_offset, v0 + vec3<i32>(0, 1, 0), dims, total_voxels, vs);
    let c110 = sample_voxel_at(obj.brick_map_offset, v0 + vec3<i32>(1, 1, 0), dims, total_voxels, vs);
    let c001 = sample_voxel_at(obj.brick_map_offset, v0 + vec3<i32>(0, 0, 1), dims, total_voxels, vs);
    let c101 = sample_voxel_at(obj.brick_map_offset, v0 + vec3<i32>(1, 0, 1), dims, total_voxels, vs);
    let c011 = sample_voxel_at(obj.brick_map_offset, v0 + vec3<i32>(0, 1, 1), dims, total_voxels, vs);
    let c111 = sample_voxel_at(obj.brick_map_offset, v0 + vec3<i32>(1, 1, 1), dims, total_voxels, vs);
    let c00 = mix(c000, c100, t.x);
    let c10 = mix(c010, c110, t.x);
    let c01 = mix(c001, c101, t.x);
    let c11 = mix(c011, c111, t.x);
    let c0 = mix(c00, c10, t.y);
    let c1 = mix(c01, c11, t.y);
    return mix(c0, c1, t.z) + outside_dist;
}

/// Evaluate object at a world-space position. Returns (distance, material_id).
fn evaluate_object(world_pos: vec3<f32>, obj_idx: u32) -> vec2<f32> {
    let obj = objects[obj_idx];
    if obj.sdf_type == SDF_TYPE_NONE {
        return vec2<f32>(MAX_FLOAT, 0.0);
    }
    // inverse_world is world-space — transform directly.
    let local_pos = (obj.inverse_world * vec4<f32>(world_pos, 1.0)).xyz;
    var dist: f32;
    if obj.sdf_type == SDF_TYPE_ANALYTICAL {
        dist = evaluate_analytical(local_pos, obj);
    } else {
        dist = sample_voxelized(local_pos, obj);
    }
    let min_scale = min(obj.accumulated_scale_x, min(obj.accumulated_scale_y, obj.accumulated_scale_z));
    return vec2<f32>(dist * min_scale, f32(obj.material_id));
}

// ---------- Coarse Field Sampling ----------

fn sample_coarse_field(cam_rel_pos: vec3<f32>) -> f32 {
    let field_pos = cam_rel_pos - coarse_info.origin_cam_rel.xyz;
    let uvw = field_pos * coarse_info.inv_voxel_size / vec3<f32>(coarse_info.dims.xyz);
    if any(uvw < vec3<f32>(0.0)) || any(uvw > vec3<f32>(1.0)) {
        return 0.0;
    }
    return textureSampleLevel(coarse_field, coarse_sampler, uvw, 0.0).r;
}

// ---------- v2 SDF Point Query (coarse field + BVH) ----------

// Coordinate space note:
// - BVH AABBs are in WORLD space (built from Scene object AABBs).
// - GpuObject.inverse_world is WORLD-SPACE (transforms world pos to local).
// - Coarse field expects CAMERA-RELATIVE input.
// - vol.center.xyz = camera world position, so cam_rel = world_pos - vol.center.xyz.

/// Convert world-space position to camera-relative.
fn world_to_cam_rel(world_pos: vec3<f32>) -> vec3<f32> {
    return world_pos - vol.center.xyz;
}

/// Sample SDF at a world-space position. Returns minimum distance.
fn sample_sdf(world_pos: vec3<f32>) -> f32 {
    let cam_rel = world_to_cam_rel(world_pos);

    // Phase 1: coarse field check (camera-relative).
    let coarse_dist = sample_coarse_field(cam_rel);
    if coarse_dist > COARSE_NEAR_THRESHOLD {
        return coarse_dist;
    }

    // Phase 2: BVH traversal (world-space AABBs).
    if v2_scene.num_objects == 0u {
        return MAX_FLOAT;
    }

    var min_dist = MAX_FLOAT;

    var stack: array<u32, 32>;
    var stack_ptr = 0u;
    stack[0] = 0u;
    stack_ptr = 1u;

    while stack_ptr > 0u {
        stack_ptr -= 1u;
        let node_idx = stack[stack_ptr];
        let node = bvh_nodes[node_idx];

        let node_min = vec3<f32>(node.aabb_min_x, node.aabb_min_y, node.aabb_min_z);
        let node_max = vec3<f32>(node.aabb_max_x, node.aabb_max_y, node.aabb_max_z);

        // BVH AABBs are world-space; compare with world_pos.
        let closest = clamp(world_pos, node_min, node_max);
        let box_dist = length(closest - world_pos);
        if box_dist > min_dist {
            continue;
        }

        if node.left == BVH_INVALID {
            let leaf_obj_idx = node.right_or_object;
            if leaf_obj_idx < v2_scene.num_objects {
                let result = evaluate_object(world_pos, leaf_obj_idx);
                min_dist = min(min_dist, result.x);
            }
        } else {
            if stack_ptr < BVH_STACK_SIZE - 1u {
                stack[stack_ptr] = node.left;
                stack_ptr += 1u;
                stack[stack_ptr] = node.right_or_object;
                stack_ptr += 1u;
            }
        }
    }

    return min_dist;
}

/// Sample SDF and return (distance, material_id) at a world-space position.
fn sample_sdf_with_material(world_pos: vec3<f32>) -> vec2<f32> {
    let cam_rel = world_to_cam_rel(world_pos);

    if v2_scene.num_objects == 0u {
        return vec2<f32>(MAX_FLOAT, 0.0);
    }

    var min_dist = MAX_FLOAT;
    var mat_id = 0.0;

    var stack: array<u32, 32>;
    var stack_ptr = 0u;
    stack[0] = 0u;
    stack_ptr = 1u;

    while stack_ptr > 0u {
        stack_ptr -= 1u;
        let node_idx = stack[stack_ptr];
        let node = bvh_nodes[node_idx];

        let node_min = vec3<f32>(node.aabb_min_x, node.aabb_min_y, node.aabb_min_z);
        let node_max = vec3<f32>(node.aabb_max_x, node.aabb_max_y, node.aabb_max_z);

        let closest = clamp(world_pos, node_min, node_max);
        let box_dist = length(closest - world_pos);
        if box_dist > min_dist {
            continue;
        }

        if node.left == BVH_INVALID {
            let leaf_obj_idx = node.right_or_object;
            if leaf_obj_idx < v2_scene.num_objects {
                let result = evaluate_object(world_pos, leaf_obj_idx);
                if result.x < min_dist {
                    min_dist = result.x;
                    mat_id = result.y;
                }
            }
        } else {
            if stack_ptr < BVH_STACK_SIZE - 1u {
                stack[stack_ptr] = node.left;
                stack_ptr += 1u;
                stack[stack_ptr] = node.right_or_object;
                stack_ptr += 1u;
            }
        }
    }

    return vec2<f32>(min_dist, mat_id);
}

// ---------- SDF Normal via Central Differences ----------

fn sdf_normal(pos: vec3<f32>) -> vec3<f32> {
    let eps = vol.voxel_sizes.x * 0.5; // L0 voxel size / 2
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

    // Compute world position of this texel centre.
    // The radiance volume center is in world space.
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

    // Concavity probe: step along normal, check if we quickly encounter another surface.
    let probe_d1 = sample_sdf(pos + normal * voxel_size * 2.0);
    let probe_d2 = sample_sdf(pos + normal * voxel_size * 5.0);
    let c1 = smoothstep(0.0, voxel_size * 1.0, probe_d1);
    let c2 = smoothstep(0.0, voxel_size * 2.0, probe_d2);
    let concavity = min(c1, c2);

    // Get material from closest object
    let sdf_mat = sample_sdf_with_material(pos);
    let mat_id = u32(sdf_mat.y);
    let mat = materials[mat_id];
    let albedo = vec3<f32>(mat.albedo_r, mat.albedo_g, mat.albedo_b);

    // Emissive self-injection
    let emission = vec3<f32>(mat.emission_r, mat.emission_g, mat.emission_b) * mat.emission_strength;

    var radiance = vec3<f32>(0.0);
    var shadow_count = 0u;
    let max_shadow = inject.max_shadow_lights;
    let num_lights_val = inject.num_lights;

    // Evaluate direct lighting (Lambertian diffuse — no specular for GI bounce).
    // Light positions are camera-relative in the light buffer, but injection
    // works in world space. For directional lights this doesn't matter (only
    // direction). For point/spot lights, convert camera-relative → world-space
    // by adding vol.center.xyz (which equals the camera world position).
    for (var i = 0u; i < num_lights_val; i++) {
        let light = lights[i];
        let light_color = vec3<f32>(light.color_r, light.color_g, light.color_b);

        var light_dir: vec3<f32>;
        var atten = 1.0;

        if light.light_type == 0u {
            // Directional
            light_dir = normalize(vec3<f32>(light.dir_x, light.dir_y, light.dir_z));
        } else if light.light_type == 1u {
            // Point — light pos is camera-relative; convert to world-space.
            let light_pos = vec3<f32>(light.pos_x, light.pos_y, light.pos_z) + vol.center.xyz;
            let to_light = light_pos - pos;
            let dist = length(to_light);
            light_dir = to_light / max(dist, 0.0001);
            atten = distance_attenuation(dist, light.range);
        } else {
            // Spot — light pos is camera-relative; convert to world-space.
            let light_pos = vec3<f32>(light.pos_x, light.pos_y, light.pos_z) + vol.center.xyz;
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

    // Apply concavity attenuation to prevent bright halos at junctions.
    textureStore(radiance_out, vec3<i32>(gid), vec4<f32>(radiance * concavity, opacity));
}
