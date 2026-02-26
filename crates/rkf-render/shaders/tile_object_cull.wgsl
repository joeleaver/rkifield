// Tile object culling compute shader — Phase 6.
//
// Projects object AABBs to screen space and builds per-tile object lists.
// Each tile (16×16 pixels) gets a list of which objects potentially overlap it.
// The ray marcher reads these lists to only evaluate relevant objects per pixel.
//
// Dispatch: ceil(width/16) × ceil(height/16) × 1
// Workgroup: 16×16 = 256 threads (cooperative object testing)

// ---------- Constants ----------

const TILE_SIZE: u32 = 16u;
const MAX_OBJECTS_PER_TILE: u32 = 32u;
const SDF_TYPE_NONE: u32 = 0u;

// ---------- GPU Structs ----------

// Must match Rust GpuObject layout exactly (256 bytes).
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
    _pad0: f32, _pad1: f32, _pad2: f32, _pad3: f32,
    _pad4: f32, _pad5: f32, _pad6: f32, _pad7: f32,
    _pad8: f32, _pad9: f32, _pad10: f32, _pad11: f32,
    _pad12: f32, _pad13: f32, _pad14: f32, _pad15: f32,
    _pad16: f32, _pad17: f32, _pad18: f32, _pad19: f32,
    _pad20: f32,
}

struct CameraUniforms {
    position: vec4<f32>,
    forward:  vec4<f32>,
    right:    vec4<f32>,
    up:       vec4<f32>,
    resolution: vec2<f32>,
    jitter: vec2<f32>,
    prev_vp: mat4x4<f32>,
}

struct SceneUniforms {
    num_objects: u32,
    max_steps: u32,
    max_distance: f32,
    hit_threshold: f32,
}

// ---------- Bindings ----------

// Group 0: GpuScene — we only use objects, camera, and scene from the full layout.
// Bindings 0, 1, 5 are present in the bind group but unused by this shader.
@group(0) @binding(2) var<storage, read> objects: array<GpuObject>;
@group(0) @binding(3) var<uniform>       camera: CameraUniforms;
@group(0) @binding(4) var<uniform>       scene: SceneUniforms;

// Group 1: Tile output buffers
@group(1) @binding(0) var<storage, read_write> tile_object_indices: array<u32>;
@group(1) @binding(1) var<storage, read_write> tile_object_counts: array<atomic<u32>>;

// ---------- Shared Memory ----------

var<workgroup> tile_count: atomic<u32>;
var<workgroup> tile_objects: array<u32, 32>;  // MAX_OBJECTS_PER_TILE

// ---------- Projection ----------

/// Project a camera-relative position to screen-space pixel coordinates.
/// Returns vec3(pixel_x, pixel_y, depth_along_forward).
/// Depth <= 0 means the point is behind the camera.
fn project_to_pixel(cam_rel: vec3<f32>) -> vec3<f32> {
    let fwd = camera.forward.xyz;
    let r = camera.right.xyz;
    let u = camera.up.xyz;

    let depth = dot(cam_rel, fwd);
    if depth <= 0.0 {
        return vec3<f32>(-1.0, -1.0, depth);
    }

    // Projection: camera.right and camera.up are FOV-scaled, so:
    //   ndc_x = dot(cam_rel, right) / (depth * dot(right, right))
    //   ndc_y = dot(cam_rel, up)    / (depth * dot(up, up))
    let r_sq = dot(r, r);
    let u_sq = dot(u, u);

    let ndc_x = dot(cam_rel, r) / (depth * r_sq);
    let ndc_y = dot(cam_rel, u) / (depth * u_sq);

    let pixel_x = (ndc_x * 0.5 + 0.5) * camera.resolution.x;
    let pixel_y = (0.5 - ndc_y * 0.5) * camera.resolution.y;

    return vec3<f32>(pixel_x, pixel_y, depth);
}

// ---------- Entry Point ----------

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let thread_idx = lid.y * 16u + lid.x;

    // Initialize shared memory (thread 0 only).
    if thread_idx == 0u {
        atomicStore(&tile_count, 0u);
    }
    workgroupBarrier();

    // Tile bounds in pixel coordinates.
    let tile_min = vec2<f32>(f32(wid.x * TILE_SIZE), f32(wid.y * TILE_SIZE));
    let tile_max = tile_min + vec2<f32>(f32(TILE_SIZE));

    let cam_pos = camera.position.xyz;

    // Cooperative object testing: 256 threads test different objects in strided fashion.
    for (var i = thread_idx; i < scene.num_objects; i += 256u) {
        let obj = objects[i];

        // Skip empty objects.
        if obj.sdf_type == SDF_TYPE_NONE {
            continue;
        }

        let aabb_lo = obj.aabb_min.xyz;
        let aabb_hi = obj.aabb_max.xyz;

        // Project all 8 AABB corners to screen space.
        var screen_min = camera.resolution.xy;
        var screen_max = vec2<f32>(0.0);
        var any_in_front = false;
        var any_behind = false;

        for (var c = 0u; c < 8u; c++) {
            let corner = vec3<f32>(
                select(aabb_lo.x, aabb_hi.x, (c & 1u) != 0u),
                select(aabb_lo.y, aabb_hi.y, (c & 2u) != 0u),
                select(aabb_lo.z, aabb_hi.z, (c & 4u) != 0u),
            );
            // AABB is in camera-relative space (set by CPU).
            let proj = project_to_pixel(corner);

            if proj.z > 0.0 {
                any_in_front = true;
                screen_min = min(screen_min, proj.xy);
                screen_max = max(screen_max, proj.xy);
            } else {
                any_behind = true;
            }
        }

        var overlaps = false;
        if any_behind && any_in_front {
            // AABB straddles the near plane — conservatively include in this tile.
            overlaps = true;
        } else if !any_behind && any_in_front {
            // All corners in front: check screen-space bounding rect overlap with tile.
            overlaps = screen_max.x >= tile_min.x && screen_min.x < tile_max.x &&
                       screen_max.y >= tile_min.y && screen_min.y < tile_max.y;
        }
        // If all corners behind camera, skip (overlaps stays false).

        if overlaps {
            let idx = atomicAdd(&tile_count, 1u);
            if idx < MAX_OBJECTS_PER_TILE {
                tile_objects[idx] = i;
            }
        }
    }
    workgroupBarrier();

    // Write results to global memory.
    let num_tiles_x = u32(ceil(camera.resolution.x / f32(TILE_SIZE)));
    let tile_id = wid.y * num_tiles_x + wid.x;
    let count = min(atomicLoad(&tile_count), MAX_OBJECTS_PER_TILE);
    let base = tile_id * MAX_OBJECTS_PER_TILE;

    // Cooperative write: each thread writes one entry.
    if thread_idx < count {
        tile_object_indices[base + thread_idx] = tile_objects[thread_idx];
    }

    // Thread 0 writes the count.
    if thread_idx == 0u {
        atomicStore(&tile_object_counts[tile_id], count);
    }
}
