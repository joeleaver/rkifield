// Joint rebaking compute shader — Phase 17 (Skeletal Animation).
//
// For each voxel in a joint region:
// 1. Compute world position from brick coords
// 2. Transform to segment A local space, sample SDF from segment A bricks
// 3. Transform to segment B local space, sample SDF from segment B bricks
// 4. Smooth-min blend the two distances
// 5. Material from closer segment
// 6. Write result to joint brick in brick pool with JOINT_REGION flag

// ---------- Types ----------

struct VoxelSample {
    word0: u32, // lower 16 = f16 distance, upper 16 = u16 material_id
    word1: u32, // byte0 = blend_weight, byte1 = secondary_id, byte2 = flags, byte3 = reserved
}

struct JointParams {
    // Inverse bone matrix for segment A (world -> A local)
    inv_bone_a: mat4x4<f32>,
    // Inverse bone matrix for segment B (world -> B local)
    inv_bone_b: mat4x4<f32>,
    // Joint brick info
    joint_brick_base: u32,      // base index in brick pool for output
    joint_brick_count: u32,     // number of joint bricks
    // Segment A brick info
    seg_a_brick_base: u32,
    seg_a_brick_count: u32,
    // Segment B brick info
    seg_b_brick_base: u32,
    seg_b_brick_count: u32,
    // Joint parameters
    blend_k: f32,               // smooth-min k parameter
    voxel_size: f32,            // voxel size for this tier
    // World-space origin of joint brick region
    joint_world_min: vec3<f32>,
    _pad0: f32,
    // Segment A rest-pose AABB min (local space)
    seg_a_local_min: vec3<f32>,
    _pad1: f32,
    // Segment B rest-pose AABB min (local space)
    seg_b_local_min: vec3<f32>,
    _pad2: f32,
    // Segment voxel sizes (may differ from joint voxel_size if different tiers)
    seg_a_voxel_size: f32,
    seg_b_voxel_size: f32,
    // Joint brick region dimensions in bricks (x, y, z)
    region_bricks_x: u32,
    region_bricks_y: u32,
    // Segment A grid dimensions in bricks (for 3D indexing)
    seg_a_bricks_x: u32,
    seg_a_bricks_y: u32,
    seg_a_bricks_z: u32,
    // Segment B grid dimensions in bricks (for 3D indexing)
    seg_b_bricks_x: u32,
    seg_b_bricks_y: u32,
    seg_b_bricks_z: u32,
}

// ---------- Constants ----------

// Voxel flags
const FLAG_JOINT_REGION: u32 = 1u; // Lipschitz mitigation: ray marcher uses 0.8x step

// Large positive distance returned when sampling outside a segment's brick region.
const OUT_OF_BOUNDS_DIST: f32 = 1.0;

// Default material for out-of-bounds samples.
const OUT_OF_BOUNDS_MAT: u32 = 0u;

// ---------- Bindings ----------

@group(0) @binding(0) var<storage, read_write> brick_pool: array<VoxelSample>;
@group(0) @binding(1) var<uniform> params: JointParams;

// ---------- Voxel pack/unpack ----------

fn extract_distance(word0: u32) -> f32 {
    return unpack2x16float(word0).x;
}

fn extract_material_id(word0: u32) -> u32 {
    return word0 >> 16u;
}

fn pack_word0(distance: f32, material_id: u32) -> u32 {
    return (material_id << 16u) | (pack2x16float(vec2<f32>(distance, 0.0)) & 0xFFFFu);
}

fn pack_word1(blend_weight: u32, secondary_id: u32, flags: u32, reserved: u32) -> u32 {
    return (reserved << 24u) | (flags << 16u) | (secondary_id << 8u) | blend_weight;
}

// ---------- Smooth-min (architecture spec) ----------

/// Polynomial smooth minimum as specified in the architecture doc.
/// Produces a smooth blend between two SDF distances with blend radius k.
fn smooth_min(a: f32, b: f32, k: f32) -> f32 {
    let h = max(k - abs(a - b), 0.0) / max(k, 0.0001);
    return min(a, b) - h * h * k * 0.25;
}

// ---------- Segment sampling ----------

/// Sample SDF distance from a segment's bricks using nearest-neighbor lookup.
///
/// local_pos: position in segment's rest-pose local space
/// seg_min: rest-pose AABB minimum corner
/// seg_voxel_size: voxel size of the segment's bricks
/// brick_base: first brick index in the pool for this segment
/// bricks_x/y/z: grid dimensions of the segment in bricks
fn sample_segment_sdf(
    local_pos: vec3<f32>,
    seg_min: vec3<f32>,
    seg_voxel_size: f32,
    brick_base: u32,
    bricks_x: u32,
    bricks_y: u32,
    bricks_z: u32,
) -> f32 {
    // Continuous voxel coordinate within the segment grid
    let voxel_coord = (local_pos - seg_min) / seg_voxel_size;

    // Nearest voxel (integer coordinate)
    let vi = vec3<i32>(floor(voxel_coord));

    // Total grid size in voxels
    let grid_voxels = vec3<i32>(vec3<u32>(bricks_x * 8u, bricks_y * 8u, bricks_z * 8u));

    // Bounds check
    if vi.x < 0 || vi.y < 0 || vi.z < 0 ||
       vi.x >= grid_voxels.x || vi.y >= grid_voxels.y || vi.z >= grid_voxels.z {
        return OUT_OF_BOUNDS_DIST;
    }

    let uv = vec3<u32>(vi);

    // Which brick (3D)
    let brick_coord = uv / 8u;
    // Which voxel within the brick
    let local_voxel = uv % 8u;

    // Linear brick index
    let brick_idx = brick_coord.x + brick_coord.y * bricks_x + brick_coord.z * bricks_x * bricks_y;

    // Linear voxel index within the brick (x + y*8 + z*64)
    let voxel_idx = local_voxel.x + local_voxel.y * 8u + local_voxel.z * 64u;

    // Global index into brick_pool
    let global_idx = (brick_base + brick_idx) * 512u + voxel_idx;

    return extract_distance(brick_pool[global_idx].word0);
}

/// Sample material ID from a segment's bricks using nearest-neighbor lookup.
fn sample_segment_material(
    local_pos: vec3<f32>,
    seg_min: vec3<f32>,
    seg_voxel_size: f32,
    brick_base: u32,
    bricks_x: u32,
    bricks_y: u32,
    bricks_z: u32,
) -> u32 {
    let voxel_coord = (local_pos - seg_min) / seg_voxel_size;
    let vi = vec3<i32>(floor(voxel_coord));
    let grid_voxels = vec3<i32>(vec3<u32>(bricks_x * 8u, bricks_y * 8u, bricks_z * 8u));

    if vi.x < 0 || vi.y < 0 || vi.z < 0 ||
       vi.x >= grid_voxels.x || vi.y >= grid_voxels.y || vi.z >= grid_voxels.z {
        return OUT_OF_BOUNDS_MAT;
    }

    let uv = vec3<u32>(vi);
    let brick_coord = uv / 8u;
    let local_voxel = uv % 8u;
    let brick_idx = brick_coord.x + brick_coord.y * bricks_x + brick_coord.z * bricks_x * bricks_y;
    let voxel_idx = local_voxel.x + local_voxel.y * 8u + local_voxel.z * 64u;
    let global_idx = (brick_base + brick_idx) * 512u + voxel_idx;

    return extract_material_id(brick_pool[global_idx].word0);
}

// ---------- Main entry point ----------

@compute @workgroup_size(8, 8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Each workgroup processes one 8x8x8 brick in the joint region.
    // global_invocation_id spans the entire joint region in voxels.
    let bricks_per_row = params.region_bricks_x;

    // Which brick within the joint region (3D)
    let brick_local = gid / 8u;
    // Which voxel within that brick
    let voxel_local = gid % 8u;

    // Linear brick index within the joint region
    let brick_idx = brick_local.x
        + brick_local.y * bricks_per_row
        + brick_local.z * bricks_per_row * params.region_bricks_y;

    // Bounds check: skip threads beyond the joint brick count
    if brick_idx >= params.joint_brick_count {
        return;
    }

    // World position of this voxel (center of voxel cell)
    let world_pos = params.joint_world_min
        + vec3<f32>(gid) * params.voxel_size
        + vec3<f32>(0.5) * params.voxel_size;

    // Transform to segment A local space and sample SDF
    let local_a = (params.inv_bone_a * vec4<f32>(world_pos, 1.0)).xyz;
    let sdf_a = sample_segment_sdf(
        local_a,
        params.seg_a_local_min,
        params.seg_a_voxel_size,
        params.seg_a_brick_base,
        params.seg_a_bricks_x,
        params.seg_a_bricks_y,
        params.seg_a_bricks_z,
    );

    // Transform to segment B local space and sample SDF
    let local_b = (params.inv_bone_b * vec4<f32>(world_pos, 1.0)).xyz;
    let sdf_b = sample_segment_sdf(
        local_b,
        params.seg_b_local_min,
        params.seg_b_voxel_size,
        params.seg_b_brick_base,
        params.seg_b_bricks_x,
        params.seg_b_bricks_y,
        params.seg_b_bricks_z,
    );

    // Smooth-min blend the two distances
    let blended = smooth_min(sdf_a, sdf_b, params.blend_k);

    // Material from whichever segment is closer
    let mat_a = sample_segment_material(
        local_a,
        params.seg_a_local_min,
        params.seg_a_voxel_size,
        params.seg_a_brick_base,
        params.seg_a_bricks_x,
        params.seg_a_bricks_y,
        params.seg_a_bricks_z,
    );
    let mat_b = sample_segment_material(
        local_b,
        params.seg_b_local_min,
        params.seg_b_voxel_size,
        params.seg_b_brick_base,
        params.seg_b_bricks_x,
        params.seg_b_bricks_y,
        params.seg_b_bricks_z,
    );
    // select(false_val, true_val, cond) — reversed vs most languages
    let material = select(mat_b, mat_a, sdf_a <= sdf_b);

    // Compute output index in brick pool
    let voxel_idx = voxel_local.x + voxel_local.y * 8u + voxel_local.z * 64u;
    let global_idx = (params.joint_brick_base + brick_idx) * 512u + voxel_idx;

    // Write result with JOINT_REGION flag for Lipschitz mitigation
    var out: VoxelSample;
    out.word0 = pack_word0(blended, material);
    out.word1 = pack_word1(0u, 0u, FLAG_JOINT_REGION, 0u);
    brick_pool[global_idx] = out;
}
