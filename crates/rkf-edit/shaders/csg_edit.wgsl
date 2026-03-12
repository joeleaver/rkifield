// CSG edit compute shader — v2 object-local editing.
//
// One workgroup (8x8x8 = 512 threads) processes one brick.
// Each thread handles one voxel: evaluates the analytic SDF primitive
// in edit-local space, applies the CSG operation, and writes back
// the modified voxel sample to the brick pool.

// ---------- Types ----------

struct VoxelSample {
    word0: u32, // lower 16 = f16 distance, upper 8 = u8 material_id
    word1: u32, // RGBA8 per-voxel color
}

struct EditParams {
    // Spatial (3 x vec4 = 48 bytes)
    position:   vec4<f32>,  // xyz = object-local pos, w = unused
    rotation:   vec4<f32>,  // quaternion xyzw
    dimensions: vec4<f32>,  // xyz = half-extents/radius, w = unused

    // Parameters (1 x vec4 = 16 bytes)
    strength:    f32,
    blend_k:     f32,
    falloff:     u32,       // 0=Linear, 1=Smooth, 2=Sharp
    material_id: u32,

    // Type info (1 x vec4 = 16 bytes)
    edit_type:    u32,      // 0-8 (see EditType enum)
    shape_type:   u32,      // 0-5 (see ShapeType enum)
    color_packed: u32,      // RGBA8 packed
    _pad_type:    u32,

    // Brick info (1 x vec4 = 16 bytes)
    brick_base_index: u32,
    brick_local_min:  vec3<f32>,

    // Grid info (1 x vec4 = 16 bytes)
    voxel_size: f32,
    _pad:       vec3<f32>,

    // Padding (1 x vec4 = 16 bytes)
    _pad2: vec4<f32>,
}

// ---------- Constants ----------

// Edit types
const EDIT_CSG_UNION: u32         = 0u;
const EDIT_CSG_SUBTRACT: u32      = 1u;
const EDIT_CSG_INTERSECT: u32     = 2u;
const EDIT_SMOOTH_UNION: u32      = 3u;
const EDIT_SMOOTH_SUBTRACT: u32   = 4u;
const EDIT_SMOOTH: u32            = 5u;
const EDIT_FLATTEN: u32           = 6u;
const EDIT_PAINT: u32             = 7u;
const EDIT_COLOR_PAINT: u32       = 8u;

// Shape types
const SHAPE_SPHERE: u32   = 0u;
const SHAPE_BOX: u32      = 1u;
const SHAPE_CAPSULE: u32  = 2u;
const SHAPE_CYLINDER: u32 = 3u;
const SHAPE_TORUS: u32    = 4u;
const SHAPE_PLANE: u32    = 5u;

// Falloff curves
const FALLOFF_LINEAR: u32 = 0u;
const FALLOFF_SMOOTH: u32 = 1u;
const FALLOFF_SHARP: u32  = 2u;

// Surface band for paint operations: voxels within this distance are
// considered "near surface" and eligible for material/color modification.
const PAINT_SURFACE_BAND: f32 = 0.02;

// ---------- Bindings ----------

// Group 0: edit data
@group(0) @binding(0) var<storage, read_write> brick_pool: array<VoxelSample>;
@group(0) @binding(1) var<uniform> edit: EditParams;

// ---------- Voxel pack/unpack ----------

fn extract_distance(word0: u32) -> f32 {
    return unpack2x16float(word0).x;
}

fn extract_material_id(word0: u32) -> u32 {
    return (word0 >> 16u) & 0x3Fu;
}

fn pack_word0(distance: f32, material_id: u32) -> u32 {
    return (material_id << 16u) | (pack2x16float(vec2<f32>(distance, 0.0)) & 0xFFFFu);
}

// ---------- Quaternion rotation ----------

/// Rotate a point by the inverse of a unit quaternion (conjugate rotation).
/// This transforms from world space into edit-local space.
fn quat_rotate_inverse(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    // Conjugate of q = (-q.xyz, q.w)
    let q_conj = vec4<f32>(-q.xyz, q.w);
    // q_conj * v * q  (Hamilton product, vector part)
    let t = 2.0 * cross(q_conj.xyz, v);
    return v + q_conj.w * t + cross(q_conj.xyz, t);
}

// ---------- Analytic SDF primitives ----------

fn sdf_sphere(p: vec3<f32>, radius: f32) -> f32 {
    return length(p) - radius;
}

fn sdf_box(p: vec3<f32>, half_extents: vec3<f32>) -> f32 {
    let d = abs(p) - half_extents;
    return length(max(d, vec3<f32>(0.0))) + min(max(d.x, max(d.y, d.z)), 0.0);
}

fn sdf_capsule(p: vec3<f32>, radius: f32, half_height: f32) -> f32 {
    // Capsule along local Y axis
    let clamped_y = clamp(p.y, -half_height, half_height);
    return length(p - vec3<f32>(0.0, clamped_y, 0.0)) - radius;
}

fn sdf_cylinder(p: vec3<f32>, radius: f32, half_height: f32) -> f32 {
    // Cylinder along local Y axis
    let d = vec2<f32>(length(p.xz) - radius, abs(p.y) - half_height);
    return min(max(d.x, d.y), 0.0) + length(max(d, vec2<f32>(0.0)));
}

fn sdf_torus(p: vec3<f32>, major_radius: f32, minor_radius: f32) -> f32 {
    // Torus in XZ plane
    let q = vec2<f32>(length(p.xz) - major_radius, p.y);
    return length(q) - minor_radius;
}

fn sdf_plane(p: vec3<f32>, offset: f32) -> f32 {
    // Plane with normal along +Y, offset along normal
    return p.y - offset;
}

/// Evaluate the SDF primitive specified by edit.shape_type at point p
/// (in edit-local space).
fn eval_shape(p: vec3<f32>) -> f32 {
    let dims = edit.dimensions.xyz;

    switch edit.shape_type {
        case SHAPE_SPHERE: {
            return sdf_sphere(p, dims.x);
        }
        case SHAPE_BOX: {
            return sdf_box(p, dims);
        }
        case SHAPE_CAPSULE: {
            return sdf_capsule(p, dims.x, dims.y);
        }
        case SHAPE_CYLINDER: {
            return sdf_cylinder(p, dims.x, dims.y);
        }
        case SHAPE_TORUS: {
            return sdf_torus(p, dims.x, dims.y);
        }
        case SHAPE_PLANE: {
            return sdf_plane(p, dims.x);
        }
        default: {
            return sdf_sphere(p, dims.x);
        }
    }
}

// ---------- Falloff ----------

/// Compute falloff weight based on distance from edit center.
/// Returns 1.0 at center, 0.0 at or beyond the shape boundary.
fn compute_falloff(dist_from_center: f32, max_radius: f32) -> f32 {
    let t = clamp(dist_from_center / max(max_radius, 0.0001), 0.0, 1.0);

    switch edit.falloff {
        case FALLOFF_LINEAR: {
            return 1.0 - t;
        }
        case FALLOFF_SMOOTH: {
            // smoothstep: 1 at center (t=0), 0 at edge (t=1)
            let s = t * t * (3.0 - 2.0 * t);
            return 1.0 - s;
        }
        case FALLOFF_SHARP: {
            let inv = 1.0 - t;
            return inv * inv * inv;
        }
        default: {
            return 1.0 - t;
        }
    }
}

// ---------- CSG operations ----------

fn csg_union(existing: f32, shape: f32) -> f32 {
    return min(existing, shape);
}

fn csg_subtract(existing: f32, shape: f32) -> f32 {
    return max(existing, -shape);
}

fn csg_intersect(existing: f32, shape: f32) -> f32 {
    return max(existing, shape);
}

fn smooth_min(a: f32, b: f32, k: f32) -> f32 {
    let h = clamp(0.5 + 0.5 * (b - a) / max(k, 0.0001), 0.0, 1.0);
    return mix(b, a, h) - k * h * (1.0 - h);
}

fn csg_smooth_union(existing: f32, shape: f32, k: f32) -> f32 {
    return smooth_min(existing, shape, k);
}

fn csg_smooth_subtract(existing: f32, shape: f32, k: f32) -> f32 {
    return -smooth_min(-existing, shape, k);
}

/// Apply the CSG operation specified by edit.edit_type.
/// Returns the new distance value.
fn apply_csg(existing: f32, shape_dist: f32, falloff_weight: f32) -> f32 {
    let k = edit.blend_k;
    let str = edit.strength * falloff_weight;

    switch edit.edit_type {
        case EDIT_CSG_UNION: {
            let result = csg_union(existing, shape_dist);
            return mix(existing, result, str);
        }
        case EDIT_CSG_SUBTRACT: {
            let result = csg_subtract(existing, shape_dist);
            return mix(existing, result, str);
        }
        case EDIT_CSG_INTERSECT: {
            let result = csg_intersect(existing, shape_dist);
            return mix(existing, result, str);
        }
        case EDIT_SMOOTH_UNION: {
            let result = csg_smooth_union(existing, shape_dist, k);
            return mix(existing, result, str);
        }
        case EDIT_SMOOTH_SUBTRACT: {
            let result = csg_smooth_subtract(existing, shape_dist, k);
            return mix(existing, result, str);
        }
        case EDIT_FLATTEN: {
            // Pull toward shape surface (distance = 0 at shape boundary)
            let target = max(existing, shape_dist);
            return mix(existing, target, str);
        }
        default: {
            return existing;
        }
    }
}

// ---------- Smooth brush ----------

/// Smooth brush: average SDF values from 6-connected neighbors.
/// This requires reading neighboring voxels within the same brick.
fn apply_smooth(
    idx: u32,
    ix: u32, iy: u32, iz: u32,
    existing: f32,
    falloff_weight: f32,
) -> f32 {
    let base = edit.brick_base_index;
    var sum = 0.0;
    var count = 0.0;

    // +X neighbor
    if ix < 7u {
        sum += extract_distance(brick_pool[base + (ix + 1u) + iy * 8u + iz * 64u].word0);
        count += 1.0;
    }
    // -X neighbor
    if ix > 0u {
        sum += extract_distance(brick_pool[base + (ix - 1u) + iy * 8u + iz * 64u].word0);
        count += 1.0;
    }
    // +Y neighbor
    if iy < 7u {
        sum += extract_distance(brick_pool[base + ix + (iy + 1u) * 8u + iz * 64u].word0);
        count += 1.0;
    }
    // -Y neighbor
    if iy > 0u {
        sum += extract_distance(brick_pool[base + ix + (iy - 1u) * 8u + iz * 64u].word0);
        count += 1.0;
    }
    // +Z neighbor
    if iz < 7u {
        sum += extract_distance(brick_pool[base + ix + iy * 8u + (iz + 1u) * 64u].word0);
        count += 1.0;
    }
    // -Z neighbor
    if iz > 0u {
        sum += extract_distance(brick_pool[base + ix + iy * 8u + (iz - 1u) * 64u].word0);
        count += 1.0;
    }

    if count > 0.0 {
        let avg = sum / count;
        let str = edit.strength * falloff_weight;
        return mix(existing, avg, str);
    }
    return existing;
}

// ---------- Main entry point ----------

@compute @workgroup_size(8, 8, 8)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let ix = lid.x;
    let iy = lid.y;
    let iz = lid.z;

    // Linear index within the 8x8x8 brick
    let voxel_idx = ix + iy * 8u + iz * 64u;
    let global_idx = edit.brick_base_index + voxel_idx;

    // Read existing voxel
    let voxel = brick_pool[global_idx];
    let existing_dist = extract_distance(voxel.word0);
    let existing_mat = extract_material_id(voxel.word0);
    let existing_color = voxel.word1;

    // Compute object-local position of this voxel
    let voxel_pos = edit.brick_local_min + (vec3<f32>(f32(ix), f32(iy), f32(iz)) + 0.5) * edit.voxel_size;

    // Transform from object-local to edit-local space
    let local_pos = quat_rotate_inverse(edit.rotation, voxel_pos - edit.position.xyz);

    // Evaluate SDF primitive in local space
    let shape_dist = eval_shape(local_pos);

    // Compute distance from edit center for falloff
    let dist_from_center = length(voxel_pos - edit.position.xyz);
    let max_radius = max(edit.dimensions.x, max(edit.dimensions.y, edit.dimensions.z));
    let falloff_weight = compute_falloff(dist_from_center, max_radius);

    // Skip if outside falloff radius (no effect)
    if falloff_weight <= 0.0 {
        return;
    }

    // Apply the edit operation
    var new_dist = existing_dist;
    var new_mat = existing_mat;
    var new_color = existing_color;

    switch edit.edit_type {
        // CSG operations: modify distance and potentially material
        case EDIT_CSG_UNION, EDIT_CSG_SUBTRACT, EDIT_CSG_INTERSECT,
             EDIT_SMOOTH_UNION, EDIT_SMOOTH_SUBTRACT, EDIT_FLATTEN: {
            new_dist = apply_csg(existing_dist, shape_dist, falloff_weight);

            // For union/smooth_union: paint material where surface was added
            if edit.edit_type == EDIT_CSG_UNION || edit.edit_type == EDIT_SMOOTH_UNION {
                if new_dist < existing_dist {
                    // Blend material based on how much the surface moved
                    let blend_factor = clamp((existing_dist - new_dist) / max(abs(existing_dist), 0.001), 0.0, 1.0);
                    if blend_factor > 0.5 {
                        new_mat = edit.material_id;
                    }
                }
            }
        }

        // Smooth brush: average neighbors
        case EDIT_SMOOTH: {
            new_dist = apply_smooth(global_idx, ix, iy, iz, existing_dist, falloff_weight);
        }

        // Paint: set material on near-surface voxels (no geometry change)
        case EDIT_PAINT: {
            if abs(existing_dist) < PAINT_SURFACE_BAND + max_radius {
                let str = edit.strength * falloff_weight;
                if str > 0.5 {
                    new_mat = edit.material_id;
                }
            }
        }

        // Color paint: handled externally via companion color pool.
        case EDIT_COLOR_PAINT: {}

        default: {}
    }

    // Write back modified voxel
    var new_voxel: VoxelSample;
    new_voxel.word0 = pack_word0(new_dist, new_mat);
    new_voxel.word1 = new_color;
    brick_pool[global_idx] = new_voxel;
}
