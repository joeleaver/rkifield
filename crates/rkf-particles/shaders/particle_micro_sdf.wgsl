// SDF micro-object helper functions for the ray marcher.
//
// These are meant to be included (copy-pasted or @import'd) into the
// ray march pass, not dispatched standalone. They evaluate small SDF
// primitives (spheres, capsules) that represent solid particles with
// full shading, shadows, and GI.
//
// Bindings expected from the including shader:
//   @group(N) @binding(0) var<storage, read> micro_objects: array<SdfMicroObject>;
//   @group(N) @binding(1) var<uniform> micro_params: MicroParams;

/// GPU micro-object descriptor -- 48 bytes, matches Rust `SdfMicroObject`.
struct SdfMicroObject {
    position: vec3<f32>,
    radius: f32,
    end_offset: vec3<f32>,
    material_id: u32,
    color: vec3<f32>,
    emission: f32,
}

/// Uniform parameters for the micro-object pass.
struct MicroParams {
    object_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// ---- SDF primitives ----

/// Signed distance from `pos` to a sphere defined by the micro-object.
fn sdf_micro_sphere(pos: vec3<f32>, obj: SdfMicroObject) -> f32 {
    return distance(pos, obj.position) - obj.radius;
}

/// Signed distance from `pos` to a capsule defined by the micro-object.
/// The capsule runs from `obj.position` to `obj.position + obj.end_offset`
/// with thickness `obj.radius`.
fn sdf_micro_capsule(pos: vec3<f32>, obj: SdfMicroObject) -> f32 {
    let a = obj.position;
    let b = obj.position + obj.end_offset;
    let pa = pos - a;
    let ba = b - a;
    let len_sq = dot(ba, ba);

    // Degenerate capsule (a == b) -- treat as sphere.
    var h = 0.0;
    if len_sq > 1e-10 {
        h = clamp(dot(pa, ba) / len_sq, 0.0, 1.0);
    }

    return length(pa - ba * h) - obj.radius;
}

// ---- Evaluation ----

/// Evaluate a single micro-object SDF at `pos`, choosing sphere or capsule
/// based on whether `end_offset` is non-zero.
fn sdf_micro_object(pos: vec3<f32>, obj: SdfMicroObject) -> f32 {
    let offset_len_sq = dot(obj.end_offset, obj.end_offset);
    if offset_len_sq < 1e-10 {
        return sdf_micro_sphere(pos, obj);
    } else {
        return sdf_micro_capsule(pos, obj);
    }
}

/// Evaluate all micro-objects and return vec2(min_distance, object_index).
///
/// Returns vec2(MAX_DIST, -1.0) if no objects exist. The caller can use
/// the object index to look up material/color from the storage buffer.
fn evaluate_micro_objects_sdf(pos: vec3<f32>, objects: ptr<storage, array<SdfMicroObject>, read>, count: u32) -> vec2<f32> {
    let MAX_DIST = 1e10;
    var best_dist = MAX_DIST;
    var best_idx = -1.0;

    for (var i = 0u; i < count; i++) {
        let obj = objects[i];
        let d = sdf_micro_object(pos, obj);

        if d < best_dist {
            best_dist = d;
            best_idx = f32(i);
        }
    }

    return vec2<f32>(best_dist, best_idx);
}
