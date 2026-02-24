// Terrain evaluation helper for the v2 ray marcher.
// Uses a procedural height function for O(1) tile access — no BVH required.
//
// These functions are meant to be included (via concatenation or
// `#include`-style preprocessing) into the ray-march compute pass when
// terrain objects are present in the scene.

// ---------------------------------------------------------------------------
// Procedural terrain height (layered sine waves)
// ---------------------------------------------------------------------------

/// Evaluate terrain height at an XZ world position.
///
/// Returns the Y coordinate of the terrain surface directly below (or above)
/// `pos_xz`. Two frequency layers give coarse rolling hills plus fine detail.
fn terrain_height(pos_xz: vec2<f32>) -> f32 {
    let h1 = sin(pos_xz.x * 0.02) * sin(pos_xz.y * 0.02) * 20.0;
    let h2 = sin(pos_xz.x * 0.1)  * sin(pos_xz.y * 0.08) * 3.0;
    return h1 + h2;
}

// ---------------------------------------------------------------------------
// Terrain SDF
// ---------------------------------------------------------------------------

/// Evaluate the terrain SDF at a world-space position.
///
/// Returns the signed distance to the terrain surface:
/// - Positive above the ground (outside the solid).
/// - Negative below the ground (inside the solid).
///
/// This is a height-field SDF and is only exact along the Y axis; for
/// inclined surfaces the Lipschitz constant exceeds 1. Use a conservative
/// step scale (≤ 0.5) when sphere-tracing against this function alone.
fn evaluate_terrain(world_pos: vec3<f32>) -> f32 {
    let ground_height = terrain_height(world_pos.xz);
    return world_pos.y - ground_height;
}

// ---------------------------------------------------------------------------
// Terrain normal
// ---------------------------------------------------------------------------

/// Estimate the terrain surface normal at `world_pos` via central finite
/// differences on the height function.
///
/// Returns a unit normal pointing away from the terrain (upward on flat
/// ground). The epsilon `0.1` is suitable for the frequency content of
/// `terrain_height`; reduce for higher-frequency detail or increase for
/// smoother normals.
fn terrain_normal(world_pos: vec3<f32>) -> vec3<f32> {
    let eps = 0.1;
    let h  = terrain_height(world_pos.xz);
    let hx = terrain_height(world_pos.xz + vec2<f32>(eps, 0.0));
    let hz = terrain_height(world_pos.xz + vec2<f32>(0.0, eps));
    let dx = (hx - h) / eps;
    let dz = (hz - h) / eps;
    return normalize(vec3<f32>(-dx, 1.0, -dz));
}
