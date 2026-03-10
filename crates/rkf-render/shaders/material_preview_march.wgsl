// material_preview_march.wgsl — Ray-march a single analytical SDF primitive
// for the material preview panel. Writes G-buffer outputs compatible with the
// main shading pass.

// ---------- Constants ----------

const PRIM_SPHERE: u32   = 0u;
const PRIM_BOX: u32      = 1u;
const PRIM_CAPSULE: u32  = 2u;
const PRIM_TORUS: u32    = 3u;
const PRIM_CYLINDER: u32 = 4u;
const PRIM_PLANE: u32    = 5u;

const MAX_STEPS: u32  = 64u;
const MAX_DIST: f32   = 20.0;
const HIT_THRESH: f32 = 0.001;
const NORM_EPS: f32   = 0.001;

// ---------- Uniforms ----------

struct PreviewUniforms {
    camera_pos: vec4<f32>,
    camera_forward: vec4<f32>,
    camera_right: vec4<f32>,
    camera_up: vec4<f32>,
    resolution: vec2<f32>,
    primitive_type: u32,
    material_id: u32,
}

@group(0) @binding(0) var<uniform> uniforms: PreviewUniforms;

// ---------- G-buffer outputs ----------

@group(1) @binding(0) var gbuf_position: texture_storage_2d<rgba32float, write>;
@group(1) @binding(1) var gbuf_normal:   texture_storage_2d<rgba16float, write>;
@group(1) @binding(2) var gbuf_material: texture_storage_2d<r32uint, write>;
@group(1) @binding(3) var gbuf_motion:   texture_storage_2d<rgba32float, write>;

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

fn sdf_plane(p: vec3<f32>) -> f32 {
    return p.y;
}

// ---------- Scene SDF ----------

fn scene_sdf(p: vec3<f32>) -> f32 {
    switch uniforms.primitive_type {
        case PRIM_SPHERE: {
            return sdf_sphere(p, 1.0);
        }
        case PRIM_BOX: {
            return sdf_box(p, vec3<f32>(0.8, 0.8, 0.8));
        }
        case PRIM_CAPSULE: {
            return sdf_capsule(p, 0.5, 0.7);
        }
        case PRIM_TORUS: {
            return sdf_torus(p, 0.7, 0.3);
        }
        case PRIM_CYLINDER: {
            return sdf_cylinder(p, 0.6, 0.8);
        }
        case PRIM_PLANE: {
            return sdf_plane(p);
        }
        default: {
            return sdf_sphere(p, 1.0);
        }
    }
}

// ---------- Normal via central differences ----------

fn calc_normal(p: vec3<f32>) -> vec3<f32> {
    let e = vec2<f32>(NORM_EPS, 0.0);
    return normalize(vec3<f32>(
        scene_sdf(p + e.xyy) - scene_sdf(p - e.xyy),
        scene_sdf(p + e.yxy) - scene_sdf(p - e.yxy),
        scene_sdf(p + e.yyx) - scene_sdf(p - e.yyx),
    ));
}

// ---------- Entry point ----------

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Bounds check.
    if gid.x >= u32(uniforms.resolution.x) || gid.y >= u32(uniforms.resolution.y) {
        return;
    }

    let pixel = vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5);
    let uv = pixel / uniforms.resolution;
    let u = uv.x * 2.0 - 1.0;
    let v = 1.0 - uv.y * 2.0; // flip Y: top of image = +up

    let ray_origin = uniforms.camera_pos.xyz;
    let ray_dir = normalize(
        uniforms.camera_forward.xyz + u * uniforms.camera_right.xyz + v * uniforms.camera_up.xyz
    );

    // Sphere trace.
    var t = 0.0;
    var hit = false;
    for (var i = 0u; i < MAX_STEPS; i++) {
        let p = ray_origin + ray_dir * t;
        let d = scene_sdf(p);
        if d < HIT_THRESH {
            hit = true;
            break;
        }
        t += d;
        if t > MAX_DIST {
            break;
        }
    }

    let coord = vec2<i32>(gid.xy);

    if hit {
        let hit_pos = ray_origin + ray_dir * t;
        let normal = calc_normal(hit_pos);
        let packed_mat = uniforms.material_id & 0xFFFFu; // lo16 = material_id, hi16 = 0

        textureStore(gbuf_position, coord, vec4<f32>(hit_pos, t));
        textureStore(gbuf_normal,   coord, vec4<f32>(normal, 0.0));
        textureStore(gbuf_material, coord, vec4<u32>(packed_mat, 0u, 0u, 0u));
        textureStore(gbuf_motion,   coord, vec4<f32>(0.0, 0.0, 0.0, t));
    } else {
        textureStore(gbuf_position, coord, vec4<f32>(0.0, 0.0, 0.0, -1.0));
        textureStore(gbuf_normal,   coord, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        textureStore(gbuf_material, coord, vec4<u32>(0u, 0u, 0u, 0u));
        textureStore(gbuf_motion,   coord, vec4<f32>(0.0));
    }
}
