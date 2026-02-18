// Edge-aware sharpening compute shader.
//
// 5×5 cross kernel with weights based on material ID and depth similarity.
// Unsharp mask: result = center + (center - blur) * strength.
// Material ID boundaries create hard sharpening edges.

struct SharpenUniforms {
    width: u32,
    height: u32,
    strength: f32,    // sharpening strength (default 0.5)
    _pad: u32,
}

// Group 0: input (upscaled HDR, read-only)
@group(0) @binding(0) var input_color: texture_2d<f32>;
// G-buffer for edge detection (at internal res — sample nearest)
@group(0) @binding(1) var gbuf_position: texture_2d<f32>;
@group(0) @binding(2) var gbuf_material: texture_2d<u32>;

// Group 1: output (sharpened HDR, write-only)
@group(1) @binding(0) var output_color: texture_storage_2d<rgba16float, write>;

// Group 2: uniforms
@group(2) @binding(0) var<uniform> params: SharpenUniforms;

const DEPTH_SIGMA: f32 = 0.05;   // depth similarity falloff

// 5×5 cross kernel offsets and base weights (Gaussian-like)
// Cross pattern: center + 4 cardinal directions at distances 1 and 2
const CROSS_OFFSETS: array<vec2<i32>, 9> = array<vec2<i32>, 9>(
    vec2<i32>(0, 0),    // center
    vec2<i32>(1, 0),    // right 1
    vec2<i32>(-1, 0),   // left 1
    vec2<i32>(0, 1),    // down 1
    vec2<i32>(0, -1),   // up 1
    vec2<i32>(2, 0),    // right 2
    vec2<i32>(-2, 0),   // left 2
    vec2<i32>(0, 2),    // down 2
    vec2<i32>(0, -2),   // up 2
);

const CROSS_WEIGHTS: array<f32, 9> = array<f32, 9>(
    0.0,   // center (not used in blur, only as reference)
    0.25,  // distance 1
    0.25,
    0.25,
    0.25,
    0.0625, // distance 2
    0.0625,
    0.0625,
    0.0625,
);

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.width || gid.y >= params.height {
        return;
    }

    let coord = vec2<i32>(gid.xy);
    let dims = vec2<i32>(i32(params.width), i32(params.height));

    // Read center pixel
    let center_color = textureLoad(input_color, coord, 0).rgb;

    // Sample center depth and material for edge detection
    // Map display coord to internal-res coord for G-buffer lookup
    let internal_dims = vec2<i32>(textureDimensions(gbuf_position));
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);
    let internal_coord = clamp(
        vec2<i32>(uv * vec2<f32>(internal_dims)),
        vec2<i32>(0),
        internal_dims - vec2<i32>(1)
    );

    let center_depth = textureLoad(gbuf_position, internal_coord, 0).w;
    let center_mat = textureLoad(gbuf_material, internal_coord, 0).r & 0xFFFFu;

    // Weighted blur using cross kernel with edge-aware weights
    var blur = vec3<f32>(0.0);
    var total_weight = 0.0;

    for (var i = 1; i < 9; i++) {
        let offset = CROSS_OFFSETS[i];
        let sample_coord = clamp(coord + offset, vec2<i32>(0), dims - vec2<i32>(1));

        let sample_color = textureLoad(input_color, sample_coord, 0).rgb;

        // Sample neighbor's depth and material
        let neighbor_uv = (vec2<f32>(sample_coord) + 0.5) / vec2<f32>(dims);
        let neighbor_internal = clamp(
            vec2<i32>(neighbor_uv * vec2<f32>(internal_dims)),
            vec2<i32>(0),
            internal_dims - vec2<i32>(1)
        );
        let neighbor_depth = textureLoad(gbuf_position, neighbor_internal, 0).w;
        let neighbor_mat = textureLoad(gbuf_material, neighbor_internal, 0).r & 0xFFFFu;

        // Edge-aware weight: zero at material boundaries, falloff with depth difference
        var edge_weight = 1.0;

        // Material boundary: hard edge
        if neighbor_mat != center_mat {
            edge_weight = 0.0;
        }

        // Depth similarity: Gaussian falloff
        let depth_diff = abs(center_depth - neighbor_depth) / max(center_depth, 0.001);
        edge_weight *= exp(-depth_diff * depth_diff / (2.0 * DEPTH_SIGMA * DEPTH_SIGMA));

        let w = CROSS_WEIGHTS[i] * edge_weight;
        blur += sample_color * w;
        total_weight += w;
    }

    // Normalize blur
    if total_weight > 0.001 {
        blur /= total_weight;
    } else {
        blur = center_color;
    }

    // Unsharp mask: center + (center - blur) * strength
    let sharpened = center_color + (center_color - blur) * params.strength;

    // Clamp to prevent negative values
    let result = max(sharpened, vec3<f32>(0.0));

    textureStore(output_color, coord, vec4<f32>(result, 1.0));
}
