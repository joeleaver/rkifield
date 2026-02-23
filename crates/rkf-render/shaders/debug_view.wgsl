// Debug visualization — reads G-buffer and writes displayable colors.
//
// Mode 0: Normal shading (basic Lambert + ambient)
// Mode 1: Surface normals (normal * 0.5 + 0.5)
// Mode 2: World positions (abs fract)
// Mode 3: Material IDs (hash-based color)

const MAX_FLOAT: f32 = 3.402823e+38;

struct DebugUniforms {
    mode: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var gbuf_position: texture_2d<f32>;
@group(0) @binding(1) var gbuf_normal:   texture_2d<f32>;
@group(0) @binding(2) var gbuf_material: texture_2d<u32>;
@group(0) @binding(3) var<uniform> debug: DebugUniforms;

@group(1) @binding(0) var output: texture_storage_2d<rgba16float, write>;

fn hash_color(id: u32) -> vec3<f32> {
    // Simple hash to generate distinct colors for material IDs.
    let r = f32((id * 2654435761u) & 0xFFu) / 255.0;
    let g = f32((id * 2246822519u) & 0xFFu) / 255.0;
    let b = f32((id * 3266489917u) & 0xFFu) / 255.0;
    return vec3<f32>(r, g, b) * 0.8 + 0.2;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) pixel: vec3<u32>) {
    let dims = textureDimensions(gbuf_position);
    if pixel.x >= dims.x || pixel.y >= dims.y {
        return;
    }

    let coord = vec2<i32>(pixel.xy);
    let pos_data = textureLoad(gbuf_position, coord, 0);
    let normal_data = textureLoad(gbuf_normal, coord, 0);
    let mat_data = textureLoad(gbuf_material, coord, 0);

    let hit_dist = pos_data.w;
    let is_hit = hit_dist < MAX_FLOAT * 0.5;

    var color: vec3<f32>;

    if !is_hit {
        // Sky gradient.
        let uv_y = f32(pixel.y) / f32(dims.y);
        color = mix(vec3<f32>(0.4, 0.6, 0.9), vec3<f32>(0.1, 0.15, 0.3), uv_y);
    } else {
        let normal = normal_data.xyz;
        let world_pos = pos_data.xyz;
        let mat_id = mat_data.r;

        switch debug.mode {
            case 0u: {
                // Basic Lambert + ambient with fixed light.
                let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
                let ndl = max(dot(normal, light_dir), 0.0);
                let base_color = hash_color(mat_id);
                color = base_color * (ndl * 0.7 + 0.3);
            }
            case 1u: {
                // Surface normals.
                color = normal * 0.5 + 0.5;
            }
            case 2u: {
                // World positions.
                color = abs(fract(world_pos));
            }
            case 3u: {
                // Material IDs.
                color = hash_color(mat_id);
            }
            default: {
                color = normal * 0.5 + 0.5;
            }
        }
    }

    textureStore(output, coord, vec4<f32>(color, 1.0));
}
