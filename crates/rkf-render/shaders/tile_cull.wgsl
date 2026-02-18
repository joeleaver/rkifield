// Tile light culling compute shader — Phase 7.
//
// Divides the screen into 16×16 pixel tiles. For each tile, computes
// min/max depth from the G-buffer, then tests each light against the
// tile's depth range. Writes per-tile light index lists.
//
// Dispatch: ceil(width/16) × ceil(height/16) × 1
// Workgroup: 16×16 = 256 threads (one per pixel in the tile)

// ---------- Types ----------

struct Light {
    light_type: u32,  // 0=directional, 1=point, 2=spot
    pos_x: f32,
    pos_y: f32,
    pos_z: f32,
    dir_x: f32,
    dir_y: f32,
    dir_z: f32,
    color_r: f32,
    color_g: f32,
    color_b: f32,
    intensity: f32,
    range: f32,
    inner_angle: f32,
    outer_angle: f32,
    cookie_index: i32,
    shadow_caster: u32,
}

struct CullUniforms {
    num_tiles_x: u32,
    num_tiles_y: u32,
    num_lights: u32,
    screen_width: u32,
    screen_height: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    // Camera data for depth-based culling
    camera_pos: vec4<f32>,
    camera_forward: vec4<f32>,
}

// ---------- Constants ----------

const MAX_LIGHTS_PER_TILE: u32 = 64u;
const MAX_FLOAT: f32 = 3.402823e+38;
const LIGHT_TYPE_DIRECTIONAL: u32 = 0u;

// ---------- Bindings ----------

// Group 0: G-buffer position texture (for depth)
@group(0) @binding(0) var gbuf_position: texture_2d<f32>;

// Group 1: Lights + uniforms
@group(1) @binding(0) var<storage, read> lights: array<Light>;
@group(1) @binding(1) var<uniform> cull: CullUniforms;

// Group 2: Output tile data
@group(2) @binding(0) var<storage, read_write> tile_light_indices: array<u32>;
@group(2) @binding(1) var<storage, read_write> tile_light_counts: array<atomic<u32>>;

// ---------- Shared memory ----------

var<workgroup> tile_min_depth: atomic<u32>;
var<workgroup> tile_max_depth: atomic<u32>;
var<workgroup> tile_count: atomic<u32>;
var<workgroup> tile_lights: array<u32, 64>; // MAX_LIGHTS_PER_TILE

// ---------- Entry point ----------

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let thread_idx = lid.y * 16u + lid.x;

    // Initialize shared memory (thread 0 only)
    if thread_idx == 0u {
        atomicStore(&tile_min_depth, 0x7F7FFFFFu); // Large positive float as u32
        atomicStore(&tile_max_depth, 0u);
        atomicStore(&tile_count, 0u);
    }
    workgroupBarrier();

    // Phase 1: Each thread reads one pixel's depth, compute tile min/max
    let pixel = vec2<i32>(vec2<u32>(wid.x * 16u + lid.x, wid.y * 16u + lid.y));
    let screen = vec2<i32>(vec2<u32>(cull.screen_width, cull.screen_height));

    if pixel.x < screen.x && pixel.y < screen.y {
        let depth = textureLoad(gbuf_position, pixel, 0).w;

        // Skip sky pixels (no geometry)
        if depth < MAX_FLOAT * 0.5 && depth > 0.0 {
            // IEEE 754 positive floats maintain ordering as u32
            let depth_bits = bitcast<u32>(depth);
            atomicMin(&tile_min_depth, depth_bits);
            atomicMax(&tile_max_depth, depth_bits);
        }
    }
    workgroupBarrier();

    // Read final min/max depth
    let min_depth_bits = atomicLoad(&tile_min_depth);
    let max_depth_bits = atomicLoad(&tile_max_depth);
    let min_depth = bitcast<f32>(min_depth_bits);
    let max_depth = bitcast<f32>(max_depth_bits);

    // If no geometry in this tile, skip light testing
    let has_geometry = max_depth_bits > 0u;

    // Phase 2: Test lights against tile depth range (parallel across threads)
    // 256 threads, each tests different lights in strided fashion
    if has_geometry {
        let cam_pos = cull.camera_pos.xyz;
        let cam_fwd = cull.camera_forward.xyz;

        for (var i = thread_idx; i < cull.num_lights; i += 256u) {
            let light = lights[i];
            var include = false;

            if light.light_type == LIGHT_TYPE_DIRECTIONAL {
                // Directional lights always affect all tiles
                include = true;
            } else {
                // Point/Spot: project light position onto camera forward axis
                let light_pos = vec3<f32>(light.pos_x, light.pos_y, light.pos_z);
                let to_light = light_pos - cam_pos;
                let light_depth = dot(to_light, cam_fwd);

                // Conservative depth-range test: include if light sphere
                // overlaps the tile's depth range
                include = (light_depth + light.range >= min_depth) &&
                          (light_depth - light.range <= max_depth);
            }

            if include {
                let idx = atomicAdd(&tile_count, 1u);
                if idx < MAX_LIGHTS_PER_TILE {
                    tile_lights[idx] = i;
                }
            }
        }
    }
    workgroupBarrier();

    // Phase 3: Write tile data to global memory
    let tile_id = wid.y * cull.num_tiles_x + wid.x;
    let count = min(atomicLoad(&tile_count), MAX_LIGHTS_PER_TILE);
    let base = tile_id * MAX_LIGHTS_PER_TILE;

    // Cooperative write: each thread writes one (or zero) entries
    if thread_idx < count {
        tile_light_indices[base + thread_idx] = tile_lights[thread_idx];
    }

    // Thread 0 writes the count
    if thread_idx == 0u {
        atomicStore(&tile_light_counts[tile_id], count);
    }
}
