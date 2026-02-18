// Auto-exposure compute shader.
//
// Pass 1 (histogram): Builds a 256-bin luminance histogram from the HDR image.
//   Each pixel's luminance is mapped to a log2-scaled bin. Bin 0 holds
//   near-black pixels (luminance < 0.001) and is excluded from averaging.
//
// Pass 2 (average): Single 256-thread workgroup does a parallel reduction over
//   the histogram bins to compute a weighted average log luminance, derives a
//   target exposure value, and smoothly adapts the current exposure toward it.
//   Also clears the histogram for the next frame.

struct ExposureParams {
    width: u32,
    height: u32,
    min_ev: f32,          // minimum exposure value (e.g. -4.0)
    max_ev: f32,          // maximum exposure value (e.g. 16.0)
    adapt_speed: f32,     // adaptation rate per second (e.g. 2.0)
    dt: f32,              // frame delta time in seconds
    num_pixels: u32,      // total pixels for averaging (unused directly, kept for padding)
    _pad: u32,
}

const HISTOGRAM_BINS: u32 = 256u;

// ---------------------------------------------------------------------------
// Pass 1: histogram
// ---------------------------------------------------------------------------

@group(0) @binding(0) var input_hdr: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> histogram: array<atomic<u32>, 256>;
@group(0) @binding(2) var<uniform> params: ExposureParams;

// Map luminance to a histogram bin index [0, 255].
// Bin 0 is reserved for near-black pixels (luminance < 0.001).
// Bins 1..255 cover the log2 range [min_ev, max_ev].
fn luminance_to_bin(luminance: f32) -> u32 {
    if (luminance < 0.001) { return 0u; }
    let log_lum = log2(luminance);
    let normalized = (log_lum - params.min_ev) / (params.max_ev - params.min_ev);
    return u32(clamp(normalized * 254.0 + 1.0, 1.0, 255.0));
}

@compute @workgroup_size(16, 16, 1)
fn histogram_build(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }
    let color = textureLoad(input_hdr, vec2<i32>(gid.xy), 0).rgb;
    let luminance = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
    let bin = luminance_to_bin(luminance);
    atomicAdd(&histogram[bin], 1u);
}

// ---------------------------------------------------------------------------
// Pass 2: average
// ---------------------------------------------------------------------------
// NOTE: This entry point uses a *different* bind group (group 0) from pass 1.
// The histogram buffer is at binding 0, exposure_data at binding 1,
// and the uniform params at binding 2. The Rust side creates separate
// bind group layouts and bind groups for each pass.

@group(0) @binding(0) var<storage, read_write> histogram_avg: array<atomic<u32>, 256>;
@group(0) @binding(1) var<storage, read_write> exposure_data: array<f32, 2>;  // [0]=current_exposure [1]=target_exposure
@group(0) @binding(2) var<uniform> params_avg: ExposureParams;

var<workgroup> shared_sum: array<f32, 256>;
var<workgroup> shared_count: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn average(@builtin(local_invocation_index) idx: u32) {
    // Each thread processes one histogram bin.
    let count = atomicLoad(&histogram_avg[idx]);

    // Convert bin index back to log luminance (bin 0 = near-black, excluded).
    let bin_log_lum = params_avg.min_ev + (f32(idx) - 1.0) / 254.0 * (params_avg.max_ev - params_avg.min_ev);
    let weight = select(f32(count), 0.0, idx == 0u);
    shared_sum[idx] = bin_log_lum * weight;
    shared_count[idx] = weight;

    // Clear histogram bin for the next frame.
    atomicStore(&histogram_avg[idx], 0u);

    workgroupBarrier();

    // Parallel reduction — sum shared_sum and shared_count.
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (idx < stride) {
            shared_sum[idx] += shared_sum[idx + stride];
            shared_count[idx] += shared_count[idx + stride];
        }
        workgroupBarrier();
    }

    // Thread 0 computes the final exposure value.
    if (idx == 0u) {
        let total_weight = shared_count[0];
        let avg_log_lum = select(0.0, shared_sum[0] / max(total_weight, 1.0), total_weight > 0.0);

        // Target EV: key value / average luminance (key = 0.18 = middle gray).
        // In log space: target_ev = log2(0.18) - avg_log_lum
        let target_ev = log2(0.18) - avg_log_lum;
        let clamped_ev = clamp(target_ev, params_avg.min_ev, params_avg.max_ev);

        // Smooth exponential adaptation toward target.
        let current = exposure_data[0];
        let adapted = current + (clamped_ev - current) * (1.0 - exp(-params_avg.adapt_speed * params_avg.dt));

        exposure_data[0] = adapted;
        exposure_data[1] = clamped_ev;  // store target for debugging
    }
}
