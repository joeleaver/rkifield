//! rkf-convert — offline mesh-to-.rkf v2 conversion CLI.
//!
//! Converts glTF/GLB/OBJ mesh files to .rkf v2 format with per-object brick maps
//! and multi-LOD voxelization. Uses BVH-accelerated nearest-triangle lookup for
//! unsigned distance and winding-number-based sign determination.
//!
//! # Usage
//!
//! ```text
//! rkf-convert <input.glb|.gltf|.obj> -o <output.rkf> [options]
//!
//! Options:
//!   -o, --output <path>     Output .rkf file path (required)
//!   --voxel-size <float>    Finest voxel size (default: auto)
//!   --lod-levels <int>      Number of LOD levels (default: 3)
//!   --material-id <int>     Material ID for voxels (default: 1)
//!   --pool-size <int>       Brick pool capacity (default: 65536)
//!   -v, --verbose           Print progress info
//!   -h, --help              Print help
//! ```

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use anyhow::{Context, Result, bail};
use glam::Vec3;

use rkf_core::aabb::Aabb;
use rkf_core::asset_file::{SaveLodLevel, save_object};
use rkf_core::brick_map::BrickMap;
use rkf_core::constants::BRICK_DIM;
use rkf_core::scene_node::SdfPrimitive;
use rkf_core::voxel::VoxelSample;
use rkf_import::bvh::TriangleBvh;
use rkf_import::mesh::{MeshData, load_mesh};

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

struct Args {
    input: String,
    output: String,
    voxel_size: Option<f32>,
    lod_levels: usize,
    material_id: u16,
    pool_size: u32,
    verbose: bool,
}

fn print_help() {
    eprintln!("rkf-convert v2 — mesh to .rkf converter");
    eprintln!();
    eprintln!("USAGE:");
    eprintln!("  rkf-convert <input> -o <output.rkf> [options]");
    eprintln!();
    eprintln!("ARGS:");
    eprintln!("  <input>                   Input mesh file (.glb, .gltf, .obj)");
    eprintln!();
    eprintln!("OPTIONS:");
    eprintln!("  -o, --output <path>       Output .rkf file path (required)");
    eprintln!("  --voxel-size <float>      Finest voxel size in metres (default: auto)");
    eprintln!("  --lod-levels <int>        Number of LOD levels (default: 3)");
    eprintln!("  --material-id <int>       Material ID for voxels (default: 1)");
    eprintln!("  --pool-size <int>         Brick pool capacity (default: 65536)");
    eprintln!("  -v, --verbose             Print per-LOD progress");
    eprintln!("  -h, --help                Print this help message");
}

fn parse_args() -> Result<Args> {
    let raw: Vec<String> = std::env::args().collect();
    let args: Vec<&str> = raw.iter().map(|s| s.as_str()).collect();

    if args.len() < 2 {
        print_help();
        bail!("No input file specified");
    }

    if args[1] == "-h" || args[1] == "--help" {
        print_help();
        std::process::exit(0);
    }

    let mut input: Option<String> = None;
    let mut output: Option<String> = None;
    let mut voxel_size: Option<f32> = None;
    let mut lod_levels: usize = 3;
    let mut material_id: u16 = 1;
    let mut pool_size: u32 = 65536;
    let mut verbose = false;

    let mut i = 1usize;
    while i < args.len() {
        match args[i] {
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            }
            "-v" | "--verbose" => {
                verbose = true;
                i += 1;
            }
            "-o" | "--output" => {
                i += 1;
                let v = args
                    .get(i)
                    .with_context(|| format!("'{}' requires a value", args[i - 1]))?;
                output = Some(v.to_string());
                i += 1;
            }
            "--voxel-size" => {
                i += 1;
                let v = args
                    .get(i)
                    .with_context(|| "--voxel-size requires a value")?;
                voxel_size = Some(
                    v.parse::<f32>()
                        .with_context(|| format!("invalid --voxel-size: '{v}'"))?,
                );
                i += 1;
            }
            "--lod-levels" => {
                i += 1;
                let v = args
                    .get(i)
                    .with_context(|| "--lod-levels requires a value")?;
                lod_levels = v
                    .parse::<usize>()
                    .with_context(|| format!("invalid --lod-levels: '{v}'"))?;
                if lod_levels == 0 {
                    bail!("--lod-levels must be at least 1");
                }
                i += 1;
            }
            "--material-id" => {
                i += 1;
                let v = args
                    .get(i)
                    .with_context(|| "--material-id requires a value")?;
                material_id = v
                    .parse::<u16>()
                    .with_context(|| format!("invalid --material-id: '{v}'"))?;
                i += 1;
            }
            "--pool-size" => {
                i += 1;
                let v = args
                    .get(i)
                    .with_context(|| "--pool-size requires a value")?;
                pool_size = v
                    .parse::<u32>()
                    .with_context(|| format!("invalid --pool-size: '{v}'"))?;
                i += 1;
            }
            arg => {
                // Positional argument: input path
                if input.is_some() {
                    bail!("Unexpected argument: '{arg}'");
                }
                input = Some(arg.to_string());
                i += 1;
            }
        }
    }

    let input = input.with_context(|| "No input file specified")?;
    let output = output.with_context(|| "Output path required — use -o <path>")?;

    Ok(Args {
        input,
        output,
        voxel_size,
        lod_levels,
        material_id,
        pool_size,
        verbose,
    })
}

// ---------------------------------------------------------------------------
// Voxel size auto-detection
// ---------------------------------------------------------------------------

/// Estimate a sensible finest voxel size from the mesh bounding box.
///
/// Targets approximately 128 brick-widths (1024 voxels) along the longest
/// axis of the mesh, clamped to [0.001, 0.1] metres.
fn auto_voxel_size(mesh: &MeshData) -> f32 {
    let extent = mesh.bounds_max - mesh.bounds_min;
    let longest = extent.max_element();
    if longest < 1e-6 {
        return 0.02; // degenerate mesh fallback
    }
    // 1024 voxels along longest axis
    let vs = longest / 1024.0;
    vs.clamp(0.001, 0.1)
}

// ---------------------------------------------------------------------------
// SDF evaluation — BVH nearest + winding number sign
// ---------------------------------------------------------------------------

/// Estimate inside/outside sign using generalized winding number (angle-weighted).
///
/// For each triangle, computes the solid angle it subtends at `point` and
/// accumulates the signed winding number. Positive winding → inside.
///
/// This is an O(N) approximation sufficient for offline conversion.
fn winding_number(mesh: &MeshData, point: Vec3) -> f32 {
    let mut winding = 0.0f32;
    for i in 0..mesh.triangle_count() {
        let [a, b, c] = mesh.triangle_positions(i);
        let a = a - point;
        let b = b - point;
        let c = c - point;
        let la = a.length();
        let lb = b.length();
        let lc = c.length();
        if la < 1e-10 || lb < 1e-10 || lc < 1e-10 {
            continue;
        }
        let na = a / la;
        let nb = b / lb;
        let nc = c / lc;
        // Solid angle of triangle: 2 * atan2(|a·(b×c)|, 1 + a·b + b·c + a·c)
        let num = na.dot(nb.cross(nc));
        let den = 1.0 + na.dot(nb) + nb.dot(nc) + na.dot(nc);
        winding += 2.0 * num.atan2(den);
    }
    winding / (4.0 * std::f32::consts::PI)
}

// ---------------------------------------------------------------------------
// Per-LOD voxelization
// ---------------------------------------------------------------------------

/// Voxelize the mesh SDF into a `SaveLodLevel` at the given voxel resolution.
///
/// Performs a narrow-band optimization: bricks whose center is farther than
/// `brick_world_size * 1.8` from the nearest surface triangle are skipped.
///
/// Slot indices in the returned `BrickMap` are sequential local indices
/// (0..brick_count-1) matching the order of `brick_data` entries, which is
/// exactly what `save_object` expects.
fn voxelize_to_lod(
    mesh: &MeshData,
    bvh: &TriangleBvh,
    aabb: &Aabb,
    voxel_size: f32,
    material_id: u16,
    pool_size: u32,
    verbose: bool,
) -> Result<SaveLodLevel> {
    let brick_world_size = voxel_size * BRICK_DIM as f32;

    // Compute brick grid dimensions from the (padded) AABB.
    let aabb_size = aabb.max - aabb.min;
    let dims = glam::UVec3::new(
        ((aabb_size.x / brick_world_size).ceil() as u32).max(1),
        ((aabb_size.y / brick_world_size).ceil() as u32).max(1),
        ((aabb_size.z / brick_world_size).ceil() as u32).max(1),
    );

    // Narrow-band threshold: matches voxelize_sdf's margin.
    let narrow_band = brick_world_size * 1.8;

    let mut brick_map = BrickMap::new(dims);
    let mut brick_data: Vec<[VoxelSample; 512]> = Vec::new();
    let mut next_slot: u32 = 0;

    for bz in 0..dims.z {
        for by in 0..dims.y {
            for bx in 0..dims.x {
                let brick_min = aabb.min
                    + Vec3::new(
                        bx as f32 * brick_world_size,
                        by as f32 * brick_world_size,
                        bz as f32 * brick_world_size,
                    );
                let brick_center = brick_min + Vec3::splat(brick_world_size * 0.5);

                // Narrow-band cull: sample unsigned distance at brick center.
                let center_nearest = bvh.nearest(brick_center);
                if center_nearest.distance >= narrow_band {
                    // Entirely outside the surface band — skip.
                    continue;
                }

                // Guard against exceeding the pool size hint.
                if next_slot >= pool_size {
                    bail!(
                        "Brick count exceeded pool_size={pool_size} at ({bx},{by},{bz}). \
                         Use --pool-size to increase the limit."
                    );
                }

                // Assign a sequential local slot and populate brick voxels.
                let slot = next_slot;
                next_slot += 1;
                brick_map.set(bx, by, bz, slot);

                let mut samples = [VoxelSample::default(); 512];
                let half_voxel = voxel_size * 0.5;

                let mut idx = 0;
                for vz in 0..BRICK_DIM {
                    for vy in 0..BRICK_DIM {
                        for vx in 0..BRICK_DIM {
                            let pos = brick_min
                                + Vec3::new(
                                    vx as f32 * voxel_size + half_voxel,
                                    vy as f32 * voxel_size + half_voxel,
                                    vz as f32 * voxel_size + half_voxel,
                                );

                            // Unsigned distance via BVH nearest-triangle.
                            let nearest = bvh.nearest(pos);
                            let unsigned_dist = nearest.distance;

                            // Sign via winding number.
                            let winding = winding_number(mesh, pos);
                            let sign = if winding < -0.5 { -1.0f32 } else { 1.0f32 };
                            let signed_dist = sign * unsigned_dist;

                            samples[idx] = VoxelSample::new(signed_dist, material_id, [255, 255, 255, 255]);
                            idx += 1;
                        }
                    }
                }

                brick_data.push(samples);
            }
        }
    }

    if verbose {
        eprintln!(
            "    dims={}x{}x{}  bricks={}",
            dims.x, dims.y, dims.z, brick_data.len()
        );
    }

    Ok(SaveLodLevel {
        voxel_size,
        brick_map,
        brick_data,
    })
}

// ---------------------------------------------------------------------------
// LOD generation
// ---------------------------------------------------------------------------

/// Generate all LOD levels for a mesh.
///
/// Returns a `Vec<SaveLodLevel>` ordered finest-to-coarsest (level 0 = finest).
/// `save_object` sorts them coarsest-first for on-disk layout.
fn generate_lods(
    mesh: &MeshData,
    bvh: &TriangleBvh,
    finest_voxel_size: f32,
    lod_levels: usize,
    material_id: u16,
    pool_size: u32,
    verbose: bool,
) -> Result<Vec<SaveLodLevel>> {
    // Build a slightly padded AABB so surface bricks at mesh boundaries are captured.
    let margin = finest_voxel_size * 2.0;
    let aabb = Aabb::new(
        mesh.bounds_min - Vec3::splat(margin),
        mesh.bounds_max + Vec3::splat(margin),
    );

    let mut lods = Vec::with_capacity(lod_levels);

    for level in 0..lod_levels {
        // Each coarser level doubles the voxel size.
        let voxel_size = finest_voxel_size * (1u32 << level) as f32;

        if verbose {
            eprintln!("  LOD {}: voxel_size={:.4}m ...", level, voxel_size);
        }

        let lod =
            voxelize_to_lod(mesh, bvh, &aabb, voxel_size, material_id, pool_size, verbose)
                .with_context(|| format!("LOD {level} (voxel_size={voxel_size:.4}m) failed"))?;

        if !verbose {
            eprintln!(
                "  LOD {}: voxel_size={:.4}m  bricks={}",
                level,
                lod.voxel_size,
                lod.brick_data.len()
            );
        }

        lods.push(lod);
    }

    Ok(lods)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn run() -> Result<()> {
    eprintln!("rkf-convert v2 — mesh to .rkf converter");
    eprintln!();

    let args = parse_args()?;

    // --- Step 1: Load mesh ---------------------------------------------------

    eprintln!("Loading mesh: {}", args.input);
    let mesh = load_mesh(&args.input)
        .with_context(|| format!("Failed to load '{}'", args.input))?;

    eprintln!(
        "  vertices: {}  triangles: {}",
        mesh.positions.len(),
        mesh.triangle_count()
    );
    eprintln!(
        "  bounds:   ({:.3}, {:.3}, {:.3}) .. ({:.3}, {:.3}, {:.3})",
        mesh.bounds_min.x,
        mesh.bounds_min.y,
        mesh.bounds_min.z,
        mesh.bounds_max.x,
        mesh.bounds_max.y,
        mesh.bounds_max.z
    );
    eprintln!("  materials: {}", mesh.materials.len());
    eprintln!();

    if mesh.triangle_count() == 0 {
        bail!("Input mesh has no triangles");
    }

    // --- Step 2: Determine voxel size ----------------------------------------

    let finest_voxel_size = match args.voxel_size {
        Some(vs) => {
            if vs <= 0.0 {
                bail!("--voxel-size must be positive, got {vs}");
            }
            vs
        }
        None => {
            let auto = auto_voxel_size(&mesh);
            eprintln!("Auto voxel size: {:.4}m", auto);
            auto
        }
    };
    eprintln!("Finest voxel size: {:.4}m", finest_voxel_size);
    eprintln!("LOD levels: {}", args.lod_levels);
    eprintln!();

    // --- Step 3: Build BVH ---------------------------------------------------

    eprintln!("Building BVH over {} triangles...", mesh.triangle_count());
    let bvh = TriangleBvh::build(&mesh);
    eprintln!("  done");
    eprintln!();

    // --- Step 4: Generate LODs -----------------------------------------------

    eprintln!("Generating {} LOD level(s)...", args.lod_levels);
    let lods = generate_lods(
        &mesh,
        &bvh,
        finest_voxel_size,
        args.lod_levels,
        args.material_id,
        args.pool_size,
        args.verbose,
    )
    .context("LOD generation failed")?;
    eprintln!();

    // --- Step 5: Build file header metadata ----------------------------------

    let aabb = Aabb::new(mesh.bounds_min, mesh.bounds_max);

    // Analytical bound: bounding sphere (radius = half-diagonal of the AABB)
    let center = aabb.center();
    let bounding_radius = (aabb.max - center).length();
    let analytical_bound = SdfPrimitive::Sphere {
        radius: bounding_radius,
    };

    let material_ids: Vec<u16> = vec![args.material_id];

    // --- Step 6: Save to .rkf v2 ---------------------------------------------

    eprintln!("Writing output: {}", args.output);

    let output_path = Path::new(&args.output);
    if let Some(parent) = output_path.parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() {
            std::fs::create_dir_all(parent).with_context(|| {
                format!("Failed to create output directory '{}'", parent.display())
            })?;
        }
    }

    let file = File::create(output_path)
        .with_context(|| format!("Failed to create output file '{}'", args.output))?;
    let mut writer = BufWriter::new(file);

    save_object(
        &mut writer,
        &aabb,
        Some(&analytical_bound),
        &material_ids,
        &lods,
    )
    .with_context(|| format!("Failed to write .rkf file '{}'", args.output))?;

    drop(writer);

    // --- Step 7: Print summary -----------------------------------------------

    let file_size = std::fs::metadata(&args.output)
        .map(|m| m.len())
        .unwrap_or(0);

    eprintln!();
    eprintln!("=== Conversion complete ===");
    eprintln!("Output:    {}", args.output);
    eprintln!("File size: {}", format_bytes(file_size));
    eprintln!("LODs:      {}", lods.len());
    for (i, lod) in lods.iter().enumerate() {
        eprintln!(
            "  LOD {}: voxel_size={:.4}m  bricks={}  map_dims={}x{}x{}",
            i,
            lod.voxel_size,
            lod.brick_data.len(),
            lod.brick_map.dims.x,
            lod.brick_map.dims.y,
            lod.brick_map.dims.z,
        );
    }

    Ok(())
}

/// Format a byte count as a human-readable string.
fn format_bytes(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KiB", bytes as f64 / 1024.0)
    } else if bytes < 1024 * 1024 * 1024 {
        format!("{:.1} MiB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{:.1} GiB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}
