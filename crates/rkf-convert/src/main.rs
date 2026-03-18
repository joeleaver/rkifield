//! rkf-convert — offline mesh-to-.rkf v3 conversion CLI.
//!
//! Converts glTF/GLB/OBJ mesh files to .rkf v3 format (geometry-first) with
//! per-object brick maps, multi-LOD voxelization, per-voxel material transfer,
//! and optional per-voxel color from mesh textures.
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
//!   --material-id <int>     Override material ID for all voxels (default: use mesh materials)
//!   --pool-size <int>       Max brick count per LOD (default: 65536)
//!   -v, --verbose           Print progress info
//!   -h, --help              Print help
//! ```

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use anyhow::{Context, Result, bail};
use glam::Vec3;

use rkf_core::aabb::Aabb;
use rkf_core::asset_file_v3::{SaveLodV3, save_object_v3};
use rkf_core::brick_geometry::{BrickGeometry, voxel_index};
use rkf_core::brick_map::BrickMap;
use rkf_core::companion::{ColorBrick, ColorVoxel};
use rkf_core::constants::BRICK_DIM;
use rkf_core::scene_node::SdfPrimitive;
use rkf_core::sdf_cache::SdfCache;
use rkf_core::sdf_compute::{SlotMapping, compute_sdf_from_geometry};
use rkf_import::bvh::TriangleBvh;
use rkf_import::material_transfer::sample_material;
use rkf_import::mesh::{MeshData, load_mesh};

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

struct Args {
    input: String,
    output: String,
    voxel_size: Option<f32>,
    lod_levels: usize,
    material_id_override: Option<u8>,
    pool_size: u32,
    verbose: bool,
}

fn print_help() {
    eprintln!("rkf-convert v3 — mesh to .rkf converter (geometry-first)");
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
    eprintln!("  --material-id <int>       Override material ID for all voxels (default: use mesh materials)");
    eprintln!("  --pool-size <int>         Max bricks per LOD (default: 65536)");
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
    let mut material_id_override: Option<u8> = None;
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
                let id = v
                    .parse::<u8>()
                    .with_context(|| format!("invalid --material-id: '{v}'"))?;
                if id > 63 {
                    bail!("--material-id must be 0-63 (6-bit)");
                }
                material_id_override = Some(id);
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
        material_id_override,
        pool_size,
        verbose,
    })
}

// ---------------------------------------------------------------------------
// Voxel size auto-detection
// ---------------------------------------------------------------------------

/// Standard voxel size tiers — must match the editor's VOXEL_TIERS.
const VOXEL_TIERS: [f32; 4] = [0.005, 0.02, 0.08, 0.32];

/// Pick the best voxel tier for a mesh based on its bounding box.
///
/// Uses the same heuristic as the editor: pick the coarsest tier that
/// gives at least 8 bricks on the longest axis.
fn auto_voxel_size(mesh: &MeshData) -> f32 {
    let extent = mesh.bounds_max - mesh.bounds_min;
    let longest = extent.max_element();
    if longest < 1e-6 {
        return VOXEL_TIERS[0]; // degenerate mesh fallback
    }
    // Pick coarsest tier with >= 8 bricks on longest axis (same as editor)
    for &vs in VOXEL_TIERS.iter().rev() {
        let bricks = longest / (vs * BRICK_DIM as f32);
        if bricks >= 8.0 {
            return vs;
        }
    }
    VOXEL_TIERS[0] // finest if the object is tiny
}

// ---------------------------------------------------------------------------
// SDF sign evaluation — winding number
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
// Per-LOD voxelization (geometry-first)
// ---------------------------------------------------------------------------

/// Voxelize the mesh into a `SaveLodV3` at the given voxel resolution.
///
/// Produces BrickGeometry (occupancy + surface voxels with material),
/// SdfCache (computed from geometry), and ColorBricks (from mesh textures).
fn voxelize_to_lod(
    mesh: &MeshData,
    bvh: &TriangleBvh,
    aabb: &Aabb,
    voxel_size: f32,
    material_id_override: Option<u8>,
    pool_size: u32,
    has_textures: bool,
    verbose: bool,
) -> Result<SaveLodV3> {
    let brick_world_size = voxel_size * BRICK_DIM as f32;

    // Compute brick grid dimensions from the (padded) AABB.
    let aabb_size = aabb.max - aabb.min;
    let dims = glam::UVec3::new(
        ((aabb_size.x / brick_world_size).ceil() as u32).max(1),
        ((aabb_size.y / brick_world_size).ceil() as u32).max(1),
        ((aabb_size.z / brick_world_size).ceil() as u32).max(1),
    );

    // Narrow-band threshold
    let narrow_band = brick_world_size * 1.8;

    let mut brick_map = BrickMap::new(dims);
    let mut geometry_vec: Vec<BrickGeometry> = Vec::new();
    let mut color_vec: Vec<ColorBrick> = Vec::new();
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
                    continue;
                }

                if next_slot >= pool_size {
                    bail!(
                        "Brick count exceeded pool_size={pool_size} at ({bx},{by},{bz}). \
                         Use --pool-size to increase the limit."
                    );
                }

                let slot = next_slot;
                next_slot += 1;
                brick_map.set(bx, by, bz, slot);

                let mut geo = BrickGeometry::new();
                let mut color_brick = ColorBrick { data: [ColorVoxel::new(0, 0, 0, 0); 512] };
                let half_voxel = voxel_size * 0.5;

                // Single pass: determine occupancy + sample color for every voxel.
                // Color is sampled densely (all 512 voxels) so the GPU can
                // sample texture at any ray hit point, not just surface voxels.
                let sample_color = has_textures && material_id_override.is_none();

                for vz in 0..BRICK_DIM as u8 {
                    for vy in 0..BRICK_DIM as u8 {
                        for vx in 0..BRICK_DIM as u8 {
                            let pos = brick_min
                                + Vec3::new(
                                    vx as f32 * voxel_size + half_voxel,
                                    vy as f32 * voxel_size + half_voxel,
                                    vz as f32 * voxel_size + half_voxel,
                                );

                            let winding = winding_number(mesh, pos);
                            let is_inside = winding < -0.5;
                            geo.set_solid(vx, vy, vz, is_inside);

                            // Sample texture color at every voxel position
                            if sample_color {
                                let mat_sample = sample_material(mesh, bvh, pos);
                                if let Some(c) = mat_sample.color {
                                    let flat = voxel_index(vx, vy, vz);
                                    color_brick.data[flat as usize] =
                                        ColorVoxel::new(c.r, c.g, c.b, 255);
                                }
                            }
                        }
                    }
                }

                // Build surface voxel list and assign material IDs
                geo.rebuild_surface_list();

                for sv in &mut geo.surface_voxels {
                    let idx = sv.index();
                    let (vx, vy, vz) = rkf_core::brick_geometry::index_to_xyz(idx);

                    if let Some(override_id) = material_id_override {
                        sv.material_id = override_id;
                    } else {
                        let pos = brick_min
                            + Vec3::new(
                                vx as f32 * voxel_size + half_voxel,
                                vy as f32 * voxel_size + half_voxel,
                                vz as f32 * voxel_size + half_voxel,
                            );
                        let mat_sample = sample_material(mesh, bvh, pos);
                        sv.material_id = (mat_sample.material_id as u8).min(63);
                    }
                }

                geometry_vec.push(geo);
                color_vec.push(color_brick);
            }
        }
    }

    // Compute SDF from geometry using Fast Sweeping Method
    let brick_count = geometry_vec.len();
    let mut sdf_caches: Vec<SdfCache> = vec![SdfCache::empty(); brick_count];
    let slot_mappings: Vec<SlotMapping> = (0..brick_count as u32)
        .map(|i| SlotMapping {
            brick_slot: i,
            geometry_slot: i,
            sdf_slot: i,
        })
        .collect();

    compute_sdf_from_geometry(
        &brick_map,
        &geometry_vec,
        &mut sdf_caches,
        &slot_mappings,
        voxel_size,
    );

    if verbose {
        eprintln!(
            "    dims={}x{}x{}  bricks={}",
            dims.x, dims.y, dims.z, brick_count
        );
    }

    // Only include color bricks if we have textures and no material override
    let color_bricks = if has_textures && material_id_override.is_none() {
        Some(color_vec)
    } else {
        None
    };

    Ok(SaveLodV3 {
        voxel_size,
        brick_map,
        geometry: geometry_vec,
        sdf_cache: Some(sdf_caches),
        color_bricks,
    })
}

// ---------------------------------------------------------------------------
// LOD generation
// ---------------------------------------------------------------------------

/// Generate all LOD levels for a mesh.
fn generate_lods(
    mesh: &MeshData,
    bvh: &TriangleBvh,
    finest_voxel_size: f32,
    lod_levels: usize,
    material_id_override: Option<u8>,
    pool_size: u32,
    has_textures: bool,
    verbose: bool,
) -> Result<Vec<SaveLodV3>> {
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

        let lod = voxelize_to_lod(
            mesh,
            bvh,
            &aabb,
            voxel_size,
            material_id_override,
            pool_size,
            has_textures,
            verbose,
        )
        .with_context(|| format!("LOD {level} (voxel_size={voxel_size:.4}m) failed"))?;

        if !verbose {
            eprintln!(
                "  LOD {}: voxel_size={:.4}m  bricks={}",
                level,
                lod.voxel_size,
                lod.geometry.len()
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
    eprintln!("rkf-convert v3 — mesh to .rkf converter (geometry-first)");
    eprintln!();

    let args = parse_args()?;

    // --- Step 1: Load mesh ---------------------------------------------------

    eprintln!("Loading mesh: {}", args.input);
    let mesh = load_mesh(&args.input)
        .with_context(|| format!("Failed to load '{}'", args.input))?;

    let has_textures = mesh.materials.iter().any(|m| m.albedo_texture.is_some());

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
    eprintln!(
        "  materials: {}  textures: {}",
        mesh.materials.len(),
        if has_textures { "yes" } else { "no" }
    );
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
        args.material_id_override,
        args.pool_size,
        has_textures,
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

    // Collect material IDs: either the override or all mesh material indices
    let material_ids: Vec<u8> = if let Some(id) = args.material_id_override {
        vec![id]
    } else {
        (0..mesh.materials.len().min(64) as u8).collect()
    };

    // --- Step 6: Save to .rkf v3 ---------------------------------------------

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

    save_object_v3(
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
    eprintln!("Format:    .rkf v3 (geometry-first)");
    eprintln!("LODs:      {}", lods.len());
    for (i, lod) in lods.iter().enumerate() {
        eprintln!(
            "  LOD {}: voxel_size={:.4}m  bricks={}  map_dims={}x{}x{}",
            i,
            lod.voxel_size,
            lod.geometry.len(),
            lod.brick_map.dims.x,
            lod.brick_map.dims.y,
            lod.brick_map.dims.z,
        );
    }
    if has_textures && args.material_id_override.is_none() {
        eprintln!("Color:     per-voxel color from mesh textures");
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
