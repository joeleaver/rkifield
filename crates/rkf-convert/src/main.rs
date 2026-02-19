use clap::Parser;

#[derive(Parser, Debug)]
#[command(name = "rkf-convert", about = "Convert mesh assets to .rkf format")]
struct Args {
    /// Input mesh file (glTF, GLB, OBJ)
    input: String,

    /// Output .rkf file
    #[arg(short, long)]
    output: Option<String>,

    /// Force resolution tier (0-3). Default: auto from mesh.
    #[arg(long)]
    tier: Option<u8>,

    /// Number of LOD tiers to generate (default: 3)
    #[arg(long, default_value = "3")]
    lod_levels: u32,

    /// Skip per-voxel color baking
    #[arg(long)]
    no_color: bool,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Set up logging
    env_logger::Builder::new()
        .filter_level(if args.verbose {
            log::LevelFilter::Debug
        } else {
            log::LevelFilter::Info
        })
        .init();

    // Determine output path
    let output = args.output.unwrap_or_else(|| {
        let p = std::path::Path::new(&args.input);
        p.with_extension("rkf").to_string_lossy().to_string()
    });

    log::info!("Converting {} → {}", args.input, output);

    // Load mesh
    let mesh_data = rkf_import::mesh::load_mesh(&args.input)?;
    log::info!(
        "Loaded: {} vertices, {} triangles, {} materials",
        mesh_data.positions.len(),
        mesh_data.indices.len() / 3,
        mesh_data.materials.len()
    );

    // Auto-select or use specified tier
    let tier = args.tier.map(|t| t as usize).unwrap_or_else(|| {
        rkf_import::voxelize::auto_select_tier(&mesh_data)
    });
    log::info!("Using resolution tier {tier}");

    // Voxelize
    let config = rkf_import::voxelize::VoxelizeConfig {
        tier,
        narrow_band_bricks: 3,
        compute_color: !args.no_color,
    };
    let result = rkf_import::voxelize::voxelize_mesh(&mesh_data, &config);
    log::info!("Voxelized: {} bricks", result.brick_count);

    // Generate LOD tiers
    let lod_tiers = rkf_import::lod::generate_lod_tiers(
        &result.grid,
        &result.pool,
        &result.aabb,
        tier,
        args.lod_levels.saturating_sub(1), // lod_levels includes the source tier
    );
    log::info!("Generated {} LOD tiers", lod_tiers.len());

    // Convert to chunk
    let chunk = rkf_import::voxelize::to_chunk(&result, &lod_tiers, tier, glam::IVec3::ZERO);
    log::info!(
        "Chunk: {} tiers, {} total bricks",
        chunk.grids.len(),
        chunk.brick_count
    );

    // Save .rkf
    let path = std::path::Path::new(&output);
    rkf_core::chunk::save_chunk_file(&chunk, path)?;
    log::info!("Saved {}", output);

    Ok(())
}
