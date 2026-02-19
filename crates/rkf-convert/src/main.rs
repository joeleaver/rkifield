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

    // TODO: Phase 18.3+ — BVH, voxelize, write .rkf
    log::info!("Conversion pipeline not yet complete (Phase 18.3+)");

    Ok(())
}
