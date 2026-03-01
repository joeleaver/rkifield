//! Generate a test .rkf file with a complex voxelized shape.
//!
//! Usage: cargo run -p rkf-core --example gen_test_rkf
//!
//! Produces `scenes/test_cross.rkf` — a smooth union of 5 spheres in a cross
//! pattern. The shape is complex enough to span many bricks and test cross-brick
//! interpolation, concave joints, and varying curvature, but predictable enough
//! to immediately verify visual correctness.

use std::fs::File;
use std::io::BufWriter;

use glam::Vec3;

use rkf_core::aabb::Aabb;
use rkf_core::asset_file::{save_object, SaveLodLevel};
use rkf_core::brick::Brick;
use rkf_core::brick_map::{BrickMap, BrickMapAllocator, EMPTY_SLOT};
use rkf_core::brick_pool::Pool;
use rkf_core::voxelize_object::voxelize_sdf;

fn smooth_min(a: f32, b: f32, k: f32) -> f32 {
    let h = (k - (a - b).abs()).max(0.0) / k.max(1e-6);
    a.min(b) - h * h * k * 0.25
}

fn main() {
    let voxel_size = 0.03_f32; // Fine resolution
    let smooth_k = 0.15_f32; // Smooth union radius

    // Sphere centers and radii: center + 4 arms
    let spheres: [(Vec3, f32); 5] = [
        (Vec3::new(0.0, 0.0, 0.0), 0.35),  // center
        (Vec3::new(0.55, 0.0, 0.0), 0.25),  // +X arm
        (Vec3::new(-0.55, 0.0, 0.0), 0.25), // -X arm
        (Vec3::new(0.0, 0.0, 0.55), 0.25),  // +Z arm
        (Vec3::new(0.0, 0.0, -0.55), 0.25), // -Z arm
    ];

    // Compute AABB from sphere extents + margin
    let margin = voxel_size * 4.0;
    let mut aabb_min = Vec3::splat(f32::MAX);
    let mut aabb_max = Vec3::splat(f32::MIN);
    for &(center, radius) in &spheres {
        aabb_min = aabb_min.min(center - Vec3::splat(radius + margin));
        aabb_max = aabb_max.max(center + Vec3::splat(radius + margin));
    }
    let aabb = Aabb::new(aabb_min, aabb_max);

    // SDF: smooth union of all spheres
    let sdf_fn = |pos: Vec3| -> (f32, u16) {
        let mut d = (pos - spheres[0].0).length() - spheres[0].1;
        for i in 1..5 {
            let di = (pos - spheres[i].0).length() - spheres[i].1;
            d = smooth_min(d, di, smooth_k);
        }
        (d, 6u16)
    };

    // Voxelize
    let mut pool: Pool<Brick> = Pool::new(8192);
    let mut alloc = BrickMapAllocator::new();

    let (handle, brick_count) =
        voxelize_sdf(sdf_fn, &aabb, voxel_size, &mut pool, &mut alloc)
            .expect("voxelize failed — pool too small?");

    println!(
        "Voxelized cross: {} bricks, dims={:?}",
        brick_count, handle.dims
    );

    // Collect brick data for saving
    let dims = handle.dims;
    let mut brick_map = BrickMap::new(dims);
    let mut brick_data: Vec<[rkf_core::voxel::VoxelSample; 512]> = Vec::new();

    for bz in 0..dims.z {
        for by in 0..dims.y {
            for bx in 0..dims.x {
                let slot = alloc.get_entry(&handle, bx, by, bz);
                match slot {
                    Some(s) if s != EMPTY_SLOT => {
                        // Map to sequential local index
                        let local_idx = brick_data.len() as u32;
                        brick_map.set(bx, by, bz, local_idx);

                        // Copy voxel data
                        let brick = pool.get(s);
                        let mut samples = [rkf_core::voxel::VoxelSample::default(); 512];
                        for i in 0..512 {
                            samples[i] = brick.voxels[i];
                        }
                        brick_data.push(samples);
                    }
                    _ => {
                        // Leave as EMPTY_SLOT (default in BrickMap::new)
                    }
                }
            }
        }
    }

    println!("Brick data: {} bricks collected", brick_data.len());

    // Compute grid-aligned AABB (matches GPU shader expectations)
    let brick_world_size = voxel_size * 8.0;
    let grid_half = Vec3::new(
        dims.x as f32 * brick_world_size * 0.5,
        dims.y as f32 * brick_world_size * 0.5,
        dims.z as f32 * brick_world_size * 0.5,
    );
    let grid_aabb = Aabb::new(-grid_half, grid_half);

    // Save as .rkf
    let lod = SaveLodLevel {
        voxel_size,
        brick_map,
        brick_data,
    };

    let out_path = "scenes/test_cross.rkf";
    let file = File::create(out_path).expect("failed to create output file");
    let mut writer = BufWriter::new(file);
    save_object(&mut writer, &grid_aabb, None, &[6], &[lod])
        .expect("failed to save .rkf");

    println!("Wrote {out_path}");
    println!(
        "Grid AABB: [{:.3}, {:.3}, {:.3}] to [{:.3}, {:.3}, {:.3}]",
        grid_aabb.min.x, grid_aabb.min.y, grid_aabb.min.z,
        grid_aabb.max.x, grid_aabb.max.y, grid_aabb.max.z,
    );
}
