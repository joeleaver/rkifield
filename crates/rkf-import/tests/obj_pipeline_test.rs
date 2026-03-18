//! End-to-end test: OBJ → voxelize → .rkf v3 → reload.
//!
//! Creates a minimal OBJ cube with a textured material, runs the full
//! geometry-first pipeline, saves to .rkf v3, and verifies the roundtrip.

use std::io::Cursor;

use glam::Vec3;

use rkf_core::aabb::Aabb;
use rkf_core::asset_file_v3::{SaveLodV3, load_object_header_v3, load_object_lod_v3, save_object_v3};
use rkf_core::brick_geometry::{BrickGeometry, index_to_xyz, voxel_index};
use rkf_core::brick_map::BrickMap;
use rkf_core::companion::{ColorBrick, ColorVoxel};
use rkf_core::constants::BRICK_DIM;
use rkf_core::sdf_cache::SdfCache;
use rkf_core::sdf_compute::{SlotMapping, compute_sdf_from_geometry};
use rkf_import::bvh::TriangleBvh;
use rkf_import::material_transfer::sample_material;
use rkf_import::mesh::{MeshData, load_mesh};

/// Write a minimal OBJ cube with textured material to a temp directory.
/// Returns (dir_path, obj_path). Caller should clean up the directory.
fn create_test_obj() -> (std::path::PathBuf, String) {
    let dir = std::env::temp_dir().join("rkf_obj_pipeline_test");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();

    // Write a 4x4 red/green checkerboard texture
    let mut img = image::RgbaImage::new(4, 4);
    for y in 0..4u32 {
        for x in 0..4u32 {
            let color = if (x + y) % 2 == 0 {
                image::Rgba([255, 0, 0, 255]) // red
            } else {
                image::Rgba([0, 255, 0, 255]) // green
            };
            img.put_pixel(x, y, color);
        }
    }
    img.save(dir.join("checker.png")).unwrap();

    // MTL with texture reference
    std::fs::write(
        dir.join("cube.mtl"),
        "newmtl checker_mat\nKd 0.8 0.8 0.8\nmap_Kd checker.png\n",
    )
    .unwrap();

    // Unit cube OBJ (centered at origin, spans -0.5 to 0.5)
    let obj = r#"mtllib cube.mtl
usemtl checker_mat

v -0.5 -0.5 -0.5
v  0.5 -0.5 -0.5
v  0.5  0.5 -0.5
v -0.5  0.5 -0.5
v -0.5 -0.5  0.5
v  0.5 -0.5  0.5
v  0.5  0.5  0.5
v -0.5  0.5  0.5

vt 0 0
vt 1 0
vt 1 1
vt 0 1

vn  0  0 -1
vn  0  0  1
vn -1  0  0
vn  1  0  0
vn  0 -1  0
vn  0  1  0

f 1/1/1 2/2/1 3/3/1
f 1/1/1 3/3/1 4/4/1
f 5/1/2 7/3/2 6/2/2
f 5/1/2 8/4/2 7/3/2
f 1/1/3 4/2/3 8/3/3
f 1/1/3 8/3/3 5/4/3
f 2/1/4 6/2/4 7/3/4
f 2/1/4 7/3/4 3/4/4
f 1/1/5 5/2/5 6/3/5
f 1/1/5 6/3/5 2/4/5
f 4/1/6 3/2/6 7/3/6
f 4/1/6 7/3/6 8/4/6
"#;
    std::fs::write(dir.join("cube.obj"), obj).unwrap();

    let path = dir.join("cube.obj").to_string_lossy().to_string();
    (dir, path)
}

#[test]
fn obj_load_has_texture() {
    let (_dir, path) = create_test_obj();
    let mesh = load_mesh(&path).unwrap();

    assert_eq!(mesh.triangle_count(), 12, "cube has 12 triangles");
    assert_eq!(mesh.materials.len(), 1);
    assert_eq!(mesh.materials[0].name, "checker_mat");

    let tex = mesh.materials[0]
        .albedo_texture
        .as_ref()
        .expect("texture should be loaded");
    assert_eq!(tex.width, 4);
    assert_eq!(tex.height, 4);
}

#[test]
fn obj_material_transfer() {
    let (_dir, path) = create_test_obj();
    let mesh = load_mesh(&path).unwrap();
    let bvh = TriangleBvh::build(&mesh);

    // Sample near the center of a cube face — should get material_id 0
    let sample = sample_material(&mesh, &bvh, Vec3::new(0.0, 0.0, -0.5));
    assert_eq!(sample.material_id, 0);
    assert!(sample.color.is_some(), "should have texture color");
}

/// Simplified winding number (same algorithm as rkf-convert uses).
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
        let num = na.dot(nb.cross(nc));
        let den = 1.0 + na.dot(nb) + nb.dot(nc) + na.dot(nc);
        winding += 2.0 * num.atan2(den);
    }
    winding / (4.0 * std::f32::consts::PI)
}

#[test]
fn obj_full_pipeline_roundtrip() {
    let (_dir, path) = create_test_obj();
    let mesh = load_mesh(&path).unwrap();
    let bvh = TriangleBvh::build(&mesh);

    // Voxelize at a coarse resolution for fast test
    let voxel_size = 0.15;
    let brick_world_size = voxel_size * BRICK_DIM as f32;

    let margin = voxel_size * 2.0;
    let aabb = Aabb::new(
        mesh.bounds_min - Vec3::splat(margin),
        mesh.bounds_max + Vec3::splat(margin),
    );
    let aabb_size = aabb.max - aabb.min;
    let dims = glam::UVec3::new(
        ((aabb_size.x / brick_world_size).ceil() as u32).max(1),
        ((aabb_size.y / brick_world_size).ceil() as u32).max(1),
        ((aabb_size.z / brick_world_size).ceil() as u32).max(1),
    );

    let narrow_band = brick_world_size * 1.8;
    let mut brick_map = BrickMap::new(dims);
    let mut geometry_vec: Vec<BrickGeometry> = Vec::new();
    let mut color_vec: Vec<ColorBrick> = Vec::new();
    let mut next_slot = 0u32;

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

                let center_nearest = bvh.nearest(brick_center);
                if center_nearest.distance >= narrow_band {
                    continue;
                }

                brick_map.set(bx, by, bz, next_slot);
                next_slot += 1;

                let mut geo = BrickGeometry::new();
                let mut color_brick = ColorBrick {
                    data: [ColorVoxel::new(0, 0, 0, 0); 512],
                };
                let half_voxel = voxel_size * 0.5;

                for vz in 0..BRICK_DIM as u8 {
                    for vy in 0..BRICK_DIM as u8 {
                        for vx in 0..BRICK_DIM as u8 {
                            let pos = brick_min
                                + Vec3::new(
                                    vx as f32 * voxel_size + half_voxel,
                                    vy as f32 * voxel_size + half_voxel,
                                    vz as f32 * voxel_size + half_voxel,
                                );
                            let w = winding_number(&mesh, pos);
                            geo.set_solid(vx, vy, vz, w < -0.5);
                        }
                    }
                }

                geo.rebuild_surface_list();

                for sv in &mut geo.surface_voxels {
                    let idx = sv.index();
                    let (vx, vy, vz) = index_to_xyz(idx);
                    let pos = brick_min
                        + Vec3::new(
                            vx as f32 * voxel_size + half_voxel,
                            vy as f32 * voxel_size + half_voxel,
                            vz as f32 * voxel_size + half_voxel,
                        );
                    let mat_sample = sample_material(&mesh, &bvh, pos);
                    sv.material_id = (mat_sample.material_id as u8).min(63);
                    if let Some(c) = mat_sample.color {
                        let flat = voxel_index(vx, vy, vz);
                        color_brick.data[flat as usize] = ColorVoxel::new(c.r, c.g, c.b, 255);
                    }
                }

                geometry_vec.push(geo);
                color_vec.push(color_brick);
            }
        }
    }

    let brick_count = geometry_vec.len();
    assert!(brick_count > 0, "should have allocated some bricks");

    // Compute SDF
    let mut sdf_caches: Vec<SdfCache> = vec![SdfCache::empty(); brick_count];
    let slot_mappings: Vec<SlotMapping> = (0..brick_count as u32)
        .map(|i| SlotMapping {
            brick_slot: i,
            geometry_slot: i,
            sdf_slot: i,
        })
        .collect();
    compute_sdf_from_geometry(&brick_map, &geometry_vec, &mut sdf_caches, &slot_mappings, voxel_size);

    // Verify some geometry makes sense
    let total_solid: u32 = geometry_vec.iter().map(|g| g.solid_count()).sum();
    assert!(total_solid > 0, "should have some solid voxels");
    let total_surface: usize = geometry_vec.iter().map(|g| g.surface_voxels.len()).sum();
    assert!(total_surface > 0, "should have some surface voxels");

    // Verify some color data was written
    let total_colored: usize = color_vec
        .iter()
        .map(|cb| cb.data.iter().filter(|v| v.intensity() > 0).count())
        .sum();
    assert!(total_colored > 0, "should have some colored voxels");

    // Save to .rkf v3
    let lod = SaveLodV3 {
        voxel_size,
        brick_map,
        geometry: geometry_vec,
        sdf_cache: Some(sdf_caches),
        color_bricks: Some(color_vec),
    };

    let mut buf = Cursor::new(Vec::new());
    save_object_v3(&mut buf, &aabb, None, &[0], &[lod]).unwrap();
    let file_bytes = buf.into_inner();
    assert!(!file_bytes.is_empty());

    // Reload and verify
    let mut cursor = Cursor::new(&file_bytes);
    let header = load_object_header_v3(&mut cursor).unwrap();
    assert_eq!(header.lod_entries.len(), 1);
    assert!(header.has_sdf_cache);
    assert!(header.has_color);
    assert_eq!(header.material_ids, vec![0]);

    let lod_data = load_object_lod_v3(&mut cursor, &header, 0).unwrap();
    assert_eq!(lod_data.geometry.len(), brick_count);
    assert!(lod_data.sdf_cache.is_some());
    assert!(lod_data.color_bricks.is_some());

    let loaded_colors = lod_data.color_bricks.unwrap();
    assert_eq!(loaded_colors.len(), brick_count);

    // Verify roundtrip: same number of colored voxels
    let loaded_colored: usize = loaded_colors
        .iter()
        .map(|cb| cb.data.iter().filter(|v| v.intensity() > 0).count())
        .sum();
    assert_eq!(loaded_colored, total_colored, "color data should roundtrip");

    // Verify SDF has both positive and negative values
    let sdf = lod_data.sdf_cache.unwrap();
    let has_negative = sdf.iter().any(|c| {
        (0..8u8).any(|z| (0..8u8).any(|y| (0..8u8).any(|x| c.get_distance(x, y, z) < 0.0)))
    });
    let has_positive = sdf.iter().any(|c| {
        (0..8u8).any(|z| (0..8u8).any(|y| (0..8u8).any(|x| c.get_distance(x, y, z) > 0.0)))
    });
    assert!(has_negative, "SDF should have negative (interior) values");
    assert!(has_positive, "SDF should have positive (exterior) values");

    eprintln!(
        "Pipeline roundtrip OK: {} bricks, {} solid voxels, {} surface voxels, {} colored voxels, {} bytes",
        brick_count, total_solid, total_surface, total_colored, file_bytes.len()
    );
}
