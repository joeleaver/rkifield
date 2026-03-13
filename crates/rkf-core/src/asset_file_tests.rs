//! Tests for asset_file (v2 format).

use super::*;
use std::io::Cursor;

// -- Helpers ---------------------------------------------------------------

/// Create a VoxelSample with a given f32 distance and material id.
fn vs(dist: f32, mat: u16) -> VoxelSample {
    VoxelSample::new(dist, mat, [255, 255, 255, 255])
}

/// Build a 2x2x2 brick map with two allocated bricks (slots 10 and 20)
/// and return it together with two corresponding brick data arrays.
fn make_small_lod(voxel_size: f32) -> SaveLodLevel {
    let dims = UVec3::new(2, 2, 2);
    let mut brick_map = BrickMap::new(dims);
    brick_map.set(0, 0, 0, 10); // slot 10 -> local 0
    brick_map.set(1, 0, 0, 20); // slot 20 -> local 1

    // Brick 0: all voxels with dist=1.0, mat=1
    let brick0 = [vs(1.0, 1); 512];
    // Brick 1: all voxels with dist=2.0, mat=2
    let brick1 = [vs(2.0, 2); 512];

    SaveLodLevel {
        voxel_size,
        brick_map,
        brick_data: vec![brick0, brick1],
    }
}

/// Save to a `Vec<u8>` via Cursor and return the bytes.
fn save_to_bytes(
    aabb: &Aabb,
    analytical: Option<&SdfPrimitive>,
    mat_ids: &[u16],
    lods: &[SaveLodLevel],
) -> Vec<u8> {
    let mut buf = Cursor::new(Vec::<u8>::new());
    save_object(&mut buf, aabb, analytical, mat_ids, lods).expect("save failed");
    buf.into_inner()
}

/// Standard test AABB.
fn unit_aabb() -> Aabb {
    Aabb::new(Vec3::ZERO, Vec3::ONE)
}

// -- Size checks -----------------------------------------------------------

#[test]
fn header_size_is_128_bytes() {
    assert_eq!(std::mem::size_of::<RkfFileHeader>(), 128);
}

#[test]
fn lod_entry_size_is_40_bytes() {
    assert_eq!(std::mem::size_of::<LodFileEntry>(), 40);
}

// -- 1. Single LOD roundtrip -----------------------------------------------

#[test]
fn test_roundtrip_single_lod() {
    let lod = make_small_lod(0.02);
    let bytes = save_to_bytes(&unit_aabb(), None, &[1, 2], &[lod]);

    let mut cursor = Cursor::new(bytes);
    let header = load_object_header(&mut cursor).expect("load_header failed");

    assert_eq!(header.lod_entries.len(), 1);
    assert_eq!(header.lod_entries[0].brick_count, 2);
    assert_eq!(header.lod_entries[0].brick_dims, UVec3::new(2, 2, 2));
    assert_eq!(header.lod_entries[0].voxel_size, 0.02);
    assert_eq!(header.material_ids, vec![1, 2]);

    let lod_data = load_object_lod(&mut cursor, &header, 0).expect("load_lod failed");
    assert_eq!(lod_data.brick_data.len(), 2);

    // Slot 10 was local 0, slot 20 was local 1.
    // After load, brick_map entries are local indices (0..brick_count-1).
    let map = &lod_data.brick_map;
    let slot_00 = map.get(0, 0, 0).unwrap();
    let slot_10 = map.get(1, 0, 0).unwrap();
    assert_ne!(slot_00, EMPTY_SLOT);
    assert_ne!(slot_10, EMPTY_SLOT);
    assert_ne!(slot_00, slot_10);

    // The brick at local 0 should have dist=1.0, mat=1
    let brick0 = &lod_data.brick_data[slot_00 as usize];
    assert_eq!(brick0[0].distance_f32(), 1.0);
    assert_eq!(brick0[0].material_id(), 1);

    // The brick at local 1 should have dist=2.0, mat=2
    let brick1 = &lod_data.brick_data[slot_10 as usize];
    assert_eq!(brick1[0].distance_f32(), 2.0);
    assert_eq!(brick1[0].material_id(), 2);
}

// -- 2. Multi-LOD roundtrip ------------------------------------------------

#[test]
fn test_roundtrip_multi_lod() {
    // Three LODs: fine (0.005), medium (0.02), coarse (0.08).
    // We pass them in arbitrary order; save_object sorts them coarsest-first.
    let lod_fine = make_small_lod(0.005);
    let lod_med = make_small_lod(0.02);
    let lod_coarse = make_small_lod(0.08);

    let bytes = save_to_bytes(
        &unit_aabb(),
        None,
        &[1, 2],
        &[lod_fine, lod_med, lod_coarse],
    );

    let mut cursor = Cursor::new(bytes);
    let header = load_object_header(&mut cursor).expect("load_header failed");

    assert_eq!(header.lod_entries.len(), 3);

    // Entries are sorted coarsest-first (descending voxel_size).
    assert!(header.lod_entries[0].voxel_size >= header.lod_entries[1].voxel_size);
    assert!(header.lod_entries[1].voxel_size >= header.lod_entries[2].voxel_size);
    assert!((header.lod_entries[0].voxel_size - 0.08).abs() < 1e-6);
    assert!((header.lod_entries[1].voxel_size - 0.02).abs() < 1e-6);
    assert!((header.lod_entries[2].voxel_size - 0.005).abs() < 1e-6);

    // Load each LOD independently and verify brick counts.
    for i in 0..3 {
        let lod_data = load_object_lod(&mut cursor, &header, i)
            .unwrap_or_else(|e| panic!("load_lod({i}) failed: {e}"));
        assert_eq!(
            lod_data.brick_data.len(),
            2,
            "LOD {i} should have 2 bricks"
        );
    }
}

// -- 3. Header-only load ---------------------------------------------------

#[test]
fn test_header_only_load() {
    let aabb = Aabb::new(Vec3::new(-1.0, -2.0, -3.0), Vec3::new(1.0, 2.0, 3.0));
    let lod = make_small_lod(0.02);
    let bytes = save_to_bytes(&aabb, None, &[5, 6, 7], &[lod]);

    let mut cursor = Cursor::new(bytes);
    let header = load_object_header(&mut cursor).expect("load_header failed");

    // AABB roundtrips.
    assert!((header.aabb.min.x - (-1.0)).abs() < 1e-6);
    assert!((header.aabb.max.z - 3.0).abs() < 1e-6);

    // Material IDs roundtrip.
    assert_eq!(header.material_ids, vec![5, 6, 7]);

    // No brick data was loaded -- only metadata.
    assert_eq!(header.lod_entries.len(), 1);
}

// -- 4. Invalid magic ------------------------------------------------------

#[test]
fn test_invalid_magic() {
    let mut bad = vec![0u8; 128 + 40];
    bad[0] = b'B';
    bad[1] = b'A';
    bad[2] = b'D';
    bad[3] = b'!';

    let mut cursor = Cursor::new(bad);
    let err = load_object_header(&mut cursor).unwrap_err();
    assert!(matches!(err, AssetError::InvalidMagic));
}

// -- 5. LOD index out of range ---------------------------------------------

#[test]
fn test_lod_index_out_of_range() {
    let lod = make_small_lod(0.02);
    let bytes = save_to_bytes(&unit_aabb(), None, &[], &[lod]);

    let mut cursor = Cursor::new(bytes);
    let header = load_object_header(&mut cursor).expect("load_header");

    let err = load_object_lod(&mut cursor, &header, 99).unwrap_err();
    assert!(matches!(err, AssetError::LodIndexOutOfRange(99, 1)));
}

// -- 6. Empty brick map ----------------------------------------------------

#[test]
fn test_empty_brick_map() {
    let lod = SaveLodLevel {
        voxel_size: 0.08,
        brick_map: BrickMap::new(UVec3::new(4, 4, 4)),
        brick_data: vec![],
    };

    let bytes = save_to_bytes(&unit_aabb(), None, &[], &[lod]);
    let mut cursor = Cursor::new(bytes);
    let header = load_object_header(&mut cursor).expect("load_header");

    assert_eq!(header.lod_entries[0].brick_count, 0);

    let lod_data = load_object_lod(&mut cursor, &header, 0).expect("load_lod");
    assert_eq!(lod_data.brick_data.len(), 0);
    assert_eq!(lod_data.brick_map.dims, UVec3::new(4, 4, 4));

    // All entries should be EMPTY_SLOT.
    for &e in &lod_data.brick_map.entries {
        assert_eq!(e, EMPTY_SLOT);
    }
}

// -- 7. Analytical bound roundtrip -----------------------------------------

#[test]
fn test_analytical_bound_roundtrip() {
    let bound = SdfPrimitive::Sphere { radius: 0.75 };
    let lod = make_small_lod(0.02);
    let bytes = save_to_bytes(&unit_aabb(), Some(&bound), &[3], &[lod]);

    let mut cursor = Cursor::new(bytes);
    let header = load_object_header(&mut cursor).expect("load_header");

    let decoded = header.analytical_bound.expect("should have analytical bound");
    match decoded {
        SdfPrimitive::Sphere { radius } => {
            assert!((radius - 0.75).abs() < 1e-6, "radius = {radius}");
        }
        other => panic!("unexpected primitive: {other:?}"),
    }
}

#[test]
fn test_analytical_bound_box_roundtrip() {
    let bound = SdfPrimitive::Box {
        half_extents: Vec3::new(1.0, 2.0, 3.0),
    };
    let lod = make_small_lod(0.02);
    let bytes = save_to_bytes(&unit_aabb(), Some(&bound), &[], &[lod]);

    let mut cursor = Cursor::new(bytes);
    let header = load_object_header(&mut cursor).expect("load_header");

    match header.analytical_bound.expect("bound") {
        SdfPrimitive::Box { half_extents } => {
            assert!((half_extents.x - 1.0).abs() < 1e-6);
            assert!((half_extents.y - 2.0).abs() < 1e-6);
            assert!((half_extents.z - 3.0).abs() < 1e-6);
        }
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn test_analytical_bound_capsule_roundtrip() {
    let bound = SdfPrimitive::Capsule {
        radius: 0.3,
        half_height: 0.8,
    };
    let lod = make_small_lod(0.02);
    let bytes = save_to_bytes(&unit_aabb(), Some(&bound), &[], &[lod]);

    let mut cursor = Cursor::new(bytes);
    let header = load_object_header(&mut cursor).expect("load_header");

    match header.analytical_bound.expect("bound") {
        SdfPrimitive::Capsule {
            radius,
            half_height,
        } => {
            assert!((radius - 0.3).abs() < 1e-6);
            assert!((half_height - 0.8).abs() < 1e-6);
        }
        other => panic!("unexpected: {other:?}"),
    }
}

#[test]
fn test_no_analytical_bound() {
    let lod = make_small_lod(0.02);
    let bytes = save_to_bytes(&unit_aabb(), None, &[], &[lod]);

    let mut cursor = Cursor::new(bytes);
    let header = load_object_header(&mut cursor).expect("load_header");
    assert!(header.analytical_bound.is_none());
}

// -- Additional correctness tests ------------------------------------------

#[test]
fn test_aabb_roundtrip() {
    let aabb = Aabb::new(Vec3::new(-5.0, -3.0, 0.5), Vec3::new(5.0, 3.0, 4.5));
    let lod = make_small_lod(0.04);
    let bytes = save_to_bytes(&aabb, None, &[], &[lod]);

    let mut cursor = Cursor::new(bytes);
    let header = load_object_header(&mut cursor).expect("load_header");

    assert!((header.aabb.min.x - (-5.0)).abs() < 1e-6);
    assert!((header.aabb.min.y - (-3.0)).abs() < 1e-6);
    assert!((header.aabb.min.z - 0.5).abs() < 1e-6);
    assert!((header.aabb.max.x - 5.0).abs() < 1e-6);
    assert!((header.aabb.max.y - 3.0).abs() < 1e-6);
    assert!((header.aabb.max.z - 4.5).abs() < 1e-6);
}

#[test]
fn test_material_ids_truncated_at_16() {
    // Provide 20 material IDs -- only 16 should be stored.
    let mat_ids: Vec<u16> = (0..20).collect();
    let lod = make_small_lod(0.02);
    let bytes = save_to_bytes(&unit_aabb(), None, &mat_ids, &[lod]);

    let mut cursor = Cursor::new(bytes);
    let header = load_object_header(&mut cursor).expect("load_header");
    assert_eq!(header.material_ids.len(), 16);
    assert_eq!(header.material_ids[15], 15);
}

#[test]
fn test_file_layout_starts_with_magic() {
    let lod = make_small_lod(0.02);
    let bytes = save_to_bytes(&unit_aabb(), None, &[], &[lod]);
    assert_eq!(&bytes[0..4], b"RKF2");
}

#[test]
fn test_lod_data_offset_after_header_and_entries() {
    let lod = make_small_lod(0.02);
    let bytes = save_to_bytes(&unit_aabb(), None, &[], &[lod]);

    let mut cursor = Cursor::new(&bytes[..]);
    let header = load_object_header(&mut cursor).expect("load_header");
    let entry = &header.lod_entries[0];

    // Data must start after the file header (128 bytes) + 1 LOD entry (40 bytes).
    let expected_min_offset = 128u64 + 40u64;
    assert!(
        entry.data_offset >= expected_min_offset,
        "data_offset {} should be >= {}",
        entry.data_offset,
        expected_min_offset
    );
}

#[test]
fn test_unsupported_version() {
    let lod = make_small_lod(0.02);
    let mut bytes = save_to_bytes(&unit_aabb(), None, &[], &[lod]);
    // Overwrite version field (bytes 4..8) with version 99.
    let v: u32 = 99;
    bytes[4..8].copy_from_slice(&v.to_ne_bytes());

    let mut cursor = Cursor::new(bytes);
    let err = load_object_header(&mut cursor).unwrap_err();
    assert!(matches!(err, AssetError::UnsupportedVersion(99)));
}

#[test]
fn test_lod_seek_independence() {
    // Load LODs in reverse order -- each seek should work correctly.
    let lod0 = {
        let mut m = BrickMap::new(UVec3::new(2, 2, 2));
        m.set(0, 0, 0, 5);
        SaveLodLevel {
            voxel_size: 0.08,
            brick_map: m,
            brick_data: vec![[vs(8.0, 8); 512]],
        }
    };
    let lod1 = {
        let mut m = BrickMap::new(UVec3::new(3, 3, 3));
        m.set(1, 1, 1, 7);
        SaveLodLevel {
            voxel_size: 0.02,
            brick_map: m,
            brick_data: vec![[vs(2.0, 2); 512]],
        }
    };

    let bytes = save_to_bytes(&unit_aabb(), None, &[], &[lod0, lod1]);
    let mut cursor = Cursor::new(bytes);
    let header = load_object_header(&mut cursor).expect("load_header");

    // Load LOD 1 (finer) first.
    let d1 = load_object_lod(&mut cursor, &header, 1).expect("load lod 1");
    assert_eq!(d1.brick_data[0][0].distance_f32(), 2.0);

    // Then load LOD 0 (coarser).
    let d0 = load_object_lod(&mut cursor, &header, 0).expect("load lod 0");
    assert_eq!(d0.brick_data[0][0].distance_f32(), 8.0);
}

#[test]
fn test_plane_analytical_roundtrip() {
    let bound = SdfPrimitive::Plane {
        normal: Vec3::new(0.0, 1.0, 0.0),
        distance: -2.5,
    };
    let lod = make_small_lod(0.02);
    let bytes = save_to_bytes(&unit_aabb(), Some(&bound), &[], &[lod]);

    let mut cursor = Cursor::new(bytes);
    let header = load_object_header(&mut cursor).expect("load_header");

    match header.analytical_bound.expect("bound") {
        SdfPrimitive::Plane { normal, distance } => {
            assert!((normal.y - 1.0).abs() < 1e-6);
            assert!((distance - (-2.5)).abs() < 1e-6);
        }
        other => panic!("unexpected: {other:?}"),
    }
}
