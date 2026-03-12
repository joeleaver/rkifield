//! .rkf v3 file format — geometry-first serialization.
//!
//! The v3 format stores geometry (occupancy + surface voxels) as the source of truth,
//! with an optional SDF cache for fast loading.
//!
//! # File layout
//!
//! ```text
//! [RkfV3Header]              128 bytes, fixed
//! [LodV3Entry × N]           56 bytes per entry, N = lod_count
//! [LOD data 0]               LZ4 compressed geometry + optional SDF cache
//! ...
//! [LOD data N-1]
//! ```
//!
//! # LOD data layout (uncompressed)
//!
//! ```text
//! [brick map u32 × (dims.x × dims.y × dims.z)]
//! [geometry data]:
//!     per allocated brick:
//!         occupancy: [u64; 8] = 64 bytes
//!         surface_count: u16
//!         surface_voxels: [SurfaceVoxel; surface_count] = 8 bytes each
//! [sdf cache (optional)]:
//!     per allocated brick:
//!         distances: [u16; 512] = 1024 bytes (f16 bits)
//! ```

use std::io::{Read, Seek, SeekFrom, Write};
use std::mem;

use bytemuck::{Pod, Zeroable};
use glam::{UVec3, Vec3};

use crate::aabb::Aabb;
use crate::brick_geometry::BrickGeometry;
use crate::brick_map::{BrickMap, EMPTY_SLOT, INTERIOR_SLOT};
use crate::scene_node::SdfPrimitive;
use crate::sdf_cache::SdfCache;

use crate::asset_file::{AssetError, LodEntryInfo, ObjectHeader};

// ---------------------------------------------------------------------------
// File-format structs
// ---------------------------------------------------------------------------

const MAGIC_V3: [u8; 4] = *b"RKF3";
const VERSION_V3: u32 = 3;

/// Flags for the v3 header.
const FLAG_HAS_SDF_CACHE: u32 = 1 << 0;

/// On-disk v3 header — 128 bytes.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct RkfV3Header {
    magic: [u8; 4],
    version: u32,
    lod_count: u32,
    material_count: u32,
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
    flags: u32,
    analytical_type: u32,
    analytical_params: [f32; 4],
    material_ids: [u8; 32],  // u8 material IDs (256 max), up to 32 stored
    _reserved: [u8; 32],
}

unsafe impl Zeroable for RkfV3Header {}
unsafe impl Pod for RkfV3Header {}

/// On-disk v3 LOD entry — 48 bytes.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct LodV3Entry {
    voxel_size: f32,
    brick_count: u32,
    brick_dims: [u32; 3],
    _pad: u32,
    geometry_offset: u64,
    geometry_compressed_size: u32,
    geometry_uncompressed_size: u32,
    sdf_offset: u64,
    sdf_compressed_size: u32,
    sdf_uncompressed_size: u32,
}

unsafe impl Zeroable for LodV3Entry {}
unsafe impl Pod for LodV3Entry {}

const _: () = assert!(mem::size_of::<RkfV3Header>() == 128);
const _: () = assert!(mem::size_of::<LodV3Entry>() == 56); // adjust if needed

// ---------------------------------------------------------------------------
// Public data types
// ---------------------------------------------------------------------------

/// One LOD level for v3 save.
pub struct SaveLodV3 {
    /// Voxel size for this LOD.
    pub voxel_size: f32,
    /// Brick map (entries are pool slots or EMPTY_SLOT/INTERIOR_SLOT).
    pub brick_map: BrickMap,
    /// Geometry data for each allocated brick (ordered by first-seen in brick map).
    pub geometry: Vec<BrickGeometry>,
    /// Optional SDF cache data (same order as geometry). None = compute on load.
    pub sdf_cache: Option<Vec<SdfCache>>,
}

/// Loaded v3 LOD data.
#[derive(Debug)]
pub struct LodDataV3 {
    /// Brick map with local slot indices (0..brick_count-1).
    pub brick_map: BrickMap,
    /// Geometry data per allocated brick.
    pub geometry: Vec<BrickGeometry>,
    /// SDF cache per allocated brick (None if not present in file).
    pub sdf_cache: Option<Vec<SdfCache>>,
}

/// V3 header info (extends ObjectHeader).
#[derive(Debug)]
pub struct ObjectHeaderV3 {
    /// Base header info.
    pub aabb: Aabb,
    pub analytical_bound: Option<SdfPrimitive>,
    pub material_ids: Vec<u8>,
    pub lod_entries: Vec<LodEntryInfoV3>,
    pub has_sdf_cache: bool,
}

/// V3 LOD entry info.
#[derive(Debug)]
pub struct LodEntryInfoV3 {
    pub voxel_size: f32,
    pub brick_count: u32,
    pub brick_dims: UVec3,
    pub geometry_offset: u64,
    pub geometry_compressed_size: u32,
    pub sdf_offset: u64,
    pub sdf_compressed_size: u32,
}

// ---------------------------------------------------------------------------
// Serialization helpers
// ---------------------------------------------------------------------------

fn encode_analytical(prim: &SdfPrimitive) -> (u32, [f32; 4]) {
    crate::asset_file::encode_analytical_pub(prim)
}

fn decode_analytical(type_id: u32, params: [f32; 4]) -> Option<SdfPrimitive> {
    crate::asset_file::decode_analytical_pub(type_id, params)
}

/// Pack geometry data for one LOD into bytes.
fn pack_geometry(brick_map: &BrickMap, geometry: &[BrickGeometry]) -> (Vec<u8>, Vec<u32>) {
    let total_entries = brick_map.entries.len();

    // Build slot remapping: original slot → local index
    let mut slot_to_local: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    let mut next_local = 0u32;
    for &slot in &brick_map.entries {
        if slot != EMPTY_SLOT && slot != INTERIOR_SLOT {
            slot_to_local.entry(slot).or_insert_with(|| {
                let idx = next_local;
                next_local += 1;
                idx
            });
        }
    }

    // Remap brick map entries (preserve EMPTY_SLOT and INTERIOR_SLOT)
    let remapped: Vec<u32> = brick_map
        .entries
        .iter()
        .map(|&slot| {
            if slot == EMPTY_SLOT || slot == INTERIOR_SLOT {
                slot
            } else {
                slot_to_local[&slot]
            }
        })
        .collect();

    // Pack map + geometry
    let mut buf = Vec::new();

    // Brick map
    let map_bytes: &[u8] = bytemuck::cast_slice(&remapped);
    buf.extend_from_slice(map_bytes);

    // Geometry data per allocated brick
    for geo in geometry {
        buf.extend_from_slice(&geo.to_bytes());
    }

    // Return ordered slot list for SDF cache alignment
    let mut ordered_slots: Vec<(u32, u32)> = slot_to_local.into_iter().collect();
    ordered_slots.sort_by_key(|&(_, local)| local);
    let original_slots: Vec<u32> = ordered_slots.iter().map(|&(orig, _)| orig).collect();

    (buf, original_slots)
}

/// Pack SDF cache data.
fn pack_sdf_cache(caches: &[SdfCache]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(caches.len() * 1024);
    for cache in caches {
        buf.extend_from_slice(cache.as_bytes());
    }
    buf
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Save a voxelized object to .rkf v3 format.
pub fn save_object_v3<W: Write + Seek>(
    writer: &mut W,
    aabb: &Aabb,
    analytical_bound: Option<&SdfPrimitive>,
    material_ids: &[u8],
    lod_levels: &[SaveLodV3],
) -> Result<(), AssetError> {
    let lod_count = lod_levels.len() as u32;
    let has_sdf = lod_levels.iter().any(|l| l.sdf_cache.is_some());

    // Material IDs (up to 32)
    let mut mat_ids_arr = [0u8; 32];
    let mat_count = material_ids.len().min(32);
    mat_ids_arr[..mat_count].copy_from_slice(&material_ids[..mat_count]);

    let (analytical_type, analytical_params) = match analytical_bound {
        Some(prim) => encode_analytical(prim),
        None => (0, [0.0f32; 4]),
    };

    let flags = if has_sdf { FLAG_HAS_SDF_CACHE } else { 0 };

    let header = RkfV3Header {
        magic: MAGIC_V3,
        version: VERSION_V3,
        lod_count,
        material_count: mat_count as u32,
        aabb_min: aabb.min.to_array(),
        aabb_max: aabb.max.to_array(),
        flags,
        analytical_type,
        analytical_params,
        material_ids: mat_ids_arr,
        _reserved: [0u8; 32],
    };
    writer.write_all(bytemuck::bytes_of(&header))?;

    // Write placeholder LOD entries
    let entries_start = writer.stream_position()?;
    let zero_entry: LodV3Entry = bytemuck::Zeroable::zeroed();
    for _ in 0..lod_count {
        writer.write_all(bytemuck::bytes_of(&zero_entry))?;
    }

    // Sort coarsest-first
    let mut sorted: Vec<usize> = (0..lod_levels.len()).collect();
    sorted.sort_by(|&a, &b| {
        lod_levels[b].voxel_size.partial_cmp(&lod_levels[a].voxel_size)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut final_entries: Vec<LodV3Entry> = Vec::with_capacity(lod_count as usize);

    for &idx in &sorted {
        let lod = &lod_levels[idx];
        let dims = lod.brick_map.dims;
        let brick_count = lod.geometry.len() as u32;

        // Pack and compress geometry
        let (geo_raw, _ordered_slots) = pack_geometry(&lod.brick_map, &lod.geometry);
        let geo_uncompressed = geo_raw.len() as u32;
        let geo_compressed = lz4_flex::compress_prepend_size(&geo_raw);
        let geo_offset = writer.stream_position()?;
        writer.write_all(&geo_compressed)?;

        // Pack and compress SDF cache (optional)
        let (sdf_offset, sdf_compressed_size, sdf_uncompressed_size) = if let Some(ref caches) = lod.sdf_cache {
            let sdf_raw = pack_sdf_cache(caches);
            let sdf_uncompressed = sdf_raw.len() as u32;
            let sdf_compressed = lz4_flex::compress_prepend_size(&sdf_raw);
            let offset = writer.stream_position()?;
            writer.write_all(&sdf_compressed)?;
            (offset, sdf_compressed.len() as u32, sdf_uncompressed)
        } else {
            (0, 0, 0)
        };

        final_entries.push(LodV3Entry {
            voxel_size: lod.voxel_size,
            brick_count,
            brick_dims: [dims.x, dims.y, dims.z],
            _pad: 0,
            geometry_offset: geo_offset,
            geometry_compressed_size: geo_compressed.len() as u32,
            geometry_uncompressed_size: geo_uncompressed,
            sdf_offset,
            sdf_compressed_size,
            sdf_uncompressed_size,
        });
    }

    // Write final entries
    writer.seek(SeekFrom::Start(entries_start))?;
    for entry in &final_entries {
        writer.write_all(bytemuck::bytes_of(entry))?;
    }

    Ok(())
}

/// Load v3 header.
pub fn load_object_header_v3<R: Read>(reader: &mut R) -> Result<ObjectHeaderV3, AssetError> {
    let mut header_bytes = [0u8; 128];
    reader.read_exact(&mut header_bytes)?;
    let header: RkfV3Header = *bytemuck::from_bytes(&header_bytes);

    if header.magic != MAGIC_V3 {
        return Err(AssetError::InvalidMagic);
    }
    if header.version != VERSION_V3 {
        return Err(AssetError::UnsupportedVersion(header.version));
    }

    let lod_count = header.lod_count as usize;
    let entry_size = mem::size_of::<LodV3Entry>();
    let mut lod_entries = Vec::with_capacity(lod_count);
    let mut entry_bytes = vec![0u8; entry_size];

    for _ in 0..lod_count {
        reader.read_exact(&mut entry_bytes)?;
        let entry: LodV3Entry = *bytemuck::from_bytes(&entry_bytes);
        lod_entries.push(LodEntryInfoV3 {
            voxel_size: entry.voxel_size,
            brick_count: entry.brick_count,
            brick_dims: UVec3::new(entry.brick_dims[0], entry.brick_dims[1], entry.brick_dims[2]),
            geometry_offset: entry.geometry_offset,
            geometry_compressed_size: entry.geometry_compressed_size,
            sdf_offset: entry.sdf_offset,
            sdf_compressed_size: entry.sdf_compressed_size,
        });
    }

    let aabb = Aabb::new(
        Vec3::from_array(header.aabb_min),
        Vec3::from_array(header.aabb_max),
    );
    let analytical_bound = decode_analytical(header.analytical_type, header.analytical_params);

    let mat_count = (header.material_count as usize).min(32);
    let material_ids = header.material_ids[..mat_count].to_vec();

    Ok(ObjectHeaderV3 {
        aabb,
        analytical_bound,
        material_ids,
        lod_entries,
        has_sdf_cache: header.flags & FLAG_HAS_SDF_CACHE != 0,
    })
}

/// Load a single LOD level from a v3 file.
pub fn load_object_lod_v3<R: Read + Seek>(
    reader: &mut R,
    header: &ObjectHeaderV3,
    lod_index: usize,
) -> Result<LodDataV3, AssetError> {
    if lod_index >= header.lod_entries.len() {
        return Err(AssetError::LodIndexOutOfRange(
            lod_index,
            header.lod_entries.len(),
        ));
    }

    let entry = &header.lod_entries[lod_index];
    let dims = entry.brick_dims;
    let brick_count = entry.brick_count as usize;

    // Read geometry
    reader.seek(SeekFrom::Start(entry.geometry_offset))?;
    let mut geo_compressed = vec![0u8; entry.geometry_compressed_size as usize];
    reader.read_exact(&mut geo_compressed)?;
    let geo_data = lz4_flex::decompress_size_prepended(&geo_compressed)
        .map_err(|e| AssetError::Decompression(e.to_string()))?;

    // Unpack brick map
    let total_entries = (dims.x * dims.y * dims.z) as usize;
    let map_bytes = total_entries * 4;
    if geo_data.len() < map_bytes {
        return Err(AssetError::Decompression("geometry data too short for brick map".into()));
    }
    let map_slice: &[u32] = bytemuck::cast_slice(&geo_data[..map_bytes]);
    let mut brick_map = BrickMap::new(dims);
    brick_map.entries.copy_from_slice(map_slice);

    // Unpack geometry
    let mut geometry = Vec::with_capacity(brick_count);
    let mut offset = map_bytes;
    for _ in 0..brick_count {
        let remaining = &geo_data[offset..];
        let (geo, consumed) = BrickGeometry::from_bytes(remaining)
            .ok_or_else(|| AssetError::Decompression("failed to parse BrickGeometry".into()))?;
        geometry.push(geo);
        offset += consumed;
    }

    // Read SDF cache (optional)
    let sdf_cache = if entry.sdf_compressed_size > 0 {
        reader.seek(SeekFrom::Start(entry.sdf_offset))?;
        let mut sdf_compressed = vec![0u8; entry.sdf_compressed_size as usize];
        reader.read_exact(&mut sdf_compressed)?;
        let sdf_data = lz4_flex::decompress_size_prepended(&sdf_compressed)
            .map_err(|e| AssetError::Decompression(e.to_string()))?;

        let mut caches = Vec::with_capacity(brick_count);
        for i in 0..brick_count {
            let start = i * 1024;
            if start + 1024 > sdf_data.len() {
                return Err(AssetError::Decompression("SDF cache data truncated".into()));
            }
            let cache = SdfCache::from_bytes(&sdf_data[start..start + 1024])
                .ok_or_else(|| AssetError::Decompression("failed to parse SdfCache".into()))?;
            caches.push(cache);
        }
        Some(caches)
    } else {
        None
    };

    Ok(LodDataV3 {
        brick_map,
        geometry,
        sdf_cache,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn unit_aabb() -> Aabb {
        Aabb::new(Vec3::ZERO, Vec3::ONE)
    }

    fn make_test_lod(voxel_size: f32) -> SaveLodV3 {
        // Create a brick with a half-solid plane
        let mut geo = BrickGeometry::new();
        for z in 0..4u8 {
            for y in 0..8u8 {
                for x in 0..8u8 {
                    geo.set_solid(x, y, z, true);
                }
            }
        }
        geo.rebuild_surface_list();
        // Set material on some voxels
        if let Some(sv) = geo.surface_voxels.first_mut() {
            sv.blend_weight = 255;
            sv.material_id = 3;
        }

        let mut brick_map = BrickMap::new(UVec3::new(2, 2, 2));
        brick_map.set(0, 0, 0, 10);
        brick_map.set(1, 0, 0, 20);

        let geo2 = BrickGeometry::new(); // empty brick (will be deallocated in real usage, but ok for test)

        SaveLodV3 {
            voxel_size,
            brick_map,
            geometry: vec![geo, geo2],
            sdf_cache: None,
        }
    }

    #[test]
    fn v3_roundtrip_no_sdf_cache() {
        let lod = make_test_lod(0.02);
        let mut buf = Cursor::new(Vec::new());
        save_object_v3(&mut buf, &unit_aabb(), None, &[1, 3], &[lod]).unwrap();

        let bytes = buf.into_inner();
        assert_eq!(&bytes[0..4], b"RKF3");

        let mut cursor = Cursor::new(bytes);
        let header = load_object_header_v3(&mut cursor).unwrap();
        assert_eq!(header.lod_entries.len(), 1);
        assert_eq!(header.material_ids, vec![1, 3]);
        assert!(!header.has_sdf_cache);

        let lod_data = load_object_lod_v3(&mut cursor, &header, 0).unwrap();
        assert_eq!(lod_data.geometry.len(), 2);
        assert!(lod_data.sdf_cache.is_none());

        // Verify geometry roundtrip
        let geo = &lod_data.geometry[0];
        assert_eq!(geo.solid_count(), 256); // half solid (4 * 8 * 8)
        assert!(!geo.surface_voxels.is_empty());
        // First surface voxel should have our material assignment
        let sv = geo.surface_voxels.first().unwrap();
        assert_eq!(sv.blend_weight, 255);
        assert_eq!(sv.material_id, 3);
    }

    #[test]
    fn v3_roundtrip_with_sdf_cache() {
        let mut lod = make_test_lod(0.02);

        // Add SDF cache
        let mut cache0 = SdfCache::empty();
        cache0.set_distance(0, 0, 0, -1.5);
        let cache1 = SdfCache::empty();
        lod.sdf_cache = Some(vec![cache0, cache1]);

        let mut buf = Cursor::new(Vec::new());
        save_object_v3(&mut buf, &unit_aabb(), None, &[], &[lod]).unwrap();

        let mut cursor = Cursor::new(buf.into_inner());
        let header = load_object_header_v3(&mut cursor).unwrap();
        assert!(header.has_sdf_cache);

        let lod_data = load_object_lod_v3(&mut cursor, &header, 0).unwrap();
        let caches = lod_data.sdf_cache.unwrap();
        assert_eq!(caches.len(), 2);
        assert!((caches[0].get_distance(0, 0, 0) - (-1.5)).abs() < 0.01);
    }

    #[test]
    fn v3_multi_lod() {
        let lod0 = make_test_lod(0.08);
        let lod1 = make_test_lod(0.02);

        let mut buf = Cursor::new(Vec::new());
        save_object_v3(&mut buf, &unit_aabb(), None, &[], &[lod0, lod1]).unwrap();

        let mut cursor = Cursor::new(buf.into_inner());
        let header = load_object_header_v3(&mut cursor).unwrap();
        assert_eq!(header.lod_entries.len(), 2);
        // Coarsest first
        assert!(header.lod_entries[0].voxel_size >= header.lod_entries[1].voxel_size);

        // Load both LODs
        for i in 0..2 {
            let data = load_object_lod_v3(&mut cursor, &header, i).unwrap();
            assert_eq!(data.geometry.len(), 2);
        }
    }

    #[test]
    fn v3_invalid_magic() {
        let mut bad = vec![0u8; 256];
        bad[0..4].copy_from_slice(b"BAD!");
        let mut cursor = Cursor::new(bad);
        assert!(matches!(load_object_header_v3(&mut cursor).unwrap_err(), AssetError::InvalidMagic));
    }

    #[test]
    fn v3_lod_out_of_range() {
        let lod = make_test_lod(0.02);
        let mut buf = Cursor::new(Vec::new());
        save_object_v3(&mut buf, &unit_aabb(), None, &[], &[lod]).unwrap();

        let mut cursor = Cursor::new(buf.into_inner());
        let header = load_object_header_v3(&mut cursor).unwrap();
        assert!(matches!(
            load_object_lod_v3(&mut cursor, &header, 5).unwrap_err(),
            AssetError::LodIndexOutOfRange(5, 1)
        ));
    }

    #[test]
    fn v3_interior_slot_preserved() {
        let mut brick_map = BrickMap::new(UVec3::new(2, 1, 1));
        brick_map.set(0, 0, 0, 10);
        brick_map.set(1, 0, 0, INTERIOR_SLOT);

        let mut geo = BrickGeometry::new();
        geo.set_solid(4, 4, 4, true);
        geo.rebuild_surface_list();

        let lod = SaveLodV3 {
            voxel_size: 0.1,
            brick_map,
            geometry: vec![geo],
            sdf_cache: None,
        };

        let mut buf = Cursor::new(Vec::new());
        save_object_v3(&mut buf, &unit_aabb(), None, &[], &[lod]).unwrap();

        let mut cursor = Cursor::new(buf.into_inner());
        let header = load_object_header_v3(&mut cursor).unwrap();
        let data = load_object_lod_v3(&mut cursor, &header, 0).unwrap();

        // Brick (1,0,0) should be INTERIOR_SLOT
        assert_eq!(data.brick_map.get(1, 0, 0), Some(INTERIOR_SLOT));
        // Brick (0,0,0) should be allocated (local index 0)
        assert_eq!(data.brick_map.get(0, 0, 0), Some(0));
    }

    #[test]
    fn v3_brick_map_roundtrip_multi_brick() {
        // Test with multiple bricks in different positions to catch index mapping bugs.
        let dims = UVec3::new(3, 3, 3);
        let mut brick_map = BrickMap::new(dims);
        // Place bricks at specific coordinates with non-sequential slot IDs
        brick_map.set(0, 0, 0, 42);
        brick_map.set(2, 1, 0, 99);
        brick_map.set(1, 2, 2, 7);
        brick_map.set(1, 1, 1, INTERIOR_SLOT);

        // Create 3 unique geometry bricks (one per non-interior, non-empty slot)
        let mut geos = Vec::new();
        for i in 0..3u8 {
            let mut geo = BrickGeometry::new();
            // Each brick has different solid pattern for verification
            for z in 0..((i + 1) * 2) {
                for y in 0..8 {
                    for x in 0..8 {
                        geo.set_solid(x, y, z, true);
                    }
                }
            }
            geo.rebuild_surface_list();
            geos.push(geo);
        }

        let lod = SaveLodV3 {
            voxel_size: 0.05,
            brick_map,
            geometry: geos.clone(),
            sdf_cache: None,
        };

        let mut buf = Cursor::new(Vec::new());
        save_object_v3(&mut buf, &unit_aabb(), None, &[], &[lod]).unwrap();

        let mut cursor = Cursor::new(buf.into_inner());
        let header = load_object_header_v3(&mut cursor).unwrap();
        let data = load_object_lod_v3(&mut cursor, &header, 0).unwrap();

        assert_eq!(data.brick_map.dims, dims);
        assert_eq!(data.geometry.len(), 3);

        // Verify brick map: non-empty non-interior entries should be local indices 0,1,2
        // The order depends on iteration through entries (flat order)
        // (0,0,0)=42 is first encountered → local 0
        // (2,1,0)=99 is second → local 1
        // (1,2,2)=7 is third → local 2
        assert_eq!(data.brick_map.get(0, 0, 0), Some(0));
        assert_eq!(data.brick_map.get(2, 1, 0), Some(1));
        assert_eq!(data.brick_map.get(1, 2, 2), Some(2));
        assert_eq!(data.brick_map.get(1, 1, 1), Some(INTERIOR_SLOT));

        // Empty slots should be EMPTY_SLOT
        assert_eq!(data.brick_map.get(0, 0, 1), Some(EMPTY_SLOT));

        // Verify geometry data matches original ORDER
        // local 0 = slot 42 = geos[?] ... but which original geometry goes with slot 42?
        // In the save, geometry is [geos[0], geos[1], geos[2]] and
        // pack_geometry iterates entries, encounters slot 42 first (gets local 0),
        // then 99 (local 1), then 7 (local 2).
        // The geometry array is written as-is: geos[0], geos[1], geos[2].
        // So loaded geometry[0] = original geos[0], geometry[1] = geos[1], etc.
        assert_eq!(data.geometry[0].solid_count(), geos[0].solid_count());
        assert_eq!(data.geometry[1].solid_count(), geos[1].solid_count());
        assert_eq!(data.geometry[2].solid_count(), geos[2].solid_count());

        // Cross-check: local index 0 (at (0,0,0)) should map to geometry[0]
        // which has 2*8*8 = 128 solid voxels
        assert_eq!(data.geometry[0].solid_count(), 128);
        assert_eq!(data.geometry[1].solid_count(), 256);
        assert_eq!(data.geometry[2].solid_count(), 384);
    }

}
