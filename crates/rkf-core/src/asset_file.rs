//! .rkf v2 file format — serialization and deserialization for voxelized objects.
//!
//! The .rkf v2 format stores a single voxelized object with multiple LOD levels.
//! Each LOD level contains a brick map and the corresponding voxel brick data,
//! compressed with LZ4.
//!
//! # File layout
//!
//! ```text
//! [RkfFileHeader]          128 bytes, fixed
//! [LodFileEntry × N]       40 bytes per entry, N = lod_count
//! [LOD data 0]             LZ4 compressed, coarsest (largest voxel_size)
//! [LOD data 1]
//! ...
//! [LOD data N-1]           finest (smallest voxel_size)
//! ```
//!
//! # LOD data layout (uncompressed)
//!
//! ```text
//! [brick map u32 × (dims.x × dims.y × dims.z)]   local slot indices (0..brick_count-1) or EMPTY_SLOT
//! [brick data VoxelSample × (512 × brick_count)]  8 bytes per sample, 4096 bytes per brick
//! ```

use std::io::{Read, Seek, SeekFrom, Write};
use std::mem;

use bytemuck::{Pod, Zeroable};
use glam::{UVec3, Vec3};

use crate::aabb::Aabb;
use crate::brick_map::{BrickMap, EMPTY_SLOT};
use crate::scene_node::SdfPrimitive;
use crate::voxel::VoxelSample;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur when reading or writing .rkf v2 files.
#[derive(Debug, thiserror::Error)]
pub enum AssetError {
    /// Underlying I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// File does not start with the expected "RKF2" magic bytes.
    #[error("invalid magic: expected RKF2")]
    InvalidMagic,

    /// File uses a version number this code does not understand.
    #[error("unsupported version: {0}")]
    UnsupportedVersion(u32),

    /// Caller requested a LOD index that does not exist.
    #[error("LOD index {0} out of range (max {1})")]
    LodIndexOutOfRange(usize, usize),

    /// LZ4 decompression failed.
    #[error("decompression error: {0}")]
    Decompression(String),
}

// ---------------------------------------------------------------------------
// File-format structs  (bytemuck Pod — must be repr(C) with no padding gaps)
// ---------------------------------------------------------------------------

/// Magic bytes for .rkf v2 files.
const MAGIC: [u8; 4] = *b"RKF2";
/// Supported file version.
const VERSION: u32 = 2;
/// Number of samples per brick (8 × 8 × 8).
const BRICK_SAMPLES: usize = 512;

/// On-disk file header — exactly 128 bytes.
///
/// All multi-byte integers are stored in native endianness. The format is not
/// designed for cross-platform interchange; it is a local GPU asset cache.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct RkfFileHeader {
    /// "RKF2"
    magic: [u8; 4],
    /// Format version (2).
    version: u32,
    /// Number of LOD levels stored in this file.
    lod_count: u32,
    /// Number of unique material IDs referenced by this object.
    material_count: u32,
    /// Object-local AABB minimum corner.
    aabb_min: [f32; 3],
    /// Object-local AABB maximum corner.
    aabb_max: [f32; 3],
    /// Analytical bound type: 0=none, 1=sphere, 2=box, 3=capsule.
    analytical_type: u32,
    /// Analytical bound parameters (meaning depends on `analytical_type`).
    analytical_params: [f32; 4],
    /// Up to 16 material IDs referenced by this object.
    material_ids: [u16; 16],
    /// Explicit padding to reach exactly 128 bytes.
    ///
    /// Byte count: 4+4+4+4+12+12+4+16+32 = 92 → need 36 bytes padding.
    _reserved: [u8; 36],
}

// SAFETY: All fields are Pod-compatible. The explicit `_reserved` padding
// ensures the struct is exactly 128 bytes with no implicit padding.
unsafe impl Zeroable for RkfFileHeader {}
unsafe impl Pod for RkfFileHeader {}

/// On-disk LOD entry descriptor — exactly 40 bytes.
///
/// The explicit `_pad` field covers the 4 bytes of alignment padding that
/// `#[repr(C)]` would insert before the `u64 data_offset` anyway.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
struct LodFileEntry {
    /// World-space size of one voxel for this LOD (metres).
    voxel_size: f32,
    /// Number of allocated (non-empty) bricks.
    brick_count: u32,
    /// Brick map grid dimensions (number of bricks per axis).
    brick_dims: [u32; 3],
    /// Explicit padding to align `data_offset` to 8 bytes.
    _pad: u32,
    /// Byte offset from the start of the file to the compressed LOD data.
    data_offset: u64,
    /// Size of the compressed data block in bytes.
    compressed_size: u32,
    /// Size of the data after decompression in bytes.
    uncompressed_size: u32,
}

// SAFETY: All fields are Pod-compatible. The explicit `_pad` makes alignment
// padding visible so bytemuck can safely cast the bytes.
unsafe impl Zeroable for LodFileEntry {}
unsafe impl Pod for LodFileEntry {}

// Compile-time size assertions.
const _: () = assert!(mem::size_of::<RkfFileHeader>() == 128);
const _: () = assert!(mem::size_of::<LodFileEntry>() == 40);

// ---------------------------------------------------------------------------
// Public data types
// ---------------------------------------------------------------------------

/// One LOD level supplied to [`save_object`].
pub struct SaveLodLevel {
    /// World-space voxel size for this LOD.
    pub voxel_size: f32,
    /// Brick map: maps 3D brick coordinates to pool slot indices.
    ///
    /// Entries equal to [`EMPTY_SLOT`] indicate no brick at that position.
    /// Entries that are not [`EMPTY_SLOT`] are remapped to sequential local
    /// indices (0 .. brick_count-1) during serialization.
    pub brick_map: BrickMap,
    /// Brick voxel data — one array of 512 [`VoxelSample`]s per allocated
    /// brick, in the same order as the non-`EMPTY_SLOT` entries in the map.
    pub brick_data: Vec<[VoxelSample; 512]>,
}

/// Metadata about a single LOD level, returned by [`load_object_header`].
#[derive(Debug)]
pub struct LodEntryInfo {
    /// World-space voxel size for this LOD.
    pub voxel_size: f32,
    /// Number of allocated bricks.
    pub brick_count: u32,
    /// Brick map grid dimensions.
    pub brick_dims: UVec3,
    /// Byte offset to the compressed data (used by [`load_object_lod`]).
    pub data_offset: u64,
    /// Compressed data size in bytes.
    pub compressed_size: u32,
}

/// Header information loaded from a .rkf v2 file without reading any brick data.
#[derive(Debug)]
pub struct ObjectHeader {
    /// Object-local axis-aligned bounding box.
    pub aabb: Aabb,
    /// Optional analytical SDF fallback for far-field rendering.
    pub analytical_bound: Option<SdfPrimitive>,
    /// Material IDs referenced by voxels in this object.
    pub material_ids: Vec<u16>,
    /// Per-LOD metadata (sorted coarsest → finest, i.e. descending voxel_size).
    pub lod_entries: Vec<LodEntryInfo>,
}

/// Brick map and voxel data for one LOD level, returned by [`load_object_lod`].
#[derive(Debug)]
pub struct LodData {
    /// Brick map with pool slot indices restored (local 0..N → caller's pool).
    ///
    /// Note: slot indices are sequential local indices (0 .. brick_count-1).
    /// The caller is responsible for mapping them to actual pool slots.
    pub brick_map: BrickMap,
    /// Brick voxel data — one array of 512 samples per brick.
    pub brick_data: Vec<[VoxelSample; 512]>,
}

// ---------------------------------------------------------------------------
// Serialization helpers
// ---------------------------------------------------------------------------

/// Encode an [`SdfPrimitive`] to `(type_id, [f32; 4])`.
pub(crate) fn encode_analytical_pub(prim: &SdfPrimitive) -> (u32, [f32; 4]) {
    encode_analytical(prim)
}

/// Decode `(type_id, [f32; 4])` to an [`SdfPrimitive`], or `None` if type_id == 0.
pub(crate) fn decode_analytical_pub(type_id: u32, params: [f32; 4]) -> Option<SdfPrimitive> {
    decode_analytical(type_id, params)
}

fn encode_analytical(prim: &SdfPrimitive) -> (u32, [f32; 4]) {
    match *prim {
        SdfPrimitive::Sphere { radius } => (1, [radius, 0.0, 0.0, 0.0]),
        SdfPrimitive::Box { half_extents } => {
            (2, [half_extents.x, half_extents.y, half_extents.z, 0.0])
        }
        SdfPrimitive::Capsule {
            radius,
            half_height,
        } => (3, [radius, half_height, 0.0, 0.0]),
        SdfPrimitive::Torus {
            major_radius,
            minor_radius,
        } => (4, [major_radius, minor_radius, 0.0, 0.0]),
        SdfPrimitive::Cylinder {
            radius,
            half_height,
        } => (5, [radius, half_height, 0.0, 0.0]),
        SdfPrimitive::Plane { normal, distance } => {
            (6, [normal.x, normal.y, normal.z, distance])
        }
    }
}

/// Decode `(type_id, [f32; 4])` to an [`SdfPrimitive`], or `None` if type_id == 0.
fn decode_analytical(type_id: u32, params: [f32; 4]) -> Option<SdfPrimitive> {
    match type_id {
        0 => None,
        1 => Some(SdfPrimitive::Sphere { radius: params[0] }),
        2 => Some(SdfPrimitive::Box {
            half_extents: Vec3::new(params[0], params[1], params[2]),
        }),
        3 => Some(SdfPrimitive::Capsule {
            radius: params[0],
            half_height: params[1],
        }),
        4 => Some(SdfPrimitive::Torus {
            major_radius: params[0],
            minor_radius: params[1],
        }),
        5 => Some(SdfPrimitive::Cylinder {
            radius: params[0],
            half_height: params[1],
        }),
        6 => Some(SdfPrimitive::Plane {
            normal: Vec3::new(params[0], params[1], params[2]),
            distance: params[3],
        }),
        _ => None,
    }
}

/// Pack a LOD level's brick map and voxel data into a byte buffer.
///
/// Layout:
/// 1. Brick map — `dims.x * dims.y * dims.z` u32 entries (local indices).
/// 2. Brick data — `brick_count * 512 * 8` bytes.
///
/// Entries in the brick map that were originally pool slots are remapped to
/// sequential local indices (0..brick_count-1). `EMPTY_SLOT` stays `EMPTY_SLOT`.
fn pack_lod_data(lod: &SaveLodLevel) -> Vec<u8> {
    let total_entries = lod.brick_map.entries.len();
    let brick_count = lod.brick_data.len();

    // Build a mapping from original pool slot → local index.
    // We iterate entries in order and assign each unique non-EMPTY slot a
    // sequential local index. This preserves the correspondence with brick_data.
    let mut slot_to_local: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    let mut next_local: u32 = 0;
    for &slot in &lod.brick_map.entries {
        if slot != EMPTY_SLOT {
            slot_to_local.entry(slot).or_insert_with(|| {
                let idx = next_local;
                next_local += 1;
                idx
            });
        }
    }
    debug_assert_eq!(next_local as usize, brick_count);

    // Remap the brick map entries.
    let remapped: Vec<u32> = lod
        .brick_map
        .entries
        .iter()
        .map(|&slot| {
            if slot == EMPTY_SLOT {
                EMPTY_SLOT
            } else {
                slot_to_local[&slot]
            }
        })
        .collect();

    // Pack into bytes.
    let map_bytes = total_entries * mem::size_of::<u32>();
    let brick_bytes = brick_count * BRICK_SAMPLES * mem::size_of::<VoxelSample>();
    let mut buf = Vec::with_capacity(map_bytes + brick_bytes);

    // Write brick map.
    let map_byte_slice: &[u8] = bytemuck::cast_slice(&remapped);
    buf.extend_from_slice(map_byte_slice);

    // Write brick data.
    for brick in &lod.brick_data {
        let brick_byte_slice: &[u8] = bytemuck::cast_slice(brick.as_ref());
        buf.extend_from_slice(brick_byte_slice);
    }

    buf
}

/// Unpack a decompressed LOD data buffer into a [`LodData`].
fn unpack_lod_data(data: &[u8], dims: UVec3, brick_count: usize) -> Result<LodData, AssetError> {
    let total_entries = (dims.x * dims.y * dims.z) as usize;
    let map_bytes = total_entries * mem::size_of::<u32>();
    let brick_bytes = brick_count * BRICK_SAMPLES * mem::size_of::<VoxelSample>();
    let expected_total = map_bytes + brick_bytes;

    if data.len() < expected_total {
        return Err(AssetError::Decompression(format!(
            "decompressed data is {} bytes, expected at least {}",
            data.len(),
            expected_total
        )));
    }

    // Parse brick map.
    let map_slice: &[u32] = bytemuck::cast_slice(&data[..map_bytes]);
    let mut brick_map = BrickMap::new(dims);
    brick_map.entries.copy_from_slice(map_slice);

    // Parse brick data.
    let brick_data_raw = &data[map_bytes..map_bytes + brick_bytes];
    let sample_slice: &[VoxelSample] = bytemuck::cast_slice(brick_data_raw);

    let mut brick_data: Vec<[VoxelSample; 512]> = Vec::with_capacity(brick_count);
    for i in 0..brick_count {
        let start = i * BRICK_SAMPLES;
        let end = start + BRICK_SAMPLES;
        let arr: [VoxelSample; 512] = sample_slice[start..end]
            .try_into()
            .map_err(|_| AssetError::Decompression("brick data slice length mismatch".into()))?;
        brick_data.push(arr);
    }

    Ok(LodData {
        brick_map,
        brick_data,
    })
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Save a voxelized object to .rkf v2 format.
///
/// LOD levels are written coarsest-first (sorted by `voxel_size` descending).
/// Up to 16 material IDs can be stored in the header; extras are silently
/// truncated.
///
/// # Errors
///
/// Returns [`AssetError::Io`] if any write fails.
pub fn save_object<W: Write + Seek>(
    writer: &mut W,
    aabb: &Aabb,
    analytical_bound: Option<&SdfPrimitive>,
    material_ids: &[u16],
    lod_levels: &[SaveLodLevel],
) -> Result<(), AssetError> {
    // Sort LOD levels coarsest-first (descending voxel_size).
    let mut sorted_indices: Vec<usize> = (0..lod_levels.len()).collect();
    sorted_indices.sort_by(|&a, &b| {
        lod_levels[b]
            .voxel_size
            .partial_cmp(&lod_levels[a].voxel_size)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let lod_count = lod_levels.len() as u32;

    // Build material_ids array (up to 16 entries, padded with zeros).
    let mut mat_ids_arr = [0u16; 16];
    let mat_count = material_ids.len().min(16);
    mat_ids_arr[..mat_count].copy_from_slice(&material_ids[..mat_count]);

    // Build analytical bound encoding.
    let (analytical_type, analytical_params) = match analytical_bound {
        Some(prim) => encode_analytical(prim),
        None => (0, [0.0f32; 4]),
    };

    // Write header.
    let header = RkfFileHeader {
        magic: MAGIC,
        version: VERSION,
        lod_count,
        material_count: mat_count as u32,
        aabb_min: aabb.min.to_array(),
        aabb_max: aabb.max.to_array(),
        analytical_type,
        analytical_params,
        material_ids: mat_ids_arr,
        _reserved: [0u8; 36],
    };
    writer.write_all(bytemuck::bytes_of(&header))?;

    // Write placeholder LodFileEntry array — we'll seek back to fill offsets.
    let entries_start = writer.stream_position()?;
    let zero_entry = LodFileEntry {
        voxel_size: 0.0,
        brick_count: 0,
        brick_dims: [0; 3],
        _pad: 0,
        data_offset: 0,
        compressed_size: 0,
        uncompressed_size: 0,
    };
    for _ in 0..lod_count {
        writer.write_all(bytemuck::bytes_of(&zero_entry))?;
    }

    // Compress and write each LOD level, recording offsets.
    let mut final_entries: Vec<LodFileEntry> = Vec::with_capacity(lod_count as usize);

    for &idx in &sorted_indices {
        let lod = &lod_levels[idx];
        let dims = lod.brick_map.dims;
        let brick_count = lod.brick_data.len() as u32;

        // Pack raw data.
        let raw = pack_lod_data(lod);
        let uncompressed_size = raw.len() as u32;

        // LZ4 compress (prepend size for decompressor).
        let compressed = lz4_flex::compress_prepend_size(&raw);
        let compressed_size = compressed.len() as u32;

        // Record offset before writing.
        let data_offset = writer.stream_position()?;

        writer.write_all(&compressed)?;

        final_entries.push(LodFileEntry {
            voxel_size: lod.voxel_size,
            brick_count,
            brick_dims: [dims.x, dims.y, dims.z],
            _pad: 0,
            data_offset,
            compressed_size,
            uncompressed_size,
        });
    }

    // Seek back and overwrite the placeholder entries with the real data.
    writer.seek(SeekFrom::Start(entries_start))?;
    for entry in &final_entries {
        writer.write_all(bytemuck::bytes_of(entry))?;
    }

    Ok(())
}

/// Load only the header (metadata) from a .rkf v2 file.
///
/// This is fast — it reads the fixed header and LOD entry array without
/// touching any brick data. Use [`load_object_lod`] to load individual LODs
/// on demand.
///
/// # Errors
///
/// Returns [`AssetError::InvalidMagic`] or [`AssetError::UnsupportedVersion`]
/// if the file does not match expectations.
pub fn load_object_header<R: Read>(reader: &mut R) -> Result<ObjectHeader, AssetError> {
    // Read and validate header.
    let mut header_bytes = [0u8; mem::size_of::<RkfFileHeader>()];
    reader.read_exact(&mut header_bytes)?;
    let header: RkfFileHeader = *bytemuck::from_bytes(&header_bytes);

    if header.magic != MAGIC {
        return Err(AssetError::InvalidMagic);
    }
    if header.version != VERSION {
        return Err(AssetError::UnsupportedVersion(header.version));
    }

    // Read LOD entries.
    let lod_count = header.lod_count as usize;
    let mut lod_entries = Vec::with_capacity(lod_count);
    let entry_size = mem::size_of::<LodFileEntry>();
    let mut entry_bytes = vec![0u8; entry_size];

    for _ in 0..lod_count {
        reader.read_exact(&mut entry_bytes)?;
        let entry: LodFileEntry = *bytemuck::from_bytes(&entry_bytes);
        lod_entries.push(LodEntryInfo {
            voxel_size: entry.voxel_size,
            brick_count: entry.brick_count,
            brick_dims: UVec3::new(entry.brick_dims[0], entry.brick_dims[1], entry.brick_dims[2]),
            data_offset: entry.data_offset,
            compressed_size: entry.compressed_size,
        });
    }

    // Decode AABB.
    let aabb = Aabb::new(
        Vec3::from_array(header.aabb_min),
        Vec3::from_array(header.aabb_max),
    );

    // Decode analytical bound.
    let analytical_bound = decode_analytical(header.analytical_type, header.analytical_params);

    // Decode material IDs (only `material_count` are valid).
    let mat_count = (header.material_count as usize).min(16);
    let material_ids = header.material_ids[..mat_count].to_vec();

    Ok(ObjectHeader {
        aabb,
        analytical_bound,
        material_ids,
        lod_entries,
    })
}

/// Load a single LOD level's brick data from a .rkf v2 file.
///
/// `header` must be the result of a prior [`load_object_header`] call on the
/// same file. `lod_index` is 0-based, ordered coarsest-first.
///
/// # Errors
///
/// Returns [`AssetError::LodIndexOutOfRange`] if `lod_index` is out of bounds,
/// or [`AssetError::Decompression`] if the compressed data is corrupt.
pub fn load_object_lod<R: Read + Seek>(
    reader: &mut R,
    header: &ObjectHeader,
    lod_index: usize,
) -> Result<LodData, AssetError> {
    if lod_index >= header.lod_entries.len() {
        return Err(AssetError::LodIndexOutOfRange(
            lod_index,
            header.lod_entries.len(),
        ));
    }

    let entry = &header.lod_entries[lod_index];

    // Seek to the compressed data block.
    reader.seek(SeekFrom::Start(entry.data_offset))?;

    // Read the compressed bytes.
    let mut compressed = vec![0u8; entry.compressed_size as usize];
    reader.read_exact(&mut compressed)?;

    // LZ4 decompress.
    let decompressed = lz4_flex::decompress_size_prepended(&compressed)
        .map_err(|e| AssetError::Decompression(e.to_string()))?;

    // Unpack.
    unpack_lod_data(
        &decompressed,
        entry.brick_dims,
        entry.brick_count as usize,
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    // ── Helpers ─────────────────────────────────────────────────────────────

    /// Create a VoxelSample with a given f32 distance and material id.
    fn vs(dist: f32, mat: u16) -> VoxelSample {
        VoxelSample::new(dist, mat, 0, 0, 0)
    }

    /// Build a 2×2×2 brick map with two allocated bricks (slots 10 and 20)
    /// and return it together with two corresponding brick data arrays.
    fn make_small_lod(voxel_size: f32) -> SaveLodLevel {
        let dims = UVec3::new(2, 2, 2);
        let mut brick_map = BrickMap::new(dims);
        brick_map.set(0, 0, 0, 10); // slot 10 → local 0
        brick_map.set(1, 0, 0, 20); // slot 20 → local 1

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

    // ── Size checks ─────────────────────────────────────────────────────────

    #[test]
    fn header_size_is_128_bytes() {
        assert_eq!(mem::size_of::<RkfFileHeader>(), 128);
    }

    #[test]
    fn lod_entry_size_is_40_bytes() {
        assert_eq!(mem::size_of::<LodFileEntry>(), 40);
    }

    // ── 1. Single LOD roundtrip ──────────────────────────────────────────────

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

    // ── 2. Multi-LOD roundtrip ───────────────────────────────────────────────

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

    // ── 3. Header-only load ──────────────────────────────────────────────────

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

        // No brick data was loaded — only metadata.
        assert_eq!(header.lod_entries.len(), 1);
    }

    // ── 4. Invalid magic ─────────────────────────────────────────────────────

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

    // ── 5. LOD index out of range ────────────────────────────────────────────

    #[test]
    fn test_lod_index_out_of_range() {
        let lod = make_small_lod(0.02);
        let bytes = save_to_bytes(&unit_aabb(), None, &[], &[lod]);

        let mut cursor = Cursor::new(bytes);
        let header = load_object_header(&mut cursor).expect("load_header");

        let err = load_object_lod(&mut cursor, &header, 99).unwrap_err();
        assert!(matches!(err, AssetError::LodIndexOutOfRange(99, 1)));
    }

    // ── 6. Empty brick map ───────────────────────────────────────────────────

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

    // ── 7. Analytical bound roundtrip ────────────────────────────────────────

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

    // ── Additional correctness tests ──────────────────────────────────────────

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
        // Provide 20 material IDs — only 16 should be stored.
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
        // Load LODs in reverse order — each seek should work correctly.
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
}
