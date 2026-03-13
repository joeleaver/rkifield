//! .rkf v2 file format — DEPRECATED, retained for backwards-compatible loading.
//!
//! **Use `asset_file_v3` for all new saves.** The v2 format stores raw VoxelSample
//! data (GPU format) and lacks support for secondary materials and blend weights.
//! The v3 format stores geometry-first data (occupancy + surface voxels) and is
//! the current/authoritative format.
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

#[cfg(test)]
#[path = "asset_file_tests.rs"]
mod tests;
