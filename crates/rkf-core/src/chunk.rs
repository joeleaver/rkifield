//! Chunk data structure and `.rkf` binary format for streaming.
//!
//! A [`Chunk`] represents one 8m x 8m x 8m region of the world, containing per-tier
//! sparse grids and their associated brick data. Chunks are the unit of streaming:
//! loaded from disk on demand and evicted when no longer needed.
//!
//! The `.rkf` binary format stores a chunk's grids and bricks with LZ4 compression.
//! Use [`save_chunk`] / [`load_chunk`] for `Read`/`Write` streams, or
//! [`save_chunk_file`] / [`load_chunk_file`] for filesystem paths.

use std::io::{Read, Write};
use std::path::Path;

use bytemuck::{Pod, Zeroable};
use glam::{IVec3, UVec3};

use crate::brick::Brick;
use crate::brick_pool::BrickPool;
use crate::populate::PopulateError;
use crate::sparse_grid::SparseGrid;

// ---------------------------------------------------------------------------
// Magic and version
// ---------------------------------------------------------------------------

/// Magic bytes identifying the `.rkf` format.
const RKF_MAGIC: [u8; 4] = *b"RKF\0";

/// Current format version.
const RKF_VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// On-disk header types
// ---------------------------------------------------------------------------

/// File header for the `.rkf` binary format (32 bytes).
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct RkfHeader {
    /// Magic bytes: `b"RKF\0"`.
    pub magic: [u8; 4],
    /// Format version (currently 1).
    pub version: u32,
    /// Chunk X coordinate.
    pub chunk_x: i32,
    /// Chunk Y coordinate.
    pub chunk_y: i32,
    /// Chunk Z coordinate.
    pub chunk_z: i32,
    /// Number of tier sections that follow.
    pub tier_count: u32,
    /// Reserved flags (must be 0).
    pub flags: u32,
    /// Reserved (must be 0).
    pub _reserved: u32,
}

// SAFETY: RkfHeader is repr(C), all fields are plain integers/byte arrays.
unsafe impl Zeroable for RkfHeader {}
unsafe impl Pod for RkfHeader {}

/// Per-tier header in the `.rkf` format (16 bytes).
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct RkfTierHeader {
    /// Resolution tier index (0-3).
    pub tier: u32,
    /// Grid dimension X (in cells/bricks).
    pub dims_x: u32,
    /// Grid dimension Y (in cells/bricks).
    pub dims_y: u32,
    /// Grid dimension Z (in cells/bricks).
    pub dims_z: u32,
}

// SAFETY: RkfTierHeader is repr(C), all fields are u32.
unsafe impl Zeroable for RkfTierHeader {}
unsafe impl Pod for RkfTierHeader {}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during chunk I/O.
#[derive(Debug)]
pub enum ChunkIoError {
    /// Underlying I/O error.
    Io(std::io::Error),
    /// File does not start with the expected magic bytes.
    InvalidMagic,
    /// File version is newer than what this code supports.
    UnsupportedVersion(u32),
    /// LZ4 decompression failed.
    DecompressError(String),
    /// Data is structurally invalid (wrong sizes, etc.).
    InvalidData(String),
}

impl std::fmt::Display for ChunkIoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ChunkIoError::Io(e) => write!(f, "I/O error: {e}"),
            ChunkIoError::InvalidMagic => write!(f, "invalid RKF magic bytes"),
            ChunkIoError::UnsupportedVersion(v) => {
                write!(f, "unsupported RKF version: {v}")
            }
            ChunkIoError::DecompressError(msg) => {
                write!(f, "LZ4 decompression error: {msg}")
            }
            ChunkIoError::InvalidData(msg) => write!(f, "invalid data: {msg}"),
        }
    }
}

impl std::error::Error for ChunkIoError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ChunkIoError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for ChunkIoError {
    fn from(e: std::io::Error) -> Self {
        ChunkIoError::Io(e)
    }
}

// ---------------------------------------------------------------------------
// Chunk and TierGrid
// ---------------------------------------------------------------------------

/// A single resolution tier within a chunk: its sparse grid and local brick data.
#[derive(Debug, Clone)]
pub struct TierGrid {
    /// Resolution tier index (0-3).
    pub tier: u8,
    /// The sparse grid for this tier.
    pub grid: SparseGrid,
    /// Brick data for this tier. Indices are grid-local (0-based), not pool-global.
    pub bricks: Vec<Brick>,
}

/// A world chunk — one 8m x 8m x 8m region with per-tier sparse grids and bricks.
#[derive(Debug, Clone)]
pub struct Chunk {
    /// Chunk coordinates in the world grid (8m per unit).
    pub coords: IVec3,
    /// Per-tier sparse grids. Only populated tiers are present.
    pub grids: Vec<TierGrid>,
    /// Total number of bricks across all tiers.
    pub brick_count: u32,
}

impl Chunk {
    /// Create a new empty chunk at the given coordinates.
    pub fn new(coords: IVec3) -> Self {
        Self {
            coords,
            grids: Vec::new(),
            brick_count: 0,
        }
    }

    /// Recompute `brick_count` from the tier grids.
    pub fn recount_bricks(&mut self) {
        self.brick_count = self.grids.iter().map(|tg| tg.bricks.len() as u32).sum();
    }

    /// Load this chunk's bricks into a pool, returning the global slot mapping.
    ///
    /// For each tier grid, allocates pool slots for every brick and copies the data.
    /// Updates the grid's slot references from local indices to pool-global indices.
    ///
    /// Returns a flat list of all allocated pool slots (across all tiers).
    pub fn load_into_pool(&mut self, pool: &mut BrickPool) -> Result<Vec<u32>, PopulateError> {
        let total_bricks: u32 = self.grids.iter().map(|tg| tg.bricks.len() as u32).sum();
        let mut global_slots = Vec::with_capacity(total_bricks as usize);

        for tier_grid in &mut self.grids {
            // Build local-to-global slot mapping for this tier.
            let mut local_to_global: Vec<u32> = Vec::with_capacity(tier_grid.bricks.len());

            for brick in &tier_grid.bricks {
                let slot = pool.allocate().ok_or(PopulateError::PoolExhausted {
                    allocated_so_far: global_slots.len() as u32,
                })?;
                *pool.get_mut(slot) = *brick;
                local_to_global.push(slot);
                global_slots.push(slot);
            }

            // Update grid slot references from local to global.
            let dims = tier_grid.grid.dimensions();
            for z in 0..dims.z {
                for y in 0..dims.y {
                    for x in 0..dims.x {
                        if let Some(local_slot) = tier_grid.grid.brick_slot(x, y, z) {
                            if (local_slot as usize) < local_to_global.len() {
                                tier_grid
                                    .grid
                                    .set_brick_slot(x, y, z, local_to_global[local_slot as usize]);
                            }
                        }
                    }
                }
            }
        }

        Ok(global_slots)
    }
}

// ---------------------------------------------------------------------------
// Save / Load
// ---------------------------------------------------------------------------

/// Save a chunk to `.rkf` format.
///
/// Writes: file header, tier headers, then for each tier: LZ4-compressed
/// occupancy + slot + brick data.
pub fn save_chunk(chunk: &Chunk, writer: &mut impl Write) -> Result<(), ChunkIoError> {
    // --- File header ---
    let header = RkfHeader {
        magic: RKF_MAGIC,
        version: RKF_VERSION,
        chunk_x: chunk.coords.x,
        chunk_y: chunk.coords.y,
        chunk_z: chunk.coords.z,
        tier_count: chunk.grids.len() as u32,
        flags: 0,
        _reserved: 0,
    };
    writer.write_all(bytemuck::bytes_of(&header))?;

    // --- Tier headers ---
    for tier_grid in &chunk.grids {
        let dims = tier_grid.grid.dimensions();
        let tier_header = RkfTierHeader {
            tier: tier_grid.tier as u32,
            dims_x: dims.x,
            dims_y: dims.y,
            dims_z: dims.z,
        };
        writer.write_all(bytemuck::bytes_of(&tier_header))?;
    }

    // --- Tier data (each tier individually LZ4-compressed) ---
    for tier_grid in &chunk.grids {
        let mut uncompressed = Vec::new();

        // Occupancy data (raw u32s)
        let occ_bytes: &[u8] = bytemuck::cast_slice(tier_grid.grid.occupancy_data());
        uncompressed.extend_from_slice(occ_bytes);

        // Slot data (raw u32s)
        let slot_bytes: &[u8] = bytemuck::cast_slice(tier_grid.grid.slot_data());
        uncompressed.extend_from_slice(slot_bytes);

        // Brick data (each brick is 4096 bytes, Pod)
        for brick in &tier_grid.bricks {
            uncompressed.extend_from_slice(bytemuck::bytes_of(brick));
        }

        // LZ4 compress
        let compressed = lz4_flex::compress_prepend_size(&uncompressed);

        // Write compressed size (u32) then compressed data
        let compressed_len = compressed.len() as u32;
        writer.write_all(&compressed_len.to_le_bytes())?;
        writer.write_all(&compressed)?;
    }

    Ok(())
}

/// Load a chunk from `.rkf` format.
pub fn load_chunk(reader: &mut impl Read) -> Result<Chunk, ChunkIoError> {
    // --- File header ---
    let mut header_bytes = [0u8; std::mem::size_of::<RkfHeader>()];
    reader.read_exact(&mut header_bytes)?;
    let header: &RkfHeader = bytemuck::from_bytes(&header_bytes);

    if header.magic != RKF_MAGIC {
        return Err(ChunkIoError::InvalidMagic);
    }
    if header.version != RKF_VERSION {
        return Err(ChunkIoError::UnsupportedVersion(header.version));
    }

    let coords = IVec3::new(header.chunk_x, header.chunk_y, header.chunk_z);
    let tier_count = header.tier_count as usize;

    // --- Tier headers ---
    let mut tier_headers = Vec::with_capacity(tier_count);
    for _ in 0..tier_count {
        let mut th_bytes = [0u8; std::mem::size_of::<RkfTierHeader>()];
        reader.read_exact(&mut th_bytes)?;
        let th: RkfTierHeader = *bytemuck::from_bytes(&th_bytes);
        tier_headers.push(th);
    }

    // --- Tier data ---
    let mut grids = Vec::with_capacity(tier_count);
    let mut total_bricks = 0u32;

    for th in &tier_headers {
        // Read compressed size
        let mut len_bytes = [0u8; 4];
        reader.read_exact(&mut len_bytes)?;
        let compressed_len = u32::from_le_bytes(len_bytes) as usize;

        // Read compressed data
        let mut compressed = vec![0u8; compressed_len];
        reader.read_exact(&mut compressed)?;

        // Decompress
        let uncompressed = lz4_flex::decompress_size_prepended(&compressed)
            .map_err(|e| ChunkIoError::DecompressError(e.to_string()))?;

        let dims = UVec3::new(th.dims_x, th.dims_y, th.dims_z);
        let total_cells = (dims.x as usize) * (dims.y as usize) * (dims.z as usize);
        let occ_words = total_cells.div_ceil(16);
        let occ_bytes = occ_words * 4;
        let slot_bytes = total_cells * 4;

        // Validate minimum data size (occupancy + slots, bricks may be zero)
        let min_size = occ_bytes + slot_bytes;
        if uncompressed.len() < min_size {
            return Err(ChunkIoError::InvalidData(format!(
                "tier {} data too short: {} < {} (occ={} + slots={})",
                th.tier,
                uncompressed.len(),
                min_size,
                occ_bytes,
                slot_bytes,
            )));
        }

        // Parse occupancy — copy into aligned buffer (LZ4 output may have alignment 1)
        let mut occupancy = vec![0u32; occ_words];
        let occ_dst: &mut [u8] = bytemuck::cast_slice_mut(&mut occupancy);
        occ_dst.copy_from_slice(&uncompressed[..occ_bytes]);

        // Parse slots — copy into aligned buffer
        let mut slots = vec![0u32; total_cells];
        let slot_dst: &mut [u8] = bytemuck::cast_slice_mut(&mut slots);
        slot_dst.copy_from_slice(&uncompressed[occ_bytes..occ_bytes + slot_bytes]);

        // Parse bricks — copy into aligned Brick structs
        let brick_data = &uncompressed[occ_bytes + slot_bytes..];
        let brick_size = std::mem::size_of::<Brick>();
        if brick_data.len() % brick_size != 0 {
            return Err(ChunkIoError::InvalidData(format!(
                "tier {} brick data not aligned: {} bytes is not a multiple of {}",
                th.tier,
                brick_data.len(),
                brick_size,
            )));
        }
        let num_bricks = brick_data.len() / brick_size;
        let mut bricks = Vec::with_capacity(num_bricks);
        for i in 0..num_bricks {
            let start = i * brick_size;
            let mut brick = Brick::default();
            bytemuck::bytes_of_mut(&mut brick)
                .copy_from_slice(&brick_data[start..start + brick_size]);
            bricks.push(brick);
        }

        // Reconstruct grid
        let grid = SparseGrid::from_raw_parts(dims, occupancy, slots).ok_or_else(|| {
            ChunkIoError::InvalidData(format!(
                "tier {} grid reconstruction failed for dims {:?}",
                th.tier, dims,
            ))
        })?;

        total_bricks += num_bricks as u32;
        grids.push(TierGrid {
            tier: th.tier as u8,
            grid,
            bricks,
        });
    }

    Ok(Chunk {
        coords,
        grids,
        brick_count: total_bricks,
    })
}

/// Save a chunk to a file path.
pub fn save_chunk_file(chunk: &Chunk, path: &Path) -> Result<(), ChunkIoError> {
    let mut file = std::fs::File::create(path)?;
    save_chunk(chunk, &mut file)
}

/// Load a chunk from a file path.
pub fn load_chunk_file(path: &Path) -> Result<Chunk, ChunkIoError> {
    let mut file = std::fs::File::open(path)?;
    load_chunk(&mut file)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell_state::CellState;
    use crate::sparse_grid::EMPTY_SLOT;
    use crate::voxel::VoxelSample;
    use std::io::Cursor;

    /// Create a test brick with a recognizable pattern at voxel (0,0,0).
    fn make_test_brick(material_id: u16) -> Brick {
        let mut brick = Brick::default();
        brick.set(0, 0, 0, VoxelSample::new(-0.5, material_id, 128, 7, 0));
        brick.set(7, 7, 7, VoxelSample::new(0.25, material_id, 64, 3, 0));
        brick
    }

    /// Create a simple single-tier chunk with a few bricks.
    fn make_test_chunk() -> Chunk {
        let dims = UVec3::new(4, 4, 4);
        let mut grid = SparseGrid::new(dims);

        // Place 3 bricks
        let brick0 = make_test_brick(1);
        let brick1 = make_test_brick(2);
        let brick2 = make_test_brick(3);

        grid.set_cell_state(0, 0, 0, CellState::Surface);
        grid.set_brick_slot(0, 0, 0, 0); // local slot 0

        grid.set_cell_state(1, 0, 0, CellState::Surface);
        grid.set_brick_slot(1, 0, 0, 1); // local slot 1

        grid.set_cell_state(3, 3, 3, CellState::Surface);
        grid.set_brick_slot(3, 3, 3, 2); // local slot 2

        grid.set_cell_state(2, 2, 2, CellState::Interior);
        // Interior cells have no brick

        Chunk {
            coords: IVec3::new(10, -5, 3),
            grids: vec![TierGrid {
                tier: 1,
                grid,
                bricks: vec![brick0, brick1, brick2],
            }],
            brick_count: 3,
        }
    }

    // ------ Header size checks ------

    #[test]
    fn header_size() {
        assert_eq!(
            std::mem::size_of::<RkfHeader>(),
            32,
            "RkfHeader must be exactly 32 bytes"
        );
        assert_eq!(
            std::mem::size_of::<RkfTierHeader>(),
            16,
            "RkfTierHeader must be exactly 16 bytes"
        );
    }

    // ------ Single-tier roundtrip ------

    #[test]
    fn chunk_roundtrip() {
        let chunk = make_test_chunk();
        let mut buf = Vec::new();
        save_chunk(&chunk, &mut buf).unwrap();

        let mut cursor = Cursor::new(&buf);
        let loaded = load_chunk(&mut cursor).unwrap();

        assert_eq!(loaded.coords, chunk.coords);
        assert_eq!(loaded.grids.len(), 1);
        assert_eq!(loaded.brick_count, 3);

        let tg = &loaded.grids[0];
        assert_eq!(tg.tier, 1);
        assert_eq!(tg.bricks.len(), 3);
        assert_eq!(tg.grid.dimensions(), UVec3::new(4, 4, 4));

        // Verify cell states
        assert_eq!(tg.grid.cell_state(0, 0, 0), CellState::Surface);
        assert_eq!(tg.grid.cell_state(1, 0, 0), CellState::Surface);
        assert_eq!(tg.grid.cell_state(3, 3, 3), CellState::Surface);
        assert_eq!(tg.grid.cell_state(2, 2, 2), CellState::Interior);
        assert_eq!(tg.grid.cell_state(0, 1, 0), CellState::Empty);

        // Verify brick slots
        assert_eq!(tg.grid.brick_slot(0, 0, 0), Some(0));
        assert_eq!(tg.grid.brick_slot(1, 0, 0), Some(1));
        assert_eq!(tg.grid.brick_slot(3, 3, 3), Some(2));

        // Verify brick data
        assert_eq!(tg.bricks[0].sample(0, 0, 0).material_id(), 1);
        assert_eq!(tg.bricks[1].sample(0, 0, 0).material_id(), 2);
        assert_eq!(tg.bricks[2].sample(0, 0, 0).material_id(), 3);

        // Verify corner voxel
        let v = tg.bricks[0].sample(7, 7, 7);
        assert_eq!(v.material_id(), 1);
        assert_eq!(v.blend_weight(), 64);
        assert_eq!(v.secondary_id(), 3);
    }

    // ------ Empty chunk roundtrip ------

    #[test]
    fn empty_chunk_roundtrip() {
        let chunk = Chunk::new(IVec3::new(0, 0, 0));
        let mut buf = Vec::new();
        save_chunk(&chunk, &mut buf).unwrap();

        let mut cursor = Cursor::new(&buf);
        let loaded = load_chunk(&mut cursor).unwrap();

        assert_eq!(loaded.coords, IVec3::ZERO);
        assert_eq!(loaded.grids.len(), 0);
        assert_eq!(loaded.brick_count, 0);
    }

    // ------ Multi-tier roundtrip ------

    #[test]
    fn multi_tier_roundtrip() {
        let mut grids = Vec::new();

        // Tier 1: small grid with 1 brick
        {
            let dims = UVec3::new(2, 2, 2);
            let mut grid = SparseGrid::new(dims);
            grid.set_cell_state(0, 0, 0, CellState::Surface);
            grid.set_brick_slot(0, 0, 0, 0);
            grids.push(TierGrid {
                tier: 1,
                grid,
                bricks: vec![make_test_brick(10)],
            });
        }

        // Tier 2: larger grid with 2 bricks
        {
            let dims = UVec3::new(3, 3, 3);
            let mut grid = SparseGrid::new(dims);
            grid.set_cell_state(0, 0, 0, CellState::Surface);
            grid.set_brick_slot(0, 0, 0, 0);
            grid.set_cell_state(2, 2, 2, CellState::Surface);
            grid.set_brick_slot(2, 2, 2, 1);
            grids.push(TierGrid {
                tier: 2,
                grid,
                bricks: vec![make_test_brick(20), make_test_brick(30)],
            });
        }

        let chunk = Chunk {
            coords: IVec3::new(-1, 0, 1),
            grids,
            brick_count: 3,
        };

        let mut buf = Vec::new();
        save_chunk(&chunk, &mut buf).unwrap();

        let mut cursor = Cursor::new(&buf);
        let loaded = load_chunk(&mut cursor).unwrap();

        assert_eq!(loaded.coords, IVec3::new(-1, 0, 1));
        assert_eq!(loaded.grids.len(), 2);
        assert_eq!(loaded.brick_count, 3);

        // Tier 1
        assert_eq!(loaded.grids[0].tier, 1);
        assert_eq!(loaded.grids[0].bricks.len(), 1);
        assert_eq!(loaded.grids[0].grid.dimensions(), UVec3::new(2, 2, 2));
        assert_eq!(
            loaded.grids[0].bricks[0].sample(0, 0, 0).material_id(),
            10
        );

        // Tier 2
        assert_eq!(loaded.grids[1].tier, 2);
        assert_eq!(loaded.grids[1].bricks.len(), 2);
        assert_eq!(loaded.grids[1].grid.dimensions(), UVec3::new(3, 3, 3));
        assert_eq!(
            loaded.grids[1].bricks[0].sample(0, 0, 0).material_id(),
            20
        );
        assert_eq!(
            loaded.grids[1].bricks[1].sample(0, 0, 0).material_id(),
            30
        );
        assert_eq!(loaded.grids[1].grid.brick_slot(2, 2, 2), Some(1));
    }

    // ------ Invalid magic rejected ------

    #[test]
    fn invalid_magic_rejected() {
        let mut buf = vec![0u8; 64];
        buf[0..4].copy_from_slice(b"BAD\0");
        let mut cursor = Cursor::new(&buf);
        let result = load_chunk(&mut cursor);
        assert!(matches!(result, Err(ChunkIoError::InvalidMagic)));
    }

    // ------ Unsupported version ------

    #[test]
    fn unsupported_version_rejected() {
        let header = RkfHeader {
            magic: RKF_MAGIC,
            version: 99,
            chunk_x: 0,
            chunk_y: 0,
            chunk_z: 0,
            tier_count: 0,
            flags: 0,
            _reserved: 0,
        };
        let buf = bytemuck::bytes_of(&header).to_vec();
        let mut cursor = Cursor::new(&buf);
        let result = load_chunk(&mut cursor);
        assert!(matches!(result, Err(ChunkIoError::UnsupportedVersion(99))));
    }

    // ------ load_into_pool ------

    #[test]
    fn load_into_pool() {
        let mut chunk = make_test_chunk();
        let mut pool: BrickPool = crate::brick_pool::Pool::new(64);

        let global_slots = chunk.load_into_pool(&mut pool).unwrap();

        // Should have allocated 3 bricks
        assert_eq!(global_slots.len(), 3);
        assert_eq!(pool.allocated_count(), 3);

        // Verify brick data was copied into pool
        assert_eq!(pool.get(global_slots[0]).sample(0, 0, 0).material_id(), 1);
        assert_eq!(pool.get(global_slots[1]).sample(0, 0, 0).material_id(), 2);
        assert_eq!(pool.get(global_slots[2]).sample(0, 0, 0).material_id(), 3);

        // Verify grid slots were updated to global indices
        let tg = &chunk.grids[0];
        assert_eq!(tg.grid.brick_slot(0, 0, 0), Some(global_slots[0]));
        assert_eq!(tg.grid.brick_slot(1, 0, 0), Some(global_slots[1]));
        assert_eq!(tg.grid.brick_slot(3, 3, 3), Some(global_slots[2]));

        // Interior cell should still have no slot
        assert_eq!(tg.grid.brick_slot(2, 2, 2), None);
    }

    #[test]
    fn load_into_pool_exhaustion() {
        let mut chunk = make_test_chunk(); // 3 bricks
        let mut pool: BrickPool = crate::brick_pool::Pool::new(2); // Only 2 slots

        let result = chunk.load_into_pool(&mut pool);
        assert!(result.is_err());
        if let Err(PopulateError::PoolExhausted { allocated_so_far }) = result {
            assert_eq!(allocated_so_far, 2);
        } else {
            panic!("expected PoolExhausted error");
        }
    }

    // ------ File roundtrip ------

    #[test]
    fn chunk_file_roundtrip() {
        let chunk = make_test_chunk();
        let dir = std::env::temp_dir();
        let path = dir.join("rkf_test_chunk_roundtrip.rkf");

        save_chunk_file(&chunk, &path).unwrap();
        let loaded = load_chunk_file(&path).unwrap();

        assert_eq!(loaded.coords, chunk.coords);
        assert_eq!(loaded.brick_count, chunk.brick_count);
        assert_eq!(loaded.grids.len(), chunk.grids.len());
        assert_eq!(
            loaded.grids[0].bricks[0].sample(0, 0, 0).material_id(),
            1
        );

        // Clean up
        let _ = std::fs::remove_file(&path);
    }

    // ------ LZ4 compression reduces size ------

    #[test]
    fn lz4_compression_reduces_size() {
        // A chunk with bricks full of default data (lots of repeated infinity values)
        // should compress well.
        let dims = UVec3::new(4, 4, 4);
        let mut grid = SparseGrid::new(dims);
        let mut bricks = Vec::new();

        // Fill 10 bricks with default data (very compressible)
        for i in 0..10u32 {
            let x = i % 4;
            let y = (i / 4) % 4;
            let z = i / 16;
            grid.set_cell_state(x, y, z, CellState::Surface);
            grid.set_brick_slot(x, y, z, i);
            bricks.push(Brick::default());
        }

        let chunk = Chunk {
            coords: IVec3::ZERO,
            grids: vec![TierGrid {
                tier: 1,
                grid,
                bricks,
            }],
            brick_count: 10,
        };

        let mut buf = Vec::new();
        save_chunk(&chunk, &mut buf).unwrap();

        // Uncompressed size: 10 bricks * 4096 bytes = 40960 bytes just for bricks
        // Plus occupancy + slots. The compressed file should be much smaller.
        let uncompressed_brick_bytes = 10 * 4096;
        assert!(
            buf.len() < uncompressed_brick_bytes,
            "compressed size {} should be less than uncompressed brick data {} bytes",
            buf.len(),
            uncompressed_brick_bytes,
        );
    }

    // ------ Truncated data rejected ------

    #[test]
    fn truncated_header_rejected() {
        let buf = vec![0u8; 10]; // Too short for header
        let mut cursor = Cursor::new(&buf);
        let result = load_chunk(&mut cursor);
        assert!(matches!(result, Err(ChunkIoError::Io(_))));
    }

    // ------ from_raw_parts validation ------

    #[test]
    fn sparse_grid_from_raw_parts_valid() {
        let dims = UVec3::new(4, 4, 4);
        let grid = SparseGrid::new(dims);
        let occ = grid.occupancy_data().to_vec();
        let slots = grid.slot_data().to_vec();
        let rebuilt = SparseGrid::from_raw_parts(dims, occ, slots);
        assert!(rebuilt.is_some());
        let rebuilt = rebuilt.unwrap();
        assert_eq!(rebuilt.dimensions(), dims);
    }

    #[test]
    fn sparse_grid_from_raw_parts_invalid() {
        let dims = UVec3::new(4, 4, 4);
        // Wrong occupancy length
        let result = SparseGrid::from_raw_parts(dims, vec![0u32; 1], vec![EMPTY_SLOT; 64]);
        assert!(result.is_none());
        // Wrong slot length
        let result = SparseGrid::from_raw_parts(dims, vec![0u32; 4], vec![EMPTY_SLOT; 10]);
        assert!(result.is_none());
    }

    // ------ Recount bricks ------

    #[test]
    fn recount_bricks() {
        let mut chunk = make_test_chunk();
        chunk.brick_count = 0; // intentionally wrong
        chunk.recount_bricks();
        assert_eq!(chunk.brick_count, 3);
    }
}
