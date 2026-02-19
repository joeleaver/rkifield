//! Per-chunk append-only edit journal (`.rkj` files).
//!
//! Each chunk maintains an [`EditJournal`] of [`CompactEditOp`] entries
//! (64 bytes each).  Journals are replayed on chunk load to reconstruct
//! edits on top of the base `.rkf` data.  When a journal grows past
//! the compaction threshold (1000 entries), the edits should be baked
//! into the base `.rkf` and the journal cleared.
//!
//! # File Format (`.rkj`)
//!
//! ```text
//! Offset  Size  Field
//! 0       4     magic: u32 = 0x004A4B52 ("RKJ\0" little-endian)
//! 4       4     version: u32 = 1
//! 8       4     entry_count: u32
//! 12      64*N  entries: [CompactEditOp; N]
//! ```

use bytemuck::{Pod, Zeroable};
use half::f16;
use std::io::{self, Read, Seek, SeekFrom, Write};

use crate::edit_op::EditOp;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Magic number for `.rkj` files: "RKJ\0" in little-endian.
pub const RKJ_MAGIC: u32 = 0x004A4B52;

/// Current journal format version.
pub const RKJ_VERSION: u32 = 1;

/// Compaction threshold: journal should be compacted when entry count exceeds this.
pub const COMPACTION_THRESHOLD: usize = 1000;

// ---------------------------------------------------------------------------
// RkjHeader
// ---------------------------------------------------------------------------

/// `.rkj` file header — 12 bytes.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct RkjHeader {
    /// Magic number ([`RKJ_MAGIC`]).
    pub magic: u32,
    /// Format version ([`RKJ_VERSION`]).
    pub version: u32,
    /// Number of [`CompactEditOp`] entries following the header.
    pub entry_count: u32,
}

// Compile-time size assertions.
const _: () = assert!(std::mem::size_of::<RkjHeader>() == 12);

// ---------------------------------------------------------------------------
// CompactEditOp — 64 bytes fixed-size journal entry
// ---------------------------------------------------------------------------

/// Compact edit operation — 64 bytes, fixed size, suitable for disk journaling.
///
/// All fields are packed for minimal size. Float16 values are stored as raw
/// `u16` bits.  Quaternion components are normalized `i16` values.
#[repr(C)]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct CompactEditOp {
    // -- Identity (8 bytes) --
    /// Operation ID (monotonically increasing within a chunk).
    pub op_id: u32,
    /// Flags: bit 0 = has_rotation, bit 1 = has_secondary_material, bit 2 = is_multi_frame.
    pub flags: u16,
    /// Edit type discriminant (see [`EditType`]).
    pub edit_type: u8,
    /// Shape type discriminant (see [`ShapeType`]).
    pub shape_type: u8,

    // -- Spatial (28 bytes) --
    /// Edit center position `[x, y, z]` in chunk-local space.
    pub position: [f32; 3],
    /// Rotation quaternion as normalized `i16` values (multiply by 1/32767 to recover).
    pub rotation: [i16; 4],
    /// Shape half-extents / dimensions as `f16` bit patterns.
    pub dimensions: [u16; 3],
    /// Padding to align spatial section.
    pub _pad_spatial: u16,

    // -- Parameters (12 bytes) --
    /// Brush strength as `f16` bit pattern.
    pub strength: u16,
    /// Smooth-min blend radius as `f16` bit pattern.
    pub blend_k: u16,
    /// Falloff curve discriminant (see [`FalloffCurve`]).
    pub falloff: u8,
    /// Primary material ID — low byte.
    pub material_id_lo: u8,
    /// Primary material ID — high byte.
    pub material_id_hi: u8,
    /// Secondary material ID.
    pub secondary_id: u8,
    /// RGBA8 color `[r, g, b, a]`.
    pub color: [u8; 4],

    // -- Metadata (16 bytes) --
    /// Timestamp (e.g., Unix millis or frame number).
    pub timestamp: u64,
    /// Source identifier (user/tool/agent that produced this edit).
    pub source_id: u32,
    /// Reserved for future use.
    pub _reserved: u32,
}

// Compile-time size assertion.
const _: () = assert!(std::mem::size_of::<CompactEditOp>() == 64);

impl CompactEditOp {
    /// Create a [`CompactEditOp`] from an [`EditOp`] and metadata.
    pub fn from_edit_op(
        op: &EditOp,
        op_id: u32,
        timestamp: u64,
        source_id: u32,
    ) -> Self {
        let edit_type = op.brush.edit_type() as u8;
        let shape_type = op.brush.shape_type() as u8;

        // Flags.
        let has_rotation = op.rotation != glam::Quat::IDENTITY;
        let has_secondary = op.brush.secondary_id != 0;
        let flags: u16 = (has_rotation as u16) | ((has_secondary as u16) << 1);

        // Rotation as normalized i16.
        let rotation = [
            (op.rotation.x * 32767.0) as i16,
            (op.rotation.y * 32767.0) as i16,
            (op.rotation.z * 32767.0) as i16,
            (op.rotation.w * 32767.0) as i16,
        ];

        // Dimensions: use brush radius for all axes (sphere-like).
        let r = op.brush.radius;
        let dimensions = [
            f16::from_f32(r).to_bits(),
            f16::from_f32(r).to_bits(),
            f16::from_f32(r).to_bits(),
        ];

        let material_id = op.brush.material_id;

        Self {
            op_id,
            flags,
            edit_type,
            shape_type,
            position: [op.position.x, op.position.y, op.position.z],
            rotation,
            dimensions,
            _pad_spatial: 0,
            strength: f16::from_f32(op.brush.strength).to_bits(),
            blend_k: f16::from_f32(op.brush.blend_k).to_bits(),
            falloff: op.brush.falloff as u8,
            material_id_lo: (material_id & 0xFF) as u8,
            material_id_hi: ((material_id >> 8) & 0xFF) as u8,
            secondary_id: op.brush.secondary_id,
            color: [0, 0, 0, 0],
            timestamp,
            source_id,
            _reserved: 0,
        }
    }

    /// Extract the primary material ID as `u16`.
    #[inline]
    pub fn material_id(&self) -> u16 {
        (self.material_id_lo as u16) | ((self.material_id_hi as u16) << 8)
    }

    /// Set the primary material ID from `u16`.
    #[inline]
    pub fn set_material_id(&mut self, id: u16) {
        self.material_id_lo = (id & 0xFF) as u8;
        self.material_id_hi = ((id >> 8) & 0xFF) as u8;
    }

    /// Returns `true` if the rotation flag is set.
    #[inline]
    pub fn has_rotation(&self) -> bool {
        self.flags & 1 != 0
    }

    /// Returns `true` if the secondary material flag is set.
    #[inline]
    pub fn has_secondary_material(&self) -> bool {
        self.flags & 2 != 0
    }

    /// Returns `true` if the multi-frame flag is set.
    #[inline]
    pub fn is_multi_frame(&self) -> bool {
        self.flags & 4 != 0
    }

    /// Recover strength as `f32`.
    #[inline]
    pub fn strength_f32(&self) -> f32 {
        f16::from_bits(self.strength).to_f32()
    }

    /// Recover blend_k as `f32`.
    #[inline]
    pub fn blend_k_f32(&self) -> f32 {
        f16::from_bits(self.blend_k).to_f32()
    }

    /// Recover dimensions as `[f32; 3]`.
    #[inline]
    pub fn dimensions_f32(&self) -> [f32; 3] {
        [
            f16::from_bits(self.dimensions[0]).to_f32(),
            f16::from_bits(self.dimensions[1]).to_f32(),
            f16::from_bits(self.dimensions[2]).to_f32(),
        ]
    }
}

// ---------------------------------------------------------------------------
// JournalError
// ---------------------------------------------------------------------------

/// Errors produced by journal I/O operations.
#[derive(Debug, thiserror::Error)]
pub enum JournalError {
    /// Underlying I/O error.
    #[error("journal I/O error: {0}")]
    Io(#[from] io::Error),
    /// Invalid magic number in the file header.
    #[error("invalid journal magic: expected 0x{expected:08X}, got 0x{got:08X}")]
    InvalidMagic {
        /// Expected magic value.
        expected: u32,
        /// Actual magic value read.
        got: u32,
    },
    /// Unsupported journal version.
    #[error("unsupported journal version: expected {expected}, got {got}")]
    UnsupportedVersion {
        /// Expected version.
        expected: u32,
        /// Actual version read.
        got: u32,
    },
}

// ---------------------------------------------------------------------------
// EditJournal
// ---------------------------------------------------------------------------

/// In-memory edit journal for a single chunk.
///
/// Entries are appended during editing and persisted to `.rkj` files.
/// On chunk load, the journal is replayed to reconstruct edits on top
/// of the base `.rkf` data.
pub struct EditJournal {
    entries: Vec<CompactEditOp>,
}

impl EditJournal {
    /// Create an empty journal.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Append an edit operation to the journal.
    pub fn append(&mut self, op: CompactEditOp) {
        self.entries.push(op);
    }

    /// Number of entries in the journal.
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the journal has no entries.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Slice of all entries.
    #[inline]
    pub fn entries(&self) -> &[CompactEditOp] {
        &self.entries
    }

    /// Check if the journal exceeds the compaction threshold.
    pub fn needs_compaction(&self) -> bool {
        self.entries.len() > COMPACTION_THRESHOLD
    }

    /// Clear all entries (e.g., after compaction bakes them into the base `.rkf`).
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Write the entire journal to a writer as a `.rkj` file.
    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<(), JournalError> {
        let header = RkjHeader {
            magic: RKJ_MAGIC,
            version: RKJ_VERSION,
            entry_count: self.entries.len() as u32,
        };
        writer.write_all(bytemuck::bytes_of(&header))?;
        if !self.entries.is_empty() {
            writer.write_all(bytemuck::cast_slice(&self.entries))?;
        }
        Ok(())
    }

    /// Read a `.rkj` file, replacing the current journal contents.
    pub fn read_from<R: Read>(&mut self, reader: &mut R) -> Result<(), JournalError> {
        // Read header.
        let mut header_bytes = [0u8; 12];
        reader.read_exact(&mut header_bytes)?;
        let header: &RkjHeader = bytemuck::from_bytes(&header_bytes);

        if header.magic != RKJ_MAGIC {
            return Err(JournalError::InvalidMagic {
                expected: RKJ_MAGIC,
                got: header.magic,
            });
        }
        if header.version != RKJ_VERSION {
            return Err(JournalError::UnsupportedVersion {
                expected: RKJ_VERSION,
                got: header.version,
            });
        }

        let count = header.entry_count as usize;
        if count == 0 {
            self.entries.clear();
            return Ok(());
        }

        // Read entries.
        let mut entry_bytes = vec![0u8; count * 64];
        reader.read_exact(&mut entry_bytes)?;
        let entries: &[CompactEditOp] = bytemuck::cast_slice(&entry_bytes);
        self.entries = entries.to_vec();

        Ok(())
    }

    /// Append new entries to an existing `.rkj` file.
    ///
    /// Updates the header's `entry_count` and appends the new entries at the end.
    /// If the file is empty (writer at position 0), writes a fresh header + entries.
    pub fn append_to_file<W: Write + Seek>(
        &self,
        writer: &mut W,
        new_entries: &[CompactEditOp],
    ) -> Result<(), JournalError> {
        if new_entries.is_empty() {
            return Ok(());
        }

        // Check current file size by seeking to end.
        let end_pos = writer.seek(SeekFrom::End(0))?;

        if end_pos == 0 {
            // Empty file — write fresh header + entries.
            let header = RkjHeader {
                magic: RKJ_MAGIC,
                version: RKJ_VERSION,
                entry_count: new_entries.len() as u32,
            };
            writer.write_all(bytemuck::bytes_of(&header))?;
            writer.write_all(bytemuck::cast_slice(new_entries))?;
        } else {
            // Derive existing entry count from file size (can't read with Write trait).
            // File layout: 12-byte header + N * 64-byte entries.
            let existing_count = ((end_pos - 12) / 64) as u32;
            let new_count = existing_count + new_entries.len() as u32;

            // Write updated header.
            let header = RkjHeader {
                magic: RKJ_MAGIC,
                version: RKJ_VERSION,
                entry_count: new_count,
            };
            writer.seek(SeekFrom::Start(0))?;
            writer.write_all(bytemuck::bytes_of(&header))?;

            // Seek to end and append entries.
            writer.seek(SeekFrom::End(0))?;
            writer.write_all(bytemuck::cast_slice(new_entries))?;
        }

        Ok(())
    }
}

impl Default for EditJournal {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brush::Brush;
    use crate::edit_op::EditOp;
    use glam::{Quat, Vec3};
    use std::io::Cursor;

    // -- Size assertions --

    #[test]
    fn compact_edit_op_is_64_bytes() {
        assert_eq!(std::mem::size_of::<CompactEditOp>(), 64);
    }

    #[test]
    fn rkj_header_is_12_bytes() {
        assert_eq!(std::mem::size_of::<RkjHeader>(), 12);
    }

    // -- Pod/Zeroable --

    #[test]
    fn compact_edit_op_pod_zeroable() {
        let op: CompactEditOp = bytemuck::Zeroable::zeroed();
        let bytes = bytemuck::bytes_of(&op);
        assert_eq!(bytes.len(), 64);
        assert!(bytes.iter().all(|&b| b == 0));

        // Round-trip through bytes.
        let _: &CompactEditOp = bytemuck::from_bytes(bytes);
    }

    #[test]
    fn rkj_header_pod_zeroable() {
        let h: RkjHeader = bytemuck::Zeroable::zeroed();
        let bytes = bytemuck::bytes_of(&h);
        assert_eq!(bytes.len(), 12);
    }

    // -- from_edit_op --

    #[test]
    fn from_edit_op_basic_fields() {
        let op = EditOp {
            brush: Brush::add_sphere(0.5, 42),
            position: Vec3::new(1.0, 2.0, 3.0),
            rotation: Quat::IDENTITY,
        };

        let compact = CompactEditOp::from_edit_op(&op, 7, 1000, 99);

        assert_eq!(compact.op_id, 7);
        assert_eq!(compact.timestamp, 1000);
        assert_eq!(compact.source_id, 99);
        assert_eq!(compact.edit_type, op.brush.edit_type() as u8);
        assert_eq!(compact.shape_type, op.brush.shape_type() as u8);
        assert!((compact.position[0] - 1.0).abs() < 1e-6);
        assert!((compact.position[1] - 2.0).abs() < 1e-6);
        assert!((compact.position[2] - 3.0).abs() < 1e-6);
    }

    // -- material_id get/set --

    #[test]
    fn material_id_roundtrip() {
        let mut op: CompactEditOp = bytemuck::Zeroable::zeroed();
        op.set_material_id(0);
        assert_eq!(op.material_id(), 0);

        op.set_material_id(42);
        assert_eq!(op.material_id(), 42);

        op.set_material_id(256);
        assert_eq!(op.material_id(), 256);

        op.set_material_id(u16::MAX);
        assert_eq!(op.material_id(), u16::MAX);
    }

    #[test]
    fn material_id_from_edit_op() {
        let op = EditOp {
            brush: Brush::add_sphere(0.5, 1234),
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
        };
        let compact = CompactEditOp::from_edit_op(&op, 0, 0, 0);
        assert_eq!(compact.material_id(), 1234);
    }

    // -- flags --

    #[test]
    fn flags_identity_rotation_not_set() {
        let op = EditOp {
            brush: Brush::add_sphere(0.5, 1),
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
        };
        let compact = CompactEditOp::from_edit_op(&op, 0, 0, 0);
        assert!(!compact.has_rotation());
    }

    #[test]
    fn flags_nonidentity_rotation_set() {
        let op = EditOp {
            brush: Brush::add_sphere(0.5, 1),
            position: Vec3::ZERO,
            rotation: Quat::from_rotation_y(std::f32::consts::FRAC_PI_4),
        };
        let compact = CompactEditOp::from_edit_op(&op, 0, 0, 0);
        assert!(compact.has_rotation());
    }

    #[test]
    fn flags_secondary_material() {
        let mut brush = Brush::add_sphere(0.5, 1);
        brush.secondary_id = 5;
        let op = EditOp {
            brush,
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
        };
        let compact = CompactEditOp::from_edit_op(&op, 0, 0, 0);
        assert!(compact.has_secondary_material());
    }

    #[test]
    fn flags_no_secondary_material() {
        let op = EditOp {
            brush: Brush::add_sphere(0.5, 1),
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
        };
        let compact = CompactEditOp::from_edit_op(&op, 0, 0, 0);
        assert!(!compact.has_secondary_material());
    }

    #[test]
    fn is_multi_frame_default_false() {
        let op: CompactEditOp = bytemuck::Zeroable::zeroed();
        assert!(!op.is_multi_frame());
    }

    // -- f16 value recovery --

    #[test]
    fn strength_roundtrip() {
        let op = EditOp {
            brush: Brush::add_sphere(0.5, 1),
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
        };
        let compact = CompactEditOp::from_edit_op(&op, 0, 0, 0);
        // Brush strength defaults to 1.0.
        assert!((compact.strength_f32() - 1.0).abs() < 0.01);
    }

    #[test]
    fn blend_k_roundtrip() {
        let brush = Brush::add_sphere(1.0, 1); // blend_k = 0.3
        let op = EditOp {
            brush,
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
        };
        let compact = CompactEditOp::from_edit_op(&op, 0, 0, 0);
        assert!((compact.blend_k_f32() - 0.3).abs() < 0.01);
    }

    #[test]
    fn dimensions_roundtrip() {
        let brush = Brush::add_sphere(2.5, 1);
        let op = EditOp {
            brush,
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
        };
        let compact = CompactEditOp::from_edit_op(&op, 0, 0, 0);
        let dims = compact.dimensions_f32();
        assert!((dims[0] - 2.5).abs() < 0.01);
        assert!((dims[1] - 2.5).abs() < 0.01);
        assert!((dims[2] - 2.5).abs() < 0.01);
    }

    // -- Journal operations --

    #[test]
    fn journal_new_is_empty() {
        let j = EditJournal::new();
        assert!(j.is_empty());
        assert_eq!(j.len(), 0);
        assert!(!j.needs_compaction());
    }

    #[test]
    fn journal_append_and_len() {
        let mut j = EditJournal::new();
        let op: CompactEditOp = bytemuck::Zeroable::zeroed();
        j.append(op);
        assert_eq!(j.len(), 1);
        assert!(!j.is_empty());
        j.append(op);
        assert_eq!(j.len(), 2);
    }

    #[test]
    fn journal_entries_slice() {
        let mut j = EditJournal::new();
        let mut op: CompactEditOp = bytemuck::Zeroable::zeroed();
        op.op_id = 42;
        j.append(op);
        assert_eq!(j.entries()[0].op_id, 42);
    }

    #[test]
    fn journal_clear() {
        let mut j = EditJournal::new();
        let op: CompactEditOp = bytemuck::Zeroable::zeroed();
        j.append(op);
        j.append(op);
        j.clear();
        assert!(j.is_empty());
        assert_eq!(j.len(), 0);
    }

    #[test]
    fn journal_needs_compaction() {
        let mut j = EditJournal::new();
        let op: CompactEditOp = bytemuck::Zeroable::zeroed();
        for _ in 0..=COMPACTION_THRESHOLD {
            j.append(op);
        }
        assert!(j.needs_compaction());
    }

    #[test]
    fn journal_below_compaction_threshold() {
        let mut j = EditJournal::new();
        let op: CompactEditOp = bytemuck::Zeroable::zeroed();
        for _ in 0..COMPACTION_THRESHOLD {
            j.append(op);
        }
        assert!(!j.needs_compaction());
    }

    // -- Write / Read round-trip --

    #[test]
    fn write_read_roundtrip_empty() {
        let j = EditJournal::new();
        let mut buf = Vec::new();
        j.write_to(&mut buf).unwrap();

        // Should be header only (12 bytes).
        assert_eq!(buf.len(), 12);

        let mut j2 = EditJournal::new();
        j2.read_from(&mut Cursor::new(&buf)).unwrap();
        assert!(j2.is_empty());
    }

    #[test]
    fn write_read_roundtrip_with_entries() {
        let mut j = EditJournal::new();

        let op1 = CompactEditOp::from_edit_op(
            &EditOp {
                brush: Brush::add_sphere(1.0, 42),
                position: Vec3::new(1.0, 2.0, 3.0),
                rotation: Quat::IDENTITY,
            },
            0,
            100,
            1,
        );
        let op2 = CompactEditOp::from_edit_op(
            &EditOp {
                brush: Brush::subtract_sphere(0.5),
                position: Vec3::new(4.0, 5.0, 6.0),
                rotation: Quat::from_rotation_z(1.0),
            },
            1,
            200,
            2,
        );

        j.append(op1);
        j.append(op2);

        let mut buf = Vec::new();
        j.write_to(&mut buf).unwrap();

        // Should be header (12) + 2 entries (128) = 140 bytes.
        assert_eq!(buf.len(), 12 + 2 * 64);

        let mut j2 = EditJournal::new();
        j2.read_from(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(j2.len(), 2);

        // Verify fields survived round-trip.
        assert_eq!(j2.entries()[0].op_id, 0);
        assert_eq!(j2.entries()[0].timestamp, 100);
        assert_eq!(j2.entries()[0].source_id, 1);
        assert_eq!(j2.entries()[0].material_id(), 42);
        assert!((j2.entries()[0].position[0] - 1.0).abs() < 1e-6);

        assert_eq!(j2.entries()[1].op_id, 1);
        assert_eq!(j2.entries()[1].timestamp, 200);
        assert_eq!(j2.entries()[1].source_id, 2);
    }

    // -- Read error cases --

    #[test]
    fn read_invalid_magic_returns_error() {
        let mut header_bytes = [0u8; 12];
        // Write bad magic.
        header_bytes[0..4].copy_from_slice(&0xDEADBEEFu32.to_le_bytes());
        header_bytes[4..8].copy_from_slice(&RKJ_VERSION.to_le_bytes());
        header_bytes[8..12].copy_from_slice(&0u32.to_le_bytes());

        let mut j = EditJournal::new();
        let result = j.read_from(&mut Cursor::new(&header_bytes));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, JournalError::InvalidMagic { .. }),
            "expected InvalidMagic, got: {err}"
        );
    }

    #[test]
    fn read_invalid_version_returns_error() {
        let mut header_bytes = [0u8; 12];
        header_bytes[0..4].copy_from_slice(&RKJ_MAGIC.to_le_bytes());
        header_bytes[4..8].copy_from_slice(&99u32.to_le_bytes()); // bad version
        header_bytes[8..12].copy_from_slice(&0u32.to_le_bytes());

        let mut j = EditJournal::new();
        let result = j.read_from(&mut Cursor::new(&header_bytes));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, JournalError::UnsupportedVersion { .. }),
            "expected UnsupportedVersion, got: {err}"
        );
    }

    #[test]
    fn read_truncated_data_returns_io_error() {
        let mut header_bytes = [0u8; 12];
        header_bytes[0..4].copy_from_slice(&RKJ_MAGIC.to_le_bytes());
        header_bytes[4..8].copy_from_slice(&RKJ_VERSION.to_le_bytes());
        header_bytes[8..12].copy_from_slice(&1u32.to_le_bytes()); // claims 1 entry

        // But no entry data follows.
        let mut j = EditJournal::new();
        let result = j.read_from(&mut Cursor::new(&header_bytes));
        assert!(result.is_err());
    }

    // -- append_to_file --

    #[test]
    fn append_to_file_on_empty_file() {
        let j = EditJournal::new();
        let mut buf = Cursor::new(Vec::new());

        let mut op: CompactEditOp = bytemuck::Zeroable::zeroed();
        op.op_id = 10;

        j.append_to_file(&mut buf, &[op]).unwrap();

        // Read back.
        let data = buf.into_inner();
        assert_eq!(data.len(), 12 + 64);

        let mut j2 = EditJournal::new();
        j2.read_from(&mut Cursor::new(&data)).unwrap();
        assert_eq!(j2.len(), 1);
        assert_eq!(j2.entries()[0].op_id, 10);
    }

    #[test]
    fn append_to_file_on_existing_file() {
        // Write initial journal with 1 entry.
        let mut j = EditJournal::new();
        let mut op1: CompactEditOp = bytemuck::Zeroable::zeroed();
        op1.op_id = 1;
        j.append(op1);

        let mut buf = Cursor::new(Vec::new());
        j.write_to(&mut buf).unwrap();

        // Now append a second entry.
        let mut op2: CompactEditOp = bytemuck::Zeroable::zeroed();
        op2.op_id = 2;
        j.append_to_file(&mut buf, &[op2]).unwrap();

        // Read back — should have 2 entries.
        let data = buf.into_inner();
        assert_eq!(data.len(), 12 + 2 * 64);

        let mut j2 = EditJournal::new();
        j2.read_from(&mut Cursor::new(&data)).unwrap();
        assert_eq!(j2.len(), 2);
        assert_eq!(j2.entries()[0].op_id, 1);
        assert_eq!(j2.entries()[1].op_id, 2);
    }

    #[test]
    fn append_to_file_empty_entries_is_noop() {
        let j = EditJournal::new();
        let mut buf = Cursor::new(Vec::new());
        j.append_to_file(&mut buf, &[]).unwrap();
        assert!(buf.into_inner().is_empty());
    }

    // -- bytemuck::cast_slice round-trip --

    #[test]
    fn cast_slice_roundtrip() {
        let mut op: CompactEditOp = bytemuck::Zeroable::zeroed();
        op.op_id = 99;
        op.set_material_id(1000);
        op.timestamp = 12345678;

        let ops = [op];
        let bytes: &[u8] = bytemuck::cast_slice(&ops);
        assert_eq!(bytes.len(), 64);

        let ops_back: &[CompactEditOp] = bytemuck::cast_slice(bytes);
        assert_eq!(ops_back[0].op_id, 99);
        assert_eq!(ops_back[0].material_id(), 1000);
        assert_eq!(ops_back[0].timestamp, 12345678);
    }

    // -- Default impl --

    #[test]
    fn journal_default_is_empty() {
        let j = EditJournal::default();
        assert!(j.is_empty());
    }

    // -- Magic constant --

    #[test]
    fn magic_is_rkj_null() {
        // "RKJ\0" in little-endian: R=0x52, K=0x4B, J=0x4A, \0=0x00
        let bytes = RKJ_MAGIC.to_le_bytes();
        assert_eq!(bytes[0], b'R');
        assert_eq!(bytes[1], b'K');
        assert_eq!(bytes[2], b'J');
        assert_eq!(bytes[3], 0);
    }
}
