//! # rkf-edit
//!
//! Procedural editing operations for the RKIField SDF engine.
//!
//! This crate implements GPU-accelerated CSG operations and sculpting tools
//! that modify voxel data in-place. All edits are recorded in an append-only
//! journal for undo/redo and persistence.
//!
//! Key components:
//! - GPU CSG operations (union, subtraction, intersection) with SDF primitives
//! - Sculpt brushes (add, remove, smooth, paint material)
//! - Undo/redo stack backed by edit journal
//! - Per-chunk append-only edit journals (`.rkj` files)
//! - `CompactEditOp` (64 bytes) for efficient journal entries

#![warn(missing_docs)]
