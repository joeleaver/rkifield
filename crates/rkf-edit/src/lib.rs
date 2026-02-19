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
//! - Analytic SDF primitives (sphere, box, capsule, cylinder, torus, plane)
//! - Sculpt brushes (add, remove, smooth, paint material)
//! - Undo/redo stack backed by edit journal
//! - Per-chunk append-only edit journals (`.rkj` files)
//! - `CompactEditOp` (64 bytes) for efficient journal entries
//!
//! # Modules
//!
//! - [`types`] — Edit operation enums and GPU-compatible [`types::EditParams`] struct
//! - [`pipeline`] — [`pipeline::CsgEditPipeline`] GPU compute pipeline for CSG edits
//! - [`brush`] — Sculpting [`brush::Brush`] with type, shape, and parameter presets
//! - [`edit_op`] — CPU-side orchestration: AABB overlap, brick allocation, GPU dispatch prep

#![warn(missing_docs)]

pub mod brush;
pub mod edit_op;
pub mod journal;
pub mod pipeline;
pub mod types;
pub mod undo;
