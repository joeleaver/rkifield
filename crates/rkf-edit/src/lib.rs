//! # rkf-edit
//!
//! Procedural editing operations for the RKIField SDF engine.
//!
//! This crate implements GPU-accelerated CSG operations and sculpting tools
//! that modify voxel data in-place.
//!
//! Key components:
//! - GPU CSG operations (union, subtraction, intersection) with SDF primitives
//! - Analytic SDF primitives (sphere, box, capsule, cylinder, torus, plane)
//! - Sculpt brushes (add, remove, smooth, paint material)
//! - Undo/redo stack (per-object in v2)
//!
//! # Modules
//!
//! - [`types`] — Edit operation enums and GPU-compatible [`types::EditParams`] struct
//! - [`pipeline`] — [`pipeline::CsgEditPipeline`] GPU compute pipeline for CSG edits
//! - [`brush`] — Sculpting [`brush::Brush`] with type, shape, and parameter presets

#![warn(missing_docs)]

pub mod brush;
pub mod edit_op;
pub mod pipeline;
pub mod transform_ops;
pub mod types;
pub mod undo;
