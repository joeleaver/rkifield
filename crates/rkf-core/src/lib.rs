//! # rkf-core
//!
//! Core data structures for the RKIField SDF graphics engine.
//!
//! This crate provides the foundational types shared across the engine:
//! - [`WorldPosition`] for float-precision-safe world-space coordinates
//! - Voxel sample types and brick data structures (8x8x8 voxel bricks)
//! - Brick pool management and GPU resource coordination
//! - Sparse spatial index for world occupancy
//! - Material table types and constants
//! - Resolution tier definitions and chunk geometry

#![warn(missing_docs)]

pub mod automation;
pub mod cell_state;
pub mod constants;

pub use cell_state::CellState;
