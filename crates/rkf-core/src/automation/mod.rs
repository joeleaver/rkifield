//! Automation API — the engine's control surface for MCP tools and agents.
//!
//! This module defines the [`AutomationApi`] trait implemented by `rkf-runtime`
//! and called by `rkf-mcp`. All observation methods are safe in any mode;
//! mutation methods require editor mode.
//!
//! A [`StubAutomationApi`] is provided for testing and as a placeholder until
//! runtime implements the trait.

mod stub;
#[cfg(test)]
mod tests;
mod trait_def;
mod types;

pub use stub::StubAutomationApi;
pub use trait_def::AutomationApi;
pub use types::{
    AssetStatusReport, AutomationError, AutomationResult, BlueprintInfo, BrickPoolStats,
    CameraSnapshot, ComponentDef, ComponentInfo, EntityDef, EntityNode, EntitySnapshot, FieldInfo,
    LogEntry, LogLevel, MaterialDef, MaterialInfo, MaterialSnapshot, ObjectShapeResult,
    QualityPreset, RenderStats, SceneGraphSnapshot, ShaderInfo, SpatialQueryResult, SystemInfo,
    VoxelSliceResult,
};
