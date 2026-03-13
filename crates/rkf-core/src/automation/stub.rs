//! Stub implementation of [`AutomationApi`].

use super::trait_def::AutomationApi;
use super::types::*;

/// Stub implementation that returns [`AutomationError::NotImplemented`] for
/// every method.
///
/// Used during testing and as a placeholder before `rkf-runtime` provides a
/// real implementation.
pub struct StubAutomationApi;

impl AutomationApi for StubAutomationApi {
    fn screenshot(&self, _width: u32, _height: u32) -> AutomationResult<Vec<u8>> {
        Err(AutomationError::NotImplemented("screenshot"))
    }

    fn scene_graph(&self) -> AutomationResult<SceneGraphSnapshot> {
        Err(AutomationError::NotImplemented("scene_graph"))
    }

    fn entity_inspect(&self, _entity_id: &str) -> AutomationResult<EntitySnapshot> {
        Err(AutomationError::NotImplemented("entity_inspect"))
    }

    fn render_stats(&self) -> AutomationResult<RenderStats> {
        Err(AutomationError::NotImplemented("render_stats"))
    }

    fn asset_status(&self) -> AutomationResult<AssetStatusReport> {
        Err(AutomationError::NotImplemented("asset_status"))
    }

    fn read_log(&self, _lines: usize) -> AutomationResult<Vec<LogEntry>> {
        Err(AutomationError::NotImplemented("read_log"))
    }

    fn camera_state(&self) -> AutomationResult<CameraSnapshot> {
        Err(AutomationError::NotImplemented("camera_state"))
    }

    fn brick_pool_stats(&self) -> AutomationResult<BrickPoolStats> {
        Err(AutomationError::NotImplemented("brick_pool_stats"))
    }

    fn spatial_query(
        &self,
        _chunk: [i32; 3],
        _local: [f32; 3],
    ) -> AutomationResult<SpatialQueryResult> {
        Err(AutomationError::NotImplemented("spatial_query"))
    }

    fn entity_spawn(&self, _def: EntityDef) -> AutomationResult<String> {
        Err(AutomationError::NotImplemented("entity_spawn"))
    }

    fn entity_despawn(&self, _entity_id: &str) -> AutomationResult<()> {
        Err(AutomationError::NotImplemented("entity_despawn"))
    }

    fn entity_set_component(
        &self,
        _entity_id: &str,
        _component: ComponentDef,
    ) -> AutomationResult<()> {
        Err(AutomationError::NotImplemented("entity_set_component"))
    }

    fn material_set(&self, _id: u16, _material: MaterialDef) -> AutomationResult<()> {
        Err(AutomationError::NotImplemented("material_set"))
    }

    fn brush_apply(&self, _op: serde_json::Value) -> AutomationResult<()> {
        Err(AutomationError::NotImplemented("brush_apply"))
    }

    fn scene_load(&self, _path: &str) -> AutomationResult<()> {
        Err(AutomationError::NotImplemented("scene_load"))
    }

    fn scene_save(&self, _path: &str) -> AutomationResult<()> {
        Err(AutomationError::NotImplemented("scene_save"))
    }

    fn camera_set(
        &self,
        _chunk: [i32; 3],
        _local: [f32; 3],
        _rotation: [f32; 4],
    ) -> AutomationResult<()> {
        Err(AutomationError::NotImplemented("camera_set"))
    }

    fn quality_preset(&self, _preset: QualityPreset) -> AutomationResult<()> {
        Err(AutomationError::NotImplemented("quality_preset"))
    }

    fn execute_command(&self, _command: &str) -> AutomationResult<String> {
        Err(AutomationError::NotImplemented("execute_command"))
    }

    // The v2 methods have default implementations in the trait that return
    // Err("... not supported").  StubAutomationApi inherits those defaults and
    // does not need to override them; the trait defaults are tested directly.
}
