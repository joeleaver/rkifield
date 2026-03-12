//! Edit processing helpers for the engine loop (steps e0-e7).
//!
//! Extracted from `engine_loop.rs` to keep file sizes manageable.

use std::sync::{Arc, Mutex};

use crate::automation::SharedState;
use crate::editor_state::EditorState;
use crate::engine::EditorEngine;

/// Rebuild a merged scene clone from all objects in EditorState.
pub(crate) fn rebuild_scene_clone(es: &EditorState) -> rkf_core::scene::Scene {
    let mut merged = rkf_core::scene::Scene::new("merged");
    merged.objects = es.world.all_objects().cloned().collect();
    for obj in &mut merged.objects {
        obj.aabb = crate::placement::compute_object_local_aabb(obj);
    }
    merged
}

/// e0: Process pending convert-to-voxel (analytical -> geometry-first).
pub(crate) fn process_convert_to_voxel(
    obj_id: u32,
    engine: &mut EditorEngine,
    editor_state: &Arc<Mutex<EditorState>>,
    scene_clone: &mut rkf_core::scene::Scene,
    frame_dirty_objects: &mut Vec<u32>,
) {
    if let Ok(mut es) = editor_state.lock() {
        let scene = es.world.scene_mut();
        if let Some(obj) = scene.objects.iter_mut().find(|o| o.id == obj_id) {
            if let rkf_core::SdfSource::Analytical { primitive, material_id } =
                obj.root_node.sdf_source.clone()
            {
                // Bake non-uniform scale into the voxelized volume so the
                // resulting object is the same world-space size as the scaled
                // primitive, with uniform voxel density and scale reset to 1.
                let obj_scale = obj.scale;
                let scaled_half = crate::engine::primitive_half_extents(&primitive) * obj_scale;
                // Size voxels so the shortest axis still gets reasonable resolution
                // (at least ~8 voxels = 1 brick), capped so the longest axis doesn't
                // exceed ~128 voxels (16 bricks) to keep brick count manageable.
                let min_extent = scaled_half.x.min(scaled_half.y).min(scaled_half.z).max(0.001);
                let max_extent = scaled_half.x.max(scaled_half.y).max(scaled_half.z);
                let vs_from_short = min_extent * 2.0 / 8.0;   // ≥8 voxels on shortest axis
                let vs_from_long  = max_extent * 2.0 / 128.0; // ≤128 voxels on longest axis
                let voxel_size = vs_from_short.min(vs_from_long).max(0.005);
                if let Some((handle, vs, grid_aabb, _count)) =
                    engine.convert_to_geometry_first(
                        &primitive, material_id as u8, voxel_size, obj_id,
                        Some(obj_scale),
                    )
                {
                    obj.root_node.sdf_source = rkf_core::SdfSource::Voxelized {
                        brick_map_handle: handle,
                        voxel_size: vs,
                        aabb: grid_aabb,
                    };
                    obj.aabb = grid_aabb;
                    // Reset scale — shape is now baked into the voxel volume.
                    obj.scale = glam::Vec3::ONE;
                    engine.reupload_brick_data();
                    let map_data = engine.cpu_brick_map_alloc.as_slice();
                    if !map_data.is_empty() {
                        engine.gpu_scene.upload_brick_maps(
                            &engine.ctx.device, &engine.ctx.queue, map_data,
                        );
                    }
                    log::info!(
                        "Converted object {} to voxel (vs={}, scale {:?} baked in)",
                        obj.name, voxel_size, obj_scale,
                    );
                }
            }
        }

        // Rebuild scene_clone so the dirty path sees the Voxelized SdfSource
        // this frame (not the stale Analytical one captured before conversion).
        *scene_clone = rebuild_scene_clone(&es);
    }
    frame_dirty_objects.push(obj_id);
}

/// e1b: Process pending material remap.
pub(crate) fn process_remap_material(
    object_id: u64,
    from_material: u16,
    to_material: u16,
    engine: &mut EditorEngine,
    scene_clone: &rkf_core::scene::Scene,
    frame_dirty_objects: &mut Vec<u32>,
) {
    use rkf_core::scene_node::SdfSource;
    if let Some(obj) = scene_clone.objects.iter().find(|o| o.id as u64 == object_id) {
        if let SdfSource::Voxelized { brick_map_handle, .. } = &obj.root_node.sdf_source {
            let handle = *brick_map_handle;
            let dims = handle.dims;
            for bz in 0..dims.z {
                for by in 0..dims.y {
                    for bx in 0..dims.x {
                        if let Some(slot) = engine.cpu_brick_map_alloc.get_entry(&handle, bx, by, bz) {
                            if EditorEngine::is_unallocated(slot) {
                                continue;
                            }
                            let brick = engine.cpu_brick_pool.get_mut(slot);
                            for vz in 0..8u32 {
                                for vy in 0..8u32 {
                                    for vx in 0..8u32 {
                                        let mut sample = brick.sample(vx, vy, vz);
                                        if sample.material_id() == from_material {
                                            sample.set_material_id(to_material);
                                            brick.set(vx, vy, vz, sample);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            engine.reupload_brick_data();
            frame_dirty_objects.push(object_id as u32);
        }
    }
}

/// e1c: Process pending primitive material change.
pub(crate) fn process_set_primitive_material(
    object_id: u64,
    new_mat_id: u16,
    editor_state: &Arc<Mutex<EditorState>>,
    scene_clone: &mut rkf_core::scene::Scene,
    frame_dirty_objects: &mut Vec<u32>,
) {
    if let Ok(mut es) = editor_state.lock() {
        let scene = es.world.scene_mut();
        if let Some(obj) = scene.objects.iter_mut().find(|o| o.id as u64 == object_id) {
            if let rkf_core::SdfSource::Analytical { ref mut material_id, .. } =
                obj.root_node.sdf_source
            {
                *material_id = new_mat_id;
            }
        }
    }
    // Update scene_clone so the render pass sees the new material_id.
    if let Some(obj) = scene_clone.objects.iter_mut().find(|o| o.id as u64 == object_id) {
        if let rkf_core::SdfSource::Analytical { ref mut material_id, .. } =
            obj.root_node.sdf_source
        {
            *material_id = new_mat_id;
        }
    }
    frame_dirty_objects.push(object_id as u32);
}

/// e3: Process sculpt edits (CPU-side CSG, targeted GPU upload).
pub(crate) fn process_sculpt_edits(
    sculpt_edits: &[crate::sculpt::SculptEditRequest],
    engine: &mut EditorEngine,
    editor_state: &Arc<Mutex<EditorState>>,
    scene_clone: &mut rkf_core::scene::Scene,
    frame_dirty_objects: &mut Vec<u32>,
) {
    if let Ok(mut es) = editor_state.lock() {
        // Ensure undo accumulator exists for this stroke.
        if es.sculpt_undo_accumulator.is_none() {
            let obj_id = sculpt_edits[0].object_id as u64;
            es.sculpt_undo_accumulator = Some(
                crate::editor_state::SculptUndoAccumulator {
                    object_id: obj_id,
                    captured_slots: std::collections::HashSet::new(),
                    snapshots: Vec::new(),
                },
            );
        }

        let mut undo_acc = es.sculpt_undo_accumulator.take();
        let scene = es.world.scene_mut();
        let _modified = engine.apply_sculpt_edits(
            scene,
            sculpt_edits,
            undo_acc.as_mut(),
        );
        es.sculpt_undo_accumulator = undo_acc;

        // Rebuild the render clone since scene may have changed.
        *scene_clone = rebuild_scene_clone(&es);
    }
    // Mark sculpted objects dirty for incremental GPU update.
    for edit in sculpt_edits {
        frame_dirty_objects.push(edit.object_id);
    }
}

/// e3b: Process paint edits (CPU-side surface voxel modification, targeted GPU upload).
pub(crate) fn process_paint_edits(
    paint_edits: &[crate::paint::PaintEditRequest],
    engine: &mut EditorEngine,
    editor_state: &Arc<Mutex<EditorState>>,
    frame_dirty_objects: &mut Vec<u32>,
    dirty_scene: &mut bool,
) {
    if let Ok(mut es) = editor_state.lock() {
        // Ensure undo accumulator exists for this stroke.
        if es.paint_undo_accumulator.is_none() {
            let obj_id = paint_edits[0].object_id as u64;
            es.paint_undo_accumulator = Some(
                crate::editor_state::SculptUndoAccumulator {
                    object_id: obj_id,
                    captured_slots: std::collections::HashSet::new(),
                    snapshots: Vec::new(),
                },
            );
        }

        let mut undo_acc = es.paint_undo_accumulator.take();
        let scene = es.world.scene_mut();
        engine.apply_paint_edits(
            scene,
            paint_edits,
            undo_acc.as_mut(),
        );
        es.paint_undo_accumulator = undo_acc;
    }
    // Mark painted objects dirty for incremental GPU update.
    for edit in paint_edits {
        frame_dirty_objects.push(edit.object_id);
    }
    // Trigger UI refresh (materials list, etc.).
    *dirty_scene = true;
}

/// e4: Process pending voxel_slice request (CPU-side brick pool lookup).
pub(crate) fn process_voxel_slice(
    req: crate::automation::VoxelSliceRequest,
    engine: &EditorEngine,
    scene_clone: &rkf_core::scene::Scene,
    shared_state: &Arc<Mutex<SharedState>>,
) {
    let result = engine.sample_voxel_slice(scene_clone, req.object_id, req.y_coord);
    if let Ok(mut ss) = shared_state.lock() {
        match result {
            Ok(slice) => ss.voxel_slice_result = Some(slice),
            Err(e) => {
                ss.push_log(rkf_core::automation::LogLevel::Error,
                    format!("voxel_slice error: {e}"));
                // Store an empty result so the polling doesn't hang.
                ss.voxel_slice_result = Some(rkf_core::automation::VoxelSliceResult {
                    origin: [0.0, 0.0],
                    spacing: 0.0,
                    width: 0,
                    height: 0,
                    y_coord: req.y_coord,
                    distances: vec![],
                    slot_status: vec![],
                });
            }
        }
    }
}

/// e5: Process pending spatial_query request (CPU-side SDF evaluation).
pub(crate) fn process_spatial_query(
    req: crate::automation::SpatialQueryRequest,
    engine: &EditorEngine,
    scene_clone: &rkf_core::scene::Scene,
    shared_state: &Arc<Mutex<SharedState>>,
) {
    let result = engine.sample_spatial_query(scene_clone, req.world_pos);
    if let Ok(mut ss) = shared_state.lock() {
        ss.spatial_query_result = Some(result);
    }
}

/// e6: Process pending MCP sculpt request (one-shot brush hit + undo).
pub(crate) fn process_mcp_sculpt(
    req: crate::automation::McpSculptRequest,
    engine: &mut EditorEngine,
    editor_state: &Arc<Mutex<EditorState>>,
    scene_clone: &mut rkf_core::scene::Scene,
    shared_state: &Arc<Mutex<SharedState>>,
    frame_dirty_objects: &mut Vec<u32>,
) {
    let mcp_sculpt_obj_id = req.object_id;
    let sculpt_result = (|| -> Result<(), String> {
        let brush_type = match req.mode.as_str() {
            "add" => crate::sculpt::BrushType::Add,
            "subtract" => crate::sculpt::BrushType::Subtract,
            "smooth" => crate::sculpt::BrushType::Smooth,
            other => return Err(format!("invalid mode: {other}")),
        };

        let edit_request = crate::sculpt::SculptEditRequest {
            object_id: req.object_id,
            world_position: req.position,
            settings: crate::sculpt::BrushSettings {
                brush_type,
                shape: crate::sculpt::BrushShape::Sphere,
                radius: req.radius,
                strength: req.strength,
                material_id: req.material_id,
                falloff: 0.5,
            },
        };

        // Create one-shot undo accumulator.
        let mut undo_acc = crate::editor_state::SculptUndoAccumulator {
            object_id: req.object_id as u64,
            captured_slots: std::collections::HashSet::new(),
            snapshots: Vec::new(),
        };

        if let Ok(mut es) = editor_state.lock() {
            let scene = es.world.scene_mut();
            engine.apply_sculpt_edits(
                scene,
                &[edit_request],
                Some(&mut undo_acc),
            );

            // Push undo entry immediately.
            if !undo_acc.snapshots.is_empty() {
                es.undo.push(crate::undo::UndoAction {
                    kind: crate::undo::UndoActionKind::SculptStroke {
                        object_id: req.object_id as u64,
                        geometry_snapshots: undo_acc.snapshots,
                    },
                    timestamp_ms: 0,
                    description: format!("MCP sculpt ({}) on obj {}", req.mode, req.object_id),
                });
            }

            // Rebuild scene_clone since voxel data changed.
            *scene_clone = rebuild_scene_clone(&es);
        }

        Ok(())
    })();

    if let Ok(mut ss) = shared_state.lock() {
        ss.mcp_sculpt_result = Some(sculpt_result);
    }
    frame_dirty_objects.push(mcp_sculpt_obj_id);
}

/// e7: Process pending object_shape request (CPU-side brick map lookup).
pub(crate) fn process_object_shape(
    obj_id: u32,
    engine: &EditorEngine,
    scene_clone: &rkf_core::scene::Scene,
    shared_state: &Arc<Mutex<SharedState>>,
) {
    let result = engine.sample_object_shape(scene_clone, obj_id);
    if let Ok(mut ss) = shared_state.lock() {
        match result {
            Ok(shape) => ss.object_shape_result = Some(shape),
            Err(e) => {
                ss.push_log(rkf_core::automation::LogLevel::Error,
                    format!("object_shape error: {e}"));
                // Store a minimal result so polling doesn't hang.
                ss.object_shape_result = Some(rkf_core::automation::ObjectShapeResult {
                    object_id: obj_id,
                    dims: [0, 0, 0],
                    voxel_size: 0.0,
                    aabb_min: [0.0; 3],
                    aabb_max: [0.0; 3],
                    empty_count: 0,
                    interior_count: 0,
                    surface_count: 0,
                    y_slices: vec![],
                });
            }
        }
    }
}
