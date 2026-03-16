//! Edit processing helpers for the engine loop (steps e0-e7).
//!
//! Extracted from `engine_loop.rs` to keep file sizes manageable.

use std::sync::{Arc, Mutex};

use crate::automation::SharedState;
use crate::editor_state::EditorState;
use crate::engine::EditorEngine;

/// Compute a default voxel_size for a primitive with the given scale.
///
/// Targets ~16 bricks on the longest axis (128 voxels), clamped to [0.005, ∞).
pub(crate) fn compute_default_voxel_size(
    primitive: &rkf_core::SdfPrimitive,
    scale: glam::Vec3,
) -> f32 {
    let scaled_half = crate::engine::primitive_half_extents(primitive) * scale;
    let max_extent = scaled_half.x.max(scaled_half.y).max(scaled_half.z);
    // Target ~16 bricks on the longest axis.
    (max_extent * 2.0 / 16.0 / 8.0).max(0.005)
}

/// Estimate voxelization results for a given primitive, scale, and voxel_size.
///
/// Returns `(grid_dims, estimated_surface_bricks, memory_bytes)`.
/// Used by the UI to show a preview before committing to voxelization.
pub fn estimate_voxelization(
    primitive: &rkf_core::SdfPrimitive,
    scale: glam::Vec3,
    voxel_size: f32,
) -> (glam::UVec3, u32, u64) {
    let scaled_half = crate::engine::primitive_half_extents(primitive) * scale;
    let margin = voxel_size * 2.0;
    let aabb_size = (scaled_half + glam::Vec3::splat(margin)) * 2.0;
    let brick_size = voxel_size * 8.0;
    let dims = glam::UVec3::new(
        (aabb_size.x / brick_size).ceil().max(1.0) as u32,
        (aabb_size.y / brick_size).ceil().max(1.0) as u32,
        (aabb_size.z / brick_size).ceil().max(1.0) as u32,
    );
    // Rough estimate: surface bricks ≈ 2 * (xy + xz + yz) faces of the grid,
    // capped at total grid volume.
    let total = dims.x * dims.y * dims.z;
    let surface_est = (2 * (dims.x * dims.y + dims.x * dims.z + dims.y * dims.z)).min(total);
    let mem_bytes = surface_est as u64 * 4096; // 4KB per brick
    (dims, surface_est, mem_bytes)
}

/// Rebuild a render scene clone from hecs (authoritative source of truth).
pub(crate) fn rebuild_scene_clone(es: &EditorState) -> rkf_core::scene::Scene {
    let mut merged = es.world.build_render_scene();
    for obj in &mut merged.objects {
        obj.aabb = crate::placement::compute_object_local_aabb(obj);
    }
    merged
}

/// e0: Process pending convert-to-voxel (analytical -> geometry-first).
pub(crate) fn process_convert_to_voxel(
    entity_uuid: uuid::Uuid,
    voxel_size: f32,
    engine: &mut EditorEngine,
    editor_state: &Arc<Mutex<EditorState>>,
    scene_clone: &mut rkf_core::scene::Scene,
    frame_dirty_objects: &mut Vec<u32>,
) {
    if let Ok(mut es) = editor_state.lock() {
        // Resolve entity UUID → SDF object ID.
        let obj_id = match es.world.entity_records()
            .find(|(uid, _)| **uid == entity_uuid)
            .and_then(|(_, r)| r.sdf_object_id)
        {
            Some(id) => id,
            None => return,
        };
        let (primitive, material_id, obj_scale) = {
            let sdf_tree = match es.world.get::<rkf_runtime::components::SdfTree>(entity_uuid) {
                Ok(s) => s,
                Err(_) => return,
            };
            match sdf_tree.root.sdf_source.clone() {
                rkf_core::SdfSource::Analytical { primitive, material_id } => {
                    let scale = es.world.scale(entity_uuid).unwrap_or(glam::Vec3::ONE);
                    (primitive, material_id, scale)
                }
                _ => return, // Already voxelized or none.
            }
        };

        // Auto-compute voxel_size if not specified (0.0 = auto).
        let voxel_size = if voxel_size <= 0.0 {
            compute_default_voxel_size(&primitive, obj_scale)
        } else {
            voxel_size
        };

        if let Some((handle, vs, grid_aabb, _count)) =
            engine.convert_to_geometry_first(
                &primitive, material_id as u8, voxel_size, obj_id,
                Some(obj_scale),
            )
        {
            // Write hecs (authoritative source of truth).
            if let Ok(mut sdf_tree) = es.world.get_mut::<rkf_runtime::components::SdfTree>(entity_uuid) {
                sdf_tree.root.sdf_source = rkf_core::SdfSource::Voxelized {
                    brick_map_handle: handle,
                    voxel_size: vs,
                    aabb: grid_aabb,
                };
                sdf_tree.aabb = grid_aabb;
            }
            let _ = es.world.set_scale(entity_uuid, glam::Vec3::ONE);

            engine.reupload_brick_data();
            let map_data = engine.cpu_brick_map_alloc.as_slice();
            if !map_data.is_empty() {
                engine.gpu_scene.upload_brick_maps(
                    &engine.ctx.device, &engine.ctx.queue, map_data,
                );
            }
            log::info!(
                "Converted object {} to voxel (vs={}, scale {:?} baked in)",
                obj_id, voxel_size, obj_scale,
            );
        }

        // Rebuild scene_clone from hecs.
        *scene_clone = rebuild_scene_clone(&es);
        frame_dirty_objects.push(obj_id);
    }
}

/// e1b: Process pending material remap.
pub(crate) fn process_remap_material(
    entity_uuid: uuid::Uuid,
    from_material: u16,
    to_material: u16,
    engine: &mut EditorEngine,
    editor_state: &Arc<Mutex<EditorState>>,
    scene_clone: &rkf_core::scene::Scene,
    frame_dirty_objects: &mut Vec<u32>,
) {
    use rkf_core::scene_node::SdfSource;
    // Resolve UUID -> SDF object ID.
    let sdf_obj_id = if let Ok(es) = editor_state.lock() {
        es.world.entity_records()
            .find(|(uid, _)| **uid == entity_uuid)
            .and_then(|(_, r)| r.sdf_object_id)
    } else {
        None
    };
    let Some(object_id) = sdf_obj_id else { return; };
    if let Some(obj) = scene_clone.objects.iter().find(|o| o.id == object_id) {
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
            frame_dirty_objects.push(object_id);
        }
    }
}

/// e1c: Process pending primitive material change.
pub(crate) fn process_set_primitive_material(
    entity_uuid: uuid::Uuid,
    new_mat_id: u16,
    editor_state: &Arc<Mutex<EditorState>>,
    scene_clone: &mut rkf_core::scene::Scene,
    frame_dirty_objects: &mut Vec<u32>,
) {
    let sdf_obj_id = if let Ok(es) = editor_state.lock() {
        // Write hecs (authoritative source of truth).
        if let Ok(mut sdf_tree) = es.world.get_mut::<rkf_runtime::components::SdfTree>(entity_uuid) {
            if let rkf_core::SdfSource::Analytical { ref mut material_id, .. } =
                sdf_tree.root.sdf_source
            {
                *material_id = new_mat_id;
            }
        }
        // Rebuild scene_clone from hecs.
        *scene_clone = rebuild_scene_clone(&es);
        es.world.entity_records()
            .find(|(uid, _)| **uid == entity_uuid)
            .and_then(|(_, r)| r.sdf_object_id)
    } else {
        None
    };
    if let Some(oid) = sdf_obj_id {
        frame_dirty_objects.push(oid);
    }
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
            let entity_uuid = es.world.find_by_sdf_id(sculpt_edits[0].object_id)
                .unwrap_or(uuid::Uuid::nil());
            es.sculpt_undo_accumulator = Some(
                crate::editor_state::SculptUndoAccumulator {
                    object_id: entity_uuid,
                    captured_slots: std::collections::HashSet::new(),
                    snapshots: Vec::new(),
                },
            );
        }

        // Build temporary Scene from hecs for the sculpt engine.
        let mut temp_scene = es.world.build_render_scene();
        let mut undo_acc = es.sculpt_undo_accumulator.take();
        let _modified = engine.apply_sculpt_edits(
            &mut temp_scene,
            sculpt_edits,
            undo_acc.as_mut(),
        );
        es.sculpt_undo_accumulator = undo_acc;

        // Sync sculpt results (temp_scene) back to hecs (authoritative).
        for edit in sculpt_edits {
            if let Some(obj) = temp_scene.objects.iter().find(|o| o.id == edit.object_id) {
                if let Some(entity_uuid) = es.world.find_by_sdf_id(edit.object_id) {
                    if let Ok(mut sdf_tree) = es.world.get_mut::<rkf_runtime::components::SdfTree>(entity_uuid) {
                        sdf_tree.root = obj.root_node.clone();
                        sdf_tree.aabb = obj.aabb;
                    }
                    // Scale may have been reset by auto-voxelization.
                    let _ = es.world.set_scale(entity_uuid, obj.scale);
                }
            }
        }

        // Rebuild the render clone from hecs.
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
            let entity_uuid = es.world.find_by_sdf_id(paint_edits[0].object_id)
                .unwrap_or(uuid::Uuid::nil());
            es.paint_undo_accumulator = Some(
                crate::editor_state::SculptUndoAccumulator {
                    object_id: entity_uuid,
                    captured_slots: std::collections::HashSet::new(),
                    snapshots: Vec::new(),
                },
            );
        }

        let mut undo_acc = es.paint_undo_accumulator.take();
        // Build temporary Scene from hecs for the paint engine.
        let mut temp_scene = es.world.build_render_scene();
        engine.apply_paint_edits(
            &mut temp_scene,
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
        let entity_uuid_for_undo = if let Ok(es_ref) = editor_state.lock() {
            es_ref.world.find_by_sdf_id(req.object_id).unwrap_or(uuid::Uuid::nil())
        } else {
            uuid::Uuid::nil()
        };
        let mut undo_acc = crate::editor_state::SculptUndoAccumulator {
            object_id: entity_uuid_for_undo,
            captured_slots: std::collections::HashSet::new(),
            snapshots: Vec::new(),
        };

        if let Ok(mut es) = editor_state.lock() {
            // Build temporary Scene from hecs for the sculpt engine.
            let mut temp_scene = es.world.build_render_scene();
            engine.apply_sculpt_edits(
                &mut temp_scene,
                &[edit_request],
                Some(&mut undo_acc),
            );

            // Push undo entry immediately.
            if !undo_acc.snapshots.is_empty() {
                es.undo.push(crate::undo::UndoAction {
                    kind: crate::undo::UndoActionKind::SculptStroke {
                        object_id: entity_uuid_for_undo,
                        geometry_snapshots: undo_acc.snapshots,
                    },
                    timestamp_ms: 0,
                    description: format!("MCP sculpt ({}) on obj {}", req.mode, req.object_id),
                });
            }

            // Sync sculpt results back to hecs (authoritative).
            if let Some(obj) = temp_scene.objects.iter().find(|o| o.id == req.object_id) {
                if let Some(entity_uuid) = es.world.find_by_sdf_id(req.object_id) {
                    if let Ok(mut sdf_tree) = es.world.get_mut::<rkf_runtime::components::SdfTree>(entity_uuid) {
                        sdf_tree.root = obj.root_node.clone();
                        sdf_tree.aabb = obj.aabb;
                    }
                    let _ = es.world.set_scale(entity_uuid, obj.scale);
                }
            }

            // Rebuild scene_clone from hecs.
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
