//! Sequence system — advances Sequence components each frame.
//!
//! Designed to run as a LateUpdate system. Processes each entity with a
//! [`Sequence`] component, advancing the timer, executing completed steps,
//! and removing the component when all steps finish.

use super::command_queue::CommandQueue;
use super::game_store::GameStore;
use super::registry::GameplayRegistry;
use super::sequence::{Sequence, SequenceStep, StepStartValues};
use crate::components::Transform;
use glam::{Quat, Vec3};
use rkf_core::WorldPosition;

/// A deferred blueprint spawn request collected during step execution.
struct DeferredBlueprintSpawn {
    /// Entity that requested the spawn (used to read position).
    source_entity: hecs::Entity,
    /// Blueprint name to look up in the catalog.
    blueprint_name: String,
}

/// Advance all Sequence components by `delta_time`.
///
/// This is designed to be called as a LateUpdate system. It processes each
/// entity with a Sequence component, advancing the timer and executing
/// completed steps. When a sequence finishes, the Sequence component is
/// removed via the command queue.
pub fn tick_sequences(
    world: &mut hecs::World,
    commands: &mut CommandQueue,
    store: &mut GameStore,
    delta_time: f32,
) {
    tick_sequences_with_registry(world, commands, store, delta_time, None);
}

/// Like [`tick_sequences`], but with access to the gameplay registry for
/// blueprint spawning.
pub fn tick_sequences_with_registry(
    world: &mut hecs::World,
    commands: &mut CommandQueue,
    store: &mut GameStore,
    delta_time: f32,
    registry: Option<&GameplayRegistry>,
) {
    // Collect entities with sequences to avoid borrow issues
    let entities: Vec<(hecs::Entity, Sequence)> = world
        .query::<&Sequence>()
        .iter()
        .filter(|(_, seq)| !seq.is_complete())
        .map(|(e, seq)| (e, seq.clone()))
        .collect();

    let mut deferred_spawns: Vec<DeferredBlueprintSpawn> = Vec::new();

    for (entity, mut seq) in entities {
        advance_sequence(
            entity,
            &mut seq,
            world,
            commands,
            store,
            delta_time,
            &mut deferred_spawns,
        );

        if seq.is_complete() {
            // Remove the Sequence component when done
            commands.remove::<Sequence>(entity);
        } else {
            // Write back the advanced sequence
            commands.insert(entity, seq);
        }
    }

    // Process deferred blueprint spawns (requires &mut World + registry).
    if let Some(registry) = registry {
        for spawn_req in deferred_spawns {
            let position = world
                .get::<&Transform>(spawn_req.source_entity)
                .map(|t| t.position.to_vec3())
                .unwrap_or(Vec3::ZERO);

            if let Some(blueprint) = registry.blueprint_catalog.get(&spawn_req.blueprint_name) {
                if let Err(e) =
                    super::blueprint::spawn_from_blueprint(world, blueprint, position, registry)
                {
                    log::warn!(
                        "SpawnBlueprint '{}' failed: {}",
                        spawn_req.blueprint_name,
                        e
                    );
                }
            } else {
                log::warn!(
                    "SpawnBlueprint: blueprint '{}' not found in catalog",
                    spawn_req.blueprint_name
                );
            }
        }
    } else if !deferred_spawns.is_empty() {
        log::warn!(
            "SpawnBlueprint steps skipped: no GameplayRegistry provided to tick_sequences"
        );
    }
}

/// Process a single sequence, advancing through steps by `delta_time`.
fn advance_sequence(
    entity: hecs::Entity,
    seq: &mut Sequence,
    world: &hecs::World,
    commands: &mut CommandQueue,
    store: &mut GameStore,
    delta_time: f32,
    deferred_spawns: &mut Vec<DeferredBlueprintSpawn>,
) {
    seq.timer += delta_time;

    while !seq.is_complete() {
        // If the current step is a Repeat, expand it inline before processing.
        if let SequenceStep::Repeat { count, steps } = &seq.steps[seq.current] {
            let count = *count;
            let inner_steps = steps.clone();
            seq.steps.remove(seq.current);
            let mut insert_pos = seq.current;
            for _ in 0..count {
                for step in &inner_steps {
                    seq.steps.insert(insert_pos, step.clone());
                    insert_pos += 1;
                }
            }
            // If count was 0, we removed the Repeat and inserted nothing.
            // Continue the loop to process whatever is now at seq.current.
            continue;
        }

        let step = &seq.steps[seq.current];
        let step_duration = step.duration();

        // Capture start values when an interpolated step first begins.
        capture_start_values_if_needed(entity, step, &mut seq.start_values, world);

        if seq.timer >= step_duration {
            // Step complete — execute it at t=1.0
            execute_step(
                entity,
                step,
                1.0,
                &seq.start_values,
                world,
                commands,
                store,
                deferred_spawns,
            );
            seq.timer -= step_duration;
            seq.current += 1;
            // Clear start values for the next step.
            seq.start_values = StepStartValues::default();
        } else {
            // Step in progress — compute progress and apply
            let t = if step_duration > 0.0 {
                seq.timer / step_duration
            } else {
                1.0
            };
            execute_step(
                entity,
                step,
                t,
                &seq.start_values,
                world,
                commands,
                store,
                deferred_spawns,
            );
            break;
        }
    }
}

/// Capture the entity's current transform values when an interpolated step
/// begins, so we can lerp/slerp from start to target.
fn capture_start_values_if_needed(
    entity: hecs::Entity,
    step: &SequenceStep,
    start: &mut StepStartValues,
    world: &hecs::World,
) {
    match step {
        SequenceStep::MoveTo { .. } if start.position.is_none() => {
            if let Ok(transform) = world.get::<&Transform>(entity) {
                start.position = Some(transform.position.clone());
            }
        }
        SequenceStep::MoveBy { .. } if start.move_by_base.is_none() => {
            if let Ok(transform) = world.get::<&Transform>(entity) {
                start.move_by_base = Some(transform.position.clone());
            }
        }
        SequenceStep::RotateTo { .. } if start.rotation.is_none() => {
            if let Ok(transform) = world.get::<&Transform>(entity) {
                start.rotation = Some(transform.rotation);
            }
        }
        SequenceStep::RotateBy { .. } if start.rotate_by_base.is_none() => {
            if let Ok(transform) = world.get::<&Transform>(entity) {
                start.rotate_by_base = Some(transform.rotation);
            }
        }
        SequenceStep::ScaleTo { .. } if start.scale.is_none() => {
            if let Ok(transform) = world.get::<&Transform>(entity) {
                start.scale = Some(transform.scale);
            }
        }
        _ => {}
    }
}

/// Lerp between two WorldPositions using f64 arithmetic for precision.
fn lerp_world_position(from: &WorldPosition, to: &WorldPosition, t: f32) -> WorldPosition {
    // Compute displacement from→to in f64, scale by t, apply to `from`.
    let displacement = to.relative_to(from);
    let offset = displacement * t;
    from.translate(offset)
}

/// Execute a single step at progress `t` (0.0..1.0).
///
/// Instant steps (Emit, SetState, Despawn, etc.) only fire when `t >= 1.0`.
/// Interpolated steps (MoveTo, MoveBy, RotateTo, RotateBy, ScaleTo) lerp/slerp
/// based on eased `t` from captured start values to the target.
#[allow(clippy::too_many_arguments)]
fn execute_step(
    entity: hecs::Entity,
    step: &SequenceStep,
    t: f32,
    start_values: &StepStartValues,
    world: &hecs::World,
    commands: &mut CommandQueue,
    store: &mut GameStore,
    deferred_spawns: &mut Vec<DeferredBlueprintSpawn>,
) {
    match step {
        SequenceStep::Wait { .. } => {
            // Nothing to do — just wait for the timer to advance.
        }

        SequenceStep::MoveTo {
            target, ease, ..
        } => {
            if let Some(ref start_pos) = start_values.position {
                let eased_t = ease.eval(t);
                let interpolated = lerp_world_position(start_pos, target, eased_t);
                if let Ok(transform) = world.get::<&Transform>(entity) {
                    let mut new_t = Transform { ..*transform };
                    new_t.position = interpolated;
                    commands.insert(entity, new_t);
                }
            } else if t >= 1.0 {
                // Fallback: no start values captured (no Transform on entity at step start).
                if let Ok(transform) = world.get::<&Transform>(entity) {
                    let mut new_t = Transform { ..*transform };
                    new_t.position = target.clone();
                    commands.insert(entity, new_t);
                }
            }
        }

        SequenceStep::MoveBy {
            offset, ease, ..
        } => {
            if let Some(ref base_pos) = start_values.move_by_base {
                let eased_t = ease.eval(t);
                let current_offset = *offset * eased_t;
                let interpolated = base_pos.translate(current_offset);
                if let Ok(transform) = world.get::<&Transform>(entity) {
                    let mut new_t = Transform { ..*transform };
                    new_t.position = interpolated;
                    commands.insert(entity, new_t);
                }
            } else if t >= 1.0 {
                if let Ok(transform) = world.get::<&Transform>(entity) {
                    let mut new_t = Transform { ..*transform };
                    new_t.position.local += *offset;
                    commands.insert(entity, new_t);
                }
            }
        }

        SequenceStep::RotateTo {
            target, ease, ..
        } => {
            if let Some(start_rot) = start_values.rotation {
                let eased_t = ease.eval(t);
                let interpolated = start_rot.slerp(*target, eased_t);
                if let Ok(transform) = world.get::<&Transform>(entity) {
                    let mut new_t = Transform { ..*transform };
                    new_t.rotation = interpolated;
                    commands.insert(entity, new_t);
                }
            } else if t >= 1.0 {
                if let Ok(transform) = world.get::<&Transform>(entity) {
                    let mut new_t = Transform { ..*transform };
                    new_t.rotation = *target;
                    commands.insert(entity, new_t);
                }
            }
        }

        SequenceStep::RotateBy {
            rotation, ease, ..
        } => {
            if let Some(base_rot) = start_values.rotate_by_base {
                let eased_t = ease.eval(t);
                // Slerp from identity to the relative rotation, then apply to base.
                let partial_rot = Quat::IDENTITY.slerp(*rotation, eased_t);
                let interpolated = partial_rot * base_rot;
                if let Ok(transform) = world.get::<&Transform>(entity) {
                    let mut new_t = Transform { ..*transform };
                    new_t.rotation = interpolated;
                    commands.insert(entity, new_t);
                }
            } else if t >= 1.0 {
                if let Ok(transform) = world.get::<&Transform>(entity) {
                    let cur_rotation = transform.rotation;
                    let mut new_t = Transform { ..*transform };
                    new_t.rotation = *rotation * cur_rotation;
                    commands.insert(entity, new_t);
                }
            }
        }

        SequenceStep::ScaleTo {
            target, ease, ..
        } => {
            if let Some(start_scale) = start_values.scale {
                let eased_t = ease.eval(t);
                let interpolated = start_scale.lerp(*target, eased_t);
                if let Ok(transform) = world.get::<&Transform>(entity) {
                    let mut new_t = Transform { ..*transform };
                    new_t.scale = interpolated;
                    commands.insert(entity, new_t);
                }
            } else if t >= 1.0 {
                if let Ok(transform) = world.get::<&Transform>(entity) {
                    let mut new_t = Transform { ..*transform };
                    new_t.scale = *target;
                    commands.insert(entity, new_t);
                }
            }
        }

        SequenceStep::Emit { name } => {
            if t >= 1.0 {
                store.emit(name, Some(entity), None);
            }
        }

        SequenceStep::EmitWith { name, data } => {
            if t >= 1.0 {
                store.emit(name, Some(entity), Some(data.clone()));
            }
        }

        SequenceStep::EmitFrom {
            name,
            source,
            data,
        } => {
            if t >= 1.0 {
                // Validate that the source entity still exists; substitute None
                // if it has been despawned to avoid dangling references.
                let validated_source = match *source {
                    Some(e) if world.contains(e) => Some(e),
                    Some(_) => None,
                    None => None,
                };
                store.emit(name, validated_source, data.clone());
            }
        }

        SequenceStep::SetState { key, value } => {
            if t >= 1.0 {
                store.set(key, value.clone());
            }
        }

        SequenceStep::SpawnBlueprint { name } => {
            if t >= 1.0 {
                deferred_spawns.push(DeferredBlueprintSpawn {
                    source_entity: entity,
                    blueprint_name: name.clone(),
                });
            }
        }

        SequenceStep::Despawn => {
            if t >= 1.0 {
                commands.despawn(entity);
            }
        }

        SequenceStep::Repeat { .. } => {
            // Repeat steps are expanded inline by advance_sequence before
            // execute_step is called. If we somehow reach here, it means
            // the Repeat is the very first step — handle it in the next
            // advance_sequence iteration.
        }
    }
}

