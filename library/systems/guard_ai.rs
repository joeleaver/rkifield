//! Guard AI system — patrol/chase/return state machine.

use rkf_runtime::behavior::system_context::SystemContext;
use rkf_runtime::components::Transform;
use crate::components::{GuardAi, GuardState};

/// Guard AI state machine: patrol near origin, chase detected targets, return when lost.
pub fn guard_ai_system(ctx: &mut SystemContext) {
    let dt = ctx.delta_time();
    let total_time = ctx.total_time() as f32;

    // Dummy player target position — in a real game this would come from a Player query.
    let player_pos = glam::Vec3::new(0.0, 0.0, 0.0);

    let guards: Vec<_> = ctx
        .query::<(&GuardAi, &Transform)>()
        .iter()
        .map(|(entity, (ai, transform))| {
            let pos = transform.position.local;
            (entity, ai.state, ai.patrol_origin, ai.patrol_radius, ai.chase_speed, ai.detection_range, pos)
        })
        .collect();

    for (entity, state, patrol_origin, patrol_radius, chase_speed, detection_range, pos) in guards {
        let dist_to_player = (player_pos - pos).length();
        let dist_to_origin = (patrol_origin - pos).length();

        let (new_state, new_pos) = match state {
            GuardState::Patrol => {
                let angle = total_time * 0.5;
                let target = patrol_origin + glam::Vec3::new(
                    angle.cos() * patrol_radius,
                    0.0,
                    angle.sin() * patrol_radius,
                );
                let dir = target - pos;
                let step = if dir.length() > 0.01 {
                    dir.normalize() * chase_speed * 0.5 * dt
                } else {
                    glam::Vec3::ZERO
                };
                let moved = pos + step;

                if dist_to_player < detection_range {
                    (GuardState::Chase, moved)
                } else {
                    (GuardState::Patrol, moved)
                }
            }
            GuardState::Chase => {
                let dir = player_pos - pos;
                let step = if dir.length() > 0.01 {
                    dir.normalize() * chase_speed * dt
                } else {
                    glam::Vec3::ZERO
                };
                let moved = pos + step;

                if dist_to_player > detection_range * 1.5 {
                    (GuardState::Return, moved)
                } else {
                    (GuardState::Chase, moved)
                }
            }
            GuardState::Return => {
                let dir = patrol_origin - pos;
                let step = if dir.length() > 0.5 {
                    dir.normalize() * chase_speed * 0.75 * dt
                } else {
                    glam::Vec3::ZERO
                };
                let moved = pos + step;

                if dist_to_origin < 1.0 {
                    (GuardState::Patrol, moved)
                } else {
                    (GuardState::Return, moved)
                }
            }
        };

        ctx.insert(
            entity,
            GuardAi {
                state: new_state,
                patrol_origin,
                patrol_radius,
                chase_speed,
                detection_range,
            },
        );
        ctx.insert(
            entity,
            Transform {
                position: new_pos.into(),
                ..Transform::default()
            },
        );
    }
}
