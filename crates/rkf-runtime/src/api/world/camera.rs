//! Camera entity operations.

use uuid::Uuid;

use rkf_core::WorldPosition;

use crate::components::CameraComponent;

use super::{World, WorldError};

impl World {
    // ── Camera entities ─────────────────────────────────────────────────

    /// Spawn a camera entity (ECS-only, no SDF geometry).
    ///
    /// `environment_profile` is an optional path to a `.rkenv` file.
    pub fn spawn_camera(
        &mut self,
        label: impl Into<String>,
        position: WorldPosition,
        yaw: f32,
        pitch: f32,
        fov_degrees: f32,
        environment_profile: Option<&str>,
    ) -> Uuid {
        let label = label.into();
        let uuid = self.finalize_ecs_spawn(label.clone());
        let cam = CameraComponent {
            fov_degrees,
            active: false,
            label,
            yaw,
            pitch,
            environment_profile: environment_profile.unwrap_or("").to_string(),
            ..Default::default()
        };
        let record = self.entities.get(&uuid).unwrap();
        let ecs_entity = record.ecs_entity;
        let _ = self.ecs.insert_one(ecs_entity, cam);
        // Every camera gets its own EnvironmentSettings.
        let _ = self.ecs.insert_one(ecs_entity, crate::environment::EnvironmentSettings::default());
        // Store position in both EntityRecord and hecs Transform
        if let Some(r) = self.entities.get_mut(&uuid) {
            r.position = position;
        }
        if let Ok(mut t) = self.ecs.get::<&mut crate::components::Transform>(ecs_entity) {
            t.position = position;
        }
        uuid
    }

    /// List all camera entities.
    pub fn cameras(&self) -> Vec<Uuid> {
        self.entities
            .iter()
            .filter(|(_, r)| {
                self.ecs.get::<&CameraComponent>(r.ecs_entity).is_ok()
            })
            .map(|(id, _)| *id)
            .collect()
    }

    /// Find the active camera entity, if any.
    pub fn active_camera(&self) -> Option<Uuid> {
        self.entities
            .iter()
            .find(|(_, r)| {
                self.ecs
                    .get::<&CameraComponent>(r.ecs_entity)
                    .map(|c| c.active)
                    .unwrap_or(false)
            })
            .map(|(id, _)| *id)
    }

    /// Set a camera entity as the active camera (deactivates all others).
    pub fn set_active_camera(&mut self, entity_id: Uuid) -> Result<(), WorldError> {
        let record = self
            .entities
            .get(&entity_id)
            .ok_or(WorldError::NoSuchEntity(entity_id))?;
        // Verify it has a CameraComponent
        if self.ecs.get::<&CameraComponent>(record.ecs_entity).is_err() {
            return Err(WorldError::MissingComponent(entity_id, "CameraComponent"));
        }

        // Deactivate all cameras
        let ecs_entities: Vec<hecs::Entity> = self
            .entities
            .values()
            .filter(|r| self.ecs.get::<&CameraComponent>(r.ecs_entity).is_ok())
            .map(|r| r.ecs_entity)
            .collect();
        for ee in ecs_entities {
            if let Ok(mut cam) = self.ecs.get::<&mut CameraComponent>(ee) {
                cam.active = false;
            }
        }

        // Activate target
        let record = self.entities.get(&entity_id).unwrap();
        if let Ok(mut cam) = self.ecs.get::<&mut CameraComponent>(record.ecs_entity) {
            cam.active = true;
        }
        Ok(())
    }
}
