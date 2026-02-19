#![allow(dead_code)]
//! Showcase scene definition for rkf-game.
//!
//! Defines a rich demo scene with terrain, animated characters, static objects,
//! lights, particle emitters, and physics objects.

use glam::{Quat, Vec3};

/// Type of particle effect for an emitter.
#[derive(Debug, Clone, PartialEq)]
pub enum ParticleEffectType {
    /// Flames with volumetric density splats.
    Fire,
    /// Solid debris micro-objects.
    Debris,
    /// Screen-space rain overlay.
    Rain,
    /// Volumetric smoke plume.
    Smoke,
}

/// Type of entity in the showcase scene.
#[derive(Debug, Clone, PartialEq)]
pub enum EntityType {
    /// Terrain SDF volume.
    Terrain {
        /// Material table index for the terrain surface.
        material_id: u16,
    },
    /// Animated character with skeletal animation and optional blend shapes.
    AnimatedCharacter {
        /// Path to the animation asset.
        animation_path: String,
        /// Names of active blend shapes.
        blend_shapes: Vec<String>,
    },
    /// Static (non-animated) SDF object.
    StaticObject {
        /// Path to the .rkf asset.
        asset_path: String,
        /// Material table index.
        material_id: u16,
    },
    /// Omnidirectional point light.
    PointLight {
        /// Light color (linear RGB).
        color: Vec3,
        /// Luminous intensity.
        intensity: f32,
        /// Attenuation range in meters.
        range: f32,
    },
    /// Cone-shaped spot light.
    SpotLight {
        /// Light color (linear RGB).
        color: Vec3,
        /// Luminous intensity.
        intensity: f32,
        /// Attenuation range in meters.
        range: f32,
        /// Inner cone half-angle in radians.
        inner_angle: f32,
        /// Outer cone half-angle in radians.
        outer_angle: f32,
        /// Direction the spot light points.
        direction: Vec3,
    },
    /// Directional (sun/moon) light.
    DirectionalLight {
        /// Light color (linear RGB).
        color: Vec3,
        /// Luminous intensity.
        intensity: f32,
        /// Direction the light travels (toward surface).
        direction: Vec3,
    },
    /// GPU particle emitter.
    ParticleEmitter {
        /// Which particle effect to spawn.
        effect_type: ParticleEffectType,
    },
    /// Rigid body for physics simulation.
    PhysicsObject {
        /// Shape descriptor (e.g. "box", "sphere").
        shape: String,
        /// Mass in kilograms.
        mass: f32,
    },
}

/// A single entity in the showcase scene.
#[derive(Debug, Clone)]
pub struct SceneEntity {
    /// Human-readable name.
    pub name: String,
    /// World-space position.
    pub position: Vec3,
    /// Orientation.
    pub rotation: Quat,
    /// Uniform scale factor.
    pub scale: f32,
    /// What kind of entity this is.
    pub entity_type: EntityType,
}

/// Environment/atmosphere configuration for the showcase.
#[derive(Debug, Clone)]
pub struct EnvironmentSetup {
    /// Fog density coefficient.
    pub fog_density: f32,
    /// Fog color (linear RGB).
    pub fog_color: Vec3,
    /// Cloud coverage fraction (0-1).
    pub cloud_coverage: f32,
    /// Sun direction (normalized, pointing toward light source).
    pub sun_direction: Vec3,
    /// Sun intensity multiplier.
    pub sun_intensity: f32,
}

/// Complete showcase scene definition.
#[derive(Debug, Clone)]
pub struct ShowcaseScene {
    /// All entities in the scene.
    pub entities: Vec<SceneEntity>,
    /// Environment/atmosphere settings.
    pub environment: EnvironmentSetup,
    /// Character controller spawn position.
    pub spawn_point: Vec3,
}

impl ShowcaseScene {
    /// Total number of entities in the scene.
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Number of light entities (point, spot, directional).
    pub fn light_count(&self) -> usize {
        self.entities
            .iter()
            .filter(|e| {
                matches!(
                    e.entity_type,
                    EntityType::PointLight { .. }
                        | EntityType::SpotLight { .. }
                        | EntityType::DirectionalLight { .. }
                )
            })
            .count()
    }

    /// Number of particle emitter entities.
    pub fn particle_emitter_count(&self) -> usize {
        self.entities
            .iter()
            .filter(|e| matches!(e.entity_type, EntityType::ParticleEmitter { .. }))
            .count()
    }
}

/// Build the default showcase scene with a rich variety of entities.
pub fn build_showcase() -> ShowcaseScene {
    let entities = vec![
        // --- Terrain ---
        SceneEntity {
            name: "Terrain".into(),
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: 1.0,
            entity_type: EntityType::Terrain { material_id: 1 },
        },
        // --- Animated characters ---
        SceneEntity {
            name: "Character_Knight".into(),
            position: Vec3::new(3.0, 0.0, 2.0),
            rotation: Quat::IDENTITY,
            scale: 1.0,
            entity_type: EntityType::AnimatedCharacter {
                animation_path: "assets/anims/knight_idle.rkf".into(),
                blend_shapes: vec!["smile".into(), "blink".into()],
            },
        },
        SceneEntity {
            name: "Character_Mage".into(),
            position: Vec3::new(-4.0, 0.0, 5.0),
            rotation: Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
            scale: 1.0,
            entity_type: EntityType::AnimatedCharacter {
                animation_path: "assets/anims/mage_cast.rkf".into(),
                blend_shapes: vec!["jaw_open".into()],
            },
        },
        SceneEntity {
            name: "Character_Guard".into(),
            position: Vec3::new(0.0, 0.0, -6.0),
            rotation: Quat::from_rotation_y(std::f32::consts::PI),
            scale: 1.2,
            entity_type: EntityType::AnimatedCharacter {
                animation_path: "assets/anims/guard_patrol.rkf".into(),
                blend_shapes: vec![],
            },
        },
        // --- Static objects (5 pillars/walls) ---
        SceneEntity {
            name: "Pillar_A".into(),
            position: Vec3::new(6.0, 0.0, 0.0),
            rotation: Quat::IDENTITY,
            scale: 1.0,
            entity_type: EntityType::StaticObject {
                asset_path: "assets/objects/pillar.rkf".into(),
                material_id: 3,
            },
        },
        SceneEntity {
            name: "Pillar_B".into(),
            position: Vec3::new(-6.0, 0.0, 0.0),
            rotation: Quat::IDENTITY,
            scale: 1.0,
            entity_type: EntityType::StaticObject {
                asset_path: "assets/objects/pillar.rkf".into(),
                material_id: 3,
            },
        },
        SceneEntity {
            name: "Wall_North".into(),
            position: Vec3::new(0.0, 0.0, 10.0),
            rotation: Quat::IDENTITY,
            scale: 2.0,
            entity_type: EntityType::StaticObject {
                asset_path: "assets/objects/wall_segment.rkf".into(),
                material_id: 10,
            },
        },
        SceneEntity {
            name: "Wall_East".into(),
            position: Vec3::new(10.0, 0.0, 5.0),
            rotation: Quat::from_rotation_y(std::f32::consts::FRAC_PI_2),
            scale: 2.0,
            entity_type: EntityType::StaticObject {
                asset_path: "assets/objects/wall_segment.rkf".into(),
                material_id: 12,
            },
        },
        SceneEntity {
            name: "Statue_Center".into(),
            position: Vec3::new(0.0, 1.0, 0.0),
            rotation: Quat::IDENTITY,
            scale: 0.8,
            entity_type: EntityType::StaticObject {
                asset_path: "assets/objects/statue.rkf".into(),
                material_id: 11,
            },
        },
        // --- Point lights ---
        SceneEntity {
            name: "PointLight_Warm".into(),
            position: Vec3::new(3.0, 4.0, 2.0),
            rotation: Quat::IDENTITY,
            scale: 1.0,
            entity_type: EntityType::PointLight {
                color: Vec3::new(1.0, 0.85, 0.6),
                intensity: 50.0,
                range: 15.0,
            },
        },
        SceneEntity {
            name: "PointLight_Cool".into(),
            position: Vec3::new(-5.0, 3.0, -3.0),
            rotation: Quat::IDENTITY,
            scale: 1.0,
            entity_type: EntityType::PointLight {
                color: Vec3::new(0.6, 0.8, 1.0),
                intensity: 35.0,
                range: 12.0,
            },
        },
        // --- Spot light ---
        SceneEntity {
            name: "SpotLight_Stage".into(),
            position: Vec3::new(0.0, 8.0, 0.0),
            rotation: Quat::IDENTITY,
            scale: 1.0,
            entity_type: EntityType::SpotLight {
                color: Vec3::new(1.0, 1.0, 0.9),
                intensity: 100.0,
                range: 20.0,
                inner_angle: 0.3,
                outer_angle: 0.6,
                direction: Vec3::new(0.0, -1.0, 0.0),
            },
        },
        // --- Directional sun ---
        SceneEntity {
            name: "Sun".into(),
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: 1.0,
            entity_type: EntityType::DirectionalLight {
                color: Vec3::new(1.0, 0.95, 0.85),
                intensity: 3.0,
                direction: Vec3::new(-0.5, -0.8, -0.3).normalize(),
            },
        },
        // --- Particle emitters ---
        SceneEntity {
            name: "Fire_Brazier".into(),
            position: Vec3::new(6.0, 1.5, 0.0),
            rotation: Quat::IDENTITY,
            scale: 1.0,
            entity_type: EntityType::ParticleEmitter {
                effect_type: ParticleEffectType::Fire,
            },
        },
        SceneEntity {
            name: "Debris_Rubble".into(),
            position: Vec3::new(-3.0, 0.5, 7.0),
            rotation: Quat::IDENTITY,
            scale: 1.0,
            entity_type: EntityType::ParticleEmitter {
                effect_type: ParticleEffectType::Debris,
            },
        },
        SceneEntity {
            name: "Rain_Ambient".into(),
            position: Vec3::new(0.0, 15.0, 0.0),
            rotation: Quat::IDENTITY,
            scale: 1.0,
            entity_type: EntityType::ParticleEmitter {
                effect_type: ParticleEffectType::Rain,
            },
        },
        // --- Physics objects (falling crates) ---
        SceneEntity {
            name: "Crate_A".into(),
            position: Vec3::new(2.0, 5.0, -2.0),
            rotation: Quat::IDENTITY,
            scale: 1.0,
            entity_type: EntityType::PhysicsObject {
                shape: "box".into(),
                mass: 10.0,
            },
        },
        SceneEntity {
            name: "Crate_B".into(),
            position: Vec3::new(2.5, 8.0, -1.5),
            rotation: Quat::from_rotation_z(0.3),
            scale: 0.8,
            entity_type: EntityType::PhysicsObject {
                shape: "box".into(),
                mass: 8.0,
            },
        },
        SceneEntity {
            name: "Crate_C".into(),
            position: Vec3::new(1.5, 11.0, -2.5),
            rotation: Quat::from_rotation_x(0.15),
            scale: 1.2,
            entity_type: EntityType::PhysicsObject {
                shape: "box".into(),
                mass: 15.0,
            },
        },
    ];

    // --- Environment ---
    let environment = EnvironmentSetup {
        fog_density: 0.02,
        fog_color: Vec3::new(0.7, 0.75, 0.85),
        cloud_coverage: 0.4,
        sun_direction: Vec3::new(-0.5, 0.8, -0.3).normalize(),
        sun_intensity: 3.0,
    };

    ShowcaseScene {
        entities,
        environment,
        spawn_point: Vec3::new(0.0, 2.0, -10.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_showcase_entity_count() {
        let scene = build_showcase();
        // 1 terrain + 3 characters + 5 static + 2 point + 1 spot + 1 dir + 3 particle + 3 physics = 19
        assert_eq!(scene.entity_count(), 19);
    }

    #[test]
    fn build_showcase_light_count() {
        let scene = build_showcase();
        // 2 point + 1 spot + 1 directional = 4
        assert_eq!(scene.light_count(), 4);
    }

    #[test]
    fn build_showcase_particle_emitter_count() {
        let scene = build_showcase();
        // fire + debris + rain = 3
        assert_eq!(scene.particle_emitter_count(), 3);
    }

    #[test]
    fn all_positions_are_finite() {
        let scene = build_showcase();
        for entity in &scene.entities {
            assert!(
                entity.position.is_finite(),
                "Entity '{}' has non-finite position: {:?}",
                entity.name,
                entity.position
            );
        }
    }

    #[test]
    fn spawn_point_is_finite() {
        let scene = build_showcase();
        assert!(scene.spawn_point.is_finite());
    }

    #[test]
    fn environment_values_are_valid() {
        let scene = build_showcase();
        assert!(scene.environment.fog_density >= 0.0);
        assert!(scene.environment.fog_color.is_finite());
        assert!(scene.environment.cloud_coverage >= 0.0 && scene.environment.cloud_coverage <= 1.0);
        assert!(scene.environment.sun_direction.is_finite());
        assert!(scene.environment.sun_intensity > 0.0);
    }

    #[test]
    fn terrain_is_at_origin() {
        let scene = build_showcase();
        let terrain = scene
            .entities
            .iter()
            .find(|e| matches!(e.entity_type, EntityType::Terrain { .. }))
            .expect("showcase must have terrain");
        assert_eq!(terrain.position, Vec3::ZERO);
        if let EntityType::Terrain { material_id } = &terrain.entity_type {
            assert_eq!(*material_id, 1);
        }
    }

    #[test]
    fn all_scales_are_positive() {
        let scene = build_showcase();
        for entity in &scene.entities {
            assert!(
                entity.scale > 0.0,
                "Entity '{}' has non-positive scale: {}",
                entity.name,
                entity.scale
            );
        }
    }

    #[test]
    fn entity_names_are_unique() {
        let scene = build_showcase();
        let mut names: Vec<&str> = scene.entities.iter().map(|e| e.name.as_str()).collect();
        names.sort();
        let before = names.len();
        names.dedup();
        assert_eq!(before, names.len(), "duplicate entity names found");
    }

    #[test]
    fn directional_light_direction_is_normalized() {
        let scene = build_showcase();
        for entity in &scene.entities {
            if let EntityType::DirectionalLight { direction, .. } = &entity.entity_type {
                let len = direction.length();
                assert!(
                    (len - 1.0).abs() < 1e-5,
                    "DirectionalLight direction not normalized: length = {}",
                    len
                );
            }
        }
    }

    #[test]
    fn spot_light_angles_are_valid() {
        let scene = build_showcase();
        for entity in &scene.entities {
            if let EntityType::SpotLight {
                inner_angle,
                outer_angle,
                ..
            } = &entity.entity_type
            {
                assert!(*inner_angle > 0.0);
                assert!(*outer_angle > *inner_angle);
            }
        }
    }

    #[test]
    fn physics_objects_have_positive_mass() {
        let scene = build_showcase();
        for entity in &scene.entities {
            if let EntityType::PhysicsObject { mass, .. } = &entity.entity_type {
                assert!(*mass > 0.0, "PhysicsObject '{}' has non-positive mass", entity.name);
            }
        }
    }
}
