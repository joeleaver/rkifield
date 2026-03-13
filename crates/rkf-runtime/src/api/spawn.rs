//! Fluent spawn builder for the World API.
//!
//! ```ignore
//! let cube_id = world.spawn("my_cube")
//!     .position_vec3(Vec3::new(0.0, 1.0, -3.0))
//!     .sdf(SdfPrimitive::Box { half_extents: Vec3::splat(0.5) })
//!     .material(2)
//!     .build();
//! ```

use glam::{IVec3, Quat, Vec3};
use uuid::Uuid;

use rkf_core::aabb::Aabb;
use rkf_core::scene_node::{BlendMode, SceneNode, SdfPrimitive};
use rkf_core::WorldPosition;

use super::world::World;

/// Describes the SDF geometry source for a new entity.
enum SdfDesc {
    /// No SDF — ECS-only entity.
    None,
    /// Analytical primitive (zero voxels).
    Primitive(SdfPrimitive, u16),
    /// Pre-built SceneNode tree.
    Tree(SceneNode),
}

/// Pending ECS component to attach after spawn.
trait PendingComponent: Send + 'static {
    fn insert(self: Box<Self>, ecs: &mut hecs::World, entity: hecs::Entity);
}

struct PendingComponentImpl<C: hecs::Component> {
    component: Option<C>,
}

impl<C: hecs::Component> PendingComponent for PendingComponentImpl<C> {
    fn insert(mut self: Box<Self>, ecs: &mut hecs::World, entity: hecs::Entity) {
        if let Some(c) = self.component.take() {
            let _ = ecs.insert_one(entity, c);
        }
    }
}

/// Builder for spawning entities in a [`World`].
///
/// Created by [`World::spawn`]. Configure with method chains, then call
/// [`.build()`](SpawnBuilder::build) to finalize.
pub struct SpawnBuilder<'w> {
    world: &'w mut World,
    name: String,
    position: WorldPosition,
    rotation: Quat,
    scale: Vec3,
    sdf: SdfDesc,
    blend_mode: Option<BlendMode>,
    parent: Option<Uuid>,
    pending_components: Vec<Box<dyn PendingComponent>>,
}

impl<'w> SpawnBuilder<'w> {
    pub(crate) fn new(world: &'w mut World, name: String) -> Self {
        Self {
            world,
            name,
            position: WorldPosition::default(),
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            sdf: SdfDesc::None,
            blend_mode: None,
            parent: None,
            pending_components: Vec::new(),
        }
    }

    /// Set the world-space position.
    pub fn position(mut self, pos: WorldPosition) -> Self {
        self.position = pos;
        self
    }

    /// Set the world-space position from a Vec3 (convenience for editor-scale scenes).
    pub fn position_vec3(mut self, v: Vec3) -> Self {
        self.position = WorldPosition::new(IVec3::ZERO, v);
        self
    }

    /// Set the rotation.
    pub fn rotation(mut self, rot: Quat) -> Self {
        self.rotation = rot;
        self
    }

    /// Set per-axis scale.
    pub fn scale(mut self, scale: Vec3) -> Self {
        self.scale = scale;
        self
    }

    /// Set uniform scale on all axes.
    pub fn uniform_scale(mut self, s: f32) -> Self {
        self.scale = Vec3::splat(s);
        self
    }

    /// Set an analytical SDF primitive as the geometry source.
    ///
    /// Must be paired with `.material()` to set the material index.
    /// If `.material()` is not called, material 0 is used.
    pub fn sdf(mut self, primitive: SdfPrimitive) -> Self {
        // Preserve existing material if already set
        let mat = match &self.sdf {
            SdfDesc::Primitive(_, m) => *m,
            _ => 0,
        };
        self.sdf = SdfDesc::Primitive(primitive, mat);
        self
    }

    /// Set a pre-built SceneNode tree as the geometry source.
    pub fn sdf_tree(mut self, root: SceneNode) -> Self {
        self.sdf = SdfDesc::Tree(root);
        self
    }

    /// Set the material table index (used with `.sdf()`).
    pub fn material(mut self, material_id: u16) -> Self {
        match &mut self.sdf {
            SdfDesc::Primitive(_, m) => *m = material_id,
            SdfDesc::None => {
                // Default to a sphere if material is set without an SDF
                self.sdf = SdfDesc::Primitive(SdfPrimitive::Sphere { radius: 0.5 }, material_id);
            }
            SdfDesc::Tree(_) => {
                // Tree nodes have their own materials — ignore
            }
        }
        self
    }

    /// Set the blend mode (how this object combines with siblings).
    pub fn blend(mut self, mode: BlendMode) -> Self {
        self.blend_mode = Some(mode);
        self
    }

    /// Set the parent entity (establishes hierarchy).
    pub fn parent(mut self, parent: Uuid) -> Self {
        self.parent = Some(parent);
        self
    }

    /// Attach an ECS component to the entity.
    pub fn with<C: hecs::Component>(mut self, component: C) -> Self {
        self.pending_components.push(Box::new(PendingComponentImpl {
            component: Some(component),
        }));
        self
    }

    /// Finalize the entity and return its UUID.
    pub fn build(self) -> Uuid {
        let uuid = match self.sdf {
            SdfDesc::None => self.world.finalize_ecs_spawn(self.name),
            SdfDesc::Primitive(primitive, material_id) => {
                let aabb = compute_primitive_aabb(&primitive);
                let root_node = SceneNode::analytical(&self.name, primitive, material_id);
                self.world.finalize_sdf_spawn(
                    self.name,
                    self.position,
                    self.rotation,
                    self.scale,
                    root_node,
                    material_id,
                    self.blend_mode,
                    self.parent,
                    aabb,
                )
            }
            SdfDesc::Tree(root_node) => {
                let aabb = Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0));
                self.world.finalize_sdf_spawn(
                    self.name,
                    self.position,
                    self.rotation,
                    self.scale,
                    root_node,
                    0,
                    self.blend_mode,
                    self.parent,
                    aabb,
                )
            }
        };

        // Insert pending ECS components
        if !self.pending_components.is_empty() {
            if let Some(ecs_entity) = self.world.ecs_entity_for(uuid) {
                for comp in self.pending_components {
                    comp.insert(self.world.ecs_mut(), ecs_entity);
                }
            }
        }

        uuid
    }
}

/// Compute a local-space AABB for an analytical primitive.
fn compute_primitive_aabb(primitive: &SdfPrimitive) -> Aabb {
    match primitive {
        SdfPrimitive::Sphere { radius } => {
            Aabb::new(Vec3::splat(-*radius), Vec3::splat(*radius))
        }
        SdfPrimitive::Box { half_extents } => {
            Aabb::new(-*half_extents, *half_extents)
        }
        SdfPrimitive::Capsule {
            radius,
            half_height,
        } => {
            let r = *radius;
            let h = *half_height + r;
            Aabb::new(Vec3::new(-r, -h, -r), Vec3::new(r, h, r))
        }
        SdfPrimitive::Torus {
            major_radius,
            minor_radius,
        } => {
            let r = *major_radius + *minor_radius;
            let h = *minor_radius;
            Aabb::new(Vec3::new(-r, -h, -r), Vec3::new(r, h, r))
        }
        SdfPrimitive::Cylinder {
            radius,
            half_height,
        } => {
            let r = *radius;
            let h = *half_height;
            Aabb::new(Vec3::new(-r, -h, -r), Vec3::new(r, h, r))
        }
        SdfPrimitive::Plane { .. } => {
            // Infinite plane — use large AABB
            Aabb::new(Vec3::splat(-1000.0), Vec3::splat(1000.0))
        }
    }
}
