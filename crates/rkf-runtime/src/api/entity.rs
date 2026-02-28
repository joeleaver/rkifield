//! Opaque entity handle for the World API.
//!
//! [`Entity`] wraps either an SDF scene object or an ECS-only entity.
//! Users never see the internal representation — they just hold the handle.
//! A generation counter protects against use-after-despawn.

/// Internal entity discriminant — hidden from users.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum EntityInner {
    /// An SDF scene object (has geometry in the Scene).
    SdfObject(u32),
    /// An ECS-only entity (no geometry — pure component bag).
    EcsOnly(hecs::Entity),
}

/// Opaque handle to an object in the [`super::World`].
///
/// Wraps either an SDF scene object or an ECS-only entity — the user never
/// sees which. Generation protects against use-after-despawn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Entity {
    pub(crate) inner: EntityInner,
    pub(crate) generation: u32,
}

impl Entity {
    /// Create a new entity handle for an SDF object.
    pub(crate) fn sdf(object_id: u32, generation: u32) -> Self {
        Self {
            inner: EntityInner::SdfObject(object_id),
            generation,
        }
    }

    /// Create a new entity handle for an ECS-only entity.
    pub(crate) fn ecs_only(ecs_entity: hecs::Entity, generation: u32) -> Self {
        Self {
            inner: EntityInner::EcsOnly(ecs_entity),
            generation,
        }
    }
}

impl std::fmt::Display for Entity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.inner {
            EntityInner::SdfObject(id) => write!(f, "Entity(sdf:{}, gen:{})", id, self.generation),
            EntityInner::EcsOnly(e) => write!(f, "Entity(ecs:{:?}, gen:{})", e, self.generation),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entity_equality() {
        let a = Entity::sdf(42, 1);
        let b = Entity::sdf(42, 1);
        assert_eq!(a, b);
    }

    #[test]
    fn entity_inequality_generation() {
        let a = Entity::sdf(42, 1);
        let b = Entity::sdf(42, 2);
        assert_ne!(a, b);
    }

    #[test]
    fn entity_inequality_inner() {
        let a = Entity::sdf(1, 0);
        let b = Entity::sdf(2, 0);
        assert_ne!(a, b);
    }

    #[test]
    fn entity_hash_consistent() {
        use std::collections::HashSet;
        let a = Entity::sdf(42, 1);
        let b = Entity::sdf(42, 1);
        let mut set = HashSet::new();
        set.insert(a);
        assert!(set.contains(&b));
    }

    #[test]
    fn entity_debug_format() {
        let e = Entity::sdf(7, 3);
        let s = format!("{:?}", e);
        assert!(s.contains("SdfObject"));
        assert!(s.contains("7"));
    }

    #[test]
    fn entity_display_format() {
        let e = Entity::sdf(7, 3);
        let s = format!("{}", e);
        assert!(s.contains("sdf:7"));
        assert!(s.contains("gen:3"));
    }
}
