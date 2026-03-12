//! Entity lookup by name path and tag.
//!
//! Provides free functions for finding entities via the [`EntityNameIndex`] and
//! [`EntityTagIndex`]. Path-based traversal (multi-segment paths like
//! `"parent/child"`) is a future enhancement — for now only single-segment
//! name lookups are supported.

use super::entity_names::{EntityNameIndex, EntityTagIndex};

/// Error returned by [`find_path`] when an entity cannot be resolved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LookupError {
    /// No entity with the given name exists.
    NotFound,
    /// Multiple entities match (ambiguous lookup).
    Ambiguous,
    /// The path format is not supported (e.g. multi-segment paths).
    InvalidPath,
}

impl std::fmt::Display for LookupError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LookupError::NotFound => write!(f, "entity not found"),
            LookupError::Ambiguous => write!(f, "ambiguous entity lookup"),
            LookupError::InvalidPath => write!(f, "invalid path format"),
        }
    }
}

impl std::error::Error for LookupError {}

/// Look up an entity by name path.
///
/// For now, only single-segment names are supported (e.g. `"player"`).
/// Multi-segment hierarchy paths (e.g. `"parent/child"`) will be added
/// when hierarchy names are wired — currently they return
/// [`LookupError::InvalidPath`].
pub fn find_path(
    _world: &hecs::World,
    name_index: &EntityNameIndex,
    path: &str,
) -> Result<hecs::Entity, LookupError> {
    if path.is_empty() {
        return Err(LookupError::NotFound);
    }

    let segments: Vec<&str> = path.split('/').collect();

    if segments.len() > 1 {
        // Multi-segment path traversal is a future enhancement.
        return Err(LookupError::InvalidPath);
    }

    let name = segments[0];
    name_index
        .get_by_name(name)
        .ok_or(LookupError::NotFound)
}

/// Find all entities with a given tag.
///
/// Delegates to [`EntityTagIndex::entities_with_tag`].
pub fn find_tagged<'a>(
    tag_index: &'a EntityTagIndex,
    tag: &str,
) -> impl Iterator<Item = hecs::Entity> + 'a {
    tag_index.entities_with_tag(tag)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn find_path_single_name_found() {
        let mut world = hecs::World::new();
        let e = world.spawn(());
        let mut names = EntityNameIndex::new();
        names.set_name(e, "player");

        let result = find_path(&world, &names, "player");
        assert_eq!(result, Ok(e));
    }

    #[test]
    fn find_path_single_name_not_found() {
        let world = hecs::World::new();
        let names = EntityNameIndex::new();

        let result = find_path(&world, &names, "ghost");
        assert_eq!(result, Err(LookupError::NotFound));
    }

    #[test]
    fn find_path_empty_string_returns_not_found() {
        let world = hecs::World::new();
        let names = EntityNameIndex::new();

        let result = find_path(&world, &names, "");
        assert_eq!(result, Err(LookupError::NotFound));
    }

    #[test]
    fn find_path_multi_segment_returns_invalid_path() {
        let mut world = hecs::World::new();
        let e = world.spawn(());
        let mut names = EntityNameIndex::new();
        names.set_name(e, "parent");

        let result = find_path(&world, &names, "parent/child");
        assert_eq!(result, Err(LookupError::InvalidPath));
    }

    #[test]
    fn find_tagged_returns_matching_entities() {
        let mut world = hecs::World::new();
        let e1 = world.spawn(());
        let e2 = world.spawn(());
        let e3 = world.spawn(());
        let mut tags = EntityTagIndex::new();
        tags.add_tag(e1, "enemy");
        tags.add_tag(e2, "enemy");
        tags.add_tag(e3, "ally");

        let enemies: HashSet<_> = find_tagged(&tags, "enemy").collect();
        assert_eq!(enemies.len(), 2);
        assert!(enemies.contains(&e1));
        assert!(enemies.contains(&e2));
    }

    #[test]
    fn find_tagged_unknown_tag_returns_empty() {
        let tags = EntityTagIndex::new();

        let result: Vec<_> = find_tagged(&tags, "nonexistent").collect();
        assert!(result.is_empty());
    }
}
