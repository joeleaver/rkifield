//! Entity lookup by name path, tag, query, and relative hierarchy.
//!
//! Provides free functions for finding entities via the [`EntityNameIndex`],
//! [`EntityTagIndex`], hecs queries, and parent/child hierarchy traversal.

use super::entity_names::{EntityNameIndex, EntityTagIndex, children_of, parent_of};
use super::registry::QueryError;

/// Error returned by [`find_path`] and [`find_relative`] when an entity cannot be resolved.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LookupError {
    /// No entity with the given name exists.
    NotFound,
    /// Multiple entities match (ambiguous lookup).
    Ambiguous,
    /// The path format is not supported.
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
/// Supports multi-segment `/`-separated paths (e.g. `"parent/child/grandchild"`).
/// The first segment is resolved via the name index, then each subsequent
/// segment walks children by name.
pub fn find_path(
    world: &hecs::World,
    name_index: &EntityNameIndex,
    path: &str,
) -> Result<hecs::Entity, LookupError> {
    if path.is_empty() {
        return Err(LookupError::NotFound);
    }

    let segments: Vec<&str> = path.split('/').collect();

    // Resolve the first segment via the name index.
    let first = segments[0];
    let mut current = name_index
        .get_by_name(first)
        .ok_or(LookupError::NotFound)?;

    // Walk remaining segments by searching children by name.
    for &segment in &segments[1..] {
        let children = children_of(world, current);
        let mut found = None;
        for child in children {
            if name_index.get_name(child) == Some(segment) {
                if found.is_some() {
                    return Err(LookupError::Ambiguous);
                }
                found = Some(child);
            }
        }
        current = found.ok_or(LookupError::NotFound)?;
    }

    Ok(current)
}

/// Find exactly one entity matching a hecs query.
///
/// Returns an error if zero or more than one entity matches.
pub fn find_one<Q: hecs::Query>(world: &hecs::World) -> Result<hecs::Entity, QueryError> {
    let mut query = world.query::<Q>();
    let mut iter = query.iter();

    let (entity, _) = iter.next().ok_or(QueryError::NotFound)?;

    if iter.next().is_some() {
        return Err(QueryError::Multiple);
    }

    Ok(entity)
}

/// Resolve a relative path from an entity through the hierarchy.
///
/// Path segments:
/// - `"../"` walks to the parent entity.
/// - Any other segment walks to a child with that name.
///
/// Segments are separated by `/`. Example: `"../sibling/child"` goes up
/// one level, then down to "sibling", then down to "child".
pub fn find_relative(
    world: &hecs::World,
    from: Entity,
    name_index: &EntityNameIndex,
    path: &str,
) -> Result<hecs::Entity, LookupError> {
    if path.is_empty() {
        return Err(LookupError::InvalidPath);
    }

    let mut current = from;

    for segment in path.split('/') {
        if segment.is_empty() {
            continue;
        }
        if segment == ".." {
            current = parent_of(world, current).ok_or(LookupError::NotFound)?;
        } else {
            let children = children_of(world, current);
            let mut found = None;
            for child in children {
                if name_index.get_name(child) == Some(segment) {
                    if found.is_some() {
                        return Err(LookupError::Ambiguous);
                    }
                    found = Some(child);
                }
            }
            current = found.ok_or(LookupError::NotFound)?;
        }
    }

    Ok(current)
}

use hecs::Entity;

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
    use crate::components::Parent;
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
    fn find_path_multi_segment_resolves() {
        let mut world = hecs::World::new();
        let parent_e = world.spawn(());
        let child_e = world.spawn((Parent {
            entity: parent_e,
            bone_index: None,
        },));
        let grandchild_e = world.spawn((Parent {
            entity: child_e,
            bone_index: None,
        },));

        let mut names = EntityNameIndex::new();
        names.set_name(parent_e, "root");
        names.set_name(child_e, "mid");
        names.set_name(grandchild_e, "leaf");

        // Single segment still works
        assert_eq!(find_path(&world, &names, "root"), Ok(parent_e));

        // Two segments
        assert_eq!(find_path(&world, &names, "root/mid"), Ok(child_e));

        // Three segments
        assert_eq!(find_path(&world, &names, "root/mid/leaf"), Ok(grandchild_e));
    }

    #[test]
    fn find_path_multi_segment_child_not_found() {
        let mut world = hecs::World::new();
        let parent_e = world.spawn(());
        let mut names = EntityNameIndex::new();
        names.set_name(parent_e, "root");

        let result = find_path(&world, &names, "root/nonexistent");
        assert_eq!(result, Err(LookupError::NotFound));
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

    // ─── find_one tests ─────────────────────────────────────────────────

    #[test]
    fn find_one_exactly_one_match() {
        let mut world = hecs::World::new();
        let e = world.spawn((42_u32,));

        let result = find_one::<&u32>(&world);
        assert_eq!(result, Ok(e));
    }

    #[test]
    fn find_one_no_match() {
        let world = hecs::World::new();

        let result = find_one::<&u32>(&world);
        assert_eq!(result, Err(QueryError::NotFound));
    }

    #[test]
    fn find_one_multiple_matches() {
        let mut world = hecs::World::new();
        world.spawn((42_u32,));
        world.spawn((99_u32,));

        let result = find_one::<&u32>(&world);
        assert_eq!(result, Err(QueryError::Multiple));
    }

    // ─── find_relative tests ────────────────────────────────────────────

    #[test]
    fn find_relative_parent() {
        let mut world = hecs::World::new();
        let parent_e = world.spawn(());
        let child_e = world.spawn((Parent {
            entity: parent_e,
            bone_index: None,
        },));
        let names = EntityNameIndex::new();

        let result = find_relative(&world, child_e, &names, "..");
        assert_eq!(result, Ok(parent_e));
    }

    #[test]
    fn find_relative_child_by_name() {
        let mut world = hecs::World::new();
        let parent_e = world.spawn(());
        let child_e = world.spawn((Parent {
            entity: parent_e,
            bone_index: None,
        },));
        let mut names = EntityNameIndex::new();
        names.set_name(child_e, "arm");

        let result = find_relative(&world, parent_e, &names, "arm");
        assert_eq!(result, Ok(child_e));
    }

    #[test]
    fn find_relative_up_and_down() {
        let mut world = hecs::World::new();
        let parent_e = world.spawn(());
        let child_a = world.spawn((Parent {
            entity: parent_e,
            bone_index: None,
        },));
        let child_b = world.spawn((Parent {
            entity: parent_e,
            bone_index: None,
        },));
        let mut names = EntityNameIndex::new();
        names.set_name(child_a, "left");
        names.set_name(child_b, "right");

        // From child_a, go up to parent, then down to child_b
        let result = find_relative(&world, child_a, &names, "../right");
        assert_eq!(result, Ok(child_b));
    }

    #[test]
    fn find_relative_no_parent_returns_not_found() {
        let mut world = hecs::World::new();
        let root = world.spawn(());
        let names = EntityNameIndex::new();

        let result = find_relative(&world, root, &names, "..");
        assert_eq!(result, Err(LookupError::NotFound));
    }

    #[test]
    fn find_relative_empty_path_returns_invalid() {
        let mut world = hecs::World::new();
        let e = world.spawn(());
        let names = EntityNameIndex::new();

        let result = find_relative(&world, e, &names, "");
        assert_eq!(result, Err(LookupError::InvalidPath));
    }

    #[test]
    fn find_relative_child_not_found() {
        let mut world = hecs::World::new();
        let parent_e = world.spawn(());
        let names = EntityNameIndex::new();

        let result = find_relative(&world, parent_e, &names, "nonexistent");
        assert_eq!(result, Err(LookupError::NotFound));
    }
}
