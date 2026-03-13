//! Entity naming, tagging, and hierarchy lookup helpers.
//!
//! Provides bidirectional indexes for entity names and tags, plus free functions
//! for navigating the Parent-based entity hierarchy.

use std::collections::{HashMap, HashSet, VecDeque};

/// Bidirectional name-to-entity index.
///
/// Each entity can have at most one name, and each name maps to exactly one entity.
/// Setting a name that is already taken by another entity "steals" it.
pub struct EntityNameIndex {
    name_to_entity: HashMap<String, hecs::Entity>,
    entity_to_name: HashMap<hecs::Entity, String>,
    /// Monotonic counter for auto-generated entity names.
    auto_name_counter: u64,
}

impl EntityNameIndex {
    /// Create an empty name index.
    pub fn new() -> Self {
        Self {
            name_to_entity: HashMap::new(),
            entity_to_name: HashMap::new(),
            auto_name_counter: 0,
        }
    }

    /// Assign a name to an entity.
    ///
    /// - If the entity already has a different name, the old mapping is removed.
    /// - If the name is already taken by another entity, it is stolen.
    pub fn set_name(&mut self, entity: hecs::Entity, name: &str) {
        // Remove old name from this entity if it had one.
        if let Some(old_name) = self.entity_to_name.remove(&entity) {
            self.name_to_entity.remove(&old_name);
        }

        // Steal the name from any other entity that currently holds it.
        if let Some(old_entity) = self.name_to_entity.remove(name) {
            if old_entity != entity {
                self.entity_to_name.remove(&old_entity);
            }
        }

        self.name_to_entity.insert(name.to_string(), entity);
        self.entity_to_name.insert(entity, name.to_string());
    }

    /// Assign an auto-generated name to an entity if it does not already have one.
    ///
    /// Names follow the pattern `"Entity_1"`, `"Entity_2"`, etc., using a
    /// monotonic counter that never resets. If the entity already has a name,
    /// this is a no-op and returns the existing name.
    pub fn assign_auto_name(&mut self, entity: hecs::Entity) -> &str {
        if self.entity_to_name.contains_key(&entity) {
            return self.entity_to_name.get(&entity).map(|s| s.as_str()).unwrap();
        }
        self.auto_name_counter += 1;
        let name = format!("Entity_{}", self.auto_name_counter);
        self.set_name(entity, &name);
        self.entity_to_name.get(&entity).map(|s| s.as_str()).unwrap()
    }

    /// Look up an entity by name.
    pub fn get_by_name(&self, name: &str) -> Option<hecs::Entity> {
        self.name_to_entity.get(name).copied()
    }

    /// Get the name of an entity, if one is assigned.
    pub fn get_name(&self, entity: hecs::Entity) -> Option<&str> {
        self.entity_to_name.get(&entity).map(|s| s.as_str())
    }

    /// Remove the name mapping for an entity. Returns the old name if one existed.
    pub fn remove(&mut self, entity: hecs::Entity) -> Option<String> {
        if let Some(name) = self.entity_to_name.remove(&entity) {
            self.name_to_entity.remove(&name);
            Some(name)
        } else {
            None
        }
    }

    /// Number of named entities.
    pub fn len(&self) -> usize {
        self.entity_to_name.len()
    }

    /// Whether no entities have names.
    pub fn is_empty(&self) -> bool {
        self.entity_to_name.is_empty()
    }

    /// Iterate over all (name, entity) pairs.
    pub fn names(&self) -> impl Iterator<Item = (&str, hecs::Entity)> {
        self.name_to_entity.iter().map(|(n, &e)| (n.as_str(), e))
    }
}

impl Default for EntityNameIndex {
    fn default() -> Self {
        Self::new()
    }
}

/// Tag-to-entities index. Entities can have multiple tags; tags can apply to multiple entities.
pub struct EntityTagIndex {
    tag_to_entities: HashMap<String, HashSet<hecs::Entity>>,
    entity_to_tags: HashMap<hecs::Entity, HashSet<String>>,
}

impl EntityTagIndex {
    /// Create an empty tag index.
    pub fn new() -> Self {
        Self {
            tag_to_entities: HashMap::new(),
            entity_to_tags: HashMap::new(),
        }
    }

    /// Add a tag to an entity. No-op if already present.
    pub fn add_tag(&mut self, entity: hecs::Entity, tag: &str) {
        self.tag_to_entities
            .entry(tag.to_string())
            .or_default()
            .insert(entity);
        self.entity_to_tags
            .entry(entity)
            .or_default()
            .insert(tag.to_string());
    }

    /// Remove a tag from an entity. Returns `true` if the tag was present.
    pub fn remove_tag(&mut self, entity: hecs::Entity, tag: &str) -> bool {
        let had_tag = if let Some(tags) = self.entity_to_tags.get_mut(&entity) {
            tags.remove(tag)
        } else {
            false
        };

        if had_tag {
            if let Some(entities) = self.tag_to_entities.get_mut(tag) {
                entities.remove(&entity);
                if entities.is_empty() {
                    self.tag_to_entities.remove(tag);
                }
            }
            // Clean up empty tag set for entity.
            if let Some(tags) = self.entity_to_tags.get(&entity) {
                if tags.is_empty() {
                    self.entity_to_tags.remove(&entity);
                }
            }
        }

        had_tag
    }

    /// Check whether an entity has a specific tag.
    pub fn has_tag(&self, entity: hecs::Entity, tag: &str) -> bool {
        self.entity_to_tags
            .get(&entity)
            .map_or(false, |tags| tags.contains(tag))
    }

    /// Iterate over all entities with a given tag.
    pub fn entities_with_tag(&self, tag: &str) -> impl Iterator<Item = hecs::Entity> + '_ {
        self.tag_to_entities
            .get(tag)
            .into_iter()
            .flat_map(|set| set.iter().copied())
    }

    /// Iterate over all tags on an entity.
    pub fn tags_of(&self, entity: hecs::Entity) -> impl Iterator<Item = &str> {
        self.entity_to_tags
            .get(&entity)
            .into_iter()
            .flat_map(|set| set.iter().map(|s| s.as_str()))
    }

    /// Remove an entity from the index entirely (all its tags).
    ///
    /// Call this when an entity is despawned to keep the index clean.
    pub fn remove_entity(&mut self, entity: hecs::Entity) {
        if let Some(tags) = self.entity_to_tags.remove(&entity) {
            for tag in &tags {
                if let Some(entities) = self.tag_to_entities.get_mut(tag) {
                    entities.remove(&entity);
                    if entities.is_empty() {
                        self.tag_to_entities.remove(tag);
                    }
                }
            }
        }
    }

    /// Iterate over all distinct tags in the index.
    pub fn all_tags(&self) -> impl Iterator<Item = &str> {
        self.tag_to_entities.keys().map(|s| s.as_str())
    }
}

impl Default for EntityTagIndex {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Hierarchy helpers (free functions) ──────────────────────────────────────

/// Get the parent entity (reads the [`Parent`](crate::components::Parent) component).
pub fn parent_of(world: &hecs::World, entity: hecs::Entity) -> Option<hecs::Entity> {
    world
        .get::<&crate::components::Parent>(entity)
        .ok()
        .map(|p| p.entity)
}

/// Get all children of an entity (entities whose `Parent.entity` equals `parent`).
pub fn children_of(world: &hecs::World, parent: hecs::Entity) -> Vec<hecs::Entity> {
    world
        .query::<&crate::components::Parent>()
        .iter()
        .filter_map(|(child, p)| {
            if p.entity == parent {
                Some(child)
            } else {
                None
            }
        })
        .collect()
}

/// Get the root ancestor (walk up the Parent chain until no parent exists).
pub fn root_of(world: &hecs::World, mut entity: hecs::Entity) -> hecs::Entity {
    while let Some(parent) = parent_of(world, entity) {
        entity = parent;
    }
    entity
}

/// Collect all descendants recursively (breadth-first). Does not include `root` itself.
pub fn descendants_of(world: &hecs::World, root: hecs::Entity) -> Vec<hecs::Entity> {
    let mut result = Vec::new();
    let mut queue = VecDeque::new();
    queue.push_back(root);
    while let Some(entity) = queue.pop_front() {
        for child in children_of(world, entity) {
            result.push(child);
            queue.push_back(child);
        }
    }
    result
}

/// Collect the names of all siblings of `entity` in `world`.
///
/// Siblings are entities that share the same `Parent` (or all root entities if
/// `entity` has no parent). The returned set does NOT include `entity`'s own
/// name.
fn sibling_names(world: &hecs::World, entity: hecs::Entity) -> HashSet<String> {
    let my_parent = parent_of(world, entity);
    let mut names = HashSet::new();

    for (e, meta) in world.query::<&crate::components::EditorMetadata>().iter() {
        if e == entity {
            continue;
        }
        let their_parent = parent_of(world, e);
        if their_parent == my_parent && !meta.name.is_empty() {
            names.insert(meta.name.clone());
        }
    }
    names
}

/// Return a name guaranteed to be unique among the siblings of `entity`.
///
/// If `name` is not taken by any sibling, it is returned unchanged.
/// Otherwise, suffixes `_2`, `_3`, ... are tried until a unique variant is
/// found.
pub fn ensure_unique_name(world: &hecs::World, entity: hecs::Entity, name: &str) -> String {
    let taken = sibling_names(world, entity);
    if !taken.contains(name) {
        return name.to_string();
    }

    // Strip an existing numeric suffix so "Guard_2" re-entering gets "Guard_3"
    // rather than "Guard_2_2".
    let base = strip_numeric_suffix(name);

    let mut counter = 2u64;
    loop {
        let candidate = format!("{}_{}", base, counter);
        if !taken.contains(&candidate) {
            return candidate;
        }
        counter += 1;
    }
}

/// Strip a trailing `_N` numeric suffix, returning the base name.
/// e.g. `"Guard_3"` → `"Guard"`, `"Guard"` → `"Guard"`.
fn strip_numeric_suffix(name: &str) -> &str {
    if let Some(pos) = name.rfind('_') {
        let suffix = &name[pos + 1..];
        if !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_digit()) {
            return &name[..pos];
        }
    }
    name
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::Parent;

    // ─── EntityNameIndex ─────────────────────────────────────────────────

    #[test]
    fn name_index_set_and_get() {
        let mut world = hecs::World::new();
        let e = world.spawn(());
        let mut index = EntityNameIndex::new();

        index.set_name(e, "player");
        assert_eq!(index.get_by_name("player"), Some(e));
        assert_eq!(index.get_name(e), Some("player"));
        assert_eq!(index.len(), 1);
        assert!(!index.is_empty());
    }

    #[test]
    fn name_index_rename() {
        let mut world = hecs::World::new();
        let e = world.spawn(());
        let mut index = EntityNameIndex::new();

        index.set_name(e, "alpha");
        index.set_name(e, "beta");

        assert_eq!(index.get_by_name("beta"), Some(e));
        assert_eq!(index.get_name(e), Some("beta"));
        assert_eq!(index.get_by_name("alpha"), None);
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn name_index_stealing() {
        let mut world = hecs::World::new();
        let e1 = world.spawn(());
        let e2 = world.spawn(());
        let mut index = EntityNameIndex::new();

        index.set_name(e1, "hero");
        index.set_name(e2, "hero"); // steal from e1

        assert_eq!(index.get_by_name("hero"), Some(e2));
        assert_eq!(index.get_name(e1), None);
        assert_eq!(index.get_name(e2), Some("hero"));
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn name_index_remove() {
        let mut world = hecs::World::new();
        let e = world.spawn(());
        let mut index = EntityNameIndex::new();

        index.set_name(e, "npc");
        let removed = index.remove(e);

        assert_eq!(removed.as_deref(), Some("npc"));
        assert_eq!(index.get_by_name("npc"), None);
        assert_eq!(index.get_name(e), None);
        assert!(index.is_empty());
    }

    #[test]
    fn name_index_remove_nonexistent() {
        let mut world = hecs::World::new();
        let e = world.spawn(());
        let mut index = EntityNameIndex::new();

        assert_eq!(index.remove(e), None);
    }

    #[test]
    fn name_index_iteration() {
        let mut world = hecs::World::new();
        let e1 = world.spawn(());
        let e2 = world.spawn(());
        let mut index = EntityNameIndex::new();

        index.set_name(e1, "a");
        index.set_name(e2, "b");

        let mut pairs: Vec<_> = index.names().collect();
        pairs.sort_by_key(|(name, _)| name.to_string());

        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0].0, "a");
        assert_eq!(pairs[0].1, e1);
        assert_eq!(pairs[1].0, "b");
        assert_eq!(pairs[1].1, e2);
    }

    #[test]
    fn name_index_default() {
        let index = EntityNameIndex::default();
        assert!(index.is_empty());
    }

    // ─── Auto-naming ────────────────────────────────────────────────────

    #[test]
    fn auto_name_generates_sequential_names() {
        let mut world = hecs::World::new();
        let e1 = world.spawn(());
        let e2 = world.spawn(());
        let e3 = world.spawn(());
        let mut index = EntityNameIndex::new();

        index.assign_auto_name(e1);
        index.assign_auto_name(e2);
        index.assign_auto_name(e3);

        assert_eq!(index.get_name(e1), Some("Entity_1"));
        assert_eq!(index.get_name(e2), Some("Entity_2"));
        assert_eq!(index.get_name(e3), Some("Entity_3"));
        assert_eq!(index.len(), 3);
    }

    #[test]
    fn auto_name_skips_already_named_entity() {
        let mut world = hecs::World::new();
        let e = world.spawn(());
        let mut index = EntityNameIndex::new();

        index.set_name(e, "custom_name");
        let name = index.assign_auto_name(e);

        assert_eq!(name, "custom_name");
        assert_eq!(index.get_name(e), Some("custom_name"));
        // Counter should not have been incremented.
        assert_eq!(index.auto_name_counter, 0);
    }

    #[test]
    fn auto_name_counter_is_monotonic() {
        let mut world = hecs::World::new();
        let e1 = world.spawn(());
        let e2 = world.spawn(());
        let mut index = EntityNameIndex::new();

        index.assign_auto_name(e1);
        // Remove e1's name — counter should NOT go back.
        index.remove(e1);
        index.assign_auto_name(e2);

        assert_eq!(index.get_name(e2), Some("Entity_2"));
    }

    #[test]
    fn auto_name_is_findable_by_name() {
        let mut world = hecs::World::new();
        let e = world.spawn(());
        let mut index = EntityNameIndex::new();

        index.assign_auto_name(e);
        assert_eq!(index.get_by_name("Entity_1"), Some(e));
    }

    // ─── EntityTagIndex ──────────────────────────────────────────────────

    #[test]
    fn tag_index_add_and_check() {
        let mut world = hecs::World::new();
        let e = world.spawn(());
        let mut index = EntityTagIndex::new();

        index.add_tag(e, "enemy");
        assert!(index.has_tag(e, "enemy"));
        assert!(!index.has_tag(e, "ally"));
    }

    #[test]
    fn tag_index_multiple_tags_on_entity() {
        let mut world = hecs::World::new();
        let e = world.spawn(());
        let mut index = EntityTagIndex::new();

        index.add_tag(e, "enemy");
        index.add_tag(e, "flying");
        index.add_tag(e, "boss");

        assert!(index.has_tag(e, "enemy"));
        assert!(index.has_tag(e, "flying"));
        assert!(index.has_tag(e, "boss"));

        let mut tags: Vec<_> = index.tags_of(e).collect();
        tags.sort();
        assert_eq!(tags, vec!["boss", "enemy", "flying"]);
    }

    #[test]
    fn tag_index_multiple_entities_with_tag() {
        let mut world = hecs::World::new();
        let e1 = world.spawn(());
        let e2 = world.spawn(());
        let e3 = world.spawn(());
        let mut index = EntityTagIndex::new();

        index.add_tag(e1, "enemy");
        index.add_tag(e2, "enemy");
        index.add_tag(e3, "ally");

        let enemies: HashSet<_> = index.entities_with_tag("enemy").collect();
        assert_eq!(enemies.len(), 2);
        assert!(enemies.contains(&e1));
        assert!(enemies.contains(&e2));
    }

    #[test]
    fn tag_index_remove_tag() {
        let mut world = hecs::World::new();
        let e = world.spawn(());
        let mut index = EntityTagIndex::new();

        index.add_tag(e, "enemy");
        assert!(index.remove_tag(e, "enemy"));
        assert!(!index.has_tag(e, "enemy"));

        // Removing again returns false.
        assert!(!index.remove_tag(e, "enemy"));
    }

    #[test]
    fn tag_index_remove_tag_cleans_up_empty_sets() {
        let mut world = hecs::World::new();
        let e = world.spawn(());
        let mut index = EntityTagIndex::new();

        index.add_tag(e, "solo");
        index.remove_tag(e, "solo");

        // The tag should no longer appear in all_tags.
        assert_eq!(index.all_tags().count(), 0);
        // entities_with_tag for removed tag yields nothing.
        assert_eq!(index.entities_with_tag("solo").count(), 0);
    }

    #[test]
    fn tag_index_remove_entity() {
        let mut world = hecs::World::new();
        let e1 = world.spawn(());
        let e2 = world.spawn(());
        let mut index = EntityTagIndex::new();

        index.add_tag(e1, "enemy");
        index.add_tag(e1, "boss");
        index.add_tag(e2, "enemy");

        index.remove_entity(e1);

        assert!(!index.has_tag(e1, "enemy"));
        assert!(!index.has_tag(e1, "boss"));
        assert_eq!(index.tags_of(e1).count(), 0);

        // e2 still has its tag.
        assert!(index.has_tag(e2, "enemy"));

        // "boss" tag should be fully cleaned up since e1 was the only entity with it.
        assert_eq!(index.entities_with_tag("boss").count(), 0);

        // "enemy" tag still exists (e2 has it).
        let mut all: Vec<_> = index.all_tags().collect();
        all.sort();
        assert_eq!(all, vec!["enemy"]);
    }

    #[test]
    fn tag_index_remove_entity_nonexistent() {
        let mut world = hecs::World::new();
        let e = world.spawn(());
        let mut index = EntityTagIndex::new();

        // Should not panic.
        index.remove_entity(e);
    }

    #[test]
    fn tag_index_add_duplicate_tag_is_noop() {
        let mut world = hecs::World::new();
        let e = world.spawn(());
        let mut index = EntityTagIndex::new();

        index.add_tag(e, "enemy");
        index.add_tag(e, "enemy");

        assert_eq!(index.entities_with_tag("enemy").count(), 1);
        assert_eq!(index.tags_of(e).count(), 1);
    }

    #[test]
    fn tag_index_all_tags() {
        let mut world = hecs::World::new();
        let e = world.spawn(());
        let mut index = EntityTagIndex::new();

        index.add_tag(e, "a");
        index.add_tag(e, "b");
        index.add_tag(e, "c");

        let mut tags: Vec<_> = index.all_tags().collect();
        tags.sort();
        assert_eq!(tags, vec!["a", "b", "c"]);
    }

    #[test]
    fn tag_index_default() {
        let index = EntityTagIndex::default();
        assert_eq!(index.all_tags().count(), 0);
    }

    // ─── Hierarchy helpers ───────────────────────────────────────────────

    /// Build a 3-level hierarchy: grandparent -> parent -> child
    fn build_hierarchy(world: &mut hecs::World) -> (hecs::Entity, hecs::Entity, hecs::Entity) {
        let grandparent = world.spawn(());
        let parent = world.spawn((Parent {
            entity: grandparent,
            bone_index: None,
        },));
        let child = world.spawn((Parent {
            entity: parent,
            bone_index: None,
        },));
        (grandparent, parent, child)
    }

    #[test]
    fn parent_of_returns_parent() {
        let mut world = hecs::World::new();
        let (grandparent, parent, child) = build_hierarchy(&mut world);

        assert_eq!(parent_of(&world, child), Some(parent));
        assert_eq!(parent_of(&world, parent), Some(grandparent));
        assert_eq!(parent_of(&world, grandparent), None);
    }

    #[test]
    fn children_of_returns_direct_children() {
        let mut world = hecs::World::new();
        let (grandparent, parent, child) = build_hierarchy(&mut world);

        let gp_children = children_of(&world, grandparent);
        assert_eq!(gp_children, vec![parent]);

        let p_children = children_of(&world, parent);
        assert_eq!(p_children, vec![child]);

        let c_children = children_of(&world, child);
        assert!(c_children.is_empty());
    }

    #[test]
    fn children_of_multiple_children() {
        let mut world = hecs::World::new();
        let parent = world.spawn(());
        let c1 = world.spawn((Parent {
            entity: parent,
            bone_index: None,
        },));
        let c2 = world.spawn((Parent {
            entity: parent,
            bone_index: None,
        },));
        let c3 = world.spawn((Parent {
            entity: parent,
            bone_index: None,
        },));

        let mut children = children_of(&world, parent);
        children.sort_by_key(|e| e.id());

        let mut expected = vec![c1, c2, c3];
        expected.sort_by_key(|e| e.id());

        assert_eq!(children, expected);
    }

    #[test]
    fn root_of_walks_to_top() {
        let mut world = hecs::World::new();
        let (grandparent, parent, child) = build_hierarchy(&mut world);

        assert_eq!(root_of(&world, child), grandparent);
        assert_eq!(root_of(&world, parent), grandparent);
        assert_eq!(root_of(&world, grandparent), grandparent);
    }

    #[test]
    fn descendants_of_collects_all() {
        let mut world = hecs::World::new();
        let (grandparent, parent, child) = build_hierarchy(&mut world);

        // Add a second child under parent.
        let child2 = world.spawn((Parent {
            entity: parent,
            bone_index: None,
        },));

        let mut desc = descendants_of(&world, grandparent);
        desc.sort_by_key(|e| e.id());

        let mut expected = vec![parent, child, child2];
        expected.sort_by_key(|e| e.id());

        assert_eq!(desc, expected);
    }

    #[test]
    fn descendants_of_leaf_is_empty() {
        let mut world = hecs::World::new();
        let leaf = world.spawn(());

        assert!(descendants_of(&world, leaf).is_empty());
    }

    #[test]
    fn descendants_of_excludes_root() {
        let mut world = hecs::World::new();
        let root = world.spawn(());
        let child = world.spawn((Parent {
            entity: root,
            bone_index: None,
        },));

        let desc = descendants_of(&world, root);
        assert_eq!(desc, vec![child]);
        assert!(!desc.contains(&root));
    }

    // ─── Name uniqueness (spec 4.3) ─────────────────────────────────────

    use crate::components::EditorMetadata;

    #[test]
    fn ensure_unique_name_no_collision() {
        let mut world = hecs::World::new();
        let e = world.spawn((EditorMetadata {
            name: "Guard".into(),
            tags: vec![],
            locked: false,
        },));

        assert_eq!(super::ensure_unique_name(&world, e, "Guard"), "Guard");
    }

    #[test]
    fn ensure_unique_name_collision_adds_suffix() {
        let mut world = hecs::World::new();
        // Existing sibling (root level)
        world.spawn((EditorMetadata {
            name: "Guard".into(),
            tags: vec![],
            locked: false,
        },));

        // New entity to check
        let e2 = world.spawn((EditorMetadata {
            name: "Guard".into(),
            tags: vec![],
            locked: false,
        },));

        assert_eq!(super::ensure_unique_name(&world, e2, "Guard"), "Guard_2");
    }

    #[test]
    fn ensure_unique_name_increments_suffix() {
        let mut world = hecs::World::new();
        world.spawn((EditorMetadata {
            name: "Guard".into(),
            tags: vec![],
            locked: false,
        },));
        world.spawn((EditorMetadata {
            name: "Guard_2".into(),
            tags: vec![],
            locked: false,
        },));

        let e3 = world.spawn((EditorMetadata {
            name: "Guard".into(),
            tags: vec![],
            locked: false,
        },));

        assert_eq!(super::ensure_unique_name(&world, e3, "Guard"), "Guard_3");
    }

    #[test]
    fn strip_numeric_suffix_works() {
        assert_eq!(super::strip_numeric_suffix("Guard_3"), "Guard");
        assert_eq!(super::strip_numeric_suffix("Guard"), "Guard");
        assert_eq!(super::strip_numeric_suffix("A_B_42"), "A_B");
        assert_eq!(super::strip_numeric_suffix("_2"), "");
        assert_eq!(super::strip_numeric_suffix("NoSuffix_abc"), "NoSuffix_abc");
    }
}
