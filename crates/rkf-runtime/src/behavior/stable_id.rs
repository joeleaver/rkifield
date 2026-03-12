use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Persistent entity identifier — auto-assigned, globally unique.
///
/// Every entity receives a StableId at creation. It never changes across
/// the entity's lifetime and is used for serialization (entity references
/// become StableId UUIDs in scene files).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StableId(pub Uuid);

impl StableId {
    /// Create a new random StableId (UUID v4).
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Create a StableId from an existing UUID.
    pub fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }

    /// Get the inner UUID.
    pub fn uuid(&self) -> Uuid {
        self.0
    }
}

impl Default for StableId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for StableId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn new_creates_unique_ids() {
        let a = StableId::new();
        let b = StableId::new();
        assert_ne!(a, b);
    }

    #[test]
    fn from_uuid_roundtrip() {
        let uuid = Uuid::new_v4();
        let id = StableId::from_uuid(uuid);
        assert_eq!(id.uuid(), uuid);
    }

    #[test]
    fn display_matches_uuid() {
        let uuid = Uuid::new_v4();
        let id = StableId::from_uuid(uuid);
        assert_eq!(format!("{id}"), format!("{uuid}"));
    }

    #[test]
    fn equality_by_uuid() {
        let uuid = Uuid::new_v4();
        let a = StableId::from_uuid(uuid);
        let b = StableId::from_uuid(uuid);
        assert_eq!(a, b);
    }

    #[test]
    fn hashable() {
        let a = StableId::new();
        let b = StableId::new();
        let mut set = HashSet::new();
        set.insert(a);
        set.insert(b);
        set.insert(a); // duplicate
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn default_creates_new() {
        let a = StableId::default();
        let b = StableId::default();
        assert_ne!(a, b);
    }

    #[test]
    fn serde_ron_roundtrip() {
        let id = StableId::new();
        let serialized = ron::to_string(&id).expect("serialize");
        let deserialized: StableId = ron::from_str(&serialized).expect("deserialize");
        assert_eq!(id, deserialized);
    }

    #[test]
    fn serde_json_roundtrip() {
        let id = StableId::new();
        let serialized = serde_json::to_string(&id).expect("serialize");
        let deserialized: StableId = serde_json::from_str(&serialized).expect("deserialize");
        assert_eq!(id, deserialized);
    }

    #[test]
    fn inner_uuid_accessible() {
        let id = StableId::new();
        let _uuid: Uuid = id.0;
        let _uuid2: Uuid = id.uuid();
        assert_eq!(_uuid, _uuid2);
    }
}
