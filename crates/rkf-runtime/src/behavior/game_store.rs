//! Gameplay state store and event system for the behavior system.
//!
//! [`GameStore`] is the central key-value store for gameplay state. It holds
//! [`GameValue`] entries indexed by string keys, provides typed get/set with
//! automatic conversion, a RON escape hatch for arbitrary serde types, a
//! per-frame event buffer, and snapshot/restore for Play/Stop.

use std::collections::HashMap;

use serde::{de::DeserializeOwned, Serialize};

use super::game_value::GameValue;

// ─── StoreEvent ──────────────────────────────────────────────────────────────

/// A gameplay event emitted during a frame.
///
/// Events are immediate — visible to systems that run later in the same frame.
/// They persist across phases within a frame and are drained once at frame end
/// via [`GameStore::drain_events`].
#[derive(Debug, Clone)]
pub struct StoreEvent {
    /// Event name (e.g. `"player_died"`, `"state_loaded"`).
    pub name: String,
    /// The entity that emitted this event, if any.
    pub source: Option<hecs::Entity>,
    /// Optional payload data.
    pub data: Option<GameValue>,
}

// ─── StoreSnapshot ───────────────────────────────────────────────────────────

/// A deep clone of the store's key-value data, used for Play/Stop.
///
/// Events are NOT included in snapshots.
#[derive(Debug, Clone)]
pub struct StoreSnapshot {
    store: HashMap<String, GameValue>,
}

// ─── GameStore ───────────────────────────────────────────────────────────────

/// The central gameplay state store.
///
/// Holds a `HashMap<String, GameValue>` for arbitrary gameplay state, plus a
/// per-frame event buffer. Designed to be owned by the runtime and accessed by
/// behavior systems each frame.
#[derive(Debug)]
pub struct GameStore {
    store: HashMap<String, GameValue>,
    events: Vec<StoreEvent>,
}

impl Default for GameStore {
    fn default() -> Self {
        Self::new()
    }
}

impl GameStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self {
            store: HashMap::new(),
            events: Vec::new(),
        }
    }

    // ── Core get/set ─────────────────────────────────────────────────────

    /// Set a key to a value, converting via `Into<GameValue>`.
    pub fn set<T: Into<GameValue>>(&mut self, key: &str, value: T) {
        self.store.insert(key.to_owned(), value.into());
    }

    /// Get a typed value by key. Returns `None` if the key is missing or the
    /// stored value cannot be converted to `T`.
    ///
    /// Internally clones the [`GameValue`] before attempting `TryFrom`, since
    /// `TryFrom` consumes the value.
    pub fn get<T: TryFrom<GameValue>>(&self, key: &str) -> Option<T> {
        self.store
            .get(key)
            .cloned()
            .and_then(|v| T::try_from(v).ok())
    }

    /// Get a reference to the raw [`GameValue`] without conversion.
    pub fn get_raw(&self, key: &str) -> Option<&GameValue> {
        self.store.get(key)
    }

    /// Remove a key, returning its value if it existed.
    pub fn remove(&mut self, key: &str) -> Option<GameValue> {
        self.store.remove(key)
    }

    /// Remove all keys that start with the given prefix.
    pub fn remove_prefix(&mut self, prefix: &str) {
        self.store.retain(|k, _| !k.starts_with(prefix));
    }

    /// Iterate over all entries whose key starts with the given prefix.
    pub fn list<'a>(&'a self, prefix: &'a str) -> impl Iterator<Item = (&'a str, &'a GameValue)> {
        self.store
            .iter()
            .filter(move |(k, _)| k.starts_with(prefix))
            .map(|(k, v)| (k.as_str(), v))
    }

    // ── RON escape hatch ─────────────────────────────────────────────────

    /// Serialize an arbitrary `Serialize` value to RON and store it as
    /// `GameValue::Ron`.
    pub fn set_ron<T: Serialize>(&mut self, key: &str, value: &T) {
        let ron_str = ron::to_string(value).expect("set_ron: RON serialization failed");
        self.store.insert(key.to_owned(), GameValue::Ron(ron_str));
    }

    /// Deserialize a `GameValue::Ron` entry back to a concrete type.
    /// Returns `None` if the key is missing, the value is not `Ron`, or
    /// deserialization fails.
    pub fn get_ron<T: DeserializeOwned>(&self, key: &str) -> Option<T> {
        match self.store.get(key) {
            Some(GameValue::Ron(s)) => ron::from_str(s).ok(),
            _ => None,
        }
    }

    // ── Event system ─────────────────────────────────────────────────────

    /// Emit a gameplay event. Events are appended to the buffer and are
    /// visible to systems that run later in the same frame.
    pub fn emit(
        &mut self,
        name: &str,
        source: Option<hecs::Entity>,
        data: Option<GameValue>,
    ) {
        self.events.push(StoreEvent {
            name: name.to_owned(),
            source,
            data,
        });
    }

    /// Iterate over events matching the given name.
    pub fn events<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a StoreEvent> {
        self.events.iter().filter(move |e| e.name == name)
    }

    /// Clear the event buffer. Called by the engine at frame end.
    pub fn drain_events(&mut self) {
        self.events.clear();
    }

    // ── Save / load / snapshot ───────────────────────────────────────────

    /// Serialize the entire store to a RON string. Events are NOT included.
    pub fn save_to_ron(&self) -> String {
        ron::to_string(&self.store).expect("save_to_ron: RON serialization failed")
    }

    /// Deserialize a RON string, overwriting the entire store. Events are
    /// cleared, then a `"state_loaded"` event is emitted.
    pub fn load_from_ron(&mut self, data: &str) -> Result<(), String> {
        let parsed: HashMap<String, GameValue> =
            ron::from_str(data).map_err(|e| format!("load_from_ron: {e}"))?;
        self.store = parsed;
        self.events.clear();
        self.emit("state_loaded", None, None);
        Ok(())
    }

    /// Deep-clone the store for Play/Stop. Events are NOT included.
    pub fn snapshot(&self) -> StoreSnapshot {
        StoreSnapshot {
            store: self.store.clone(),
        }
    }

    /// Restore the store from a snapshot, clearing events.
    pub fn restore(&mut self, snapshot: StoreSnapshot) {
        self.store = snapshot.store;
        self.events.clear();
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{IVec3, Quat, Vec3};
    use rkf_core::WorldPosition;
    use serde::Deserialize;

    // ── behavior-1.1: core get/set ───────────────────────────────────────

    #[test]
    fn set_get_bool() {
        let mut store = GameStore::new();
        store.set("flag", true);
        assert_eq!(store.get::<bool>("flag"), Some(true));
    }

    #[test]
    fn set_get_int() {
        let mut store = GameStore::new();
        store.set("score", 42_i64);
        assert_eq!(store.get::<i64>("score"), Some(42));
        // i32 conversion also works
        assert_eq!(store.get::<i32>("score"), Some(42));
    }

    #[test]
    fn set_get_float() {
        let mut store = GameStore::new();
        store.set("speed", 3.14_f64);
        let val: f64 = store.get("speed").unwrap();
        assert!((val - 3.14).abs() < 1e-12);
        // f32 conversion
        let val32: f32 = store.get("speed").unwrap();
        assert!((val32 - 3.14).abs() < 1e-5);
    }

    #[test]
    fn set_get_string() {
        let mut store = GameStore::new();
        store.set("name", "Alice");
        assert_eq!(store.get::<String>("name"), Some("Alice".to_owned()));
    }

    #[test]
    fn set_get_vec3() {
        let mut store = GameStore::new();
        let v = Vec3::new(1.0, 2.0, 3.0);
        store.set("velocity", v);
        assert_eq!(store.get::<Vec3>("velocity"), Some(v));
    }

    #[test]
    fn set_get_world_position() {
        let mut store = GameStore::new();
        let wp = WorldPosition {
            chunk: IVec3::new(10, 0, -5),
            local: Vec3::new(1.0, 2.0, 3.0),
        };
        store.set("spawn_point", wp.clone());
        assert_eq!(store.get::<WorldPosition>("spawn_point"), Some(wp));
    }

    #[test]
    fn set_get_quat() {
        let mut store = GameStore::new();
        let q = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        store.set("orientation", q);
        let got: Quat = store.get("orientation").unwrap();
        assert!((got.x - q.x).abs() < 1e-6);
        assert!((got.y - q.y).abs() < 1e-6);
    }

    #[test]
    fn get_missing_key_returns_none() {
        let store = GameStore::new();
        assert_eq!(store.get::<bool>("nonexistent"), None);
    }

    #[test]
    fn get_type_mismatch_returns_none() {
        let mut store = GameStore::new();
        store.set("flag", true);
        // Stored as Bool, try to get as i64
        assert_eq!(store.get::<i64>("flag"), None);
    }

    #[test]
    fn get_raw() {
        let mut store = GameStore::new();
        store.set("x", 7_i64);
        assert_eq!(store.get_raw("x"), Some(&GameValue::Int(7)));
        assert_eq!(store.get_raw("missing"), None);
    }

    #[test]
    fn set_overwrites() {
        let mut store = GameStore::new();
        store.set("key", 1_i64);
        store.set("key", 2_i64);
        assert_eq!(store.get::<i64>("key"), Some(2));
    }

    #[test]
    fn remove() {
        let mut store = GameStore::new();
        store.set("key", 42_i64);
        let removed = store.remove("key");
        assert_eq!(removed, Some(GameValue::Int(42)));
        assert_eq!(store.get::<i64>("key"), None);
        // Remove missing key
        assert_eq!(store.remove("key"), None);
    }

    #[test]
    fn remove_prefix() {
        let mut store = GameStore::new();
        store.set("player.health", 100_i64);
        store.set("player.name", "Bob");
        store.set("enemy.health", 50_i64);
        store.set("score", 0_i64);

        store.remove_prefix("player.");

        assert_eq!(store.get::<i64>("player.health"), None);
        assert_eq!(store.get::<String>("player.name"), None);
        // Other keys untouched
        assert_eq!(store.get::<i64>("enemy.health"), Some(50));
        assert_eq!(store.get::<i64>("score"), Some(0));
    }

    #[test]
    fn list_prefix() {
        let mut store = GameStore::new();
        store.set("inv.sword", 1_i64);
        store.set("inv.shield", 1_i64);
        store.set("inv.potion", 3_i64);
        store.set("score", 100_i64);

        let mut items: Vec<(&str, &GameValue)> = store.list("inv.").collect();
        items.sort_by_key(|(k, _)| *k);
        assert_eq!(items.len(), 3);
        assert_eq!(items[0].0, "inv.potion");
        assert_eq!(items[1].0, "inv.shield");
        assert_eq!(items[2].0, "inv.sword");
    }

    #[test]
    fn list_empty_prefix_returns_all() {
        let mut store = GameStore::new();
        store.set("a", 1_i64);
        store.set("b", 2_i64);
        assert_eq!(store.list("").count(), 2);
    }

    // ── behavior-1.2: RON escape hatch ───────────────────────────────────

    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
    struct CustomData {
        x: f32,
        y: f32,
        label: String,
    }

    #[test]
    fn set_ron_get_ron_roundtrip() {
        let mut store = GameStore::new();
        let data = CustomData {
            x: 1.5,
            y: -3.0,
            label: "test".to_owned(),
        };
        store.set_ron("custom", &data);
        let got: CustomData = store.get_ron("custom").unwrap();
        assert_eq!(got, data);
    }

    #[test]
    fn get_ron_missing_key() {
        let store = GameStore::new();
        assert_eq!(store.get_ron::<CustomData>("nope"), None);
    }

    #[test]
    fn get_ron_wrong_variant() {
        let mut store = GameStore::new();
        store.set("not_ron", 42_i64);
        assert_eq!(store.get_ron::<CustomData>("not_ron"), None);
    }

    #[test]
    fn get_ron_bad_data() {
        let mut store = GameStore::new();
        // Store invalid RON for the target type
        store
            .store
            .insert("bad".to_owned(), GameValue::Ron("not valid ron {{{".into()));
        assert_eq!(store.get_ron::<CustomData>("bad"), None);
    }

    // ── behavior-1.3: event system ───────────────────────────────────────

    #[test]
    fn emit_and_query_events() {
        let mut store = GameStore::new();
        store.emit("player_died", None, None);
        store.emit("player_died", None, Some(GameValue::Int(3)));
        store.emit("enemy_spawned", None, None);

        let deaths: Vec<_> = store.events("player_died").collect();
        assert_eq!(deaths.len(), 2);
        assert_eq!(deaths[0].name, "player_died");
        assert!(deaths[0].data.is_none());
        assert_eq!(deaths[1].data, Some(GameValue::Int(3)));

        let spawns: Vec<_> = store.events("enemy_spawned").collect();
        assert_eq!(spawns.len(), 1);
    }

    #[test]
    fn events_visible_within_same_frame() {
        let mut store = GameStore::new();

        // "Phase 1" emits
        store.emit("phase1_done", None, None);

        // "Phase 2" can see phase 1's events
        assert_eq!(store.events("phase1_done").count(), 1);

        // "Phase 2" also emits
        store.emit("phase2_done", None, None);

        // Both visible
        assert_eq!(store.events("phase1_done").count(), 1);
        assert_eq!(store.events("phase2_done").count(), 1);
    }

    #[test]
    fn drain_events_clears() {
        let mut store = GameStore::new();
        store.emit("tick", None, None);
        store.emit("tick", None, None);
        assert_eq!(store.events("tick").count(), 2);

        store.drain_events();
        assert_eq!(store.events("tick").count(), 0);
    }

    #[test]
    fn event_with_source_entity() {
        let mut store = GameStore::new();
        // hecs::Entity can be constructed via a World for testing
        let mut world = hecs::World::new();
        let entity = world.spawn(());
        store.emit("hit", Some(entity), Some(GameValue::Float(25.0)));

        let hits: Vec<_> = store.events("hit").collect();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].source, Some(entity));
        assert_eq!(hits[0].data, Some(GameValue::Float(25.0)));
    }

    #[test]
    fn no_matching_events() {
        let store = GameStore::new();
        assert_eq!(store.events("anything").count(), 0);
    }

    // ── behavior-1.4: save / load / snapshot / restore ───────────────────

    #[test]
    fn save_load_roundtrip() {
        let mut store = GameStore::new();
        store.set("health", 100_i64);
        store.set("name", "Player1");
        store.set("pos", Vec3::new(1.0, 2.0, 3.0));

        let ron_data = store.save_to_ron();

        let mut store2 = GameStore::new();
        store2.load_from_ron(&ron_data).unwrap();

        assert_eq!(store2.get::<i64>("health"), Some(100));
        assert_eq!(store2.get::<String>("name"), Some("Player1".to_owned()));
        assert_eq!(store2.get::<Vec3>("pos"), Some(Vec3::new(1.0, 2.0, 3.0)));
    }

    #[test]
    fn load_overwrites_existing() {
        let mut store = GameStore::new();
        store.set("old_key", 999_i64);
        store.set("shared_key", 1_i64);

        let mut source = GameStore::new();
        source.set("shared_key", 2_i64);
        source.set("new_key", 3_i64);
        let ron_data = source.save_to_ron();

        store.load_from_ron(&ron_data).unwrap();

        // Old key is gone (full overwrite)
        assert_eq!(store.get::<i64>("old_key"), None);
        // New values present
        assert_eq!(store.get::<i64>("shared_key"), Some(2));
        assert_eq!(store.get::<i64>("new_key"), Some(3));
    }

    #[test]
    fn load_from_ron_emits_state_loaded() {
        let mut store = GameStore::new();
        let source = GameStore::new();
        let ron_data = source.save_to_ron();

        store.load_from_ron(&ron_data).unwrap();

        let loaded: Vec<_> = store.events("state_loaded").collect();
        assert_eq!(loaded.len(), 1);
        assert!(loaded[0].source.is_none());
        assert!(loaded[0].data.is_none());
    }

    #[test]
    fn load_from_ron_invalid_data() {
        let mut store = GameStore::new();
        store.set("keep", 1_i64);

        let result = store.load_from_ron("this is not valid RON {{{");
        assert!(result.is_err());

        // Store should be unchanged on error
        assert_eq!(store.get::<i64>("keep"), Some(1));
    }

    #[test]
    fn load_clears_prior_events() {
        let mut store = GameStore::new();
        store.emit("old_event", None, None);
        assert_eq!(store.events("old_event").count(), 1);

        let source = GameStore::new();
        let ron_data = source.save_to_ron();
        store.load_from_ron(&ron_data).unwrap();

        // Old events cleared; only state_loaded remains
        assert_eq!(store.events("old_event").count(), 0);
        assert_eq!(store.events("state_loaded").count(), 1);
    }

    #[test]
    fn snapshot_restore_roundtrip() {
        let mut store = GameStore::new();
        store.set("health", 100_i64);
        store.set("name", "Hero");
        store.emit("some_event", None, None);

        let snap = store.snapshot();

        // Mutate the store
        store.set("health", 50_i64);
        store.remove("name");
        store.emit("another_event", None, None);

        // Restore
        store.restore(snap);

        assert_eq!(store.get::<i64>("health"), Some(100));
        assert_eq!(store.get::<String>("name"), Some("Hero".to_owned()));
        // Events cleared on restore
        assert_eq!(store.events("some_event").count(), 0);
        assert_eq!(store.events("another_event").count(), 0);
    }

    #[test]
    fn snapshot_is_independent_clone() {
        let mut store = GameStore::new();
        store.set("x", 1_i64);

        let snap = store.snapshot();

        // Modify original after snapshot
        store.set("x", 999_i64);
        store.set("y", 2_i64);

        // Restore brings back the snapshot state
        store.restore(snap);
        assert_eq!(store.get::<i64>("x"), Some(1));
        assert_eq!(store.get::<i64>("y"), None);
    }

    #[test]
    fn save_excludes_events() {
        let mut store = GameStore::new();
        store.set("val", 42_i64);
        store.emit("noise", None, Some(GameValue::String("data".into())));

        let ron_data = store.save_to_ron();

        // Load into fresh store — should have val but no events (except state_loaded)
        let mut store2 = GameStore::new();
        store2.load_from_ron(&ron_data).unwrap();
        assert_eq!(store2.get::<i64>("val"), Some(42));
        assert_eq!(store2.events("noise").count(), 0);
        assert_eq!(store2.events("state_loaded").count(), 1);
    }

    // ── Edge cases ───────────────────────────────────────────────────────

    #[test]
    fn empty_store_operations() {
        let mut store = GameStore::new();
        assert_eq!(store.get::<bool>("x"), None);
        assert_eq!(store.get_raw("x"), None);
        assert_eq!(store.remove("x"), None);
        store.remove_prefix("anything");
        assert_eq!(store.list("").count(), 0);
        assert_eq!(store.events("x").count(), 0);
        store.drain_events(); // no-op on empty
    }

    #[test]
    fn color_and_list_values() {
        let mut store = GameStore::new();

        // Color (no From impl, must use set with GameValue directly)
        store
            .store
            .insert("color".to_owned(), GameValue::Color([1.0, 0.0, 0.0, 1.0]));
        match store.get_raw("color") {
            Some(GameValue::Color(c)) => assert_eq!(*c, [1.0, 0.0, 0.0, 1.0]),
            other => panic!("expected Color, got {:?}", other),
        }

        // List
        let list = GameValue::List(vec![GameValue::Int(1), GameValue::Bool(false)]);
        store.store.insert("items".to_owned(), list.clone());
        assert_eq!(store.get_raw("items"), Some(&list));
    }

    #[test]
    fn ron_escape_hatch_with_nested_struct() {
        #[derive(Debug, PartialEq, Serialize, Deserialize)]
        struct Nested {
            inner: Vec<(String, f64)>,
        }

        let mut store = GameStore::new();
        let data = Nested {
            inner: vec![("a".into(), 1.0), ("b".into(), 2.0)],
        };
        store.set_ron("nested", &data);
        let got: Nested = store.get_ron("nested").unwrap();
        assert_eq!(got, data);
    }

    #[test]
    fn default_trait() {
        let store = GameStore::default();
        assert_eq!(store.list("").count(), 0);
    }
}
