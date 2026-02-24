//! Game state management — scene tracking, environment control, and a typed key-value store.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// A dynamically-typed game state value.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GameValue {
    /// Boolean value.
    Bool(bool),
    /// 64-bit signed integer value.
    Int(i64),
    /// 64-bit floating-point value.
    Float(f64),
    /// UTF-8 string value.
    String(String),
    /// Ordered list of game values.
    List(Vec<GameValue>),
}

impl GameValue {
    /// Returns the contained `bool`, or `None` if this is not a `Bool`.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            GameValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Returns the contained `i64`, or `None` if this is not an `Int`.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            GameValue::Int(i) => Some(*i),
            _ => None,
        }
    }

    /// Returns the contained `f64`, or `None` if this is not a `Float`.
    pub fn as_float(&self) -> Option<f64> {
        match self {
            GameValue::Float(f) => Some(*f),
            _ => None,
        }
    }

    /// Returns a reference to the contained string slice, or `None` if this is not a `String`.
    pub fn as_string(&self) -> Option<&str> {
        match self {
            GameValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Returns a reference to the contained list slice, or `None` if this is not a `List`.
    pub fn as_list(&self) -> Option<&[GameValue]> {
        match self {
            GameValue::List(v) => Some(v.as_slice()),
            _ => None,
        }
    }
}

/// Game state store — typed key-value pairs.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GameState {
    values: HashMap<String, GameValue>,
}

impl GameState {
    /// Create an empty game state store.
    pub fn new() -> Self {
        Self {
            values: HashMap::new(),
        }
    }

    /// Insert or overwrite a value.
    pub fn set(&mut self, key: impl Into<String>, value: GameValue) {
        self.values.insert(key.into(), value);
    }

    /// Look up a value by key.
    pub fn get(&self, key: &str) -> Option<&GameValue> {
        self.values.get(key)
    }

    /// Remove and return a value by key.
    pub fn remove(&mut self, key: &str) -> Option<GameValue> {
        self.values.remove(key)
    }

    /// Returns `true` if the key exists.
    pub fn contains(&self, key: &str) -> bool {
        self.values.contains_key(key)
    }

    /// Iterate over all keys.
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.values.keys()
    }

    /// Number of stored entries.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns `true` if there are no stored entries.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Remove all entries.
    pub fn clear(&mut self) {
        self.values.clear();
    }

    /// Merge another state into this one; keys from `other` take precedence.
    pub fn merge(&mut self, other: &GameState) {
        for (key, value) in &other.values {
            self.values.insert(key.clone(), value.clone());
        }
    }
}

/// Event emitted by the game manager within a single frame.
#[derive(Debug, Clone)]
pub enum GameEvent {
    /// A scene was added to the loaded-scene list.
    SceneLoaded {
        /// Scene name.
        name: String,
    },
    /// A scene was removed from the loaded-scene list.
    SceneUnloaded {
        /// Scene name.
        name: String,
    },
    /// The active environment profile index changed.
    EnvironmentChanged {
        /// Previous index.
        from: usize,
        /// New index.
        to: usize,
    },
    /// A game-state key was created or updated.
    StateChanged {
        /// The key that changed.
        key: String,
        /// Previous value (None if the key did not previously exist).
        old: Option<GameValue>,
        /// New value.
        new: GameValue,
    },
}

/// Central game manager — ties together scene management, environment control,
/// and the typed game-state store.
pub struct GameManager {
    /// Key-value game state store.
    pub state: GameState,
    /// Scene names that are currently loaded.
    loaded_scenes: Vec<String>,
    /// Index of the currently active environment profile.
    pub active_environment: usize,
    /// Event queue accumulated during the current frame; drained each frame.
    events: Vec<GameEvent>,
    /// Total number of frames that have elapsed since this manager was created.
    pub frame: u64,
    /// Total elapsed time in seconds since this manager was created.
    pub elapsed: f64,
}

impl Default for GameManager {
    fn default() -> Self {
        Self::new()
    }
}

impl GameManager {
    /// Create a new game manager with empty state at frame 0.
    pub fn new() -> Self {
        Self {
            state: GameState::new(),
            loaded_scenes: Vec::new(),
            active_environment: 0,
            events: Vec::new(),
            frame: 0,
            elapsed: 0.0,
        }
    }

    /// Advance time by `dt` seconds and increment the frame counter.
    ///
    /// Call once per frame before processing game logic.
    pub fn update(&mut self, dt: f64) {
        self.elapsed += dt;
        self.frame += 1;
    }

    /// Set a game state value and emit a [`GameEvent::StateChanged`] event.
    pub fn set_state(&mut self, key: impl Into<String>, value: GameValue) {
        let key: String = key.into();
        let old = self.state.get(&key).cloned();
        self.state.set(key.clone(), value.clone());
        self.events.push(GameEvent::StateChanged {
            key,
            old,
            new: value,
        });
    }

    /// Look up a game state value without emitting any events.
    pub fn get_state(&self, key: &str) -> Option<&GameValue> {
        self.state.get(key)
    }

    /// Record that a scene has been loaded, emitting a [`GameEvent::SceneLoaded`] event.
    pub fn on_scene_loaded(&mut self, name: impl Into<String>) {
        let name: String = name.into();
        self.loaded_scenes.push(name.clone());
        self.events.push(GameEvent::SceneLoaded { name });
    }

    /// Record that a scene has been unloaded, emitting a [`GameEvent::SceneUnloaded`] event.
    ///
    /// Removes the first occurrence of `name` from the loaded-scenes list.
    pub fn on_scene_unloaded(&mut self, name: &str) {
        if let Some(pos) = self.loaded_scenes.iter().position(|s| s == name) {
            self.loaded_scenes.remove(pos);
        }
        self.events.push(GameEvent::SceneUnloaded {
            name: name.to_owned(),
        });
    }

    /// Change the active environment profile, emitting a [`GameEvent::EnvironmentChanged`] event.
    pub fn set_environment(&mut self, index: usize) {
        let from = self.active_environment;
        self.active_environment = index;
        self.events.push(GameEvent::EnvironmentChanged { from, to: index });
    }

    /// Returns the names of all currently loaded scenes.
    pub fn loaded_scenes(&self) -> &[String] {
        &self.loaded_scenes
    }

    /// Drain and return all events queued during this frame, clearing the internal queue.
    pub fn drain_events(&mut self) -> Vec<GameEvent> {
        std::mem::take(&mut self.events)
    }

    /// Returns `true` if the named scene is currently loaded.
    pub fn is_scene_loaded(&self, name: &str) -> bool {
        self.loaded_scenes.iter().any(|s| s == name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_manager() {
        let gm = GameManager::new();
        assert!(gm.state.is_empty());
        assert_eq!(gm.frame, 0);
        assert_eq!(gm.elapsed, 0.0);
        assert_eq!(gm.active_environment, 0);
        assert!(gm.loaded_scenes().is_empty());
    }

    #[test]
    fn update_advances_time() {
        let mut gm = GameManager::new();
        gm.update(0.016);
        assert_eq!(gm.frame, 1);
        assert!((gm.elapsed - 0.016).abs() < 1e-12);

        gm.update(0.016);
        assert_eq!(gm.frame, 2);
        assert!((gm.elapsed - 0.032).abs() < 1e-12);
    }

    #[test]
    fn set_and_get_state() {
        let mut gm = GameManager::new();
        gm.set_state("score", GameValue::Int(42));
        let _ = gm.drain_events(); // discard events
        assert_eq!(gm.get_state("score"), Some(&GameValue::Int(42)));
        assert_eq!(gm.get_state("missing"), None);
    }

    #[test]
    fn state_changed_event() {
        let mut gm = GameManager::new();
        gm.set_state("health", GameValue::Float(100.0));
        let events = gm.drain_events();
        assert_eq!(events.len(), 1);
        match &events[0] {
            GameEvent::StateChanged { key, old, new } => {
                assert_eq!(key, "health");
                assert_eq!(*old, None);
                assert_eq!(*new, GameValue::Float(100.0));
            }
            other => panic!("unexpected event: {:?}", other),
        }

        // Overwrite should record the old value.
        gm.set_state("health", GameValue::Float(50.0));
        let events = gm.drain_events();
        assert_eq!(events.len(), 1);
        match &events[0] {
            GameEvent::StateChanged { key, old, new } => {
                assert_eq!(key, "health");
                assert_eq!(*old, Some(GameValue::Float(100.0)));
                assert_eq!(*new, GameValue::Float(50.0));
            }
            other => panic!("unexpected event: {:?}", other),
        }
    }

    #[test]
    fn remove_state() {
        let mut gs = GameState::new();
        gs.set("x", GameValue::Bool(true));
        let removed = gs.remove("x");
        assert_eq!(removed, Some(GameValue::Bool(true)));
        assert!(!gs.contains("x"));
        // Removing a missing key returns None without panic.
        assert_eq!(gs.remove("x"), None);
    }

    #[test]
    fn scene_loaded_event() {
        let mut gm = GameManager::new();
        gm.on_scene_loaded("level_01");
        let events = gm.drain_events();
        assert_eq!(events.len(), 1);
        match &events[0] {
            GameEvent::SceneLoaded { name } => assert_eq!(name, "level_01"),
            other => panic!("unexpected event: {:?}", other),
        }
        assert!(gm.is_scene_loaded("level_01"));
    }

    #[test]
    fn scene_unloaded_event() {
        let mut gm = GameManager::new();
        gm.on_scene_loaded("level_01");
        let _ = gm.drain_events();

        gm.on_scene_unloaded("level_01");
        let events = gm.drain_events();
        assert_eq!(events.len(), 1);
        match &events[0] {
            GameEvent::SceneUnloaded { name } => assert_eq!(name, "level_01"),
            other => panic!("unexpected event: {:?}", other),
        }
    }

    #[test]
    fn is_scene_loaded() {
        let mut gm = GameManager::new();
        assert!(!gm.is_scene_loaded("level_01"));
        gm.on_scene_loaded("level_01");
        assert!(gm.is_scene_loaded("level_01"));
        gm.on_scene_unloaded("level_01");
        assert!(!gm.is_scene_loaded("level_01"));
    }

    #[test]
    fn environment_changed_event() {
        let mut gm = GameManager::new();
        gm.set_environment(3);
        let events = gm.drain_events();
        assert_eq!(events.len(), 1);
        match &events[0] {
            GameEvent::EnvironmentChanged { from, to } => {
                assert_eq!(*from, 0);
                assert_eq!(*to, 3);
            }
            other => panic!("unexpected event: {:?}", other),
        }
        assert_eq!(gm.active_environment, 3);
    }

    #[test]
    fn drain_events_clears() {
        let mut gm = GameManager::new();
        gm.set_state("k", GameValue::Bool(false));
        gm.on_scene_loaded("s");
        let first = gm.drain_events();
        assert_eq!(first.len(), 2);
        // Second drain must be empty.
        let second = gm.drain_events();
        assert!(second.is_empty());
    }

    #[test]
    fn game_value_accessors() {
        let b = GameValue::Bool(true);
        assert_eq!(b.as_bool(), Some(true));
        assert_eq!(b.as_int(), None);

        let i = GameValue::Int(-7);
        assert_eq!(i.as_int(), Some(-7));
        assert_eq!(i.as_float(), None);

        let f = GameValue::Float(3.14);
        assert!((f.as_float().unwrap() - 3.14).abs() < 1e-12);
        assert_eq!(f.as_string(), None);

        let s = GameValue::String("hello".to_owned());
        assert_eq!(s.as_string(), Some("hello"));
        assert_eq!(s.as_list(), None);

        let l = GameValue::List(vec![GameValue::Int(1), GameValue::Int(2)]);
        let slice = l.as_list().unwrap();
        assert_eq!(slice.len(), 2);
        assert_eq!(l.as_bool(), None);
    }

    #[test]
    fn state_merge() {
        let mut base = GameState::new();
        base.set("a", GameValue::Int(1));
        base.set("b", GameValue::Int(2));

        let mut overlay = GameState::new();
        overlay.set("b", GameValue::Int(99)); // override
        overlay.set("c", GameValue::Int(3));  // new key

        base.merge(&overlay);

        assert_eq!(base.get("a"), Some(&GameValue::Int(1)));  // untouched
        assert_eq!(base.get("b"), Some(&GameValue::Int(99))); // overridden by other
        assert_eq!(base.get("c"), Some(&GameValue::Int(3)));  // added from other
    }
}
