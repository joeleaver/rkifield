//! Asset registry with generational handles for type-safe asset references.
//!
//! Provides [`AssetRegistry<T>`] for tracking assets through their lifecycle
//! (Unloaded → Loading → Loaded → Error) with reference counting for eviction
//! eligibility.

use std::marker::PhantomData;

// ─── AssetState ────────────────────────────────────────────────────────────

/// Lifecycle state of an asset in the registry.
#[derive(Debug, Clone, PartialEq)]
pub enum AssetState {
    /// Asset metadata known but data not loaded.
    Unloaded,
    /// Asset is being loaded asynchronously.
    Loading,
    /// Asset is fully loaded and available.
    Loaded,
    /// Asset load failed with an error message.
    Error(String),
}

// ─── Handle ────────────────────────────────────────────────────────────────

/// Generational handle for type-safe asset references.
///
/// The handle encodes an index and a generation counter to detect stale
/// references. `T` is a phantom type parameter that prevents mixing handle
/// types (e.g. a `Handle<Chunk>` cannot be used where `Handle<Character>` is
/// expected).
pub struct Handle<T> {
    /// Slot index into the registry.
    pub index: u32,
    /// Generation counter — incremented when a slot is recycled.
    pub generation: u32,
    _marker: PhantomData<T>,
}

impl<T> Handle<T> {
    /// Construct a new handle. Internal use only; callers get handles from
    /// [`AssetRegistry`] methods.
    fn new(index: u32, generation: u32) -> Self {
        Self {
            index,
            generation,
            _marker: PhantomData,
        }
    }
}

// Manual trait impls so that T does not need to satisfy Clone/Copy/etc.

impl<T> std::fmt::Debug for Handle<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Handle")
            .field("index", &self.index)
            .field("generation", &self.generation)
            .finish()
    }
}

impl<T> Clone for Handle<T> {
    fn clone(&self) -> Self {
        Self::new(self.index, self.generation)
    }
}

impl<T> Copy for Handle<T> {}

impl<T> PartialEq for Handle<T> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index && self.generation == other.generation
    }
}

impl<T> Eq for Handle<T> {}

impl<T> std::hash::Hash for Handle<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.index.hash(state);
        self.generation.hash(state);
    }
}

// ─── AssetEntry ────────────────────────────────────────────────────────────

/// A single asset slot in the registry.
#[derive(Debug)]
pub struct AssetEntry<T> {
    /// The asset data (present only when state is [`AssetState::Loaded`]).
    pub data: Option<T>,
    /// Current lifecycle state.
    pub state: AssetState,
    /// Generation counter for this slot — incremented on recycle.
    pub generation: u32,
    /// Reference count (number of active logical handles held by callers).
    pub ref_count: u32,
}


// ─── AssetRegistry ─────────────────────────────────────────────────────────

/// Central registry for assets with generational handles.
///
/// Provides type-safe [`Handle<T>`] references, state tracking, and reference
/// counting for eviction eligibility.
///
/// # Slot lifecycle
///
/// Slots are reused via a free list. When a slot is recycled its generation
/// counter is incremented, invalidating all previously issued handles that
/// point to it.
pub struct AssetRegistry<T> {
    /// Asset slots (sparse; free slots are returned to `free_list`).
    entries: Vec<AssetEntry<T>>,
    /// Indices of slots that have been freed and may be reused.
    free_list: Vec<u32>,
}

impl<T> AssetRegistry<T> {
    /// Create an empty registry with no pre-allocation.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            free_list: Vec::new(),
        }
    }

    /// Create a registry with pre-allocated capacity for `capacity` slots.
    ///
    /// No handles are issued; the registry is still empty. This avoids
    /// repeated reallocations when the expected asset count is known upfront.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
            free_list: Vec::new(),
        }
    }

    // ── Internal helpers ────────────────────────────────────────────────

    /// Allocate a slot (reusing from the free list or appending) and return
    /// its index. The caller is responsible for populating the entry.
    fn allocate_slot(&mut self) -> u32 {
        if let Some(idx) = self.free_list.pop() {
            idx
        } else {
            let idx = self.entries.len() as u32;
            // Push a placeholder; caller overwrites immediately.
            self.entries.push(AssetEntry {
                data: None,
                state: AssetState::Unloaded,
                generation: 0,
                ref_count: 0,
            });
            idx
        }
    }

    /// Returns `true` if `handle` points to the live entry at that slot.
    fn generation_matches(&self, handle: &Handle<T>) -> bool {
        self.entries
            .get(handle.index as usize)
            .map_or(false, |e| e.generation == handle.generation)
    }

    // ── Public API ──────────────────────────────────────────────────────

    /// Insert a fully loaded asset and return a handle to it.
    ///
    /// Reuses a previously freed slot if available, incrementing its
    /// generation counter so that any old handles are invalidated.
    pub fn insert(&mut self, data: T) -> Handle<T> {
        let idx = self.allocate_slot();
        let entry = &mut self.entries[idx as usize];
        let generation = entry.generation; // already incremented by `remove`
        *entry = AssetEntry {
            data: Some(data),
            state: AssetState::Loaded,
            generation,
            ref_count: 0,
        };
        Handle::new(idx, generation)
    }

    /// Reserve a slot in [`AssetState::Loading`] state without providing data.
    ///
    /// Use this when kicking off an asynchronous load; later call
    /// [`set_data`](Self::set_data) to populate.
    pub fn insert_pending(&mut self) -> Handle<T> {
        let idx = self.allocate_slot();
        let entry = &mut self.entries[idx as usize];
        let generation = entry.generation;
        *entry = AssetEntry {
            data: None,
            state: AssetState::Loading,
            generation,
            ref_count: 0,
        };
        Handle::new(idx, generation)
    }

    /// Get a shared reference to asset data.
    ///
    /// Returns `None` if the handle is stale or the asset is not yet loaded.
    pub fn get(&self, handle: &Handle<T>) -> Option<&T> {
        if !self.generation_matches(handle) {
            return None;
        }
        let entry = &self.entries[handle.index as usize];
        if entry.state == AssetState::Loaded {
            entry.data.as_ref()
        } else {
            None
        }
    }

    /// Get a mutable reference to asset data.
    ///
    /// Returns `None` if the handle is stale or the asset is not yet loaded.
    pub fn get_mut(&mut self, handle: &Handle<T>) -> Option<&mut T> {
        if !self.generation_matches(handle) {
            return None;
        }
        let entry = &mut self.entries[handle.index as usize];
        if entry.state == AssetState::Loaded {
            entry.data.as_mut()
        } else {
            None
        }
    }

    /// Return the current [`AssetState`] for a handle.
    ///
    /// Returns `None` if the handle is stale.
    pub fn state(&self, handle: &Handle<T>) -> Option<&AssetState> {
        if !self.generation_matches(handle) {
            return None;
        }
        Some(&self.entries[handle.index as usize].state)
    }

    /// Overwrite the state for an existing valid handle.
    ///
    /// Does nothing if the handle is stale.
    pub fn set_state(&mut self, handle: &Handle<T>, state: AssetState) {
        if !self.generation_matches(handle) {
            return;
        }
        self.entries[handle.index as usize].state = state;
    }

    /// Provide asset data and transition the slot to [`AssetState::Loaded`].
    ///
    /// Typically called after an asynchronous load completes. Does nothing if
    /// the handle is stale.
    pub fn set_data(&mut self, handle: &Handle<T>, data: T) {
        if !self.generation_matches(handle) {
            return;
        }
        let entry = &mut self.entries[handle.index as usize];
        entry.data = Some(data);
        entry.state = AssetState::Loaded;
    }

    /// Transition the slot to [`AssetState::Error`] and clear any stored data.
    ///
    /// Does nothing if the handle is stale.
    pub fn set_error(&mut self, handle: &Handle<T>, error: String) {
        if !self.generation_matches(handle) {
            return;
        }
        let entry = &mut self.entries[handle.index as usize];
        entry.data = None;
        entry.state = AssetState::Error(error);
    }

    /// Remove an asset, returning its data.
    ///
    /// The slot is marked free (generation incremented, state reset to
    /// [`AssetState::Unloaded`]) and added to the free list for reuse. Returns
    /// `None` if the handle is stale or the slot had no data.
    pub fn remove(&mut self, handle: &Handle<T>) -> Option<T> {
        if !self.generation_matches(handle) {
            return None;
        }
        let idx = handle.index as usize;
        let entry = &mut self.entries[idx];
        let data = entry.data.take();
        entry.state = AssetState::Unloaded;
        entry.ref_count = 0;
        // Increment generation to invalidate outstanding handles.
        entry.generation = entry.generation.wrapping_add(1);
        self.free_list.push(handle.index);
        data
    }

    /// Check whether a handle is still valid (generation matches the slot).
    pub fn is_valid(&self, handle: &Handle<T>) -> bool {
        self.generation_matches(handle)
    }

    /// Increment the logical reference count for a handle.
    ///
    /// Does nothing if the handle is stale.
    pub fn add_ref(&mut self, handle: &Handle<T>) {
        if !self.generation_matches(handle) {
            return;
        }
        self.entries[handle.index as usize].ref_count =
            self.entries[handle.index as usize].ref_count.saturating_add(1);
    }

    /// Decrement the logical reference count for a handle (floor of 0).
    ///
    /// Does nothing if the handle is stale.
    pub fn release_ref(&mut self, handle: &Handle<T>) {
        if !self.generation_matches(handle) {
            return;
        }
        let rc = &mut self.entries[handle.index as usize].ref_count;
        *rc = rc.saturating_sub(1);
    }

    /// Return the current reference count for a handle.
    ///
    /// Returns `None` if the handle is stale.
    pub fn ref_count(&self, handle: &Handle<T>) -> Option<u32> {
        if !self.generation_matches(handle) {
            return None;
        }
        Some(self.entries[handle.index as usize].ref_count)
    }

    /// Number of occupied slots (total slots minus free slots).
    pub fn len(&self) -> usize {
        self.entries.len() - self.free_list.len()
    }

    /// Returns `true` if no slots are occupied.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterate over all valid (non-free) entries, yielding a handle and a
    /// reference to the [`AssetEntry`] for each.
    pub fn iter(&self) -> impl Iterator<Item = (Handle<T>, &AssetEntry<T>)> {
        let free: std::collections::HashSet<u32> = self.free_list.iter().copied().collect();
        self.entries
            .iter()
            .enumerate()
            .filter(move |(idx, _)| !free.contains(&(*idx as u32)))
            .map(|(idx, entry)| (Handle::new(idx as u32, entry.generation), entry))
    }
}

impl<T> Default for AssetRegistry<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // 1. Empty registry reports len = 0.
    #[test]
    fn new_registry_empty() {
        let reg: AssetRegistry<u32> = AssetRegistry::new();
        assert_eq!(reg.len(), 0);
        assert!(reg.is_empty());
    }

    // 2. insert returns a handle with index=0, generation=0 for first insert.
    #[test]
    fn insert_returns_handle() {
        let mut reg: AssetRegistry<u32> = AssetRegistry::new();
        let h = reg.insert(42u32);
        assert_eq!(h.index, 0);
        assert_eq!(h.generation, 0);
    }

    // 3. get returns the inserted data.
    #[test]
    fn get_returns_data() {
        let mut reg: AssetRegistry<String> = AssetRegistry::new();
        let h = reg.insert("hello".to_string());
        assert_eq!(reg.get(&h), Some(&"hello".to_string()));
    }

    // 4. get_mut allows modifying data in place.
    #[test]
    fn get_mut_modifies_data() {
        let mut reg: AssetRegistry<u32> = AssetRegistry::new();
        let h = reg.insert(1u32);
        *reg.get_mut(&h).unwrap() = 99;
        assert_eq!(reg.get(&h), Some(&99u32));
    }

    // 5. A handle with the wrong generation returns None from get.
    #[test]
    fn invalid_handle_returns_none() {
        let mut reg: AssetRegistry<u32> = AssetRegistry::new();
        let h = reg.insert(7u32);
        let bad = Handle::<u32>::new(h.index, h.generation + 1);
        assert!(reg.get(&bad).is_none());
    }

    // 6. insert_pending creates a slot in Loading state; get returns None.
    #[test]
    fn insert_pending_creates_loading() {
        let mut reg: AssetRegistry<u32> = AssetRegistry::new();
        let h = reg.insert_pending();
        assert_eq!(reg.state(&h), Some(&AssetState::Loading));
        assert!(reg.get(&h).is_none());
    }

    // 7. set_data transitions a pending slot to Loaded and makes data accessible.
    #[test]
    fn set_data_transitions_to_loaded() {
        let mut reg: AssetRegistry<u32> = AssetRegistry::new();
        let h = reg.insert_pending();
        reg.set_data(&h, 55u32);
        assert_eq!(reg.state(&h), Some(&AssetState::Loaded));
        assert_eq!(reg.get(&h), Some(&55u32));
    }

    // 8. set_error transitions the slot to Error and clears data.
    #[test]
    fn set_error_transitions() {
        let mut reg: AssetRegistry<u32> = AssetRegistry::new();
        let h = reg.insert(10u32);
        reg.set_error(&h, "io error".to_string());
        assert_eq!(
            reg.state(&h),
            Some(&AssetState::Error("io error".to_string()))
        );
        assert!(reg.get(&h).is_none());
    }

    // 9. remove returns the data and invalidates the handle.
    #[test]
    fn remove_frees_slot() {
        let mut reg: AssetRegistry<u32> = AssetRegistry::new();
        let h = reg.insert(42u32);
        let data = reg.remove(&h);
        assert_eq!(data, Some(42));
        assert!(!reg.is_valid(&h));
        assert!(reg.get(&h).is_none());
    }

    // 10. After remove + re-insert the new handle has generation=1.
    #[test]
    fn slot_reuse_with_generation() {
        let mut reg: AssetRegistry<u32> = AssetRegistry::new();
        let h1 = reg.insert(1u32);
        reg.remove(&h1);
        let h2 = reg.insert(2u32);
        // Slot 0 was freed and reused.
        assert_eq!(h2.index, 0);
        assert_eq!(h2.generation, 1);
        assert_eq!(reg.get(&h2), Some(&2u32));
    }

    // 11. The original handle is stale after the slot is reused.
    #[test]
    fn stale_handle_invalid() {
        let mut reg: AssetRegistry<u32> = AssetRegistry::new();
        let h1 = reg.insert(1u32);
        reg.remove(&h1);
        let _h2 = reg.insert(2u32);
        // h1 points to generation 0; slot now has generation 1.
        assert!(!reg.is_valid(&h1));
        assert!(reg.get(&h1).is_none());
    }

    // 12. add_ref and release_ref update the ref count correctly.
    #[test]
    fn ref_count_tracking() {
        let mut reg: AssetRegistry<u32> = AssetRegistry::new();
        let h = reg.insert(5u32);
        assert_eq!(reg.ref_count(&h), Some(0));
        reg.add_ref(&h);
        reg.add_ref(&h);
        assert_eq!(reg.ref_count(&h), Some(2));
        reg.release_ref(&h);
        assert_eq!(reg.ref_count(&h), Some(1));
        reg.release_ref(&h);
        assert_eq!(reg.ref_count(&h), Some(0));
        // Should not underflow below 0.
        reg.release_ref(&h);
        assert_eq!(reg.ref_count(&h), Some(0));
    }

    // 13. len tracks inserted and removed assets.
    #[test]
    fn len_tracks_correctly() {
        let mut reg: AssetRegistry<u32> = AssetRegistry::new();
        let h1 = reg.insert(1u32);
        let _h2 = reg.insert(2u32);
        let _h3 = reg.insert(3u32);
        assert_eq!(reg.len(), 3);
        reg.remove(&h1);
        assert_eq!(reg.len(), 2);
    }

    // 14. iter yields all valid entries.
    #[test]
    fn iter_yields_valid_entries() {
        let mut reg: AssetRegistry<u32> = AssetRegistry::new();
        let _h1 = reg.insert(10u32);
        let _h2 = reg.insert(20u32);
        let _h3 = reg.insert(30u32);
        let values: Vec<u32> = reg
            .iter()
            .filter_map(|(h, _)| reg.get(&h).copied())
            .collect();
        assert_eq!(values.len(), 3);
        // All three values present (order may vary).
        for v in [10u32, 20, 30] {
            assert!(values.contains(&v));
        }
    }

    // 15. with_capacity creates an empty but pre-allocated registry.
    #[test]
    fn with_capacity_preallocates() {
        let reg: AssetRegistry<u32> = AssetRegistry::with_capacity(10);
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
    }
}
