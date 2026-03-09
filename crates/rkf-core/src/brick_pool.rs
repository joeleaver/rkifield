//! CPU-side brick pool with free-list allocation.
//!
//! [`Pool<T>`] is a fixed-capacity, contiguous pool of items managed by a free list.
//! Used for the CPU side of the GPU brick pool — items are stored contiguously for
//! bulk upload. Allocation pops from the free list; deallocation pushes back.
//!
//! Type aliases provide named pools for each brick type:
//! - [`BrickPool`] — core geometry ([`Brick`])
//! - [`BonePool`] — bone companion ([`BoneBrick`])
//! - [`VolumetricPool`] — volumetric companion ([`VolumetricBrick`])
//! - [`ColorPool`] — color companion ([`ColorBrick`])

use crate::brick::Brick;
use crate::brick_geometry::BrickGeometry;
use crate::companion::{BoneBrick, ColorBrick, VolumetricBrick};
use crate::sdf_cache::SdfCache;

/// A fixed-capacity pool of items managed by a free list.
///
/// All slots are pre-allocated at construction. [`allocate`](Pool::allocate) pops a
/// slot from the free list; [`deallocate`](Pool::deallocate) pushes it back and
/// resets the item to its default value.
///
/// The backing storage is a contiguous `Vec<T>` suitable for GPU buffer upload
/// via [`as_slice`](Pool::as_slice).
#[derive(Debug)]
pub struct Pool<T> {
    items: Vec<T>,
    free_list: Vec<u32>,
}

impl<T: Default + Clone> Pool<T> {
    /// Create a new pool with the given capacity.
    ///
    /// All slots start on the free list (unallocated). Items are initialized
    /// to `T::default()`.
    pub fn new(capacity: u32) -> Self {
        let items = vec![T::default(); capacity as usize];
        // Reverse order so pop() yields 0, 1, 2, … (low slots first)
        let free_list: Vec<u32> = (0..capacity).rev().collect();
        Self { items, free_list }
    }

    /// Allocate a slot from the pool. Returns `None` if the pool is full.
    pub fn allocate(&mut self) -> Option<u32> {
        self.free_list.pop()
    }

    /// Return a slot to the free list, resetting its contents to default.
    ///
    /// # Panics
    ///
    /// Panics if `slot` is out of range (>= capacity).
    pub fn deallocate(&mut self, slot: u32) {
        assert!(
            (slot as usize) < self.items.len(),
            "deallocate: slot {slot} out of range (capacity {})",
            self.items.len()
        );
        self.items[slot as usize] = T::default();
        self.free_list.push(slot);
    }

    /// Get a reference to the item at the given slot.
    ///
    /// # Panics
    ///
    /// Panics if `slot` is out of range.
    #[inline]
    pub fn get(&self, slot: u32) -> &T {
        &self.items[slot as usize]
    }

    /// Get a mutable reference to the item at the given slot.
    ///
    /// # Panics
    ///
    /// Panics if `slot` is out of range.
    #[inline]
    pub fn get_mut(&mut self, slot: u32) -> &mut T {
        &mut self.items[slot as usize]
    }

    /// Total capacity (number of slots) in the pool.
    #[inline]
    pub fn capacity(&self) -> u32 {
        self.items.len() as u32
    }

    /// Number of free (unallocated) slots.
    #[inline]
    pub fn free_count(&self) -> u32 {
        self.free_list.len() as u32
    }

    /// Number of currently allocated slots.
    #[inline]
    pub fn allocated_count(&self) -> u32 {
        self.capacity() - self.free_count()
    }

    /// Returns `true` if no slots are allocated.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.allocated_count() == 0
    }

    /// Returns `true` if all slots are allocated (free list exhausted).
    #[inline]
    pub fn is_full(&self) -> bool {
        self.free_list.is_empty()
    }

    /// Allocate multiple slots at once. Returns `None` if the pool doesn't
    /// have enough free slots.
    ///
    /// On success, returns a `Vec` of `count` slot indices.
    /// On failure, no slots are consumed.
    pub fn allocate_range(&mut self, count: u32) -> Option<Vec<u32>> {
        if self.free_count() < count {
            return None;
        }
        let mut slots = Vec::with_capacity(count as usize);
        for _ in 0..count {
            // Safe: we checked free_count >= count above.
            slots.push(self.free_list.pop().unwrap());
        }
        Some(slots)
    }

    /// Return multiple slots to the free list, resetting their contents to default.
    ///
    /// # Panics
    ///
    /// Panics if any slot is out of range (>= capacity).
    pub fn deallocate_range(&mut self, slots: &[u32]) {
        for &slot in slots {
            self.deallocate(slot);
        }
    }

    /// Backing slice — suitable for GPU buffer upload.
    #[inline]
    pub fn as_slice(&self) -> &[T] {
        &self.items
    }

    /// Mutable backing slice — for in-place SDF computation.
    #[inline]
    pub fn as_slice_mut(&mut self) -> &mut [T] {
        &mut self.items
    }

    /// Grow the pool to `new_capacity` slots, preserving existing allocations.
    ///
    /// Slots `old_capacity..new_capacity` are added to the free list.
    /// The backing `Vec` is extended with `T::default()` values so the
    /// contiguous layout required for GPU upload is maintained.
    ///
    /// # Panics
    ///
    /// Panics if `new_capacity < current capacity`.
    pub fn grow(&mut self, new_capacity: u32) {
        let old = self.capacity();
        assert!(new_capacity >= old, "pool cannot shrink");
        if new_capacity == old { return; }
        self.items.resize(new_capacity as usize, T::default());
        // Push new slots in reverse so pop() yields them in ascending order.
        for i in (old..new_capacity).rev() {
            self.free_list.push(i);
        }
    }
}

/// Core geometry brick pool — stores [`Brick`]s (4 KB each).
pub type BrickPool = Pool<Brick>;

/// Bone data companion pool — stores [`BoneBrick`]s (4 KB each).
pub type BonePool = Pool<BoneBrick>;

/// Volumetric data companion pool — stores [`VolumetricBrick`]s (2 KB each).
pub type VolumetricPool = Pool<VolumetricBrick>;

/// Color data companion pool — stores [`ColorBrick`]s (2 KB each).
pub type ColorPool = Pool<ColorBrick>;

/// Geometry-first brick pool — stores [`BrickGeometry`] (variable size, CPU-only).
pub type GeometryPool = Pool<BrickGeometry>;

/// SDF cache pool — stores [`SdfCache`] (1 KB each, derived from geometry).
pub type SdfCachePool = Pool<SdfCache>;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::VoxelSample;
    use half::f16;

    // ------ Construction ------

    #[test]
    fn new_pool_has_correct_capacity() {
        let pool: BrickPool = Pool::new(64);
        assert_eq!(pool.capacity(), 64);
    }

    #[test]
    fn new_pool_all_slots_free() {
        let pool: BrickPool = Pool::new(64);
        assert_eq!(pool.free_count(), 64);
        assert_eq!(pool.allocated_count(), 0);
        assert!(pool.is_empty());
        assert!(!pool.is_full());
    }

    #[test]
    fn new_pool_zero_capacity() {
        let pool: BrickPool = Pool::new(0);
        assert_eq!(pool.capacity(), 0);
        assert_eq!(pool.free_count(), 0);
        assert!(pool.is_empty());
        assert!(pool.is_full());
    }

    // ------ Allocation ------

    #[test]
    fn allocate_returns_sequential_slots() {
        let mut pool: BrickPool = Pool::new(4);
        assert_eq!(pool.allocate(), Some(0));
        assert_eq!(pool.allocate(), Some(1));
        assert_eq!(pool.allocate(), Some(2));
        assert_eq!(pool.allocate(), Some(3));
    }

    #[test]
    fn allocate_returns_none_when_full() {
        let mut pool: BrickPool = Pool::new(2);
        assert!(pool.allocate().is_some());
        assert!(pool.allocate().is_some());
        assert_eq!(pool.allocate(), None);
    }

    #[test]
    fn allocate_updates_counts() {
        let mut pool: BrickPool = Pool::new(4);
        pool.allocate();
        pool.allocate();
        assert_eq!(pool.allocated_count(), 2);
        assert_eq!(pool.free_count(), 2);
        assert!(!pool.is_empty());
        assert!(!pool.is_full());
    }

    #[test]
    fn allocate_all_makes_pool_full() {
        let mut pool: BrickPool = Pool::new(3);
        for _ in 0..3 {
            pool.allocate();
        }
        assert!(pool.is_full());
        assert!(!pool.is_empty());
    }

    // ------ Deallocation ------

    #[test]
    fn deallocate_frees_slot_for_reuse() {
        let mut pool: BrickPool = Pool::new(2);
        let s0 = pool.allocate().unwrap();
        let _s1 = pool.allocate().unwrap();
        assert!(pool.is_full());

        pool.deallocate(s0);
        assert_eq!(pool.free_count(), 1);
        assert_eq!(pool.allocated_count(), 1);

        // Re-allocate should return the freed slot
        let s2 = pool.allocate().unwrap();
        assert_eq!(s2, s0);
    }

    #[test]
    fn deallocate_resets_item_to_default() {
        let mut pool: BrickPool = Pool::new(4);
        let slot = pool.allocate().unwrap();

        // Modify the brick
        let v = VoxelSample::new(1.0, 42, 0, 0, 0);
        pool.get_mut(slot).set(0, 0, 0, v);
        assert_eq!(pool.get(slot).sample(0, 0, 0).material_id(), 42);

        // Deallocate resets
        pool.deallocate(slot);
        let slot2 = pool.allocate().unwrap();
        assert_eq!(slot2, slot);
        // Should be back to default (infinity distance)
        assert!(pool.get(slot2).sample(0, 0, 0).distance() == f16::INFINITY);
    }

    #[test]
    #[should_panic(expected = "out of range")]
    fn deallocate_out_of_range_panics() {
        let mut pool: BrickPool = Pool::new(4);
        pool.deallocate(4);
    }

    // ------ Get / Get Mut ------

    #[test]
    fn get_returns_default_brick() {
        let pool: BrickPool = Pool::new(4);
        let brick = pool.get(0);
        assert!(brick.sample(0, 0, 0).distance() == f16::INFINITY);
    }

    #[test]
    fn get_mut_allows_modification() {
        let mut pool: BrickPool = Pool::new(4);
        let slot = pool.allocate().unwrap();
        let v = VoxelSample::new(-0.5, 100, 255, 0, 0);
        pool.get_mut(slot).set(3, 4, 5, v);
        assert_eq!(pool.get(slot).sample(3, 4, 5), v);
    }

    #[test]
    #[should_panic]
    fn get_out_of_range_panics() {
        let pool: BrickPool = Pool::new(4);
        pool.get(4);
    }

    // ------ as_slice ------

    #[test]
    fn as_slice_has_correct_length() {
        let pool: BrickPool = Pool::new(16);
        assert_eq!(pool.as_slice().len(), 16);
    }

    #[test]
    fn as_slice_reflects_mutations() {
        let mut pool: BrickPool = Pool::new(4);
        let slot = pool.allocate().unwrap();
        let v = VoxelSample::new(2.0, 7, 0, 0, 0);
        pool.get_mut(slot).set(0, 0, 0, v);
        assert_eq!(pool.as_slice()[slot as usize].sample(0, 0, 0), v);
    }

    // ------ Companion pools ------

    #[test]
    fn bone_pool_allocate_deallocate() {
        let mut pool: BonePool = Pool::new(8);
        let s = pool.allocate().unwrap();
        assert_eq!(pool.allocated_count(), 1);
        pool.deallocate(s);
        assert_eq!(pool.allocated_count(), 0);
    }

    #[test]
    fn volumetric_pool_allocate_deallocate() {
        let mut pool: VolumetricPool = Pool::new(8);
        let s = pool.allocate().unwrap();
        assert_eq!(pool.allocated_count(), 1);
        pool.deallocate(s);
        assert_eq!(pool.allocated_count(), 0);
    }

    #[test]
    fn color_pool_allocate_deallocate() {
        let mut pool: ColorPool = Pool::new(8);
        let s = pool.allocate().unwrap();
        assert_eq!(pool.allocated_count(), 1);
        pool.deallocate(s);
        assert_eq!(pool.allocated_count(), 0);
    }

    #[test]
    fn companion_pool_get_mut_works() {
        use crate::companion::{BoneVoxel, ColorVoxel, VolumetricVoxel};

        let mut bone_pool: BonePool = Pool::new(4);
        let bs = bone_pool.allocate().unwrap();
        bone_pool
            .get_mut(bs)
            .set(1, 2, 3, BoneVoxel::new([10, 20, 30, 40], [100, 80, 50, 25]));
        assert_eq!(bone_pool.get(bs).sample(1, 2, 3).bone_index(0), 10);

        let mut vol_pool: VolumetricPool = Pool::new(4);
        let vs = vol_pool.allocate().unwrap();
        vol_pool
            .get_mut(vs)
            .set(0, 0, 0, VolumetricVoxel::new(0.5, 1.0));
        assert!((vol_pool.get(vs).sample(0, 0, 0).density_f32() - 0.5).abs() < 0.01);

        let mut col_pool: ColorPool = Pool::new(4);
        let cs = col_pool.allocate().unwrap();
        col_pool
            .get_mut(cs)
            .set(7, 7, 7, ColorVoxel::new(255, 128, 64, 200));
        assert_eq!(col_pool.get(cs).sample(7, 7, 7).red(), 255);
    }

    // ------ Bulk allocation ------

    #[test]
    fn allocate_range_returns_correct_count() {
        let mut pool: BrickPool = Pool::new(16);
        let slots = pool.allocate_range(5).unwrap();
        assert_eq!(slots.len(), 5);
        assert_eq!(pool.allocated_count(), 5);
        assert_eq!(pool.free_count(), 11);
    }

    #[test]
    fn allocate_range_slots_are_unique() {
        let mut pool: BrickPool = Pool::new(16);
        let slots = pool.allocate_range(10).unwrap();
        let set: std::collections::HashSet<u32> = slots.iter().copied().collect();
        assert_eq!(set.len(), 10);
    }

    #[test]
    fn allocate_range_returns_none_when_insufficient() {
        let mut pool: BrickPool = Pool::new(4);
        pool.allocate(); // use 1
        assert!(pool.allocate_range(4).is_none()); // need 4, only 3 free
        // Pool state unchanged after failed allocation.
        assert_eq!(pool.free_count(), 3);
    }

    #[test]
    fn allocate_range_zero() {
        let mut pool: BrickPool = Pool::new(4);
        let slots = pool.allocate_range(0).unwrap();
        assert!(slots.is_empty());
        assert_eq!(pool.free_count(), 4);
    }

    #[test]
    fn allocate_range_full_pool() {
        let mut pool: BrickPool = Pool::new(8);
        let slots = pool.allocate_range(8).unwrap();
        assert_eq!(slots.len(), 8);
        assert!(pool.is_full());
    }

    // ------ Bulk deallocation ------

    #[test]
    fn deallocate_range_frees_all() {
        let mut pool: BrickPool = Pool::new(8);
        let slots = pool.allocate_range(5).unwrap();
        assert_eq!(pool.free_count(), 3);

        pool.deallocate_range(&slots);
        assert_eq!(pool.free_count(), 8);
        assert!(pool.is_empty());
    }

    #[test]
    fn deallocate_range_resets_items() {
        let mut pool: BrickPool = Pool::new(4);
        let slots = pool.allocate_range(3).unwrap();

        // Modify a brick.
        let v = VoxelSample::new(1.0, 99, 0, 0, 0);
        pool.get_mut(slots[0]).set(0, 0, 0, v);
        assert_eq!(pool.get(slots[0]).sample(0, 0, 0).material_id(), 99);

        pool.deallocate_range(&slots);

        // Re-allocate and verify reset.
        let new_slots = pool.allocate_range(3).unwrap();
        for &s in &new_slots {
            assert!(pool.get(s).sample(0, 0, 0).distance() == f16::INFINITY);
        }
    }

    #[test]
    fn deallocate_range_empty_slice() {
        let mut pool: BrickPool = Pool::new(4);
        pool.allocate_range(2).unwrap();
        pool.deallocate_range(&[]);
        assert_eq!(pool.allocated_count(), 2);
    }

    #[test]
    fn bulk_alloc_dealloc_cycle() {
        let mut pool: BrickPool = Pool::new(16);

        // Allocate in batches.
        let batch1 = pool.allocate_range(6).unwrap();
        let batch2 = pool.allocate_range(6).unwrap();
        assert_eq!(pool.allocated_count(), 12);
        assert_eq!(pool.free_count(), 4);

        // Deallocate batch1.
        pool.deallocate_range(&batch1);
        assert_eq!(pool.free_count(), 10);

        // Allocate again — should reuse freed slots.
        let batch3 = pool.allocate_range(4).unwrap();
        assert_eq!(pool.allocated_count(), 10);

        // Deallocate everything.
        pool.deallocate_range(&batch2);
        pool.deallocate_range(&batch3);
        assert!(pool.is_empty());
    }

    // ------ Multiple alloc/dealloc cycles ------

    #[test]
    fn multiple_alloc_dealloc_cycles() {
        let mut pool: BrickPool = Pool::new(4);

        // Fill pool
        let slots: Vec<u32> = (0..4).map(|_| pool.allocate().unwrap()).collect();
        assert!(pool.is_full());

        // Free all
        for s in &slots {
            pool.deallocate(*s);
        }
        assert!(pool.is_empty());

        // Re-allocate all
        for _ in 0..4 {
            assert!(pool.allocate().is_some());
        }
        assert!(pool.is_full());
        assert_eq!(pool.allocate(), None);
    }
}
