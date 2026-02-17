//! Cell occupancy state for spatial index.
//!
//! Encodes the occupancy state of a cell in the sparse spatial index.
//! Determines ray marcher behavior when entering a cell (skip, march, accumulate density).

/// Cell occupancy state — stored as 2 bits in spatial index occupancy bitfields.
///
/// Determines ray marcher behavior when entering a cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum CellState {
    /// Outside all geometry, no bricks allocated. Ray marcher skips with large steps.
    Empty = 0,
    /// Has bricks in the pool. Ray marcher evaluates SDF with normal marching.
    Surface = 1,
    /// Inside solid geometry, no bricks allocated. Ray marcher skips to far side of cell.
    Interior = 2,
    /// Has volumetric companion bricks. Ray marcher accumulates density/emission.
    Volumetric = 3,
}

impl CellState {
    /// Convert from raw u8 value (0-3). Returns None for invalid values.
    pub fn from_u8(value: u8) -> Option<CellState> {
        match value {
            0 => Some(CellState::Empty),
            1 => Some(CellState::Surface),
            2 => Some(CellState::Interior),
            3 => Some(CellState::Volumetric),
            _ => None,
        }
    }

    /// Convert to raw u8 value.
    pub fn as_u8(self) -> u8 {
        self as u8
    }

    /// Returns true if this state has brick pool data (Surface or Volumetric).
    pub fn has_brick_data(self) -> bool {
        matches!(self, CellState::Surface | CellState::Volumetric)
    }

    /// Returns true only for Empty state.
    pub fn is_empty(self) -> bool {
        self == CellState::Empty
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_empty() {
        let state = CellState::Empty;
        assert_eq!(CellState::from_u8(state.as_u8()), Some(state));
    }

    #[test]
    fn test_roundtrip_surface() {
        let state = CellState::Surface;
        assert_eq!(CellState::from_u8(state.as_u8()), Some(state));
    }

    #[test]
    fn test_roundtrip_interior() {
        let state = CellState::Interior;
        assert_eq!(CellState::from_u8(state.as_u8()), Some(state));
    }

    #[test]
    fn test_roundtrip_volumetric() {
        let state = CellState::Volumetric;
        assert_eq!(CellState::from_u8(state.as_u8()), Some(state));
    }

    #[test]
    fn test_from_u8_all_variants() {
        assert_eq!(CellState::from_u8(0), Some(CellState::Empty));
        assert_eq!(CellState::from_u8(1), Some(CellState::Surface));
        assert_eq!(CellState::from_u8(2), Some(CellState::Interior));
        assert_eq!(CellState::from_u8(3), Some(CellState::Volumetric));
    }

    #[test]
    fn test_from_u8_invalid() {
        assert_eq!(CellState::from_u8(4), None);
        assert_eq!(CellState::from_u8(5), None);
        assert_eq!(CellState::from_u8(255), None);
    }

    #[test]
    fn test_as_u8() {
        assert_eq!(CellState::Empty.as_u8(), 0);
        assert_eq!(CellState::Surface.as_u8(), 1);
        assert_eq!(CellState::Interior.as_u8(), 2);
        assert_eq!(CellState::Volumetric.as_u8(), 3);
    }

    #[test]
    fn test_has_brick_data_surface() {
        assert!(CellState::Surface.has_brick_data());
    }

    #[test]
    fn test_has_brick_data_volumetric() {
        assert!(CellState::Volumetric.has_brick_data());
    }

    #[test]
    fn test_has_brick_data_empty() {
        assert!(!CellState::Empty.has_brick_data());
    }

    #[test]
    fn test_has_brick_data_interior() {
        assert!(!CellState::Interior.has_brick_data());
    }

    #[test]
    fn test_is_empty_true() {
        assert!(CellState::Empty.is_empty());
    }

    #[test]
    fn test_is_empty_false_surface() {
        assert!(!CellState::Surface.is_empty());
    }

    #[test]
    fn test_is_empty_false_interior() {
        assert!(!CellState::Interior.is_empty());
    }

    #[test]
    fn test_is_empty_false_volumetric() {
        assert!(!CellState::Volumetric.is_empty());
    }

    #[test]
    fn test_derive_debug() {
        let state = CellState::Surface;
        let debug_str = format!("{:?}", state);
        assert_eq!(debug_str, "Surface");
    }

    #[test]
    fn test_derive_clone() {
        let state = CellState::Volumetric;
        let cloned = state.clone();
        assert_eq!(state, cloned);
    }

    #[test]
    fn test_derive_eq() {
        assert_eq!(CellState::Empty, CellState::Empty);
        assert_ne!(CellState::Empty, CellState::Surface);
    }

    #[test]
    fn test_derive_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(CellState::Empty);
        set.insert(CellState::Surface);
        assert!(set.contains(&CellState::Empty));
    }
}
