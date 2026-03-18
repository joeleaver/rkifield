//! Zone-based configurable layout system.
//!
//! The editor layout consists of 4 containers (Left, Right, Bottom, Center)
//! at fixed positions with resizable boundaries via splitters. Each container
//! holds a stack of zones, and each zone holds tabs that can be dragged between
//! compatible zones.

pub mod components;
pub mod operations;
pub mod panel_registry;
pub mod state;

use serde::{Deserialize, Serialize};

// ── Panel identifiers ───────────────────────────────────────────────────────

/// Identifies a unique panel type in the layout system.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PanelId {
    // Regular panels (left/right/bottom)
    #[default]
    SceneTree,
    EditorCamera,
    Environment,
    ObjectProperties,
    AssetProperties,
    Materials,
    Shaders,
    Console,
    DebugOverlay,
    Systems,
    Library,
    // Canvas panels (center only)
    SceneView,
    GameView,
    AnimationEditor,
}

/// Category of a panel — determines which containers it can be placed in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PanelCategory {
    /// Regular property/tool panels — can go in Left, Right, Bottom, or Float.
    Regular,
    /// Canvas panels — can only go in Center.
    Canvas,
}

impl PanelId {
    /// All panel variants, for iteration (e.g. building View > Panels menu).
    pub const ALL: &[PanelId] = &[
        Self::SceneTree,
        Self::EditorCamera,
        Self::Environment,
        Self::ObjectProperties,
        Self::AssetProperties,
        Self::Materials,
        Self::Shaders,
        Self::Console,
        Self::DebugOverlay,
        Self::Systems,
        Self::Library,
        Self::SceneView,
        Self::GameView,
        Self::AnimationEditor,
    ];

    /// Get the category of this panel.
    pub fn category(self) -> PanelCategory {
        match self {
            Self::SceneView | Self::GameView | Self::AnimationEditor => PanelCategory::Canvas,
            _ => PanelCategory::Regular,
        }
    }

    /// Human-readable display name for this panel.
    pub fn display_name(self) -> &'static str {
        match self {
            Self::SceneTree => "Scene Tree",
            Self::EditorCamera => "Editor Camera",
            Self::Environment => "Environment",
            Self::ObjectProperties => "Object Properties",
            Self::AssetProperties => "Asset Properties",
            Self::Materials => "Materials",
            Self::Shaders => "Shaders",
            Self::Console => "Console",
            Self::DebugOverlay => "Console",
            Self::Systems => "Systems",
            Self::Library => "Library",
            Self::SceneView => "Scene",
            Self::GameView => "Game",
            Self::AnimationEditor => "Animation",
        }
    }

    /// Default container for this panel (used when closing a floating panel).
    pub fn default_container(self) -> ContainerKind {
        match self {
            Self::SceneTree => ContainerKind::Left,
            Self::EditorCamera | Self::Environment | Self::ObjectProperties | Self::AssetProperties => ContainerKind::Right,
            Self::Materials | Self::Shaders | Self::Console | Self::DebugOverlay | Self::Systems | Self::Library => ContainerKind::Bottom,
            Self::SceneView | Self::GameView | Self::AnimationEditor => ContainerKind::Center,
        }
    }
}

// ── Layout configuration (serializable) ─────────────────────────────────────

/// Which container kind in the layout.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ContainerKind {
    #[default]
    Left,
    Right,
    Bottom,
    Center,
    Floating,
}

impl ContainerKind {
    /// Whether this container accepts regular (non-canvas) panels.
    pub fn accepts_regular(self) -> bool {
        !matches!(self, Self::Center)
    }

    /// Whether this container accepts canvas panels.
    pub fn accepts_canvas(self) -> bool {
        matches!(self, Self::Center)
    }

    /// Whether a panel with the given category can be placed in this container.
    pub fn accepts(self, category: PanelCategory) -> bool {
        match category {
            PanelCategory::Regular => self.accepts_regular(),
            PanelCategory::Canvas => self.accepts_canvas(),
        }
    }
}

/// Configuration for a single zone within a container.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ZoneConfig {
    /// Tabs in this zone, in display order.
    pub tabs: Vec<PanelId>,
    /// Index of the currently active tab.
    pub active_tab: usize,
    /// Normalized size fraction within the container (0.0..=1.0).
    /// All zone fractions in a container should sum to ~1.0.
    pub size_fraction: f32,
}

impl ZoneConfig {
    /// Create a zone with a single tab.
    pub fn single(panel: PanelId) -> Self {
        Self {
            tabs: vec![panel],
            active_tab: 0,
            size_fraction: 1.0,
        }
    }

    /// Create a zone with multiple tabs, first one active.
    pub fn multi(tabs: Vec<PanelId>) -> Self {
        Self {
            tabs,
            active_tab: 0,
            size_fraction: 1.0,
        }
    }
}

/// Configuration for a container (Left, Right, Bottom, Center).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ContainerConfig {
    /// Zones stacked within this container.
    pub zones: Vec<ZoneConfig>,
    /// Whether this container is collapsed (hidden).
    pub collapsed: bool,
}

impl ContainerConfig {
    /// Create a container with a single zone containing the given tabs.
    pub fn single_zone(tabs: Vec<PanelId>) -> Self {
        Self {
            zones: vec![ZoneConfig::multi(tabs)],
            collapsed: false,
        }
    }

    /// Create an empty (collapsed) container.
    pub fn empty() -> Self {
        Self {
            zones: Vec::new(),
            collapsed: true,
        }
    }

    /// Whether this container has any zones with tabs.
    pub fn has_content(&self) -> bool {
        self.zones.iter().any(|z| !z.tabs.is_empty())
    }
}

/// Configuration for a floating panel overlay.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FloatingPanelConfig {
    /// Which panel is floating.
    pub panel: PanelId,
    /// Position and size (in pixels, relative to window).
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

/// Complete layout configuration — serializable to project files.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LayoutConfig {
    /// Left container (scene tree, etc.).
    pub left: ContainerConfig,
    /// Right container (properties, environment, etc.).
    pub right: ContainerConfig,
    /// Bottom container (asset browser, console, etc.).
    pub bottom: ContainerConfig,
    /// Center container (scene view, game view — canvas panels only).
    pub center: ContainerConfig,
    /// Floating panels (detached from containers).
    pub floating: Vec<FloatingPanelConfig>,
    /// Left container width as a fraction of window width.
    pub left_width_fraction: f32,
    /// Right container width as a fraction of window width.
    pub right_width_fraction: f32,
    /// Bottom container height as a fraction of window height.
    pub bottom_height_fraction: f32,
}

impl LayoutConfig {
    /// Get a mutable reference to a container by kind.
    pub fn container_mut(&mut self, kind: ContainerKind) -> Option<&mut ContainerConfig> {
        match kind {
            ContainerKind::Left => Some(&mut self.left),
            ContainerKind::Right => Some(&mut self.right),
            ContainerKind::Bottom => Some(&mut self.bottom),
            ContainerKind::Center => Some(&mut self.center),
            ContainerKind::Floating => None,
        }
    }

    /// Get a reference to a container by kind.
    pub fn container(&self, kind: ContainerKind) -> Option<&ContainerConfig> {
        match kind {
            ContainerKind::Left => Some(&self.left),
            ContainerKind::Right => Some(&self.right),
            ContainerKind::Bottom => Some(&self.bottom),
            ContainerKind::Center => Some(&self.center),
            ContainerKind::Floating => None,
        }
    }

    /// Find which container and zone a panel is currently in.
    /// Returns `(ContainerKind, zone_index, tab_index)` or None.
    pub fn find_panel(&self, panel: PanelId) -> Option<(ContainerKind, usize, usize)> {
        for (kind, container) in self.containers() {
            for (zi, zone) in container.zones.iter().enumerate() {
                if let Some(ti) = zone.tabs.iter().position(|&t| t == panel) {
                    return Some((kind, zi, ti));
                }
            }
        }
        None
    }

    /// Iterate over all containers with their kinds.
    pub fn containers(&self) -> [(ContainerKind, &ContainerConfig); 4] {
        [
            (ContainerKind::Left, &self.left),
            (ContainerKind::Right, &self.right),
            (ContainerKind::Bottom, &self.bottom),
            (ContainerKind::Center, &self.center),
        ]
    }
}

impl Default for LayoutConfig {
    fn default() -> Self {
        default_layout()
    }
}

/// Create the default layout matching the current hardcoded editor layout.
///
/// ```text
/// Left: [SceneTree]
/// Center: [SceneView]
/// Right: [ObjectProperties, AssetProperties]
/// Bottom: [Materials, Shaders]
/// ```
pub fn default_layout() -> LayoutConfig {
    LayoutConfig {
        left: ContainerConfig::single_zone(vec![PanelId::SceneTree]),
        right: ContainerConfig::single_zone(vec![PanelId::EditorCamera, PanelId::Environment, PanelId::ObjectProperties, PanelId::AssetProperties]),
        bottom: ContainerConfig::single_zone(vec![PanelId::Materials, PanelId::Shaders, PanelId::Systems, PanelId::Library, PanelId::Console]),
        center: ContainerConfig::single_zone(vec![PanelId::SceneView]),
        floating: Vec::new(),
        // 250px / 1280px ≈ 0.195
        left_width_fraction: 250.0 / 1280.0,
        // 300px / 1280px ≈ 0.234
        right_width_fraction: 300.0 / 1280.0,
        // 180px / 720px = 0.25
        bottom_height_fraction: 180.0 / 720.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn panel_id_category() {
        assert_eq!(PanelId::SceneTree.category(), PanelCategory::Regular);
        assert_eq!(PanelId::ObjectProperties.category(), PanelCategory::Regular);
        assert_eq!(PanelId::SceneView.category(), PanelCategory::Canvas);
        assert_eq!(PanelId::GameView.category(), PanelCategory::Canvas);
    }

    #[test]
    fn container_kind_accepts() {
        assert!(ContainerKind::Left.accepts(PanelCategory::Regular));
        assert!(!ContainerKind::Left.accepts(PanelCategory::Canvas));
        assert!(!ContainerKind::Center.accepts(PanelCategory::Regular));
        assert!(ContainerKind::Center.accepts(PanelCategory::Canvas));
        assert!(ContainerKind::Floating.accepts(PanelCategory::Regular));
    }

    #[test]
    fn default_layout_structure() {
        let layout = default_layout();
        assert_eq!(layout.left.zones.len(), 1);
        assert_eq!(layout.left.zones[0].tabs, vec![PanelId::SceneTree]);
        assert_eq!(layout.right.zones.len(), 1);
        assert_eq!(layout.right.zones[0].tabs, vec![PanelId::EditorCamera, PanelId::Environment, PanelId::ObjectProperties, PanelId::AssetProperties]);
        assert_eq!(layout.center.zones.len(), 1);
        assert_eq!(layout.center.zones[0].tabs, vec![PanelId::SceneView]);
        assert_eq!(layout.bottom.zones.len(), 1);
        assert_eq!(layout.bottom.zones[0].tabs, vec![PanelId::Materials, PanelId::Shaders, PanelId::Systems, PanelId::Library, PanelId::Console]);
        assert!(layout.floating.is_empty());
    }

    #[test]
    fn find_panel() {
        let layout = default_layout();
        assert_eq!(
            layout.find_panel(PanelId::SceneTree),
            Some((ContainerKind::Left, 0, 0))
        );
        assert_eq!(
            layout.find_panel(PanelId::EditorCamera),
            Some((ContainerKind::Right, 0, 0))
        );
        assert_eq!(
            layout.find_panel(PanelId::ObjectProperties),
            Some((ContainerKind::Right, 0, 1))
        );
        assert_eq!(
            layout.find_panel(PanelId::AssetProperties),
            Some((ContainerKind::Right, 0, 2))
        );
        assert_eq!(
            layout.find_panel(PanelId::Console),
            Some((ContainerKind::Bottom, 0, 4))
        );
    }

    #[test]
    fn serialization_roundtrip() {
        let layout = default_layout();
        let ron_str =
            ron::ser::to_string_pretty(&layout, ron::ser::PrettyConfig::default()).unwrap();
        let decoded: LayoutConfig = ron::from_str(&ron_str).unwrap();
        assert_eq!(layout, decoded);
    }

    #[test]
    fn panel_display_names() {
        assert_eq!(PanelId::SceneTree.display_name(), "Scene Tree");
        assert_eq!(PanelId::Materials.display_name(), "Materials");
        assert_eq!(PanelId::SceneView.display_name(), "Scene");
    }

    #[test]
    fn panel_default_containers() {
        assert_eq!(PanelId::SceneTree.default_container(), ContainerKind::Left);
        assert_eq!(PanelId::ObjectProperties.default_container(), ContainerKind::Right);
        assert_eq!(
            PanelId::Materials.default_container(),
            ContainerKind::Bottom
        );
        assert_eq!(
            PanelId::SceneView.default_container(),
            ContainerKind::Center
        );
    }

    #[test]
    fn empty_container() {
        let c = ContainerConfig::empty();
        assert!(c.collapsed);
        assert!(!c.has_content());
    }

    #[test]
    fn container_has_content() {
        let c = ContainerConfig::single_zone(vec![PanelId::SceneTree]);
        assert!(c.has_content());
        assert!(!c.collapsed);
    }
}
