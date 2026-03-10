//! Layout state store — revision-based reactive signals backed by ArcSwap config.
//!
//! Architecture:
//! - `LayoutBacking` (ArcSwap) is the sole store for `LayoutConfig` — non-reactive,
//!   thread-safe, lock-free. Both UI and engine threads access it directly.
//! - `LayoutState` holds revision counter signals + pixel-size signals. Components
//!   subscribe to revision signals for reactive rendering. Mutations go through
//!   setter methods that update the backing and bump the appropriate revision.
//! - Two revision tiers:
//!   - `structure_rev` — zones/tabs/containers added/removed/moved/collapsed (rare).
//!     LayoutRoot subscribes to this to rebuild the layout tree.
//!   - `tab_rev` — active tab changed (frequent). Zone content areas subscribe
//!     to this — only the panel content swaps, not the entire layout tree.
//! - Pixel signals (`left_width`, `right_width`, `bottom_height`) update CSS
//!   directly during splitter drags — no DOM rebuild, just CSS changes.

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

use rinch::prelude::*;

use super::{ContainerKind, LayoutConfig, PanelId, ZoneConfig};

/// Which edge of a zone a drop target is on (for auto-split).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplitEdge {
    Top,
    Bottom,
    Left,
    Right,
}

/// Where a tab can be dropped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DropTarget {
    /// Drop into an existing zone's tab bar (add as tab).
    Zone {
        container: ContainerKind,
        zone_idx: usize,
    },
    /// Drop on the edge of a zone to split it.
    Split {
        container: ContainerKind,
        zone_idx: usize,
        edge: SplitEdge,
    },
}

/// Data carried during a tab drag operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TabDragData {
    /// Which panel is being dragged.
    pub panel: PanelId,
    /// Where it came from.
    pub source_container: ContainerKind,
    pub source_zone: usize,
}

// ─── Reactive state store ────────────────────────────────────────────────────

/// Reactive layout state store.
///
/// Config lives in `LayoutBacking` (ArcSwap) — NOT in a signal. This avoids
/// re-entrant `RefCell` borrows when handlers call setters that trigger effects.
///
/// Components subscribe to revision signals:
/// - `structure_rev` — rebuilds layout tree on structural changes
/// - `tab_rev` — swaps active panel content on tab switch
///
/// All fields are `Copy`, making `LayoutState` itself `Copy`.
#[derive(Clone, Copy)]
pub struct LayoutState {
    /// Bumped on structural changes: zone/tab add/remove/move, container collapse.
    /// LayoutRoot subscribes to this.
    pub structure_rev: Signal<u64>,
    /// Bumped on active tab changes. Zone content areas subscribe to this.
    pub tab_rev: Signal<u64>,
    /// Left container width in pixels (high-freq splitter drag).
    pub left_width: Signal<f32>,
    /// Right container width in pixels.
    pub right_width: Signal<f32>,
    /// Bottom container height in pixels.
    pub bottom_height: Signal<f32>,
    /// Currently active tab drag data (None when not dragging).
    pub tab_drag: Signal<Option<TabDragData>>,
    /// Visual drop target during tab drag.
    pub drop_target: Signal<Option<DropTarget>>,
    /// Window size for fraction ↔ pixel conversions.
    pub window_size: Signal<(f32, f32)>,
    /// Cursor position during tab drag (for ghost overlay). None when not dragging.
    pub drag_cursor: Signal<Option<(f32, f32)>>,
    /// Bumped on zone fraction changes (splitter drag). Zone divs subscribe
    /// to this to update their flex CSS without a full tree rebuild.
    pub zone_frac_rev: Signal<u64>,
}

// ─── Cross-thread backing store ──────────────────────────────────────────────

/// Cross-thread shared backing store for layout config.
///
/// - UI thread: reads via `load()`, writes via `store()` in setter methods
/// - Engine thread: reads via `load()`/`to_ron()` for project save,
///   writes via `from_ron()` on project open (sets dirty flag for UI poll)
#[derive(Clone)]
pub struct LayoutBacking {
    config: Arc<arc_swap::ArcSwap<LayoutConfig>>,
    /// Set by engine thread after `from_ron()`, cleared by UI thread after `poll_dirty()`.
    dirty: Arc<AtomicBool>,
}

impl LayoutBacking {
    /// Create a new backing store with an initial config.
    pub fn new(config: LayoutConfig) -> Self {
        Self {
            config: Arc::new(arc_swap::ArcSwap::from_pointee(config)),
            dirty: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Read the current layout config (lock-free, any thread).
    pub fn load(&self) -> Arc<LayoutConfig> {
        self.config.load_full()
    }

    /// Store a new layout config (lock-free, any thread).
    pub fn store(&self, config: LayoutConfig) {
        self.config.store(Arc::new(config));
    }

    /// Store from an external thread (engine) and mark dirty for UI poll.
    pub fn store_external(&self, config: LayoutConfig) {
        self.config.store(Arc::new(config));
        self.dirty.store(true, Ordering::Release);
    }

    /// Check if the backing was updated externally. Clears the flag.
    pub fn poll_dirty(&self) -> bool {
        self.dirty.swap(false, Ordering::AcqRel)
    }

    /// Serialize the current layout config to RON.
    pub fn to_ron(&self) -> Option<String> {
        let cfg = self.load();
        ron::ser::to_string_pretty(&*cfg, ron::ser::PrettyConfig::default()).ok()
    }

    /// Deserialize a layout config from RON, store it, and mark dirty.
    pub fn from_ron(&self, ron_str: &str) -> bool {
        if let Ok(cfg) = ron::from_str::<LayoutConfig>(ron_str) {
            self.store_external(cfg);
            true
        } else {
            false
        }
    }
}

// ─── LayoutState construction + getters ──────────────────────────────────────

impl LayoutState {
    /// Create a new layout state from a config and initial window size.
    pub fn new(config: LayoutConfig, window_width: f32, window_height: f32) -> Self {
        let left_width = config.left_width_fraction * window_width;
        let right_width = config.right_width_fraction * window_width;
        let bottom_height = config.bottom_height_fraction * window_height;

        Self {
            structure_rev: Signal::new(0),
            tab_rev: Signal::new(0),
            left_width: Signal::new(left_width),
            right_width: Signal::new(right_width),
            bottom_height: Signal::new(bottom_height),
            tab_drag: Signal::new(None),
            drop_target: Signal::new(None),
            window_size: Signal::new((window_width, window_height)),
            drag_cursor: Signal::new(None),
            zone_frac_rev: Signal::new(0),
        }
    }

    /// Read config from backing (non-reactive — does NOT subscribe).
    /// Use this in handlers and reactive scopes that read config data.
    pub fn read_config(&self, backing: &LayoutBacking) -> LayoutConfig {
        (*backing.load()).clone()
    }

    /// Snapshot config with fractions synced from pixel signals.
    pub fn snapshot_config(&self, backing: &LayoutBacking) -> LayoutConfig {
        let mut cfg = self.read_config(backing);
        let (w, h) = rinch::core::untracked(|| self.window_size.get());
        if w > 0.0 && h > 0.0 {
            cfg.left_width_fraction = rinch::core::untracked(|| self.left_width.get()) / w;
            cfg.right_width_fraction = rinch::core::untracked(|| self.right_width.get()) / w;
            cfg.bottom_height_fraction = rinch::core::untracked(|| self.bottom_height.get()) / h;
        }
        cfg
    }

    /// Sync fractions from pixel signals and publish to backing.
    pub fn publish_to_backing(&self, backing: &LayoutBacking) {
        backing.store(self.snapshot_config(backing));
    }
}

// ─── Setters (mutation API) ──────────────────────────────────────────────────
//
// All setters:
// 1. Read config from backing (non-reactive — no signal subscription)
// 2. Modify config
// 3. Store back to backing
// 4. Bump the appropriate revision signal
//
// This ensures handlers never touch `Signal<LayoutConfig>` directly.
// The revision bump triggers reactive effects, which read config from
// the backing (non-reactive) — no re-entrant RefCell borrows.

impl LayoutState {
    /// Switch the active tab in a zone. Bumps `tab_rev` only.
    pub fn set_active_tab(
        &self,
        backing: &LayoutBacking,
        ck: ContainerKind,
        zone_idx: usize,
        tab_idx: usize,
    ) {
        let mut cfg = self.read_config(backing);
        if let Some(c) = cfg.container_mut(ck) {
            if let Some(zone) = c.zones.get_mut(zone_idx) {
                zone.active_tab = tab_idx;
            }
        }
        backing.store(cfg);
        self.tab_rev.update(|r| *r += 1);
    }

    /// Move a tab to a target zone. Bumps `structure_rev`.
    pub fn move_tab(
        &self,
        backing: &LayoutBacking,
        panel: PanelId,
        target_ck: ContainerKind,
        target_zi: usize,
    ) {
        let mut cfg = self.read_config(backing);
        let _ = super::operations::move_tab(&mut cfg, panel, target_ck, target_zi, None);
        super::operations::cleanup_empty_zones(&mut cfg);
        backing.store(cfg);
        self.bump_structure();
    }

    /// Toggle a container's collapsed state. Bumps `structure_rev`.
    pub fn toggle_container(&self, backing: &LayoutBacking, ck: ContainerKind) {
        let mut cfg = self.read_config(backing);
        if let Some(c) = cfg.container_mut(ck) {
            c.collapsed = !c.collapsed;
        }
        backing.store(cfg);
        self.bump_structure();
    }

    /// Ensure a panel is visible. If already docked and its container is
    /// visible, activate the tab. If the container is collapsed (user hid it),
    /// or the panel isn't in the layout at all, spawn it as a floating panel
    /// so we don't override the user's decision to hide that area.
    pub fn ensure_panel(&self, backing: &LayoutBacking, panel: PanelId) {
        use super::FloatingPanelConfig;

        let mut cfg = self.read_config(backing);

        // Already floating — nothing to do.
        if cfg.floating.iter().any(|f| f.panel == panel) {
            return;
        }

        if let Some((ck, zi, ti)) = cfg.find_panel(panel) {
            let collapsed = cfg.container_mut(ck).map_or(true, |c| c.collapsed);
            if !collapsed {
                // Container is visible — just activate the tab.
                if let Some(c) = cfg.container_mut(ck) {
                    if let Some(zone) = c.zones.get_mut(zi) {
                        zone.active_tab = ti;
                    }
                }
                backing.store(cfg);
                self.tab_rev.update(|r| *r += 1);
                return;
            }
            // Container is collapsed — remove the panel and float it instead.
            super::operations::remove_tab(&mut cfg, panel).ok();
            super::operations::cleanup_empty_zones(&mut cfg);
        }

        // Panel is not visible — spawn as floating.
        let (w, h) = rinch::core::untracked(|| self.window_size.get());
        let fw = 300.0_f32;
        let fh = 400.0_f32;
        let fx = (w * 0.5 - fw * 0.5).max(40.0);
        let fy = (h * 0.5 - fh * 0.5).max(60.0);
        cfg.floating.push(FloatingPanelConfig {
            panel,
            x: fx,
            y: fy,
            width: fw,
            height: fh,
        });
        backing.store(cfg);
        self.bump_structure();
    }

    /// Update zone size fractions (zone splitter drag). Bumps `zone_frac_rev`
    /// only — zone divs update their flex CSS reactively without a full tree rebuild.
    pub fn set_zone_fractions(
        &self,
        backing: &LayoutBacking,
        ck: ContainerKind,
        zone_a: usize,
        frac_a: f32,
        zone_b: usize,
        frac_b: f32,
    ) {
        let mut cfg = self.read_config(backing);
        if let Some(c) = cfg.container_mut(ck) {
            if let Some(za) = c.zones.get_mut(zone_a) {
                za.size_fraction = frac_a;
            }
            if let Some(zb) = c.zones.get_mut(zone_b) {
                zb.size_fraction = frac_b;
            }
        }
        backing.store(cfg);
        self.zone_frac_rev.update(|r| *r += 1);
    }

    /// Split a zone by dropping a tab on an edge. Bumps `structure_rev`.
    pub fn split_tab(
        &self,
        backing: &LayoutBacking,
        panel: PanelId,
        target_ck: ContainerKind,
        target_zi: usize,
        edge: SplitEdge,
    ) {
        let before = matches!(edge, SplitEdge::Top | SplitEdge::Left);
        let mut cfg = self.read_config(backing);
        let _ = super::operations::split_zone(&mut cfg, panel, target_ck, target_zi, before);
        super::operations::cleanup_empty_zones(&mut cfg);
        backing.store(cfg);
        self.bump_structure();
    }

    /// Float a tab — remove from its zone and add as a floating panel. Bumps `structure_rev`.
    pub fn float_tab(
        &self,
        backing: &LayoutBacking,
        panel: PanelId,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
    ) {
        let mut cfg = self.read_config(backing);
        let _ = super::operations::float_panel(&mut cfg, panel, x, y, width, height);
        super::operations::cleanup_empty_zones(&mut cfg);
        backing.store(cfg);
        self.bump_structure();
    }

    /// Dock a floating panel back to its default container. Bumps `structure_rev`.
    pub fn dock_floating_panel(&self, backing: &LayoutBacking, fi: usize) {
        let mut cfg = self.read_config(backing);
        if fi < cfg.floating.len() {
            let panel_id = cfg.floating[fi].panel;
            cfg.floating.remove(fi);
            let default = panel_id.default_container();
            if let Some(c) = cfg.container_mut(default) {
                if c.zones.is_empty() {
                    c.zones.push(ZoneConfig::single(panel_id));
                    c.collapsed = false;
                } else {
                    c.zones[0].tabs.push(panel_id);
                }
            }
        }
        backing.store(cfg);
        self.bump_structure();
    }

    /// Load config from backing and sync pixel signals (project open).
    /// Bumps both revisions.
    pub fn load_from_backing(&self, backing: &LayoutBacking) {
        let cfg = self.read_config(backing);
        let (w, h) = rinch::core::untracked(|| self.window_size.get());
        self.left_width.set(cfg.left_width_fraction * w);
        self.right_width.set(cfg.right_width_fraction * w);
        self.bottom_height.set(cfg.bottom_height_fraction * h);
        self.bump_structure();
    }

    /// Update pixel sizes from fractions when window resizes.
    pub fn on_window_resize(
        &self,
        backing: &LayoutBacking,
        new_width: f32,
        new_height: f32,
    ) {
        let cfg = self.read_config(backing);
        self.window_size.set((new_width, new_height));
        self.left_width.set(cfg.left_width_fraction * new_width);
        self.right_width.set(cfg.right_width_fraction * new_width);
        self.bottom_height
            .set(cfg.bottom_height_fraction * new_height);
    }

    /// Compute the total UI panel width (left + right) in pixels.
    pub fn total_panel_width(&self, backing: &LayoutBacking) -> f32 {
        let cfg = self.read_config(backing);
        let left = if cfg.left.collapsed {
            0.0
        } else {
            rinch::core::untracked(|| self.left_width.get())
        };
        let right = if cfg.right.collapsed {
            0.0
        } else {
            rinch::core::untracked(|| self.right_width.get())
        };
        left + right
    }

    /// Hit-test a cursor position against the layout geometry to find a drop target.
    ///
    /// Used by floating panel drag (component drag, not DnD) to detect when the
    /// cursor is over a docked container. Returns the first zone of the container.
    ///
    /// Layout geometry (titlebar = 36px):
    /// - Left:   x ∈ [0, left_width],            y ∈ [36, wh]
    /// - Right:  x ∈ [ww - right_width, ww],     y ∈ [36, wh]
    /// - Bottom: x ∈ [left_width, ww - right_width], y ∈ [wh - bottom_height, wh]
    pub fn hit_test_drop(&self, backing: &LayoutBacking, mx: f32, my: f32) -> Option<DropTarget> {
        let cfg = self.read_config(backing);
        let (ww, wh) = rinch::core::untracked(|| self.window_size.get());
        let lw = if cfg.left.collapsed { 0.0 } else { rinch::core::untracked(|| self.left_width.get()) };
        let rw = if cfg.right.collapsed { 0.0 } else { rinch::core::untracked(|| self.right_width.get()) };
        let bh = if cfg.bottom.collapsed { 0.0 } else { rinch::core::untracked(|| self.bottom_height.get()) };
        let titlebar = 36.0_f32;
        let tab_bar_h = 26.0_f32;

        // Left container (zones stack vertically → Top/Bottom splits)
        if !cfg.left.collapsed && !cfg.left.zones.is_empty()
            && mx >= 0.0 && mx < lw && my >= titlebar && my < wh
        {
            return Some(self.zone_drop_target(
                &cfg.left, ContainerKind::Left, my - titlebar, wh - titlebar, true, tab_bar_h,
            ));
        }

        // Right container (zones stack vertically → Top/Bottom splits)
        if !cfg.right.collapsed && !cfg.right.zones.is_empty()
            && mx >= ww - rw && mx < ww && my >= titlebar && my < wh
        {
            return Some(self.zone_drop_target(
                &cfg.right, ContainerKind::Right, my - titlebar, wh - titlebar, true, tab_bar_h,
            ));
        }

        // Bottom container (zones stack horizontally → Left/Right splits)
        if !cfg.bottom.collapsed && !cfg.bottom.zones.is_empty()
            && mx >= lw && mx < ww - rw && my >= wh - bh && my < wh
        {
            return Some(self.zone_drop_target(
                &cfg.bottom, ContainerKind::Bottom, mx - lw, ww - lw - rw, false, tab_bar_h,
            ));
        }

        None
    }

    /// Determine which zone the cursor is in and whether it's a Zone (tab bar) or Split target.
    ///
    /// For vertical containers (Left/Right), zones stack along `pos_axis` (y).
    /// The edge overlays are Top/Bottom halves of each zone's content area.
    /// For horizontal containers (Bottom), zones stack along `pos_axis` (x)
    /// and edge overlays are Left/Right halves.
    fn zone_drop_target(
        &self,
        container: &super::ContainerConfig,
        kind: ContainerKind,
        pos: f32,
        total: f32,
        vertical: bool,
        tab_bar_h: f32,
    ) -> DropTarget {
        // Find which zone we're in.
        let mut accum = 0.0;
        let mut zone_idx = container.zones.len().saturating_sub(1);
        let mut zone_start = 0.0_f32;
        for (i, zone) in container.zones.iter().enumerate() {
            let zone_size = zone.size_fraction * total;
            if pos < accum + zone_size {
                zone_idx = i;
                zone_start = accum;
                break;
            }
            accum += zone_size;
        }

        let zone_size = container.zones[zone_idx].size_fraction * total;
        let pos_in_zone = pos - zone_start;

        // If in the tab bar area (first tab_bar_h pixels of a vertical zone),
        // return Zone target (add as tab).
        if vertical && pos_in_zone < tab_bar_h {
            return DropTarget::Zone { container: kind, zone_idx };
        }

        // Content area: split into two halves for edge targets.
        let content_start = if vertical { tab_bar_h } else { 0.0 };
        let content_size = zone_size - content_start;
        let pos_in_content = pos_in_zone - content_start;
        let first_half = pos_in_content < content_size * 0.5;

        let edge = if vertical {
            if first_half { SplitEdge::Top } else { SplitEdge::Bottom }
        } else {
            if first_half { SplitEdge::Left } else { SplitEdge::Right }
        };

        DropTarget::Split { container: kind, zone_idx, edge }
    }

    fn bump_structure(&self) {
        self.structure_rev.update(|r| *r += 1);
        // No need to bump tab_rev — structure_rev triggers a full layout
        // rebuild which recreates all content areas from scratch. Bumping
        // both in sequence causes a re-entrant RefCell borrow panic when
        // disposing child scopes during effect flush.
    }
}
