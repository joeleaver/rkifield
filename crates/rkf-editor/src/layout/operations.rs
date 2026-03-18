//! Layout mutation operations.
//!
//! All operations take a mutable `LayoutConfig` and apply atomic changes.
//! These are called from UI event handlers (tab drag, splitter drag, etc.).

use super::{ContainerKind, FloatingPanelConfig, LayoutConfig, PanelCategory, PanelId, ZoneConfig};

/// Error type for layout operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayoutError {
    /// Panel not found in the layout.
    PanelNotFound(PanelId),
    /// Cannot move panel to incompatible container.
    IncompatibleCategory {
        panel: PanelId,
        target: ContainerKind,
    },
    /// Cannot remove the last canvas panel from Center.
    LastCanvasPanel,
    /// Target zone index is out of bounds.
    ZoneOutOfBounds {
        container: ContainerKind,
        zone_idx: usize,
    },
}

/// Remove a panel from wherever it is in the layout.
///
/// Returns the source location `(ContainerKind, zone_idx, tab_idx)` if found.
/// Does not clean up empty zones — call `cleanup_empty_zones` after.
pub fn remove_tab(
    config: &mut LayoutConfig,
    panel: PanelId,
) -> Result<(ContainerKind, usize, usize), LayoutError> {
    // Check floating panels first.
    if let Some(idx) = config.floating.iter().position(|f| f.panel == panel) {
        config.floating.remove(idx);
        return Ok((ContainerKind::Floating, 0, idx));
    }

    // Search containers.
    for kind in [
        ContainerKind::Left,
        ContainerKind::Right,
        ContainerKind::Bottom,
        ContainerKind::Center,
    ] {
        let container = config.container_mut(kind).unwrap();
        for (zi, zone) in container.zones.iter_mut().enumerate() {
            if let Some(ti) = zone.tabs.iter().position(|&t| t == panel) {
                zone.tabs.remove(ti);
                // Fix active_tab index.
                if zone.active_tab >= zone.tabs.len() && !zone.tabs.is_empty() {
                    zone.active_tab = zone.tabs.len() - 1;
                }
                return Ok((kind, zi, ti));
            }
        }
    }

    Err(LayoutError::PanelNotFound(panel))
}

/// Move a tab from its current location to a target zone.
///
/// Validates category compatibility. If the panel is the last canvas panel
/// in Center, the move is rejected.
pub fn move_tab(
    config: &mut LayoutConfig,
    panel: PanelId,
    target_container: ContainerKind,
    target_zone: usize,
    insert_at: Option<usize>,
) -> Result<(), LayoutError> {
    // Validate category.
    if !target_container.accepts(panel.category()) {
        return Err(LayoutError::IncompatibleCategory {
            panel,
            target: target_container,
        });
    }

    // Prevent removing last canvas panel from Center.
    if panel.category() == PanelCategory::Canvas {
        let center = &config.center;
        let canvas_count: usize = center
            .zones
            .iter()
            .map(|z| z.tabs.iter().filter(|t| t.category() == PanelCategory::Canvas).count())
            .sum();
        if canvas_count <= 1 && config.find_panel(panel).map(|(k, _, _)| k) == Some(ContainerKind::Center) {
            // Only block if we're moving it OUT of center
            if target_container != ContainerKind::Center {
                return Err(LayoutError::LastCanvasPanel);
            }
        }
    }

    // Remove from current location.
    let _ = remove_tab(config, panel);

    // Add to target.
    let container = config
        .container_mut(target_container)
        .ok_or(LayoutError::ZoneOutOfBounds {
            container: target_container,
            zone_idx: target_zone,
        })?;

    // Ensure target zone exists.
    if target_zone >= container.zones.len() {
        return Err(LayoutError::ZoneOutOfBounds {
            container: target_container,
            zone_idx: target_zone,
        });
    }

    let zone = &mut container.zones[target_zone];
    let idx = insert_at.unwrap_or(zone.tabs.len());
    let idx = idx.min(zone.tabs.len());
    zone.tabs.insert(idx, panel);
    zone.active_tab = idx;

    // Un-collapse if needed.
    container.collapsed = false;

    Ok(())
}

/// Split a zone by inserting a new zone with the given panel.
///
/// `before` controls whether the new zone is inserted before or after the
/// target zone. Zone size fractions are equalized.
pub fn split_zone(
    config: &mut LayoutConfig,
    panel: PanelId,
    target_container: ContainerKind,
    target_zone: usize,
    before: bool,
) -> Result<(), LayoutError> {
    // Validate category.
    if !target_container.accepts(panel.category()) {
        return Err(LayoutError::IncompatibleCategory {
            panel,
            target: target_container,
        });
    }

    // Remove from current location.
    let _ = remove_tab(config, panel);

    let container = config
        .container_mut(target_container)
        .ok_or(LayoutError::ZoneOutOfBounds {
            container: target_container,
            zone_idx: target_zone,
        })?;

    if target_zone > container.zones.len() {
        return Err(LayoutError::ZoneOutOfBounds {
            container: target_container,
            zone_idx: target_zone,
        });
    }

    let new_zone = ZoneConfig::single(panel);
    let insert_idx = if before {
        target_zone
    } else {
        (target_zone + 1).min(container.zones.len())
    };
    container.zones.insert(insert_idx, new_zone);

    // Equalize fractions.
    let n = container.zones.len() as f32;
    for zone in &mut container.zones {
        zone.size_fraction = 1.0 / n;
    }

    container.collapsed = false;

    Ok(())
}

/// Remove empty zones from all containers.
///
/// If a container becomes entirely empty, it is collapsed.
pub fn cleanup_empty_zones(config: &mut LayoutConfig) {
    for kind in [
        ContainerKind::Left,
        ContainerKind::Right,
        ContainerKind::Bottom,
        ContainerKind::Center,
    ] {
        let container = config.container_mut(kind).unwrap();
        container.zones.retain(|z| !z.tabs.is_empty());

        if container.zones.is_empty() {
            container.collapsed = true;
        } else {
            // Re-normalize fractions.
            let total: f32 = container.zones.iter().map(|z| z.size_fraction).sum();
            if total > 0.0 {
                for zone in &mut container.zones {
                    zone.size_fraction /= total;
                }
            }
        }
    }
}

/// Dock a floating panel back into its default container.
///
/// Creates a new zone in the default container if needed.
pub fn dock_floating_panel(
    config: &mut LayoutConfig,
    panel: PanelId,
) -> Result<(), LayoutError> {
    // Remove from floating.
    if let Some(idx) = config.floating.iter().position(|f| f.panel == panel) {
        config.floating.remove(idx);
    }

    let target = panel.default_container();
    let container = config.container_mut(target).unwrap();

    // Add to the first zone if one exists, otherwise create one.
    if let Some(zone) = container.zones.first_mut() {
        zone.tabs.push(panel);
        zone.active_tab = zone.tabs.len() - 1;
    } else {
        container.zones.push(ZoneConfig::single(panel));
    }

    container.collapsed = false;
    Ok(())
}

/// Float a panel — remove from its current location and add as floating.
pub fn float_panel(
    config: &mut LayoutConfig,
    panel: PanelId,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
) -> Result<(), LayoutError> {
    // Prevent floating the last canvas panel.
    if panel.category() == PanelCategory::Canvas {
        let center = &config.center;
        let canvas_count: usize = center
            .zones
            .iter()
            .map(|z| z.tabs.iter().filter(|t| t.category() == PanelCategory::Canvas).count())
            .sum();
        if canvas_count <= 1 && config.find_panel(panel).map(|(k, _, _)| k) == Some(ContainerKind::Center) {
            return Err(LayoutError::LastCanvasPanel);
        }
    }

    let _ = remove_tab(config, panel);

    config.floating.push(FloatingPanelConfig {
        panel,
        x,
        y,
        width,
        height,
    });

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::default_layout;

    #[test]
    fn move_tab_between_zones() {
        let mut layout = default_layout();
        // Move ObjectProperties from right zone 0 to left zone 0
        let result = move_tab(
            &mut layout,
            PanelId::ObjectProperties,
            ContainerKind::Left,
            0,
            None,
        );
        assert!(result.is_ok());
        assert!(layout.left.zones[0]
            .tabs
            .contains(&PanelId::ObjectProperties));
        // ObjectProperties should no longer be in right
        assert!(!layout.right.zones[0]
            .tabs
            .contains(&PanelId::ObjectProperties));
    }

    #[test]
    fn move_tab_category_enforcement() {
        let mut layout = default_layout();
        // Try to move SceneView (canvas) to Left (regular only)
        let result = move_tab(
            &mut layout,
            PanelId::SceneView,
            ContainerKind::Left,
            0,
            None,
        );
        assert_eq!(
            result,
            Err(LayoutError::IncompatibleCategory {
                panel: PanelId::SceneView,
                target: ContainerKind::Left,
            })
        );
    }

    #[test]
    fn move_tab_last_canvas_blocked() {
        let mut layout = default_layout();
        // Try to move the only SceneView out of Center
        let result = move_tab(
            &mut layout,
            PanelId::SceneView,
            ContainerKind::Left,
            0,
            None,
        );
        // Should fail due to both category and last-canvas
        assert!(result.is_err());
    }

    #[test]
    fn remove_tab_found() {
        let mut layout = default_layout();
        let result = remove_tab(&mut layout, PanelId::ObjectProperties);
        assert_eq!(result, Ok((ContainerKind::Right, 0, 2)));
        assert!(!layout.right.zones[0]
            .tabs
            .contains(&PanelId::ObjectProperties));
    }

    #[test]
    fn remove_tab_not_found() {
        let mut layout = default_layout();
        let result = remove_tab(&mut layout, PanelId::GameView);
        assert_eq!(result, Err(LayoutError::PanelNotFound(PanelId::GameView)));
    }

    #[test]
    fn split_zone_before() {
        let mut layout = default_layout();
        // Move SceneTree from left to bottom zone 0 (split before)
        let result = split_zone(
            &mut layout,
            PanelId::SceneTree,
            ContainerKind::Bottom,
            0,
            true,
        );
        assert!(result.is_ok());
        assert_eq!(layout.bottom.zones.len(), 2);
        assert_eq!(layout.bottom.zones[0].tabs, vec![PanelId::SceneTree]);
        assert_eq!(layout.bottom.zones[1].tabs, vec![PanelId::Materials, PanelId::Shaders, PanelId::Systems, PanelId::Library, PanelId::Console]);
    }

    #[test]
    fn split_zone_after() {
        let mut layout = default_layout();
        // Move SceneTree from left to bottom zone 0 (split after)
        let result = split_zone(
            &mut layout,
            PanelId::SceneTree,
            ContainerKind::Bottom,
            0,
            false,
        );
        assert!(result.is_ok());
        assert_eq!(layout.bottom.zones.len(), 2);
        assert_eq!(layout.bottom.zones[0].tabs, vec![PanelId::Materials, PanelId::Shaders, PanelId::Systems, PanelId::Library, PanelId::Console]);
        assert_eq!(layout.bottom.zones[1].tabs, vec![PanelId::SceneTree]);
    }

    #[test]
    fn cleanup_empty_zones_removes() {
        let mut layout = default_layout();
        // Empty out left zone
        layout.left.zones[0].tabs.clear();
        cleanup_empty_zones(&mut layout);
        assert!(layout.left.zones.is_empty());
        assert!(layout.left.collapsed);
    }

    #[test]
    fn cleanup_normalizes_fractions() {
        let mut layout = default_layout();
        // Add a second zone to right, then empty it
        layout.right.zones.push(ZoneConfig::single(PanelId::Console));
        layout.right.zones[1].tabs.clear();
        cleanup_empty_zones(&mut layout);
        assert_eq!(layout.right.zones.len(), 1);
        assert!((layout.right.zones[0].size_fraction - 1.0).abs() < 0.01);
    }

    #[test]
    fn dock_floating_panel() {
        let mut layout = default_layout();
        // Float ObjectProperties
        let _ = remove_tab(&mut layout, PanelId::ObjectProperties);
        layout.floating.push(FloatingPanelConfig {
            panel: PanelId::ObjectProperties,
            x: 100.0,
            y: 100.0,
            width: 300.0,
            height: 400.0,
        });

        let result = super::dock_floating_panel(&mut layout, PanelId::ObjectProperties);
        assert!(result.is_ok());
        assert!(layout.floating.is_empty());
        // ObjectProperties should be back in right container
        assert!(layout.right.zones.iter().any(|z| z.tabs.contains(&PanelId::ObjectProperties)));
    }

    #[test]
    fn float_panel_adds_to_floating() {
        let mut layout = default_layout();
        let result = float_panel(
            &mut layout,
            PanelId::ObjectProperties,
            50.0,
            50.0,
            300.0,
            400.0,
        );
        assert!(result.is_ok());
        assert_eq!(layout.floating.len(), 1);
        assert_eq!(layout.floating[0].panel, PanelId::ObjectProperties);
        // Should no longer be in right
        assert!(layout.find_panel(PanelId::ObjectProperties).is_none());
    }

    #[test]
    fn float_last_canvas_blocked() {
        let mut layout = default_layout();
        let result = float_panel(
            &mut layout,
            PanelId::SceneView,
            50.0,
            50.0,
            800.0,
            600.0,
        );
        assert_eq!(result, Err(LayoutError::LastCanvasPanel));
    }

    #[test]
    fn move_tab_within_same_container() {
        let mut layout = default_layout();
        // Add a second zone to right so we can move a tab into it
        layout.right.zones.push(ZoneConfig::single(PanelId::Console));
        // Move AssetProperties to right zone 1
        let result = move_tab(
            &mut layout,
            PanelId::AssetProperties,
            ContainerKind::Right,
            1,
            None,
        );
        assert!(result.is_ok());
        assert!(layout.right.zones[1]
            .tabs
            .contains(&PanelId::AssetProperties));
    }

    #[test]
    fn move_tab_zone_out_of_bounds() {
        let mut layout = default_layout();
        let result = move_tab(
            &mut layout,
            PanelId::ObjectProperties,
            ContainerKind::Left,
            99,
            None,
        );
        assert!(result.is_err());
    }
}
