//! Library panel — browse and manage engine library items.
//!
//! Shows items from the built-in engine library grouped by category
//! (Components, Systems, Materials). Users can copy library items into
//! their project or restore modified items to their library defaults.
//!
//! Data is loaded once on panel creation from `rkf_runtime::project::list_library_items`.

use std::sync::{Arc, Mutex};

use rinch::prelude::*;

use crate::editor_state::{EditorState, UiSignals};

// ── Style constants ────────────────────────────────────────────────────────

const CATEGORY_HEADER_STYLE: &str = "\
    font-size:10px;color:var(--rinch-color-dimmed);text-transform:uppercase;\
    letter-spacing:1px;padding:6px 12px 2px 12px;\
    border-bottom:1px solid var(--rinch-color-border);";

const ITEM_ROW_STYLE: &str = "\
    display:flex;align-items:center;gap:8px;\
    padding:3px 12px;font-size:11px;\
    font-family:var(--rinch-font-family-monospace);";

const ITEM_NAME_STYLE: &str = "\
    flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;\
    color:var(--rinch-color-text);";

const ITEM_PATH_STYLE: &str = "\
    color:var(--rinch-color-placeholder);font-size:10px;flex-shrink:0;";

const EMPTY_MSG_STYLE: &str = "\
    font-size:11px;color:var(--rinch-color-placeholder);padding:12px;";

const BUTTON_STYLE: &str = "\
    padding:1px 6px;font-size:9px;cursor:pointer;border-radius:3px;\
    background:var(--rinch-primary-color-9);color:var(--rinch-color-text);\
    border:1px solid var(--rinch-primary-color-7);flex-shrink:0;\
    text-transform:uppercase;letter-spacing:0.5px;";

const BUTTON_RESTORE_STYLE: &str = "\
    padding:1px 6px;font-size:9px;cursor:pointer;border-radius:3px;\
    background:var(--rinch-color-dark-6);color:var(--rinch-color-dimmed);\
    border:1px solid var(--rinch-color-border);flex-shrink:0;\
    text-transform:uppercase;letter-spacing:0.5px;";

// ── Component ──────────────────────────────────────────────────────────────

/// Library panel — lists all available library items by category.
#[component]
pub fn LibraryPanel() -> NodeHandle {
    let _ui = use_context::<UiSignals>();
    let editor_state = use_context::<Arc<Mutex<EditorState>>>();

    // Load library items once at panel creation.
    let items = load_all_library_items();

    let root = rsx! {
        div { style: "flex:1;min-height:0;display:flex;flex-direction:column;overflow-y:auto;" }
    };

    // Header with item count.
    let total = items.len();
    let header = rsx! {
        div {
            style: "font-size:10px;color:var(--rinch-color-placeholder);\
                    font-family:var(--rinch-font-family-monospace);padding:4px 12px;\
                    border-bottom:1px solid var(--rinch-color-border);flex-shrink:0;",
            {format!("{total} library items")}
        }
    };
    root.append_child(&header);

    if items.is_empty() {
        let msg = rsx! {
            div { style: EMPTY_MSG_STYLE, "No library items found." }
        };
        root.append_child(&msg);
        return root;
    }

    // Build display items grouped by category.
    let display_items = build_display_items(items);

    let list_container = rsx! {
        div { style: "display:flex;flex-direction:column;" }
    };

    for item in &display_items {
        let rendered = render_display_item(__scope, item, &editor_state);
        list_container.append_child(&rendered);
    }

    root.append_child(&list_container);
    root
}

// ── Data model ──────────────────────────────────────────────────────────────

/// A display item — either a category header or a library item row.
#[derive(Debug, Clone)]
enum DisplayItem {
    CategoryHeader(String, usize),
    ItemRow {
        name: String,
        relative_path: String,
        category: String,
    },
}

/// Load all library items across all categories.
fn load_all_library_items() -> Vec<rkf_runtime::project::LibraryItem> {
    let mut all = Vec::new();
    for category in &["component", "system", "material"] {
        if let Ok(items) = rkf_runtime::project::list_library_items(category) {
            all.extend(items);
        }
    }
    all
}

/// Group items by category and insert headers.
fn build_display_items(items: Vec<rkf_runtime::project::LibraryItem>) -> Vec<DisplayItem> {
    let mut result = Vec::with_capacity(items.len() + 3);
    let mut current_category: Option<String> = None;

    // Sort by category, then name.
    let mut sorted = items;
    sorted.sort_by(|a, b| a.category.cmp(&b.category).then(a.name.cmp(&b.name)));

    for item in &sorted {
        if current_category.as_ref() != Some(&item.category) {
            let count = sorted.iter().filter(|i| i.category == item.category).count();
            current_category = Some(item.category.clone());
            result.push(DisplayItem::CategoryHeader(
                category_display_name(&item.category),
                count,
            ));
        }
        result.push(DisplayItem::ItemRow {
            name: item.name.clone(),
            relative_path: item.relative_path.clone(),
            category: item.category.clone(),
        });
    }

    result
}

fn category_display_name(category: &str) -> String {
    match category {
        "component" => "Components".to_string(),
        "system" => "Systems".to_string(),
        "material" => "Materials".to_string(),
        "asset" => "Assets".to_string(),
        other => other.to_string(),
    }
}

/// Map a library relative path to the destination path within a project.
///
/// - `components/health.rs` → `assets/scripts/components/health.rs`
/// - `systems/patrol.rs` → `assets/scripts/systems/patrol.rs`
/// - `materials/stone.rkmat` → `assets/materials/stone.rkmat`
fn dest_rel_path(library_rel_path: &str, category: &str) -> Option<String> {
    match category {
        "component" => {
            let filename = library_rel_path.strip_prefix("components/")?;
            Some(format!("assets/scripts/components/{filename}"))
        }
        "system" => {
            let filename = library_rel_path.strip_prefix("systems/")?;
            Some(format!("assets/scripts/systems/{filename}"))
        }
        "material" => {
            let filename = library_rel_path.strip_prefix("materials/")?;
            Some(format!("assets/materials/{filename}"))
        }
        _ => None,
    }
}

/// Check if an item already exists in the project.
fn item_exists_in_project(
    project_root: &std::path::Path,
    library_rel_path: &str,
    category: &str,
) -> bool {
    if let Some(dest) = dest_rel_path(library_rel_path, category) {
        project_root.join(&dest).exists()
    } else {
        false
    }
}

/// Render a single display item.
fn render_display_item(
    scope: &mut RenderScope,
    item: &DisplayItem,
    editor_state: &Arc<Mutex<EditorState>>,
) -> NodeHandle {
    match item {
        DisplayItem::CategoryHeader(name, count) => {
            let el = scope.create_element("div");
            el.set_attribute("style", CATEGORY_HEADER_STYLE);
            el.append_child(&scope.create_text(&format!("{name} ({count})")));
            el.into()
        }
        DisplayItem::ItemRow { name, relative_path, category } => {
            let row = scope.create_element("div");
            row.set_attribute("style", ITEM_ROW_STYLE);

            // Category icon dot.
            let dot = scope.create_element("div");
            let dot_color = match category.as_str() {
                "component" => "background:var(--rinch-color-blue-5, #339af0);",
                "system" => "background:var(--rinch-color-green-5, #51cf66);",
                "material" => "background:var(--rinch-color-orange-5, #ff922b);",
                _ => "background:var(--rinch-color-dimmed);",
            };
            dot.set_attribute(
                "style",
                &format!("width:6px;height:6px;border-radius:50%;flex-shrink:0;{dot_color}"),
            );
            row.append_child(&dot);

            // Item name.
            let name_el = scope.create_element("div");
            name_el.set_attribute("style", ITEM_NAME_STYLE);
            name_el.append_child(&scope.create_text(name));
            row.append_child(&name_el);

            // Relative path hint.
            let path_el = scope.create_element("div");
            path_el.set_attribute("style", ITEM_PATH_STYLE);
            path_el.append_child(&scope.create_text(relative_path));
            row.append_child(&path_el);

            // Action button: "Add" if not in project, "Restore" if already present.
            let button = scope.create_element("div");

            // Read project info under lock.
            let project_root = {
                if let Ok(es) = editor_state.lock() {
                    es.current_project_path.as_ref().map(|pp| {
                        rkf_runtime::project::project_root(std::path::Path::new(pp))
                    })
                } else {
                    None
                }
            };

            if let Some(ref root) = project_root {
                let exists = item_exists_in_project(
                    root,
                    relative_path,
                    category,
                );

                let (label, style) = if exists {
                    ("Restore", BUTTON_RESTORE_STYLE)
                } else {
                    ("Add", BUTTON_STYLE)
                };

                button.set_attribute("style", style);
                button.append_child(&scope.create_text(label));

                // Click handler: copy/restore from library.
                let lib_path = relative_path.clone();
                let cat = category.clone();
                let root = root.clone();
                let status_text = if exists {
                    format!("Restored {}", name)
                } else {
                    format!("Added {}", name)
                };
                let hid = scope.register_handler(move || {
                    if let Some(dest) = dest_rel_path(&lib_path, &cat) {
                        match rkf_runtime::project::copy_from_library(&lib_path, &root, &dest) {
                            Ok(()) => {
                                log::info!("{status_text}: {dest}");
                            }
                            Err(e) => {
                                log::error!("Failed to copy library item: {e}");
                            }
                        }
                    }
                });
                button.set_attribute("data-rid", &hid.to_string());
            } else {
                // No project open — no button.
                button.set_attribute("style", "display:none;");
            }

            row.append_child(&button);
            row.into()
        }
    }
}
