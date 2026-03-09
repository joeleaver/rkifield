//! Shader properties panel — read-only display of shader name, type, ID,
//! and file path.

use rinch::prelude::*;

use crate::editor_state::UiSignals;

use super::{DIVIDER_STYLE, SECTION_STYLE};

/// Shader properties panel — displays shader name, type badge, ID, and file path.
#[component]
pub fn ShaderProperties(
    shader_name: String,
) -> NodeHandle {
    let ui = use_context::<UiSignals>();

    let container = __scope.create_element("div");
    container.set_attribute("style", "display:flex;flex-direction:column;");

    let shaders = ui.shaders.get();

    let shader = shaders.iter().find(|s| s.name == shader_name);
    let shader = match shader {
        Some(s) => s,
        None => return container,
    };

    // Header.
    let hdr = __scope.create_element("div");
    hdr.set_attribute("style", SECTION_STYLE);
    hdr.append_child(&__scope.create_text(&shader.name));
    container.append_child(&hdr);

    // Type badge.
    let type_row = __scope.create_element("div");
    type_row.set_attribute("style", "display:flex;align-items:center;justify-content:space-between;padding:4px 12px;");
    let type_label = __scope.create_element("span");
    type_label.set_attribute("style", "font-size:11px;color:var(--rinch-color-dimmed);");
    type_label.append_child(&__scope.create_text("Type"));
    type_row.append_child(&type_label);
    let type_value = __scope.create_element("span");
    let (badge_text, badge_bg) = if shader.built_in {
        ("built-in", "rgba(60,120,200,0.3)")
    } else {
        ("custom", "rgba(60,180,100,0.3)")
    };
    type_value.set_attribute(
        "style",
        &format!(
            "font-size:10px;color:var(--rinch-color-text);padding:1px 6px;\
             border-radius:3px;background:{badge_bg};"
        ),
    );
    type_value.append_child(&__scope.create_text(badge_text));
    type_row.append_child(&type_value);
    container.append_child(&type_row);

    // Shader ID.
    let id_row = __scope.create_element("div");
    id_row.set_attribute("style", "display:flex;align-items:center;justify-content:space-between;padding:4px 12px;");
    let id_label = __scope.create_element("span");
    id_label.set_attribute("style", "font-size:11px;color:var(--rinch-color-dimmed);");
    id_label.append_child(&__scope.create_text("ID"));
    id_row.append_child(&id_label);
    let id_value = __scope.create_element("span");
    id_value.set_attribute(
        "style",
        "font-size:11px;color:var(--rinch-color-text);\
         font-family:var(--rinch-font-family-monospace);",
    );
    id_value.append_child(&__scope.create_text(&format!("{}", shader.id)));
    id_row.append_child(&id_value);
    container.append_child(&id_row);

    // File path.
    if !shader.file_path.is_empty() {
        let path_row = __scope.create_element("div");
        path_row.set_attribute("style", "display:flex;align-items:center;justify-content:space-between;padding:4px 12px;");
        let path_label = __scope.create_element("span");
        path_label.set_attribute("style", "font-size:11px;color:var(--rinch-color-dimmed);");
        path_label.append_child(&__scope.create_text("File"));
        path_row.append_child(&path_label);
        let path_value = __scope.create_element("span");
        path_value.set_attribute(
            "style",
            "font-size:9px;color:var(--rinch-color-placeholder);\
             font-family:var(--rinch-font-family-monospace);\
             word-break:break-all;text-align:right;max-width:160px;",
        );
        path_value.append_child(&__scope.create_text(&shader.file_path));
        path_row.append_child(&path_value);
        container.append_child(&path_row);
    }

    append_divider(__scope, &container);

    container
}

fn append_divider(scope: &mut RenderScope, container: &NodeHandle) {
    let div = scope.create_element("div");
    div.set_attribute("style", DIVIDER_STYLE);
    container.append_child(&div);
}
