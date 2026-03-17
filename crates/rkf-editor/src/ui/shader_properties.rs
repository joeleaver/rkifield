//! Shader properties panel — read-only display of shader name, type, ID,
//! and file path.

use rinch::prelude::*;

use crate::editor_state::UiSignals;

use super::{DIVIDER_STYLE, SECTION_STYLE};

const ROW_STYLE: &str =
    "display:flex;align-items:center;justify-content:space-between;padding:4px 12px;";
const LABEL_SPAN_STYLE: &str = "font-size:11px;color:var(--rinch-color-dimmed);";
const ID_VALUE_STYLE: &str =
    "font-size:11px;color:var(--rinch-color-text);\
     font-family:var(--rinch-font-family-monospace);";
const PATH_VALUE_STYLE: &str =
    "font-size:9px;color:var(--rinch-color-placeholder);\
     font-family:var(--rinch-font-family-monospace);\
     word-break:break-all;text-align:right;max-width:160px;";

/// Shader properties panel — displays shader name, type badge, ID, and file path.
#[component]
pub fn ShaderProperties(
    shader_name: String,
) -> NodeHandle {
    let ui = use_context::<UiSignals>();

    let shaders = ui.shaders.get();

    let shader = shaders.iter().find(|s| s.name == shader_name);
    let shader = match shader {
        Some(s) => s,
        None => {
            return rsx! { div { style: "display:flex;flex-direction:column;" } };
        }
    };


    let name = shader.name.clone();
    let id_text = format!("{}", shader.id);
    let file_path = shader.file_path.clone();
    let has_file_path = !file_path.is_empty();

    rsx! {
        div { style: "display:flex;flex-direction:column;",
            div { style: {SECTION_STYLE}, {name} }

            // Shader ID row.
            div { style: {ROW_STYLE},
                span { style: {LABEL_SPAN_STYLE}, "ID" }
                span { style: {ID_VALUE_STYLE}, {id_text} }
            }

            // File path row (conditional).
            if has_file_path {
                div { style: {ROW_STYLE},
                    span { style: {LABEL_SPAN_STYLE}, "File" }
                    span { style: {PATH_VALUE_STYLE}, {file_path.clone()} }
                }
            }

            div { style: {DIVIDER_STYLE} }
        }
    }
}
