//! Compact floating brush palette — overlaid on the viewport in Sculpt/Paint modes.
//!
//! Shows only the controls relevant to the current mode:
//! - **Sculpt**: Radius, Strength, Falloff
//! - **Paint (Material)**: Radius, Falloff, Material indicator
//! - **Paint (Color)**: Radius, Falloff, ColorPicker, Erase button

use rinch::prelude::*;

use crate::editor_command::EditorCommand;
use crate::editor_state::{EditorMode, SliderSignals, UiSignals};
use crate::layout::state::{LayoutBacking, LayoutState};
use crate::CommandSender;
use super::slider_helpers::{build_slider_row, build_log_slider_row};

/// Floating brush palette — compact overlay positioned at top-right of the render viewport.
/// Hidden when not in Sculpt or Paint mode.
#[component]
pub fn BrushPalette() -> NodeHandle {
    let ui = use_context::<UiSignals>();
    let sliders = use_context::<SliderSignals>();
    let cmd = use_context::<CommandSender>();
    let layout_state = use_context::<LayoutState>();
    let layout_backing = use_context::<LayoutBacking>();

    let root = __scope.create_element("div");

    // Reactive visibility + position: only show in Sculpt/Paint modes,
    // positioned at top-right of the render viewport area.
    {
        let root_clone = root.clone();
        let backing = layout_backing.clone();
        __scope.create_effect(move || {
            let mode = ui.editor_mode.get();
            let visible = matches!(mode, EditorMode::Sculpt | EditorMode::Paint);

            // Subscribe to layout signals so we reposition when panels resize.
            let _ = layout_state.structure_rev.get();
            let rw = layout_state.right_width.get();
            let cfg = backing.load();
            let right_offset = if cfg.right.collapsed || cfg.right.zones.is_empty() { 0.0 } else { rw + 4.0 };
            // 36px titlebar + 12px margin
            let top = 36.0 + 12.0;
            let right = right_offset + 12.0;

            root_clone.set_attribute(
                "style",
                &format!(
                    "position:absolute;right:{right:.0}px;top:{top:.0}px;z-index:100;\
                     width:220px;pointer-events:auto;\
                     background:rgba(30,30,30,0.45);border-radius:6px;\
                     border:1px solid rgba(255,255,255,0.15);\
                     backdrop-filter:blur(8px);color:#fff;\
                     display:{};flex-direction:column;padding:6px 0;",
                    if visible { "flex" } else { "none" }
                ),
            );
        });
    }

    // ── Header: mode label ──
    let header = __scope.create_element("div");
    header.set_attribute(
        "style",
        "padding:2px 12px 4px;font-size:10px;text-transform:uppercase;\
         letter-spacing:0.5px;color:rgba(255,255,255,0.7);",
    );
    let header_text = __scope.create_text("");
    {
        let header_text = header_text.clone();
        __scope.create_effect(move || {
            let mode = ui.editor_mode.get();
            header_text.set_text(match mode {
                EditorMode::Sculpt => "Sculpt",
                EditorMode::Paint => "Paint",
                _ => "",
            });
        });
    }
    header.append_child(&header_text);
    root.append_child(&header);

    // ── Paint mode selector (Material / Color) — only in Paint mode ──
    {
        let mode_row = __scope.create_element("div");
        {
            let mode_row_clone = mode_row.clone();
            __scope.create_effect(move || {
                let is_paint = ui.editor_mode.get() == EditorMode::Paint;
                mode_row_clone.set_attribute(
                    "style",
                    &format!(
                        "display:{};gap:2px;margin:0 8px 4px;",
                        if is_paint { "flex" } else { "none" }
                    ),
                );
            });
        }

        for (mode, label) in [
            (crate::paint::PaintMode::Material, "Material"),
            (crate::paint::PaintMode::Color, "Color"),
        ] {
            let btn = __scope.create_element("div");
            {
                let btn_clone = btn.clone();
                __scope.create_effect(move || {
                    let current = ui.paint_mode.get();
                    // Color tab is active for both Color and Erase modes.
                    let is_active = match mode {
                        crate::paint::PaintMode::Color => {
                            current == crate::paint::PaintMode::Color
                                || current == crate::paint::PaintMode::Erase
                        }
                        _ => current == mode,
                    };
                    let bg = if is_active { "var(--rinch-primary-color)" } else { "rgba(255,255,255,0.06)" };
                    let color = if is_active { "#fff" } else { "rgba(255,255,255,0.7)" };
                    btn_clone.set_attribute(
                        "style",
                        &format!(
                            "flex:1;text-align:center;padding:3px 0;font-size:10px;\
                             cursor:pointer;background:{bg};color:{color};\
                             border-radius:3px;text-transform:uppercase;letter-spacing:0.5px;"
                        ),
                    );
                });
            }
            btn.append_child(&__scope.create_text(label));
            let cmd2 = cmd.clone();
            let hid = __scope.register_handler(move || {
                ui.paint_mode.set(mode);
                let _ = cmd2.0.send(EditorCommand::SetPaintMode { mode });
            });
            btn.set_attribute("data-rid", &hid.to_string());
            mode_row.append_child(&btn);
        }
        root.append_child(&mode_row);
    }

    // ── Material indicator (Paint + Material mode only) ──
    {
        let mat_row = __scope.create_element("div");
        {
            let mat_row_clone = mat_row.clone();
            __scope.create_effect(move || {
                let show = ui.editor_mode.get() == EditorMode::Paint
                    && ui.paint_mode.get() == crate::paint::PaintMode::Material;
                mat_row_clone.set_attribute(
                    "style",
                    &format!(
                        "display:{};padding:0 12px 2px;font-size:10px;color:rgba(255,255,255,0.7);",
                        if show { "block" } else { "none" }
                    ),
                );
            });
        }
        let mat_text = __scope.create_text("");
        {
            let mat_text = mat_text.clone();
            __scope.create_effect(move || {
                let slot = ui.selected_material.get();
                mat_text.set_text(&match slot {
                    Some(s) => format!("Material #{s}"),
                    None => "(select in browser)".into(),
                });
            });
        }
        mat_row.append_child(&mat_text);
        root.append_child(&mat_row);
    }

    // ── Color section (Paint + Color/Erase mode): ColorPicker + Erase button ──
    {
        let color_section = __scope.create_element("div");
        {
            let cs = color_section.clone();
            __scope.create_effect(move || {
                let show = ui.editor_mode.get() == EditorMode::Paint
                    && ui.paint_mode.get() != crate::paint::PaintMode::Material;
                cs.set_attribute(
                    "style",
                    &format!(
                        "display:{};flex-direction:column;padding:0 10px;",
                        if show { "flex" } else { "none" }
                    ),
                );
            });
        }

        // ColorPicker: compact, no alpha, no hex input.
        let initial_color = ui.paint_color.get();
        let initial_hex = format!(
            "#{:02x}{:02x}{:02x}",
            (initial_color.x * 255.0) as u8,
            (initial_color.y * 255.0) as u8,
            (initial_color.z * 255.0) as u8,
        );
        let picker = ColorPicker {
            format: "hex".into(),
            value: initial_hex,
            alpha: false,
            with_input: false,
            size: "sm".into(),
            swatches: vec![
                "#ff0000".into(), "#ff8800".into(), "#ffff00".into(),
                "#00cc00".into(), "#0088ff".into(), "#8800ff".into(),
                "#ffffff".into(),
            ],
            swatches_per_row: Some(7),
            onchange: Some(InputCallback::new({
                let cmd = cmd.clone();
                move |hex: String| {
                    if let Some((r, g, b)) = parse_hex_color(&hex) {
                        let rf = r as f32 / 255.0;
                        let gf = g as f32 / 255.0;
                        let bf = b as f32 / 255.0;
                        ui.paint_color.set(glam::Vec3::new(rf, gf, bf));
                        let _ = cmd.0.send(EditorCommand::SetPaintColor { r: rf, g: gf, b: bf });
                    }
                }
            })),
            ..Default::default()
        };
        let picker_node = rinch::core::untracked(|| picker.render(__scope, &[]));
        color_section.append_child(&picker_node);

        // Erase button — toggles between Color and Erase paint modes.
        let erase_btn = __scope.create_element("div");
        {
            let eb = erase_btn.clone();
            __scope.create_effect(move || {
                let is_erasing = ui.paint_mode.get() == crate::paint::PaintMode::Erase;
                let bg = if is_erasing { "var(--rinch-primary-color)" } else { "rgba(255,255,255,0.06)" };
                let color = if is_erasing { "#fff" } else { "rgba(255,255,255,0.7)" };
                eb.set_attribute(
                    "style",
                    &format!(
                        "text-align:center;padding:4px 0;margin-top:4px;font-size:10px;\
                         cursor:pointer;background:{bg};color:{color};\
                         border-radius:3px;text-transform:uppercase;letter-spacing:0.5px;"
                    ),
                );
            });
        }
        erase_btn.append_child(&__scope.create_text("Erase"));
        {
            let cmd2 = cmd.clone();
            let hid = __scope.register_handler(move || {
                let current = ui.paint_mode.get();
                let new_mode = if current == crate::paint::PaintMode::Erase {
                    crate::paint::PaintMode::Color
                } else {
                    crate::paint::PaintMode::Erase
                };
                ui.paint_mode.set(new_mode);
                let _ = cmd2.0.send(EditorCommand::SetPaintMode { mode: new_mode });
            });
            erase_btn.set_attribute("data-rid", &hid.to_string());
        }
        color_section.append_child(&erase_btn);

        root.append_child(&color_section);
    }

    // ── Radius slider (both modes) — logarithmic for fine control at small values ──
    build_log_slider_row(
        __scope, &root, "Radius", "",
        sliders.brush_radius, 0.01, 10.0, 2,
        { let cmd = cmd.clone(); move |_v| { sliders.send_brush_commands(&cmd); } },
    );

    // ── Strength slider (Sculpt only — does nothing in Paint) ──
    {
        let strength_wrap = __scope.create_element("div");
        {
            let sw = strength_wrap.clone();
            __scope.create_effect(move || {
                let show = ui.editor_mode.get() == EditorMode::Sculpt;
                sw.set_attribute(
                    "style",
                    &format!("display:{};flex-direction:column;", if show { "flex" } else { "none" }),
                );
            });
        }
        build_slider_row(
            __scope, &strength_wrap, "Strength", "",
            sliders.brush_strength, 0.0, 1.0, 0.01, 2,
            { let cmd = cmd.clone(); move |_v| { sliders.send_brush_commands(&cmd); } },
        );
        root.append_child(&strength_wrap);
    }

    // ── Falloff slider (both modes) ──
    build_slider_row(
        __scope, &root, "Falloff", "",
        sliders.brush_falloff, 0.0, 1.0, 0.01, 2,
        { let cmd = cmd.clone(); move |_v| { sliders.send_brush_commands(&cmd); } },
    );

    root.into()
}

/// Parse a hex color string (#RRGGBB or #RGB) into (r, g, b) bytes.
fn parse_hex_color(hex: &str) -> Option<(u8, u8, u8)> {
    let hex = hex.strip_prefix('#')?;
    match hex.len() {
        6 => {
            let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
            let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
            let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
            Some((r, g, b))
        }
        3 => {
            let r = u8::from_str_radix(&hex[0..1], 16).ok()? * 17;
            let g = u8::from_str_radix(&hex[1..2], 16).ok()? * 17;
            let b = u8::from_str_radix(&hex[2..3], 16).ok()? * 17;
            Some((r, g, b))
        }
        _ => None,
    }
}
