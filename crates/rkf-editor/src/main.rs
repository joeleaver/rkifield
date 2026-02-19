#![allow(dead_code)] // Editor modules are WIP — used incrementally

mod animation_preview;
mod automation;
mod camera;
mod debug_viz;
mod editor_state;
mod engine_viewport;
mod environment;
mod gizmo;
mod input;
mod light_editor;
mod overlay;
mod paint;
mod placement;
mod properties;
mod scene_io;
mod scene_tree;
mod sculpt;
mod undo;

use std::cell::Cell;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

use winit::application::ApplicationHandler;
use winit::event::{ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::PhysicalKey;
use winit::window::{Window, WindowId};

use rinch::embed::{RinchContext, RinchContextConfig, RinchOverlayRenderer};
use rinch::prelude::*;
use rinch_platform::{
    KeyCode as PlatformKeyCode, Modifiers as PlatformModifiers,
    MouseButton as PlatformMouseButton, PlatformEvent,
};

use automation::{EditorAutomationApi, SharedState};
use editor_state::{EditorMode, EditorState};
use engine_viewport::{EngineState, DISPLAY_HEIGHT, DISPLAY_WIDTH};

/// Global editor state accessible by rinch UI components.
static EDITOR_STATE: OnceLock<Arc<Mutex<EditorState>>> = OnceLock::new();

// Thread-locals to share signals between the rinch component tree
// and the winit event handler. Both run on the same (main) thread.
thread_local! {
    static MODE_SIGNAL: Cell<Option<Signal<u8>>> = const { Cell::new(None) };
    static MENU_SIGNAL: Cell<Option<Signal<u8>>> = const { Cell::new(None) };
    static FPS_SIGNAL: Cell<Option<Signal<u16>>> = const { Cell::new(None) };
    /// Bumped when environment settings change, to re-read values in UI.
    static ENV_REFRESH: Cell<Option<Signal<u32>>> = const { Cell::new(None) };
}

// ---------------------------------------------------------------------------
// Blit shader — fullscreen triangle compositing overlay onto surface
// ---------------------------------------------------------------------------

const BLIT_WGSL: &str = "
@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;

struct VsOut { @builtin(position) pos: vec4<f32>, @location(0) uv: vec2<f32> }

@vertex fn vs(@builtin(vertex_index) i: u32) -> VsOut {
    let x = f32(i32(i) / 2) * 4.0 - 1.0;
    let y = f32(i32(i) % 2) * 4.0 - 1.0;
    var o: VsOut;
    o.pos = vec4(x, y, 0.0, 1.0);
    o.uv = vec2((x + 1.0) / 2.0, 1.0 - (y + 1.0) / 2.0);
    return o;
}
@fragment fn fs(in: VsOut) -> @location(0) vec4<f32> {
    return textureSample(t, s, in.uv);
}
";

// ---------------------------------------------------------------------------
// Key translation (winit → rinch platform)
// ---------------------------------------------------------------------------

fn translate_key(key: winit::keyboard::KeyCode) -> PlatformKeyCode {
    use winit::keyboard::KeyCode as WK;
    match key {
        WK::ArrowLeft => PlatformKeyCode::ArrowLeft,
        WK::ArrowRight => PlatformKeyCode::ArrowRight,
        WK::ArrowUp => PlatformKeyCode::ArrowUp,
        WK::ArrowDown => PlatformKeyCode::ArrowDown,
        WK::Home => PlatformKeyCode::Home,
        WK::End => PlatformKeyCode::End,
        WK::PageUp => PlatformKeyCode::PageUp,
        WK::PageDown => PlatformKeyCode::PageDown,
        WK::Enter | WK::NumpadEnter => PlatformKeyCode::Enter,
        WK::Backspace => PlatformKeyCode::Backspace,
        WK::Delete => PlatformKeyCode::Delete,
        WK::Tab => PlatformKeyCode::Tab,
        WK::Escape => PlatformKeyCode::Escape,
        WK::Space => PlatformKeyCode::Space,
        WK::KeyA => PlatformKeyCode::KeyA,
        WK::KeyB => PlatformKeyCode::KeyB,
        WK::KeyC => PlatformKeyCode::KeyC,
        WK::KeyD => PlatformKeyCode::KeyD,
        WK::KeyE => PlatformKeyCode::KeyE,
        WK::KeyF => PlatformKeyCode::KeyF,
        WK::KeyG => PlatformKeyCode::KeyG,
        WK::KeyH => PlatformKeyCode::KeyH,
        WK::KeyI => PlatformKeyCode::KeyI,
        WK::KeyJ => PlatformKeyCode::KeyJ,
        WK::KeyK => PlatformKeyCode::KeyK,
        WK::KeyL => PlatformKeyCode::KeyL,
        WK::KeyM => PlatformKeyCode::KeyM,
        WK::KeyN => PlatformKeyCode::KeyN,
        WK::KeyO => PlatformKeyCode::KeyO,
        WK::KeyP => PlatformKeyCode::KeyP,
        WK::KeyQ => PlatformKeyCode::KeyQ,
        WK::KeyR => PlatformKeyCode::KeyR,
        WK::KeyS => PlatformKeyCode::KeyS,
        WK::KeyT => PlatformKeyCode::KeyT,
        WK::KeyU => PlatformKeyCode::KeyU,
        WK::KeyV => PlatformKeyCode::KeyV,
        WK::KeyW => PlatformKeyCode::KeyW,
        WK::KeyX => PlatformKeyCode::KeyX,
        WK::KeyY => PlatformKeyCode::KeyY,
        WK::KeyZ => PlatformKeyCode::KeyZ,
        WK::Digit0 => PlatformKeyCode::Digit0,
        WK::Digit1 => PlatformKeyCode::Digit1,
        WK::Digit2 => PlatformKeyCode::Digit2,
        WK::Digit3 => PlatformKeyCode::Digit3,
        WK::Digit4 => PlatformKeyCode::Digit4,
        WK::Digit5 => PlatformKeyCode::Digit5,
        WK::Digit6 => PlatformKeyCode::Digit6,
        WK::Digit7 => PlatformKeyCode::Digit7,
        WK::Digit8 => PlatformKeyCode::Digit8,
        WK::Digit9 => PlatformKeyCode::Digit9,
        WK::F1 => PlatformKeyCode::F1,
        WK::F2 => PlatformKeyCode::F2,
        WK::F3 => PlatformKeyCode::F3,
        WK::F4 => PlatformKeyCode::F4,
        WK::F5 => PlatformKeyCode::F5,
        WK::F6 => PlatformKeyCode::F6,
        WK::F7 => PlatformKeyCode::F7,
        WK::F8 => PlatformKeyCode::F8,
        WK::F9 => PlatformKeyCode::F9,
        WK::F10 => PlatformKeyCode::F10,
        WK::F11 => PlatformKeyCode::F11,
        WK::F12 => PlatformKeyCode::F12,
        WK::Equal => PlatformKeyCode::Equal,
        WK::Minus => PlatformKeyCode::Minus,
        _ => PlatformKeyCode::Other,
    }
}

fn translate_modifiers(m: winit::keyboard::ModifiersState) -> PlatformModifiers {
    PlatformModifiers {
        shift: m.shift_key(),
        ctrl: m.control_key(),
        alt: m.alt_key(),
        meta: m.super_key(),
    }
}

/// Translate winit key codes to editor input key codes (for camera/tool input).
fn translate_to_editor_key(key: winit::keyboard::KeyCode) -> Option<input::KeyCode> {
    use winit::keyboard::KeyCode as WK;
    match key {
        WK::KeyW => Some(input::KeyCode::W),
        WK::KeyA => Some(input::KeyCode::A),
        WK::KeyS => Some(input::KeyCode::S),
        WK::KeyD => Some(input::KeyCode::D),
        WK::KeyQ => Some(input::KeyCode::Q),
        WK::KeyE => Some(input::KeyCode::E),
        WK::KeyG => Some(input::KeyCode::G),
        WK::KeyR => Some(input::KeyCode::R),
        WK::KeyT => Some(input::KeyCode::T),
        WK::KeyX => Some(input::KeyCode::X),
        WK::KeyY => Some(input::KeyCode::Y),
        WK::KeyZ => Some(input::KeyCode::Z),
        WK::KeyF => Some(input::KeyCode::F),
        WK::Delete => Some(input::KeyCode::Delete),
        WK::Escape => Some(input::KeyCode::Escape),
        WK::Space => Some(input::KeyCode::Space),
        WK::Tab => Some(input::KeyCode::Tab),
        WK::Enter | WK::NumpadEnter => Some(input::KeyCode::Return),
        WK::Digit1 => Some(input::KeyCode::Num1),
        WK::Digit2 => Some(input::KeyCode::Num2),
        WK::Digit3 => Some(input::KeyCode::Num3),
        WK::F5 => Some(input::KeyCode::F5),
        WK::F12 => Some(input::KeyCode::F12),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Editor UI component
// ---------------------------------------------------------------------------

// ── Menu helper ──────────────────────────────────────────────────────────
// Shared styles for dropdown menus and items.

const MENU_LABEL_STYLE: &str = "color: #888; cursor: pointer; padding: 2px 6px; \
    border-radius: 3px; font-size: 13px; user-select: none;";
const MENU_LABEL_ACTIVE_STYLE: &str = "color: #ccc; cursor: pointer; padding: 2px 6px; \
    border-radius: 3px; font-size: 13px; user-select: none; background: #3a3a3a;";
const MENU_DROPDOWN_STYLE: &str = "position: absolute; top: 100%; left: 0; z-index: 100; \
    min-width: 180px; background: #2d2d2d; border: 1px solid #444; border-radius: 4px; \
    padding: 4px 0; box-shadow: 0 4px 12px rgba(0,0,0,0.4);";
const MENU_ITEM_STYLE: &str = "display: flex; align-items: center; width: 100%; border: none; \
    background: transparent; color: #ccc; padding: 4px 12px; font-size: 12px; \
    cursor: pointer; text-align: left; font-family: inherit; gap: 8px;";
const MENU_ITEM_DIM_STYLE: &str = "display: flex; align-items: center; width: 100%; border: none; \
    background: transparent; color: #666; padding: 4px 12px; font-size: 12px; \
    cursor: default; text-align: left; font-family: inherit; gap: 8px;";
const MENU_SEPARATOR_STYLE: &str = "height: 1px; background: #444; margin: 4px 8px;";
const MENU_SHORTCUT_STYLE: &str = "color: #666; font-size: 11px; margin-left: auto;";

/// Create a menu item button that closes the menu and runs an action.
fn build_menu_item(
    scope: &mut RenderScope,
    parent: &NodeHandle,
    label: &str,
    shortcut: &str,
    menu_open: Signal<u8>,
    action: impl Fn() + 'static,
) {
    let btn = scope.create_element("button");
    btn.set_attribute("style", MENU_ITEM_STYLE);
    let lbl = scope.create_text(label);
    btn.append_child(&lbl);
    if !shortcut.is_empty() {
        let sc = scope.create_element("span");
        sc.set_attribute("style", MENU_SHORTCUT_STYLE);
        sc.set_text(shortcut);
        btn.append_child(&sc);
    }
    let handler = scope.register_handler(move || {
        menu_open.set(0);
        action();
    });
    btn.set_attribute("data-rid", &handler.0.to_string());
    parent.append_child(&btn);
}

/// Create a disabled (grayed out) menu item.
fn build_menu_item_disabled(
    scope: &mut RenderScope,
    parent: &NodeHandle,
    label: &str,
    shortcut: &str,
) {
    let btn = scope.create_element("button");
    btn.set_attribute("style", MENU_ITEM_DIM_STYLE);
    btn.set_attribute("disabled", "");
    let lbl = scope.create_text(label);
    btn.append_child(&lbl);
    if !shortcut.is_empty() {
        let sc = scope.create_element("span");
        sc.set_attribute("style", MENU_SHORTCUT_STYLE);
        sc.set_text(shortcut);
        btn.append_child(&sc);
    }
    parent.append_child(&btn);
}

/// Create a separator line in a menu.
fn build_menu_separator(scope: &mut RenderScope, parent: &NodeHandle) {
    let sep = scope.create_element("div");
    sep.set_attribute("style", MENU_SEPARATOR_STYLE);
    parent.append_child(&sep);
}

// ── Environment panel helpers ────────────────────────────────────────────

const ENV_SECTION_STYLE: &str = "color: #aaa; font-weight: bold; font-size: 12px; \
    margin-top: 8px; margin-bottom: 2px; border-bottom: 1px solid #333; padding-bottom: 2px;";
const ENV_ROW_STYLE: &str = "display: flex; align-items: center; height: 22px; \
    padding: 0 2px; gap: 4px;";
const ENV_LABEL_STYLE: &str = "color: #999; font-size: 11px; flex: 1;";
const ENV_VALUE_STYLE: &str = "color: #ccc; font-size: 11px; min-width: 48px; \
    text-align: right;";
const ENV_BTN_STYLE: &str = "border: none; background: #333; color: #ccc; \
    width: 20px; height: 18px; font-size: 11px; cursor: pointer; border-radius: 2px; \
    font-family: inherit; padding: 0; line-height: 18px;";
const ENV_TOGGLE_ON_STYLE: &str = "border: none; background: #4a6fa5; color: #fff; \
    padding: 1px 8px; font-size: 10px; cursor: pointer; border-radius: 2px; \
    font-family: inherit; line-height: 16px;";
const ENV_TOGGLE_OFF_STYLE: &str = "border: none; background: #444; color: #888; \
    padding: 1px 8px; font-size: 10px; cursor: pointer; border-radius: 2px; \
    font-family: inherit; line-height: 16px;";

fn build_env_section_header(scope: &mut RenderScope, parent: &NodeHandle, label: &str) {
    let hdr = scope.create_element("div");
    hdr.set_attribute("style", ENV_SECTION_STYLE);
    hdr.set_text(label);
    parent.append_child(&hdr);
}

fn build_env_toggle(
    scope: &mut RenderScope,
    parent: &NodeHandle,
    label: &str,
    env_refresh: Signal<u32>,
    get: impl Fn(&environment::EnvironmentState) -> bool + 'static + Copy,
    set: impl Fn(&mut environment::EnvironmentState, bool) + 'static + Copy,
) {
    let row = scope.create_element("div");
    row.set_attribute("style", ENV_ROW_STYLE);

    let lbl = scope.create_element("span");
    lbl.set_attribute("style", ENV_LABEL_STYLE);
    lbl.set_text(label);
    row.append_child(&lbl);

    let btn = scope.create_element("button");
    scope.create_effect({
        let btn = btn.clone();
        move || {
            let _ = env_refresh.get();
            let val = EDITOR_STATE
                .get()
                .and_then(|es| es.lock().ok().map(|s| get(&s.environment)))
                .unwrap_or(false);
            if val {
                btn.set_attribute("style", ENV_TOGGLE_ON_STYLE);
                btn.set_text("ON");
            } else {
                btn.set_attribute("style", ENV_TOGGLE_OFF_STYLE);
                btn.set_text("OFF");
            }
        }
    });
    let h = scope.register_handler(move || {
        if let Some(es) = EDITOR_STATE.get() {
            if let Ok(mut state) = es.lock() {
                let cur = get(&state.environment);
                set(&mut state.environment, !cur);
                state.environment.mark_dirty();
            }
        }
        env_refresh.update(|v| *v = v.wrapping_add(1));
    });
    btn.set_attribute("data-rid", &h.0.to_string());
    row.append_child(&btn);

    parent.append_child(&row);
}

#[allow(clippy::too_many_arguments)]
fn build_env_slider(
    scope: &mut RenderScope,
    parent: &NodeHandle,
    label: &str,
    env_refresh: Signal<u32>,
    get: impl Fn(&environment::EnvironmentState) -> f32 + 'static + Copy,
    set: impl Fn(&mut environment::EnvironmentState, f32) + 'static + Copy,
    step: f32,
    min: f32,
    max: f32,
    decimals: usize,
) {
    let row = scope.create_element("div");
    row.set_attribute("style", ENV_ROW_STYLE);

    let lbl = scope.create_element("span");
    lbl.set_attribute("style", ENV_LABEL_STYLE);
    lbl.set_text(label);
    row.append_child(&lbl);

    let val_span = scope.create_element("span");
    val_span.set_attribute("style", ENV_VALUE_STYLE);
    scope.create_effect({
        let val_span = val_span.clone();
        move || {
            let _ = env_refresh.get();
            let val = EDITOR_STATE
                .get()
                .and_then(|es| es.lock().ok().map(|s| get(&s.environment)))
                .unwrap_or(0.0);
            val_span.set_text(&format!("{:.prec$}", val, prec = decimals));
        }
    });
    row.append_child(&val_span);

    let minus_btn = scope.create_element("button");
    minus_btn.set_attribute("style", ENV_BTN_STYLE);
    minus_btn.set_text("-");
    let h = scope.register_handler(move || {
        if let Some(es) = EDITOR_STATE.get() {
            if let Ok(mut state) = es.lock() {
                let cur = get(&state.environment);
                set(&mut state.environment, (cur - step).clamp(min, max));
                state.environment.mark_dirty();
            }
        }
        env_refresh.update(|v| *v = v.wrapping_add(1));
    });
    minus_btn.set_attribute("data-rid", &h.0.to_string());
    row.append_child(&minus_btn);

    let plus_btn = scope.create_element("button");
    plus_btn.set_attribute("style", ENV_BTN_STYLE);
    plus_btn.set_text("+");
    let h = scope.register_handler(move || {
        if let Some(es) = EDITOR_STATE.get() {
            if let Ok(mut state) = es.lock() {
                let cur = get(&state.environment);
                set(&mut state.environment, (cur + step).clamp(min, max));
                state.environment.mark_dirty();
            }
        }
        env_refresh.update(|v| *v = v.wrapping_add(1));
    });
    plus_btn.set_attribute("data-rid", &h.0.to_string());
    row.append_child(&plus_btn);

    parent.append_child(&row);
}

fn build_env_panel(scope: &mut RenderScope, env_refresh: Signal<u32>) -> NodeHandle {
    let panel = scope.create_element("div");
    panel.set_attribute("style", "display: flex; flex-direction: column; gap: 2px;");

    // ── Fog ──────────────────────────────────────────────────────────────
    build_env_section_header(scope, &panel, "Fog");
    build_env_toggle(
        scope, &panel, "Enabled", env_refresh,
        |env| env.fog.enabled,
        |env, v| env.fog.enabled = v,
    );
    build_env_slider(
        scope, &panel, "Density", env_refresh,
        |env| env.fog.density,
        |env, v| env.fog.density = v,
        0.005, 0.0, 1.0, 3,
    );
    build_env_slider(
        scope, &panel, "Start Dist", env_refresh,
        |env| env.fog.start_distance,
        |env, v| env.fog.start_distance = v,
        10.0, 0.0, 10000.0, 0,
    );
    build_env_slider(
        scope, &panel, "End Dist", env_refresh,
        |env| env.fog.end_distance,
        |env, v| env.fog.end_distance = v,
        10.0, 0.0, 10000.0, 0,
    );
    build_env_slider(
        scope, &panel, "Height Falloff", env_refresh,
        |env| env.fog.height_falloff,
        |env, v| env.fog.height_falloff = v,
        0.01, 0.0, 10.0, 2,
    );

    // ── Clouds ───────────────────────────────────────────────────────────
    build_env_section_header(scope, &panel, "Clouds");
    build_env_toggle(
        scope, &panel, "Enabled", env_refresh,
        |env| env.clouds.enabled,
        |env, v| env.clouds.enabled = v,
    );
    build_env_slider(
        scope, &panel, "Coverage", env_refresh,
        |env| env.clouds.coverage,
        |env, v| env.clouds.coverage = v,
        0.05, 0.0, 1.0, 2,
    );
    build_env_slider(
        scope, &panel, "Density", env_refresh,
        |env| env.clouds.density,
        |env, v| env.clouds.density = v,
        0.1, 0.0, 10.0, 1,
    );
    build_env_slider(
        scope, &panel, "Altitude", env_refresh,
        |env| env.clouds.altitude,
        |env, v| env.clouds.altitude = v,
        10.0, 0.0, 2000.0, 0,
    );
    build_env_slider(
        scope, &panel, "Thickness", env_refresh,
        |env| env.clouds.thickness,
        |env, v| env.clouds.thickness = v,
        5.0, 1.0, 500.0, 0,
    );
    build_env_slider(
        scope, &panel, "Wind Speed", env_refresh,
        |env| env.clouds.wind_speed,
        |env, v| env.clouds.wind_speed = v,
        1.0, 0.0, 100.0, 1,
    );

    // ── Post-Processing ──────────────────────────────────────────────────
    build_env_section_header(scope, &panel, "Post-Processing");
    build_env_toggle(
        scope, &panel, "Bloom", env_refresh,
        |env| env.post_process.bloom_enabled,
        |env, v| env.post_process.bloom_enabled = v,
    );
    build_env_slider(
        scope, &panel, "Bloom Intensity", env_refresh,
        |env| env.post_process.bloom_intensity,
        |env, v| env.post_process.bloom_intensity = v,
        0.05, 0.0, 5.0, 2,
    );
    build_env_slider(
        scope, &panel, "Bloom Threshold", env_refresh,
        |env| env.post_process.bloom_threshold,
        |env, v| env.post_process.bloom_threshold = v,
        0.1, 0.0, 10.0, 1,
    );
    build_env_slider(
        scope, &panel, "Exposure", env_refresh,
        |env| env.post_process.exposure,
        |env, v| env.post_process.exposure = v,
        0.1, 0.1, 10.0, 1,
    );
    build_env_slider(
        scope, &panel, "Contrast", env_refresh,
        |env| env.post_process.contrast,
        |env, v| env.post_process.contrast = v,
        0.05, 0.1, 3.0, 2,
    );
    build_env_slider(
        scope, &panel, "Saturation", env_refresh,
        |env| env.post_process.saturation,
        |env, v| env.post_process.saturation = v,
        0.05, 0.0, 3.0, 2,
    );
    build_env_slider(
        scope, &panel, "Vignette", env_refresh,
        |env| env.post_process.vignette_intensity,
        |env, v| env.post_process.vignette_intensity = v,
        0.05, 0.0, 2.0, 2,
    );

    panel
}

// ── Editor UI component ─────────────────────────────────────────────────

#[component]
fn editor_ui() -> NodeHandle {
    // ── Signals ──────────────────────────────────────────────────────
    let mode_signal = Signal::new(EditorMode::Navigate.index());
    MODE_SIGNAL.with(|cell| cell.set(Some(mode_signal)));

    let menu_open = Signal::new(0u8); // 0=closed, 1=File, 2=Edit, 3=View, 4=Tools
    MENU_SIGNAL.with(|cell| cell.set(Some(menu_open)));

    // Root fills the entire window
    let root = __scope.create_element("div");
    root.set_attribute(
        "style",
        "display: flex; flex-direction: column; width: 100%; height: 100%;",
    );

    // ═══════════════════════════════════════════════════════════════════
    // MENU BAR
    // ═══════════════════════════════════════════════════════════════════
    let menu_bar = __scope.create_element("div");
    menu_bar.set_attribute(
        "style",
        "display: flex; flex-direction: row; height: 32px; background: #2b2b2b; \
         align-items: center; padding: 0 8px; gap: 4px; \
         border-bottom: 1px solid #333;",
    );

    // App title
    let title = __scope.create_element("span");
    title.set_attribute("style", "color: #aaa; font-weight: bold; margin-right: 12px;");
    title.set_text("RKIField Editor");
    menu_bar.append_child(&title);

    // ── File menu ────────────────────────────────────────────────────
    let file_wrap = __scope.create_element("div");
    file_wrap.set_attribute("style", "position: relative;");
    let file_label = __scope.create_element("span");
    file_label.set_text("File");
    // Reactive style for active menu label
    __scope.create_effect({
        let file_label = file_label.clone();
        move || {
            let style = if menu_open.get() == 1 { MENU_LABEL_ACTIVE_STYLE } else { MENU_LABEL_STYLE };
            file_label.set_attribute("style", style);
        }
    });
    let h = __scope.register_handler(move || {
        menu_open.update(|v| *v = if *v == 1 { 0 } else { 1 });
    });
    file_label.set_attribute("data-rid", &h.0.to_string());
    file_wrap.append_child(&file_label);

    show_dom(
        __scope,
        &file_wrap,
        move || menu_open.get() == 1,
        move |scope| {
            let dd = scope.create_element("div");
            dd.set_attribute("style", MENU_DROPDOWN_STYLE);

            build_menu_item_disabled(scope, &dd, "New Scene", "");
            build_menu_item_disabled(scope, &dd, "Open...", "Ctrl+O");
            build_menu_item_disabled(scope, &dd, "Save", "Ctrl+S");
            build_menu_item_disabled(scope, &dd, "Save As...", "Ctrl+Shift+S");
            build_menu_separator(scope, &dd);
            build_menu_item(scope, &dd, "Quit", "Esc", menu_open, move || {
                if let Some(es) = EDITOR_STATE.get() {
                    if let Ok(mut state) = es.lock() {
                        state.wants_exit = true;
                    }
                }
            });

            dd
        },
        None::<fn(&mut RenderScope) -> NodeHandle>,
    );
    menu_bar.append_child(&file_wrap);

    // ── Edit menu ────────────────────────────────────────────────────
    let edit_wrap = __scope.create_element("div");
    edit_wrap.set_attribute("style", "position: relative;");
    let edit_label = __scope.create_element("span");
    edit_label.set_text("Edit");
    __scope.create_effect({
        let edit_label = edit_label.clone();
        move || {
            let style = if menu_open.get() == 2 { MENU_LABEL_ACTIVE_STYLE } else { MENU_LABEL_STYLE };
            edit_label.set_attribute("style", style);
        }
    });
    let h = __scope.register_handler(move || {
        menu_open.update(|v| *v = if *v == 2 { 0 } else { 2 });
    });
    edit_label.set_attribute("data-rid", &h.0.to_string());
    edit_wrap.append_child(&edit_label);

    show_dom(
        __scope,
        &edit_wrap,
        move || menu_open.get() == 2,
        move |scope| {
            let dd = scope.create_element("div");
            dd.set_attribute("style", MENU_DROPDOWN_STYLE);

            build_menu_item(scope, &dd, "Undo", "Ctrl+Z", menu_open, move || {
                if let Some(es) = EDITOR_STATE.get() {
                    if let Ok(mut state) = es.lock() {
                        let _ = state.undo.undo();
                    }
                }
            });
            build_menu_item(scope, &dd, "Redo", "Ctrl+Y", menu_open, move || {
                if let Some(es) = EDITOR_STATE.get() {
                    if let Ok(mut state) = es.lock() {
                        let _ = state.undo.redo();
                    }
                }
            });
            build_menu_separator(scope, &dd);
            build_menu_item_disabled(scope, &dd, "Delete", "Del");
            build_menu_item_disabled(scope, &dd, "Duplicate", "Ctrl+D");
            build_menu_item_disabled(scope, &dd, "Select All", "Ctrl+A");

            dd
        },
        None::<fn(&mut RenderScope) -> NodeHandle>,
    );
    menu_bar.append_child(&edit_wrap);

    // ── View menu ────────────────────────────────────────────────────
    let view_wrap = __scope.create_element("div");
    view_wrap.set_attribute("style", "position: relative;");
    let view_label = __scope.create_element("span");
    view_label.set_text("View");
    __scope.create_effect({
        let view_label = view_label.clone();
        move || {
            let style = if menu_open.get() == 3 { MENU_LABEL_ACTIVE_STYLE } else { MENU_LABEL_STYLE };
            view_label.set_attribute("style", style);
        }
    });
    let h = __scope.register_handler(move || {
        menu_open.update(|v| *v = if *v == 3 { 0 } else { 3 });
    });
    view_label.set_attribute("data-rid", &h.0.to_string());
    view_wrap.append_child(&view_label);

    show_dom(
        __scope,
        &view_wrap,
        move || menu_open.get() == 3,
        move |scope| {
            let dd = scope.create_element("div");
            dd.set_attribute("style", MENU_DROPDOWN_STYLE);

            let debug_modes: &[(&str, u32)] = &[
                ("Normal", 0),
                ("Normals", 1),
                ("Positions", 2),
                ("Material IDs", 3),
                ("Diffuse Only", 4),
                ("Specular Only", 5),
                ("GI Only", 6),
            ];
            for &(label, dm) in debug_modes {
                let shortcut = dm.to_string();
                build_menu_item(scope, &dd, label, &shortcut, menu_open, move || {
                    if let Some(es) = EDITOR_STATE.get() {
                        if let Ok(mut state) = es.lock() {
                            state.pending_debug_mode = Some(dm);
                        }
                    }
                });
            }

            dd
        },
        None::<fn(&mut RenderScope) -> NodeHandle>,
    );
    menu_bar.append_child(&view_wrap);

    // ── Tools menu ───────────────────────────────────────────────────
    let tools_wrap = __scope.create_element("div");
    tools_wrap.set_attribute("style", "position: relative;");
    let tools_label = __scope.create_element("span");
    tools_label.set_text("Tools");
    __scope.create_effect({
        let tools_label = tools_label.clone();
        move || {
            let style = if menu_open.get() == 4 { MENU_LABEL_ACTIVE_STYLE } else { MENU_LABEL_STYLE };
            tools_label.set_attribute("style", style);
        }
    });
    let h = __scope.register_handler(move || {
        menu_open.update(|v| *v = if *v == 4 { 0 } else { 4 });
    });
    tools_label.set_attribute("data-rid", &h.0.to_string());
    tools_wrap.append_child(&tools_label);

    show_dom(
        __scope,
        &tools_wrap,
        move || menu_open.get() == 4,
        move |scope| {
            let dd = scope.create_element("div");
            dd.set_attribute("style", MENU_DROPDOWN_STYLE);

            build_menu_item_disabled(scope, &dd, "Grid Snap", "");
            build_menu_item_disabled(scope, &dd, "Surface Snap", "");

            dd
        },
        None::<fn(&mut RenderScope) -> NodeHandle>,
    );
    menu_bar.append_child(&tools_wrap);

    root.append_child(&menu_bar);

    // ═══════════════════════════════════════════════════════════════════
    // MODE TOOLBAR
    // ═══════════════════════════════════════════════════════════════════
    let toolbar = __scope.create_element("div");
    toolbar.set_attribute(
        "style",
        "display: flex; flex-direction: row; height: 28px; background: #252525; \
         align-items: center; padding: 0 4px; gap: 2px; \
         border-bottom: 1px solid #333;",
    );

    for mode in EditorMode::ALL {
        let idx = mode.index();
        let label = mode.short_name();

        let btn = __scope.create_element("button");
        btn.set_text(label);

        __scope.create_effect({
            let btn = btn.clone();
            move || {
                let active = mode_signal.get() == idx;
                if active {
                    btn.set_attribute(
                        "style",
                        "border: none; padding: 2px 10px; font-size: 11px; cursor: pointer; \
                         border-radius: 3px; background: #4a6fa5; color: #fff; \
                         font-family: inherit; line-height: 20px;",
                    );
                } else {
                    btn.set_attribute(
                        "style",
                        "border: none; padding: 2px 10px; font-size: 11px; cursor: pointer; \
                         border-radius: 3px; background: transparent; color: #999; \
                         font-family: inherit; line-height: 20px;",
                    );
                }
            }
        });

        let handler_id = __scope.register_handler(move || {
            mode_signal.set(idx);
            if let Some(es) = EDITOR_STATE.get() {
                if let Ok(mut state) = es.lock() {
                    state.mode = EditorMode::from_index(idx);
                }
            }
        });
        btn.set_attribute("data-rid", &handler_id.0.to_string());

        toolbar.append_child(&btn);
    }

    root.append_child(&toolbar);

    // ═══════════════════════════════════════════════════════════════════
    // MAIN CONTENT AREA
    // ═══════════════════════════════════════════════════════════════════
    let content = __scope.create_element("div");
    content.set_attribute(
        "style",
        "display: flex; flex-direction: row; flex: 1; overflow: hidden;",
    );

    // ── Left panel: Scene Hierarchy ──────────────────────────────────
    let left_panel = __scope.create_element("div");
    left_panel.set_attribute(
        "style",
        "width: 250px; background: #1e1e1e; border-right: 1px solid #333; \
         overflow-y: auto; padding: 8px; display: flex; flex-direction: column; gap: 4px;",
    );
    let left_title = __scope.create_element("span");
    left_title.set_attribute(
        "style",
        "color: #ccc; font-weight: bold; font-size: 13px; margin-bottom: 4px;",
    );
    left_title.set_text("Scene Hierarchy");
    left_panel.append_child(&left_title);

    // Scene tree content — shows entities from EditorState.scene_tree.
    // Currently empty; will populate as entities are added.
    let tree_container = __scope.create_element("div");
    tree_container.set_attribute(
        "style",
        "flex: 1; display: flex; flex-direction: column;",
    );
    let empty_msg = __scope.create_element("span");
    empty_msg.set_attribute("style", "color: #555; font-size: 12px; font-style: italic;");
    empty_msg.set_text("No entities in scene");
    tree_container.append_child(&empty_msg);
    left_panel.append_child(&tree_container);

    content.append_child(&left_panel);

    // ── Engine viewport ──────────────────────────────────────────────
    let viewport = __scope.create_element("div");
    viewport.set_attribute("data-viewport", "main");
    viewport.set_attribute("style", "flex: 1; background: transparent;");
    content.append_child(&viewport);

    // ── Right panel: Properties / Tool settings ──────────────────────
    let right_panel = __scope.create_element("div");
    right_panel.set_attribute(
        "style",
        "width: 300px; background: #1e1e1e; border-left: 1px solid #333; \
         overflow-y: auto; padding: 8px; display: flex; flex-direction: column; gap: 8px;",
    );

    // Reactive title
    let right_title = __scope.create_element("span");
    right_title.set_attribute(
        "style",
        "color: #ccc; font-weight: bold; font-size: 13px; margin-bottom: 4px;",
    );
    __scope.create_effect({
        let right_title = right_title.clone();
        move || {
            let idx = mode_signal.get();
            let mode = EditorMode::from_index(idx);
            let title = match mode {
                EditorMode::Navigate | EditorMode::Select => "Properties",
                EditorMode::Place => "Asset Browser",
                EditorMode::Sculpt => "Sculpt Tools",
                EditorMode::Paint => "Paint Tools",
                EditorMode::Light => "Light Properties",
                EditorMode::Animate => "Animation",
                EditorMode::Environment => "Environment",
            };
            right_title.set_text(title);
        }
    });
    right_panel.append_child(&right_title);

    // Mode hint (shown for non-Environment modes)
    show_dom(
        __scope,
        &right_panel,
        move || mode_signal.get() != EditorMode::Environment.index(),
        move |scope| {
            let hint_div = scope.create_element("div");
            hint_div.set_attribute("style", "color: #666; font-size: 12px; font-style: italic;");
            let hint_text = scope.create_element("span");
            scope.create_effect({
                let hint_text = hint_text.clone();
                move || {
                    let idx = mode_signal.get();
                    let mode = EditorMode::from_index(idx);
                    let hint = match mode {
                        EditorMode::Navigate => "Use right-click + WASD to fly camera",
                        EditorMode::Select => "Click entities in viewport to select",
                        EditorMode::Place => "Choose an asset to place in the scene",
                        EditorMode::Sculpt => "Left-click + drag to sculpt terrain",
                        EditorMode::Paint => "Left-click + drag to paint materials",
                        EditorMode::Light => "Select a light to edit its properties",
                        EditorMode::Animate => "Select an entity to preview animations",
                        EditorMode::Environment => "",
                    };
                    hint_text.set_text(hint);
                }
            });
            hint_div.append_child(&hint_text);
            hint_div
        },
        None::<fn(&mut RenderScope) -> NodeHandle>,
    );

    // Environment panel (shown only in Environment mode)
    let env_refresh = Signal::new(0u32);
    ENV_REFRESH.with(|cell| cell.set(Some(env_refresh)));

    show_dom(
        __scope,
        &right_panel,
        move || mode_signal.get() == EditorMode::Environment.index(),
        move |scope| {
            build_env_panel(scope, env_refresh)
        },
        None::<fn(&mut RenderScope) -> NodeHandle>,
    );

    content.append_child(&right_panel);
    root.append_child(&content);

    // ═══════════════════════════════════════════════════════════════════
    // STATUS BAR
    // ═══════════════════════════════════════════════════════════════════
    let fps_signal = Signal::new(0u16);
    FPS_SIGNAL.with(|cell| cell.set(Some(fps_signal)));

    let status_bar = __scope.create_element("div");
    status_bar.set_attribute(
        "style",
        "height: 24px; background: #2b2b2b; display: flex; align-items: center; \
         padding: 0 8px; border-top: 1px solid #333; gap: 12px;",
    );

    // Mode name
    let status_mode = __scope.create_element("span");
    status_mode.set_attribute("style", "color: #888; font-size: 12px;");
    __scope.create_effect({
        let status_mode = status_mode.clone();
        move || {
            let idx = mode_signal.get();
            let mode = EditorMode::from_index(idx);
            status_mode.set_text(mode.name());
        }
    });
    status_bar.append_child(&status_mode);

    // Separator
    let sep = __scope.create_element("span");
    sep.set_attribute("style", "color: #444; font-size: 12px;");
    sep.set_text("|");
    status_bar.append_child(&sep);

    // FPS counter
    let status_fps = __scope.create_element("span");
    status_fps.set_attribute("style", "color: #666; font-size: 12px;");
    __scope.create_effect({
        let status_fps = status_fps.clone();
        move || {
            let fps = fps_signal.get();
            if fps > 0 {
                status_fps.set_text(&format!("{fps} fps"));
            } else {
                status_fps.set_text("Ready");
            }
        }
    });
    status_bar.append_child(&status_fps);

    root.append_child(&status_bar);

    root
}

// ---------------------------------------------------------------------------
// IPC server
// ---------------------------------------------------------------------------

/// Path to the JSON discovery metadata file for the current process.
fn discovery_metadata_path() -> String {
    format!("/tmp/rkifield-{}.json", std::process::id())
}

/// Write a JSON discovery file so MCP clients can identify this engine instance.
fn write_discovery_metadata(socket_path: &str) {
    let metadata = serde_json::json!({
        "type": "editor",
        "pid": std::process::id(),
        "socket": socket_path,
        "name": "RKIField Editor",
        "version": env!("CARGO_PKG_VERSION"),
    });
    let path = discovery_metadata_path();
    if let Err(e) = std::fs::write(&path, serde_json::to_string_pretty(&metadata).unwrap()) {
        log::warn!("Failed to write discovery metadata to {path}: {e}");
    } else {
        log::info!("Discovery metadata written to {path}");
    }
}

/// Remove the discovery metadata file (best-effort).
fn cleanup_discovery_metadata() {
    let path = discovery_metadata_path();
    let _ = std::fs::remove_file(&path);
}

fn spawn_ipc_server(api: Arc<dyn rkf_core::automation::AutomationApi>) -> String {
    let socket_path = rkf_mcp::ipc::IpcConfig::default_socket_path();
    let path_clone = socket_path.clone();

    std::thread::Builder::new()
        .name("ipc-server".into())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("failed to create tokio runtime for IPC server");

            rt.block_on(async {
                let mut registry = rkf_mcp::registry::ToolRegistry::new();
                rkf_mcp::tools::observation::register_observation_tools(&mut registry);
                let registry = Arc::new(registry);

                let config = rkf_mcp::ipc::IpcConfig {
                    socket_path: Some(path_clone),
                    tcp_port: 0,
                    mode: rkf_mcp::registry::ToolMode::Debug,
                };

                if let Err(e) = rkf_mcp::ipc::run_server(config, registry, api).await {
                    log::error!("IPC server error: {e}");
                }
            });
        })
        .expect("failed to spawn IPC server thread");

    // Write discovery metadata so MCP clients can find and identify us
    write_discovery_metadata(&socket_path);

    socket_path
}

// ---------------------------------------------------------------------------
// Debug mode from number keys (0-6)
// ---------------------------------------------------------------------------

/// Check if a number key was pressed, returning the debug mode index.
fn debug_mode_from_key(key: winit::keyboard::KeyCode, pressed: bool) -> Option<u32> {
    if !pressed {
        return None;
    }
    use winit::keyboard::KeyCode;
    match key {
        KeyCode::Digit0 => Some(0),
        KeyCode::Digit1 => Some(1),
        KeyCode::Digit2 => Some(2),
        KeyCode::Digit3 => Some(3),
        KeyCode::Digit4 => Some(4),
        KeyCode::Digit5 => Some(5),
        KeyCode::Digit6 => Some(6),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Application
// ---------------------------------------------------------------------------

struct App {
    // Window
    window: Option<Arc<Window>>,

    // wgpu core
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
    surface: Option<wgpu::Surface<'static>>,
    surface_format: wgpu::TextureFormat,
    surface_width: u32,
    surface_height: u32,

    // Rinch embed
    rinch_ctx: Option<RinchContext>,
    overlay: Option<RinchOverlayRenderer>,
    ui_visible: bool,

    // Engine
    engine: Option<EngineState>,

    // Blit pipeline (composites rinch overlay onto surface)
    blit_pipeline: Option<wgpu::RenderPipeline>,
    blit_layout: Option<wgpu::BindGroupLayout>,
    blit_sampler: Option<wgpu::Sampler>,

    // Editor state (shared with UI components via EDITOR_STATE static)
    editor_state: Arc<Mutex<EditorState>>,

    // Input routing
    mouse_phys: (f32, f32),
    prev_mouse: (f32, f32),
    viewport_dragging: bool,
    modifiers: winit::keyboard::ModifiersState,
    pending_events: Vec<PlatformEvent>,

    // MCP
    shared_state: Arc<Mutex<SharedState>>,
    socket_path: Option<String>,

    // Timing
    last_frame: Instant,
    frame_count: u64,
    last_title_update: Instant,
}

impl App {
    fn new() -> Self {
        let editor_state = Arc::new(Mutex::new(EditorState::new()));
        // Store in global static so rinch UI components can access it.
        let _ = EDITOR_STATE.set(Arc::clone(&editor_state));

        Self {
            window: None,
            device: None,
            queue: None,
            surface: None,
            surface_format: wgpu::TextureFormat::Bgra8Unorm,
            surface_width: DISPLAY_WIDTH,
            surface_height: DISPLAY_HEIGHT,
            rinch_ctx: None,
            overlay: None,
            ui_visible: true,
            engine: None,
            blit_pipeline: None,
            blit_layout: None,
            blit_sampler: None,
            editor_state,
            mouse_phys: (0.0, 0.0),
            prev_mouse: (0.0, 0.0),
            viewport_dragging: false,
            modifiers: winit::keyboard::ModifiersState::empty(),
            pending_events: Vec::new(),
            shared_state: Arc::new(Mutex::new(SharedState::new(
                0,
                0,
                DISPLAY_WIDTH,
                DISPLAY_HEIGHT,
            ))),
            socket_path: None,
            last_frame: Instant::now(),
            frame_count: 0,
            last_title_update: Instant::now(),
        }
    }

    fn scale_factor(&self) -> f64 {
        self.window.as_ref().map(|w| w.scale_factor()).unwrap_or(1.0)
    }

    fn logical_mouse(&self) -> (f32, f32) {
        let sf = self.scale_factor() as f32;
        (self.mouse_phys.0 / sf, self.mouse_phys.1 / sf)
    }

    // ── GPU initialization ─────────────────────────────────────────────────

    fn init_gpu(&mut self) {
        let window = self.window.as_ref().unwrap();
        let size = window.inner_size();
        let w = size.width.max(1);
        let h = size.height.max(1);

        // Create wgpu instance + surface
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let surface = instance
            .create_surface(window.clone())
            .expect("failed to create surface");

        // Use RenderContext for device creation (gets rkf features: FLOAT32_FILTERABLE, high limits)
        let render_ctx = rkf_render::RenderContext::new(&instance, &surface);
        let device = render_ctx.device;
        let queue = render_ctx.queue;

        // Configure surface with COPY_SRC for screenshot support
        let caps = surface.get_capabilities(&render_ctx.adapter);
        let format = caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);
        self.surface_format = format;

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            format,
            width: w,
            height: h,
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &config);
        self.surface_width = w;
        self.surface_height = h;

        // ── Blit pipeline (composites rinch overlay onto surface) ───────────

        let blit_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("overlay blit"),
            source: wgpu::ShaderSource::Wgsl(BLIT_WGSL.into()),
        });

        let blit_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let blit_pipe_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&blit_layout],
            push_constant_ranges: &[],
        });

        let blit_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("overlay blit"),
            layout: Some(&blit_pipe_layout),
            vertex: wgpu::VertexState {
                module: &blit_shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &blit_shader,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        });

        let blit_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // ── Rinch embed ────────────────────────────────────────────────────

        let rinch_ctx = RinchContext::new(
            RinchContextConfig {
                width: w,
                height: h,
                scale_factor: self.scale_factor(),
                theme: Some(ThemeProviderProps {
                    dark_mode: true,
                    primary_color: Some("blue".into()),
                    ..Default::default()
                }),
            },
            editor_ui,
        );

        let overlay = RinchOverlayRenderer::new(&device, w, h, wgpu::TextureFormat::Rgba8Unorm);

        // ── Engine ─────────────────────────────────────────────────────────

        let engine = EngineState::new(
            &device,
            &queue,
            format,
            Arc::clone(&self.shared_state),
        );

        // Store everything
        self.surface = Some(surface);
        self.device = Some(device);
        self.queue = Some(queue);
        self.blit_pipeline = Some(blit_pipeline);
        self.blit_layout = Some(blit_layout);
        self.blit_sampler = Some(blit_sampler);
        self.rinch_ctx = Some(rinch_ctx);
        self.overlay = Some(overlay);
        self.engine = Some(engine);

        // Spawn MCP IPC server
        let api: Arc<dyn rkf_core::automation::AutomationApi> = Arc::new(
            EditorAutomationApi::new(
                Arc::clone(&self.shared_state),
                Arc::clone(&self.editor_state),
            ),
        );
        let socket_path = spawn_ipc_server(api);
        log::info!("IPC server listening on {socket_path}");
        self.socket_path = Some(socket_path);

        // Push startup log entries into shared state for MCP read_log
        if let Ok(mut ss) = self.shared_state.lock() {
            ss.push_log(
                rkf_core::automation::LogLevel::Info,
                "Editor initialized — engine viewport active",
            );
            ss.push_log(
                rkf_core::automation::LogLevel::Info,
                format!("IPC server listening on {}", self.socket_path.as_deref().unwrap_or("?")),
            );
        }
        log::info!("Editor initialized — engine viewport active");
    }

    // ── Render one frame ───────────────────────────────────────────────────

    fn render(&mut self) {
        let surface = self.surface.as_ref().unwrap();

        // Acquire surface texture
        let frame = match surface.get_current_texture() {
            Ok(f) => f,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                // Skip this frame — will reconfigure on next resize
                return;
            }
            Err(e) => {
                log::error!("Surface error: {e:?}");
                return;
            }
        };
        let surface_view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Timing
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;
        self.frame_count += 1;

        let device = self.device.as_ref().unwrap();
        let queue = self.queue.as_ref().unwrap();

        // Update rinch context
        if let Some(ctx) = &mut self.rinch_ctx {
            let events: Vec<_> = self.pending_events.drain(..).collect();
            let _actions = ctx.update(&events);
        }

        // Update editor camera from input, then sync to engine camera
        {
            let mut es = self.editor_state.lock().unwrap();
            es.update_camera(dt);
            es.frame_time_history.push(dt * 1000.0);

            // Consume pending MCP camera teleport — apply to editor camera
            // so sync_to_engine_camera carries the correct values.
            if let Ok(mut ss) = self.shared_state.lock() {
                if let Some(cam) = ss.pending_camera.take() {
                    es.editor_camera.position = cam.position;
                    es.editor_camera.fly_yaw = cam.yaw;
                    es.editor_camera.fly_pitch = cam.pitch;
                    log::info!(
                        "Camera set via MCP: pos={:?} yaw={:.2} pitch={:.2}",
                        cam.position, cam.yaw, cam.pitch,
                    );
                }
            }

            if let Some(engine) = &mut self.engine {
                es.sync_to_engine_camera(&mut engine.camera);
            }

            // Consume pending debug mode from UI menus
            if let Some(dm) = es.pending_debug_mode.take() {
                if let Some(engine) = &self.engine {
                    engine.shading.set_debug_mode(queue, dm);
                    log::info!("Debug mode (from menu): {dm}");
                }
            }

            // Sync environment settings when dirty
            if es.environment.is_dirty() {
                if let Some(engine) = &mut self.engine {
                    engine.apply_environment(&es.environment);
                }
                es.environment.clear_dirty();
            }

            // Sync lights when dirty
            if es.light_editor.is_dirty() {
                if let Some(engine) = &mut self.engine {
                    engine.apply_lights(es.light_editor.all_lights());
                }
                es.light_editor.clear_dirty();
            }

            es.reset_frame_deltas();
        }

        // ── Step 1: Engine renders full pipeline + blits to surface ────────
        if let Some(engine) = &mut self.engine {
            engine.render(&surface_view, dt);
        }

        // ── Step 2: Render rinch UI overlay + composite onto surface ───────
        if let (Some(ctx), Some(overlay)) = (&mut self.rinch_ctx, &mut self.overlay) {
            let scene = ctx.scene();
            let overlay_view = overlay.render(device, queue, scene);

            let blit_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: self.blit_layout.as_ref().unwrap(),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&overlay_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(
                            self.blit_sampler.as_ref().unwrap(),
                        ),
                    },
                ],
            });

            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("overlay blit"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &surface_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load, // preserve engine output
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                pass.set_pipeline(self.blit_pipeline.as_ref().unwrap());
                pass.set_bind_group(0, &blit_bg, &[]);
                pass.draw(0..3, 0..1);
            }
            queue.submit(std::iter::once(encoder.finish()));
        }

        frame.present();

        // Update title bar and status bar FPS every 500ms
        if now.duration_since(self.last_title_update).as_millis() > 500 {
            let elapsed = now.duration_since(self.last_title_update).as_secs_f64();
            let fps = self.frame_count as f64 / elapsed;

            if let Some(window) = &self.window {
                window.set_title(&format!(
                    "RKIField Editor — {fps:.0} fps ({:.2} ms)",
                    1000.0 / fps
                ));
            }

            // Push FPS to reactive signal for status bar
            FPS_SIGNAL.with(|cell| {
                if let Some(sig) = cell.get() {
                    sig.set(fps as u16);
                }
            });

            self.frame_count = 0;
            self.last_title_update = now;
        }
    }

    // ── Handle resize ──────────────────────────────────────────────────────

    fn handle_resize(&mut self, w: u32, h: u32) {
        let w = w.max(1);
        let h = h.max(1);
        self.surface_width = w;
        self.surface_height = h;

        if let (Some(device), Some(surface)) = (&self.device, &self.surface) {
            let config = wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                format: self.surface_format,
                width: w,
                height: h,
                present_mode: wgpu::PresentMode::AutoVsync,
                desired_maximum_frame_latency: 2,
                alpha_mode: wgpu::CompositeAlphaMode::Auto,
                view_formats: vec![],
            };
            surface.configure(device, &config);
        }

        if let (Some(device), Some(overlay)) = (&self.device, &mut self.overlay) {
            overlay.resize(device, w, h);
        }

        if let Some(ctx) = &mut self.rinch_ctx {
            ctx.resize(w, h);
        }
    }
}

// ---------------------------------------------------------------------------
// ApplicationHandler
// ---------------------------------------------------------------------------

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attrs = Window::default_attributes()
            .with_title("RKIField Editor")
            .with_inner_size(winit::dpi::PhysicalSize::new(
                DISPLAY_WIDTH,
                DISPLAY_HEIGHT,
            ));
        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        self.window = Some(window);

        self.init_gpu();
        self.last_frame = Instant::now();
        self.last_title_update = Instant::now();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => {
                cleanup_discovery_metadata();
                event_loop.exit();
            }

            WindowEvent::Resized(size) => {
                self.handle_resize(size.width, size.height);
            }

            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                if let Some(ctx) = &mut self.rinch_ctx {
                    ctx.set_scale_factor(scale_factor);
                }
            }

            WindowEvent::ModifiersChanged(mods) => {
                self.modifiers = mods.state();
                self.pending_events.push(PlatformEvent::ModifiersChanged(
                    translate_modifiers(self.modifiers),
                ));
            }

            WindowEvent::CursorMoved { position, .. } => {
                let px = position.x as f32;
                let py = position.y as f32;

                // Accumulate mouse delta for editor camera
                let dx = px - self.mouse_phys.0;
                let dy = py - self.mouse_phys.1;
                if let Ok(mut es) = self.editor_state.lock() {
                    es.editor_input.mouse_pos = glam::Vec2::new(px, py);
                    es.editor_input.mouse_delta.x += dx;
                    es.editor_input.mouse_delta.y += dy;
                }

                self.prev_mouse = self.mouse_phys;
                self.mouse_phys = (px, py);

                // Always send mouse move to rinch (for hover effects)
                self.pending_events
                    .push(PlatformEvent::MouseMove { x: px, y: py });
            }

            WindowEvent::MouseInput { state, button, .. } => {
                let platform_btn = match button {
                    winit::event::MouseButton::Left => PlatformMouseButton::Left,
                    winit::event::MouseButton::Right => PlatformMouseButton::Right,
                    winit::event::MouseButton::Middle => PlatformMouseButton::Middle,
                    _ => return,
                };
                let btn_idx = match platform_btn {
                    PlatformMouseButton::Left => 0usize,
                    PlatformMouseButton::Right => 1,
                    PlatformMouseButton::Middle => 2,
                };
                let (px, py) = self.mouse_phys;

                match state {
                    ElementState::Pressed => {
                        let (lx, ly) = self.logical_mouse();
                        let wants_ui = self.ui_visible
                            && self
                                .rinch_ctx
                                .as_ref()
                                .is_some_and(|ctx| ctx.wants_mouse(lx, ly));

                        if wants_ui {
                            // Send to rinch UI
                            self.pending_events.push(PlatformEvent::MouseDown {
                                x: px,
                                y: py,
                                button: platform_btn,
                            });
                        } else {
                            // Viewport interaction — feed into editor input
                            if let Ok(mut es) = self.editor_state.lock() {
                                es.editor_input.mouse_buttons[btn_idx] = true;
                            }
                            self.viewport_dragging = true;
                        }
                    }
                    ElementState::Released => {
                        // Always clear the button in editor input
                        if let Ok(mut es) = self.editor_state.lock() {
                            es.editor_input.mouse_buttons[btn_idx] = false;
                        }

                        if self.viewport_dragging {
                            // Check if any viewport buttons still held
                            let any_held = self
                                .editor_state
                                .lock()
                                .map(|es| es.editor_input.mouse_buttons.iter().any(|&b| b))
                                .unwrap_or(false);
                            if !any_held {
                                self.viewport_dragging = false;
                            }
                        } else {
                            self.pending_events.push(PlatformEvent::MouseUp {
                                x: px,
                                y: py,
                                button: platform_btn,
                            });
                        }
                    }
                }
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let scroll_y = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y as f64 * 40.0,
                    winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y,
                };
                let (lx, ly) = self.logical_mouse();
                let wants_ui = self.ui_visible
                    && self
                        .rinch_ctx
                        .as_ref()
                        .is_some_and(|ctx| ctx.wants_mouse(lx, ly));

                if wants_ui {
                    let (px, py) = self.mouse_phys;
                    self.pending_events.push(PlatformEvent::MouseWheel {
                        x: px,
                        y: py,
                        delta_x: 0.0,
                        delta_y: scroll_y,
                    });
                } else {
                    // Feed scroll into editor input for camera zoom/move
                    if let Ok(mut es) = self.editor_state.lock() {
                        es.editor_input.scroll_delta += scroll_y as f32;
                    }
                }
            }

            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed;
                let key_code = match event.physical_key {
                    PhysicalKey::Code(k) => k,
                    _ => return,
                };

                // Escape: close menus first, then exit
                if key_code == winit::keyboard::KeyCode::Escape && pressed {
                    let menu_was_open = MENU_SIGNAL.with(|cell| {
                        cell.get().is_some_and(|sig| {
                            if sig.get() != 0 {
                                sig.set(0);
                                true
                            } else {
                                false
                            }
                        })
                    });
                    if !menu_was_open {
                        cleanup_discovery_metadata();
                        event_loop.exit();
                    }
                    return;
                }

                // Global shortcuts (work regardless of UI focus)
                let ctrl = self.modifiers.control_key();
                let shift = self.modifiers.shift_key();

                if pressed && ctrl {
                    use winit::keyboard::KeyCode;
                    match key_code {
                        KeyCode::KeyZ if !shift => {
                            // Ctrl+Z — Undo
                            if let Ok(mut es) = self.editor_state.lock() {
                                if let Some(action) = es.undo.undo() {
                                    log::info!("Undo: {}", action.description);
                                }
                            }
                            return;
                        }
                        KeyCode::KeyY | KeyCode::KeyZ if shift => {
                            // Ctrl+Y or Ctrl+Shift+Z — Redo
                            if let Ok(mut es) = self.editor_state.lock() {
                                if let Some(action) = es.undo.redo() {
                                    log::info!("Redo: {}", action.description);
                                }
                            }
                            return;
                        }
                        _ => {}
                    }
                }

                let wants_kb = self.ui_visible
                    && self
                        .rinch_ctx
                        .as_ref()
                        .is_some_and(|ctx| ctx.wants_keyboard());

                if wants_kb {
                    // Route to rinch for text input
                    let platform_key = translate_key(key_code);
                    let text = event.text.as_ref().map(|s| s.to_string());
                    let mods = translate_modifiers(self.modifiers);
                    if pressed {
                        self.pending_events.push(PlatformEvent::KeyDown {
                            key: platform_key,
                            text,
                            modifiers: mods,
                        });
                    }
                } else {
                    // Feed key state into editor input for camera/tools
                    if let Some(editor_key) = translate_to_editor_key(key_code) {
                        if let Ok(mut es) = self.editor_state.lock() {
                            if pressed {
                                es.editor_input.keys_pressed.insert(editor_key);
                            } else {
                                es.editor_input.keys_pressed.remove(&editor_key);
                            }
                        }
                    }

                    // Debug mode shortcuts (0-6)
                    if let Some(debug_mode) = debug_mode_from_key(key_code, pressed) {
                        if let Some(engine) = &self.engine {
                            engine.shading.set_debug_mode(
                                self.queue.as_ref().unwrap(),
                                debug_mode,
                            );
                            log::info!("Debug mode: {debug_mode}");
                        }
                    }
                }
            }

            WindowEvent::RedrawRequested => {
                self.render();
            }

            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        // Check if UI requested exit (File > Quit)
        if let Ok(es) = self.editor_state.lock() {
            if es.wants_exit {
                cleanup_discovery_metadata();
                event_loop.exit();
                return;
            }
        }

        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app).unwrap();
}
