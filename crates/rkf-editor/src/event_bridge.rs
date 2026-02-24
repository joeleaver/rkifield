//! Winit → rinch platform event translation.
//!
//! Converts winit `WindowEvent` variants into rinch `PlatformEvent` for
//! the embedded rinch UI context. Used by the editor main loop to feed
//! events into `RinchContext::update()`.

use rinch_platform::{
    KeyCode as PlatformKeyCode, Modifiers as PlatformModifiers,
    MouseButton as PlatformMouseButton, PlatformEvent,
};
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::keyboard::{KeyCode as WinitKeyCode, PhysicalKey};

/// Translate a winit `KeyCode` to a rinch `PlatformKeyCode`.
pub fn translate_platform_key(key: WinitKeyCode) -> PlatformKeyCode {
    use WinitKeyCode as WK;
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

/// Translate winit modifier state to rinch modifiers.
pub fn translate_modifiers(m: winit::keyboard::ModifiersState) -> PlatformModifiers {
    PlatformModifiers {
        shift: m.shift_key(),
        ctrl: m.control_key(),
        alt: m.alt_key(),
        meta: m.super_key(),
    }
}

/// Translate a winit `MouseButton` to a rinch `PlatformMouseButton`.
fn translate_mouse_button(button: MouseButton) -> Option<PlatformMouseButton> {
    match button {
        MouseButton::Left => Some(PlatformMouseButton::Left),
        MouseButton::Right => Some(PlatformMouseButton::Right),
        MouseButton::Middle => Some(PlatformMouseButton::Middle),
        _ => None,
    }
}

/// Convert a winit `WindowEvent` into zero or more `PlatformEvent`s.
///
/// The `modifiers` parameter should be the current winit modifier state,
/// tracked by the caller across `ModifiersChanged` events.
pub fn translate_window_event(
    event: &WindowEvent,
    modifiers: winit::keyboard::ModifiersState,
) -> Vec<PlatformEvent> {
    match event {
        WindowEvent::CursorMoved { position, .. } => {
            vec![PlatformEvent::MouseMove {
                x: position.x as f32,
                y: position.y as f32,
            }]
        }

        WindowEvent::MouseInput { state, button, .. } => {
            let Some(btn) = translate_mouse_button(*button) else {
                return vec![];
            };
            // Note: we don't have position in MouseInput — caller should
            // track the last cursor position and use it. For now, rinch
            // uses the last MouseMove position internally.
            match state {
                ElementState::Pressed => vec![PlatformEvent::MouseDown {
                    x: 0.0,
                    y: 0.0,
                    button: btn,
                }],
                ElementState::Released => vec![PlatformEvent::MouseUp {
                    x: 0.0,
                    y: 0.0,
                    button: btn,
                }],
            }
        }

        WindowEvent::MouseWheel { delta, .. } => {
            let delta_y = match delta {
                winit::event::MouseScrollDelta::LineDelta(_, y) => *y as f64 * 40.0,
                winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y,
            };
            vec![PlatformEvent::MouseWheel {
                x: 0.0,
                y: 0.0,
                delta_x: 0.0,
                delta_y,
            }]
        }

        WindowEvent::KeyboardInput { event, .. } => {
            if event.state != ElementState::Pressed {
                return vec![];
            }
            let key_code = match event.physical_key {
                PhysicalKey::Code(k) => translate_platform_key(k),
                _ => PlatformKeyCode::Other,
            };
            let text = event.text.as_ref().map(|s| s.to_string());
            let mods = translate_modifiers(modifiers);
            vec![PlatformEvent::KeyDown {
                key: key_code,
                text,
                modifiers: mods,
            }]
        }

        WindowEvent::ModifiersChanged(mods) => {
            vec![PlatformEvent::ModifiersChanged(translate_modifiers(
                mods.state(),
            ))]
        }

        WindowEvent::Resized(size) => {
            vec![PlatformEvent::Resized {
                width: size.width,
                height: size.height,
            }]
        }

        WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
            vec![PlatformEvent::ScaleFactorChanged(*scale_factor)]
        }

        _ => vec![],
    }
}
