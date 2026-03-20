//! UiValue — the dynamically-typed value used by UI widgets.
//!
//! Widgets produce and consume `UiValue`. The store converts between
//! `UiValue` and `GameValue` (the engine's typed value) at the boundary.

use rkf_runtime::behavior::game_value::GameValue;

/// A dynamically-typed value used by UI widgets.
///
/// These are the types that widgets naturally produce: sliders emit `Float`,
/// toggles emit `Bool`, text inputs emit `String`, etc. The store converts
/// to/from `GameValue` as needed.
#[derive(Debug, Clone, PartialEq)]
pub enum UiValue {
    /// Floating-point value (sliders, number inputs).
    Float(f64),
    /// Integer value (enum selects, counters).
    Int(i64),
    /// Boolean value (toggles).
    Bool(bool),
    /// String value (text inputs, color hex, asset paths).
    String(String),
    /// 3D vector value (position/rotation/scale inputs).
    Vec3([f64; 3]),
    /// No value / cleared.
    None,
}

// ─── Accessors ───────────────────────────────────────────────────────────

impl UiValue {
    /// Returns the contained `f64`, or `None` if not a `Float`.
    pub fn as_float(&self) -> Option<f64> {
        match self {
            UiValue::Float(f) => Some(*f),
            _ => None,
        }
    }

    /// Returns the contained `i64`, or `None` if not an `Int`.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            UiValue::Int(i) => Some(*i),
            _ => None,
        }
    }

    /// Returns the contained `bool`, or `None` if not a `Bool`.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            UiValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Returns the contained string slice, or `None` if not a `String`.
    pub fn as_string(&self) -> Option<&str> {
        match self {
            UiValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Returns the contained `[f64; 3]`, or `None` if not a `Vec3`.
    pub fn as_vec3(&self) -> Option<[f64; 3]> {
        match self {
            UiValue::Vec3(v) => Some(*v),
            _ => None,
        }
    }
}

// ─── UiValue → GameValue ─────────────────────────────────────────────────

impl From<UiValue> for GameValue {
    fn from(ui: UiValue) -> Self {
        match ui {
            UiValue::Float(f) => GameValue::Float(f),
            UiValue::Int(i) => GameValue::Int(i),
            UiValue::Bool(b) => GameValue::Bool(b),
            UiValue::String(s) => GameValue::String(s),
            UiValue::Vec3([x, y, z]) => {
                GameValue::Vec3(glam::Vec3::new(x as f32, y as f32, z as f32))
            }
            UiValue::None => GameValue::String(String::new()),
        }
    }
}

// ─── GameValue → UiValue ─────────────────────────────────────────────────

impl From<GameValue> for UiValue {
    fn from(gv: GameValue) -> Self {
        match gv {
            GameValue::Bool(b) => UiValue::Bool(b),
            GameValue::Int(i) => UiValue::Int(i),
            GameValue::Float(f) => UiValue::Float(f),
            GameValue::String(s) => UiValue::String(s),
            GameValue::Vec3(v) => UiValue::Vec3([v.x as f64, v.y as f64, v.z as f64]),
            GameValue::Color([r, g, b, _a]) => {
                // Color → hex string for UI color swatches.
                UiValue::String(rgb_to_hex([r as f64, g as f64, b as f64]))
            }
            GameValue::WorldPosition(wp) => {
                // Convert to absolute f64 coordinates.
                let chunk_size = rkf_core::world_position::CHUNK_SIZE as f64;
                let x = wp.chunk.x as f64 * chunk_size + wp.local.x as f64;
                let y = wp.chunk.y as f64 * chunk_size + wp.local.y as f64;
                let z = wp.chunk.z as f64 * chunk_size + wp.local.z as f64;
                UiValue::Vec3([x, y, z])
            }
            GameValue::Quat(q) => {
                let (yaw, pitch, roll) = quat_to_euler_degrees(q);
                UiValue::Vec3([yaw, pitch, roll])
            }
            GameValue::List(_) | GameValue::Struct(_) | GameValue::Ron(_) => {
                // Complex types: serialize to string for display.
                UiValue::String(format!("{:?}", gv))
            }
        }
    }
}

/// Convert quaternion to euler angles in degrees (yaw, pitch, roll).
fn quat_to_euler_degrees(q: glam::Quat) -> (f64, f64, f64) {
    let (yaw, pitch, roll) = q.to_euler(glam::EulerRot::YXZ);
    (
        yaw.to_degrees() as f64,
        pitch.to_degrees() as f64,
        roll.to_degrees() as f64,
    )
}

// ─── Color conversion helpers ────────────────────────────────────────────

/// Parse a "#rrggbb" hex string to `[0.0–1.0, 0.0–1.0, 0.0–1.0]`.
///
/// Returns `None` if the input is not a valid 7-character hex color string.
pub fn hex_to_rgb(hex: &str) -> Option<[f64; 3]> {
    let hex = hex.strip_prefix('#')?;
    if hex.len() != 6 {
        return None;
    }
    let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
    let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
    let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
    Some([r as f64 / 255.0, g as f64 / 255.0, b as f64 / 255.0])
}

/// Convert `[0.0–1.0]` RGB components to a "#rrggbb" hex string.
///
/// Values are clamped to `[0.0, 1.0]` before conversion.
pub fn rgb_to_hex(rgb: [f64; 3]) -> String {
    let clamp = |v: f64| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    format!("#{:02x}{:02x}{:02x}", clamp(rgb[0]), clamp(rgb[1]), clamp(rgb[2]))
}

// ─── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{IVec3, Quat, Vec3};
    use rkf_core::WorldPosition;

    // ── Accessor tests ──────────────────────────────────────────────────

    #[test]
    fn accessor_float() {
        let v = UiValue::Float(3.14);
        assert_eq!(v.as_float(), Some(3.14));
        assert_eq!(v.as_int(), None);
        assert_eq!(v.as_bool(), None);
        assert_eq!(v.as_string(), None);
        assert_eq!(v.as_vec3(), None);
    }

    #[test]
    fn accessor_int() {
        let v = UiValue::Int(42);
        assert_eq!(v.as_int(), Some(42));
        assert_eq!(v.as_float(), None);
    }

    #[test]
    fn accessor_bool() {
        let v = UiValue::Bool(true);
        assert_eq!(v.as_bool(), Some(true));
        assert_eq!(v.as_float(), None);
    }

    #[test]
    fn accessor_string() {
        let v = UiValue::String("hello".into());
        assert_eq!(v.as_string(), Some("hello"));
        assert_eq!(v.as_float(), None);
    }

    #[test]
    fn accessor_vec3() {
        let v = UiValue::Vec3([1.0, 2.0, 3.0]);
        assert_eq!(v.as_vec3(), Some([1.0, 2.0, 3.0]));
        assert_eq!(v.as_float(), None);
    }

    #[test]
    fn accessor_none() {
        let v = UiValue::None;
        assert_eq!(v.as_float(), None);
        assert_eq!(v.as_int(), None);
        assert_eq!(v.as_bool(), None);
        assert_eq!(v.as_string(), None);
        assert_eq!(v.as_vec3(), None);
    }

    // ── UiValue → GameValue ─────────────────────────────────────────────

    #[test]
    fn ui_to_game_float() {
        let gv: GameValue = UiValue::Float(2.5).into();
        assert_eq!(gv.as_float(), Some(2.5));
    }

    #[test]
    fn ui_to_game_int() {
        let gv: GameValue = UiValue::Int(-7).into();
        assert_eq!(gv.as_int(), Some(-7));
    }

    #[test]
    fn ui_to_game_bool() {
        let gv: GameValue = UiValue::Bool(false).into();
        assert_eq!(gv.as_bool(), Some(false));
    }

    #[test]
    fn ui_to_game_string() {
        let gv: GameValue = UiValue::String("test".into()).into();
        assert_eq!(gv.as_string(), Some("test"));
    }

    #[test]
    fn ui_to_game_vec3() {
        let gv: GameValue = UiValue::Vec3([1.0, 2.0, 3.0]).into();
        let v = gv.as_vec3().unwrap();
        assert!((v.x - 1.0).abs() < 1e-6);
        assert!((v.y - 2.0).abs() < 1e-6);
        assert!((v.z - 3.0).abs() < 1e-6);
    }

    #[test]
    fn ui_to_game_none() {
        let gv: GameValue = UiValue::None.into();
        // None maps to empty string.
        assert_eq!(gv.as_string(), Some(""));
    }

    // ── GameValue → UiValue ─────────────────────────────────────────────

    #[test]
    fn game_to_ui_bool() {
        let ui: UiValue = GameValue::Bool(true).into();
        assert_eq!(ui, UiValue::Bool(true));
    }

    #[test]
    fn game_to_ui_int() {
        let ui: UiValue = GameValue::Int(99).into();
        assert_eq!(ui, UiValue::Int(99));
    }

    #[test]
    fn game_to_ui_float() {
        let ui: UiValue = GameValue::Float(1.5).into();
        assert_eq!(ui, UiValue::Float(1.5));
    }

    #[test]
    fn game_to_ui_string() {
        let ui: UiValue = GameValue::String("abc".into()).into();
        assert_eq!(ui, UiValue::String("abc".into()));
    }

    #[test]
    fn game_to_ui_vec3() {
        let ui: UiValue = GameValue::Vec3(Vec3::new(1.0, 2.0, 3.0)).into();
        let v = ui.as_vec3().unwrap();
        assert!((v[0] - 1.0).abs() < 1e-6);
        assert!((v[1] - 2.0).abs() < 1e-6);
        assert!((v[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn game_to_ui_color() {
        let ui: UiValue = GameValue::Color([1.0, 0.0, 0.5, 1.0]).into();
        // Color converts to hex string.
        assert_eq!(ui, UiValue::String("#ff0080".into()));
    }

    #[test]
    fn game_to_ui_world_position() {
        let wp = WorldPosition {
            chunk: IVec3::new(1, 0, 0),
            local: Vec3::new(5.0, 0.0, 0.0),
        };
        let ui: UiValue = GameValue::WorldPosition(wp).into();
        let v = ui.as_vec3().unwrap();
        // chunk(1,0,0) * 8.0 + local(5,0,0) = (13, 0, 0)
        assert!((v[0] - 13.0).abs() < 1e-6);
        assert!((v[1]).abs() < 1e-6);
        assert!((v[2]).abs() < 1e-6);
    }

    #[test]
    fn game_to_ui_quat() {
        // Identity quaternion → (0, 0, 0) degrees.
        let ui: UiValue = GameValue::Quat(Quat::IDENTITY).into();
        let v = ui.as_vec3().unwrap();
        assert!(v[0].abs() < 1e-6);
        assert!(v[1].abs() < 1e-6);
        assert!(v[2].abs() < 1e-6);
    }

    #[test]
    fn game_to_ui_complex_types() {
        // List, Struct, Ron → String (debug format).
        let ui: UiValue = GameValue::List(vec![GameValue::Int(1)]).into();
        assert!(ui.as_string().is_some());

        let ui: UiValue = GameValue::Struct(vec![("x".into(), GameValue::Float(1.0))]).into();
        assert!(ui.as_string().is_some());

        let ui: UiValue = GameValue::Ron("(x: 1)".into()).into();
        assert!(ui.as_string().is_some());
    }

    // ── Round-trip tests ────────────────────────────────────────────────

    #[test]
    fn roundtrip_float() {
        let original = UiValue::Float(42.5);
        let gv: GameValue = original.clone().into();
        let back: UiValue = gv.into();
        assert_eq!(back, original);
    }

    #[test]
    fn roundtrip_int() {
        let original = UiValue::Int(-100);
        let gv: GameValue = original.clone().into();
        let back: UiValue = gv.into();
        assert_eq!(back, original);
    }

    #[test]
    fn roundtrip_bool() {
        let original = UiValue::Bool(true);
        let gv: GameValue = original.clone().into();
        let back: UiValue = gv.into();
        assert_eq!(back, original);
    }

    #[test]
    fn roundtrip_string() {
        let original = UiValue::String("hello world".into());
        let gv: GameValue = original.clone().into();
        let back: UiValue = gv.into();
        assert_eq!(back, original);
    }

    // ── NaN / special float tests ───────────────────────────────────────

    #[test]
    fn float_nan() {
        let v = UiValue::Float(f64::NAN);
        assert!(v.as_float().unwrap().is_nan());
        let gv: GameValue = v.into();
        assert!(gv.as_float().unwrap().is_nan());
    }

    #[test]
    fn float_infinity() {
        let v = UiValue::Float(f64::INFINITY);
        assert_eq!(v.as_float(), Some(f64::INFINITY));
        let gv: GameValue = v.into();
        assert_eq!(gv.as_float(), Some(f64::INFINITY));
    }

    #[test]
    fn float_negative() {
        let v = UiValue::Float(-999.99);
        assert_eq!(v.as_float(), Some(-999.99));
    }

    #[test]
    fn int_negative() {
        let v = UiValue::Int(i64::MIN);
        assert_eq!(v.as_int(), Some(i64::MIN));
        let gv: GameValue = v.into();
        assert_eq!(gv.as_int(), Some(i64::MIN));
    }

    // ── hex_to_rgb tests ────────────────────────────────────────────────

    #[test]
    fn hex_to_rgb_valid() {
        let rgb = hex_to_rgb("#ff8000").unwrap();
        assert!((rgb[0] - 1.0).abs() < 0.005);
        assert!((rgb[1] - 0.502).abs() < 0.005);
        assert!((rgb[2] - 0.0).abs() < 0.005);
    }

    #[test]
    fn hex_to_rgb_black() {
        let rgb = hex_to_rgb("#000000").unwrap();
        assert_eq!(rgb, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn hex_to_rgb_white() {
        let rgb = hex_to_rgb("#ffffff").unwrap();
        assert!((rgb[0] - 1.0).abs() < 0.005);
        assert!((rgb[1] - 1.0).abs() < 0.005);
        assert!((rgb[2] - 1.0).abs() < 0.005);
    }

    #[test]
    fn hex_to_rgb_uppercase() {
        let rgb = hex_to_rgb("#FF0000").unwrap();
        assert!((rgb[0] - 1.0).abs() < 0.005);
        assert_eq!(rgb[1], 0.0);
        assert_eq!(rgb[2], 0.0);
    }

    #[test]
    fn hex_to_rgb_missing_hash() {
        assert_eq!(hex_to_rgb("ff0000"), None);
    }

    #[test]
    fn hex_to_rgb_wrong_length_short() {
        assert_eq!(hex_to_rgb("#fff"), None);
    }

    #[test]
    fn hex_to_rgb_wrong_length_long() {
        assert_eq!(hex_to_rgb("#ff000000"), None);
    }

    #[test]
    fn hex_to_rgb_non_hex_chars() {
        assert_eq!(hex_to_rgb("#gggggg"), None);
    }

    #[test]
    fn hex_to_rgb_empty() {
        assert_eq!(hex_to_rgb(""), None);
    }

    #[test]
    fn hex_to_rgb_just_hash() {
        assert_eq!(hex_to_rgb("#"), None);
    }

    // ── rgb_to_hex tests ────────────────────────────────────────────────

    #[test]
    fn rgb_to_hex_basic() {
        assert_eq!(rgb_to_hex([1.0, 0.0, 0.0]), "#ff0000");
        assert_eq!(rgb_to_hex([0.0, 1.0, 0.0]), "#00ff00");
        assert_eq!(rgb_to_hex([0.0, 0.0, 1.0]), "#0000ff");
    }

    #[test]
    fn rgb_to_hex_black() {
        assert_eq!(rgb_to_hex([0.0, 0.0, 0.0]), "#000000");
    }

    #[test]
    fn rgb_to_hex_white() {
        assert_eq!(rgb_to_hex([1.0, 1.0, 1.0]), "#ffffff");
    }

    #[test]
    fn rgb_to_hex_clamps_negative() {
        assert_eq!(rgb_to_hex([-1.0, 0.0, 0.0]), "#000000");
    }

    #[test]
    fn rgb_to_hex_clamps_over_one() {
        assert_eq!(rgb_to_hex([2.0, 0.0, 0.0]), "#ff0000");
    }

    #[test]
    fn rgb_to_hex_nan() {
        // NaN clamps to 0 via .clamp().
        let hex = rgb_to_hex([f64::NAN, 0.0, 0.0]);
        // NaN.clamp(0,1) returns NaN in Rust, so .round() gives NaN, `as u8` gives 0.
        assert_eq!(hex, "#000000");
    }

    // ── hex ↔ rgb round-trip ────────────────────────────────────────────

    #[test]
    fn hex_rgb_roundtrip() {
        let original = "#3a7f1c";
        let rgb = hex_to_rgb(original).unwrap();
        let back = rgb_to_hex(rgb);
        assert_eq!(back, original);
    }

    #[test]
    fn rgb_hex_roundtrip() {
        // Note: not all f64 values survive the 8-bit quantization,
        // but exact u8-aligned values do.
        let original = [128.0 / 255.0, 64.0 / 255.0, 255.0 / 255.0];
        let hex = rgb_to_hex(original);
        let back = hex_to_rgb(&hex).unwrap();
        assert!((back[0] - original[0]).abs() < 0.005);
        assert!((back[1] - original[1]).abs() < 0.005);
        assert!((back[2] - original[2]).abs() < 0.005);
    }

    // ── Clone / Debug / PartialEq ───────────────────────────────────────

    #[test]
    fn clone_and_eq() {
        let v = UiValue::Vec3([1.0, 2.0, 3.0]);
        let v2 = v.clone();
        assert_eq!(v, v2);
    }

    #[test]
    fn debug_format() {
        let v = UiValue::Float(1.5);
        let s = format!("{:?}", v);
        assert!(s.contains("Float"));
        assert!(s.contains("1.5"));
    }

    #[test]
    fn not_equal_different_variants() {
        assert_ne!(UiValue::Float(1.0), UiValue::Int(1));
        assert_ne!(UiValue::Bool(true), UiValue::String("true".into()));
        assert_ne!(UiValue::None, UiValue::Float(0.0));
    }

    // ── Vec3 f64→f32 precision ──────────────────────────────────────────

    #[test]
    fn vec3_precision_loss() {
        // f64 values that don't round-trip perfectly through f32.
        let ui = UiValue::Vec3([0.1, 0.2, 0.3]);
        let gv: GameValue = ui.into();
        let v = gv.as_vec3().unwrap();
        // f32 precision: ~7 decimal digits.
        assert!((v.x as f64 - 0.1).abs() < 1e-7);
        assert!((v.y as f64 - 0.2).abs() < 1e-7);
        assert!((v.z as f64 - 0.3).abs() < 1e-7);
    }
}
