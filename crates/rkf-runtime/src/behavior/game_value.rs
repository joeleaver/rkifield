//! Dynamically-typed game values for the state store, events, and field-level access.

use glam::{Quat, Vec3};
use rkf_core::WorldPosition;
use serde::{Deserialize, Serialize};

/// A dynamically-typed value used by the game state store, events,
/// field-level component access, and undo/redo.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum GameValue {
    /// Boolean value.
    Bool(bool),
    /// 64-bit signed integer.
    Int(i64),
    /// 64-bit floating-point.
    Float(f64),
    /// UTF-8 string.
    String(String),
    /// 3D vector (for directions, velocities — NOT positions, use WorldPosition).
    Vec3(Vec3),
    /// World-space position with chunk precision.
    WorldPosition(WorldPosition),
    /// Rotation quaternion.
    Quat(Quat),
    /// RGBA color (linear, 0.0–1.0).
    Color([f32; 4]),
    /// Ordered list of values.
    List(Vec<GameValue>),
    /// Arbitrary serde type as RON string (escape hatch).
    Ron(String),
}

// ─── Accessor methods ─────────────────────────────────────────────────────

impl GameValue {
    /// Returns the contained `bool`, or `None` if not a `Bool`.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            GameValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Returns the contained `i64`, or `None` if not an `Int`.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            GameValue::Int(i) => Some(*i),
            _ => None,
        }
    }

    /// Returns the contained `f64`, or `None` if not a `Float`.
    pub fn as_float(&self) -> Option<f64> {
        match self {
            GameValue::Float(f) => Some(*f),
            _ => None,
        }
    }

    /// Returns the contained string slice, or `None` if not a `String`.
    pub fn as_string(&self) -> Option<&str> {
        match self {
            GameValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Returns the contained `Vec3`, or `None` if not a `Vec3`.
    pub fn as_vec3(&self) -> Option<Vec3> {
        match self {
            GameValue::Vec3(v) => Some(*v),
            _ => None,
        }
    }

    /// Returns the contained `WorldPosition`, or `None` if not a `WorldPosition`.
    pub fn as_world_position(&self) -> Option<&WorldPosition> {
        match self {
            GameValue::WorldPosition(wp) => Some(wp),
            _ => None,
        }
    }

    /// Returns the contained `Quat`, or `None` if not a `Quat`.
    pub fn as_quat(&self) -> Option<Quat> {
        match self {
            GameValue::Quat(q) => Some(*q),
            _ => None,
        }
    }

    /// Returns the contained color, or `None` if not a `Color`.
    pub fn as_color(&self) -> Option<[f32; 4]> {
        match self {
            GameValue::Color(c) => Some(*c),
            _ => None,
        }
    }

    /// Returns the contained list slice, or `None` if not a `List`.
    pub fn as_list(&self) -> Option<&[GameValue]> {
        match self {
            GameValue::List(v) => Some(v.as_slice()),
            _ => None,
        }
    }

    /// Returns the contained RON string, or `None` if not a `Ron`.
    pub fn as_ron(&self) -> Option<&str> {
        match self {
            GameValue::Ron(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

// ─── From conversions ─────────────────────────────────────────────────────

impl From<bool> for GameValue {
    fn from(v: bool) -> Self {
        GameValue::Bool(v)
    }
}

impl From<i32> for GameValue {
    fn from(v: i32) -> Self {
        GameValue::Int(v as i64)
    }
}

impl From<i64> for GameValue {
    fn from(v: i64) -> Self {
        GameValue::Int(v)
    }
}

impl From<f32> for GameValue {
    fn from(v: f32) -> Self {
        GameValue::Float(v as f64)
    }
}

impl From<f64> for GameValue {
    fn from(v: f64) -> Self {
        GameValue::Float(v)
    }
}

impl From<String> for GameValue {
    fn from(v: String) -> Self {
        GameValue::String(v)
    }
}

impl From<&str> for GameValue {
    fn from(v: &str) -> Self {
        GameValue::String(v.to_owned())
    }
}

impl From<Vec3> for GameValue {
    fn from(v: Vec3) -> Self {
        GameValue::Vec3(v)
    }
}

impl From<WorldPosition> for GameValue {
    fn from(v: WorldPosition) -> Self {
        GameValue::WorldPosition(v)
    }
}

impl From<Quat> for GameValue {
    fn from(v: Quat) -> Self {
        GameValue::Quat(v)
    }
}

impl From<[f32; 4]> for GameValue {
    fn from(val: [f32; 4]) -> Self {
        GameValue::Color(val)
    }
}

impl From<Vec<GameValue>> for GameValue {
    fn from(val: Vec<GameValue>) -> Self {
        GameValue::List(val)
    }
}

// ─── TryFrom conversions (GameValue → concrete type) ──────────────────────

/// Error when converting a [`GameValue`] to a concrete type.
#[derive(Debug, Clone, thiserror::Error)]
#[error("expected {expected}, got {actual}")]
pub struct GameValueTypeError {
    /// The expected type name.
    pub expected: &'static str,
    /// The actual variant name.
    pub actual: &'static str,
}

impl GameValue {
    fn variant_name(&self) -> &'static str {
        match self {
            GameValue::Bool(_) => "Bool",
            GameValue::Int(_) => "Int",
            GameValue::Float(_) => "Float",
            GameValue::String(_) => "String",
            GameValue::Vec3(_) => "Vec3",
            GameValue::WorldPosition(_) => "WorldPosition",
            GameValue::Quat(_) => "Quat",
            GameValue::Color(_) => "Color",
            GameValue::List(_) => "List",
            GameValue::Ron(_) => "Ron",
        }
    }
}

impl TryFrom<GameValue> for bool {
    type Error = GameValueTypeError;
    fn try_from(v: GameValue) -> Result<Self, Self::Error> {
        match v {
            GameValue::Bool(b) => Ok(b),
            other => Err(GameValueTypeError {
                expected: "Bool",
                actual: other.variant_name(),
            }),
        }
    }
}

impl TryFrom<GameValue> for i64 {
    type Error = GameValueTypeError;
    fn try_from(v: GameValue) -> Result<Self, Self::Error> {
        match v {
            GameValue::Int(i) => Ok(i),
            other => Err(GameValueTypeError {
                expected: "Int",
                actual: other.variant_name(),
            }),
        }
    }
}

impl TryFrom<GameValue> for i32 {
    type Error = GameValueTypeError;
    fn try_from(v: GameValue) -> Result<Self, Self::Error> {
        match v {
            GameValue::Int(i) => i32::try_from(i).map_err(|_| GameValueTypeError {
                expected: "Int(i32 range)",
                actual: "Int(out of i32 range)",
            }),
            other => Err(GameValueTypeError {
                expected: "Int",
                actual: other.variant_name(),
            }),
        }
    }
}

impl TryFrom<GameValue> for f64 {
    type Error = GameValueTypeError;
    fn try_from(v: GameValue) -> Result<Self, Self::Error> {
        match v {
            GameValue::Float(f) => Ok(f),
            other => Err(GameValueTypeError {
                expected: "Float",
                actual: other.variant_name(),
            }),
        }
    }
}

impl TryFrom<GameValue> for f32 {
    type Error = GameValueTypeError;
    fn try_from(v: GameValue) -> Result<Self, Self::Error> {
        match v {
            GameValue::Float(f) => Ok(f as f32),
            other => Err(GameValueTypeError {
                expected: "Float",
                actual: other.variant_name(),
            }),
        }
    }
}

impl TryFrom<GameValue> for String {
    type Error = GameValueTypeError;
    fn try_from(v: GameValue) -> Result<Self, Self::Error> {
        match v {
            GameValue::String(s) => Ok(s),
            other => Err(GameValueTypeError {
                expected: "String",
                actual: other.variant_name(),
            }),
        }
    }
}

impl TryFrom<GameValue> for Vec3 {
    type Error = GameValueTypeError;
    fn try_from(v: GameValue) -> Result<Self, Self::Error> {
        match v {
            GameValue::Vec3(v) => Ok(v),
            other => Err(GameValueTypeError {
                expected: "Vec3",
                actual: other.variant_name(),
            }),
        }
    }
}

impl TryFrom<GameValue> for WorldPosition {
    type Error = GameValueTypeError;
    fn try_from(v: GameValue) -> Result<Self, Self::Error> {
        match v {
            GameValue::WorldPosition(wp) => Ok(wp),
            other => Err(GameValueTypeError {
                expected: "WorldPosition",
                actual: other.variant_name(),
            }),
        }
    }
}

impl TryFrom<GameValue> for Quat {
    type Error = GameValueTypeError;
    fn try_from(v: GameValue) -> Result<Self, Self::Error> {
        match v {
            GameValue::Quat(q) => Ok(q),
            other => Err(GameValueTypeError {
                expected: "Quat",
                actual: other.variant_name(),
            }),
        }
    }
}

impl TryFrom<GameValue> for [f32; 4] {
    type Error = GameValueTypeError;
    fn try_from(v: GameValue) -> Result<Self, Self::Error> {
        match v {
            GameValue::Color(c) => Ok(c),
            other => Err(GameValueTypeError {
                expected: "Color",
                actual: other.variant_name(),
            }),
        }
    }
}

impl TryFrom<GameValue> for Vec<GameValue> {
    type Error = GameValueTypeError;
    fn try_from(v: GameValue) -> Result<Self, Self::Error> {
        match v {
            GameValue::List(l) => Ok(l),
            other => Err(GameValueTypeError {
                expected: "List",
                actual: other.variant_name(),
            }),
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{IVec3, Quat, Vec3};

    #[test]
    fn from_bool() {
        let v: GameValue = true.into();
        assert_eq!(v, GameValue::Bool(true));
        assert_eq!(v.as_bool(), Some(true));
        assert_eq!(v.as_int(), None);
    }

    #[test]
    fn from_int() {
        let v: GameValue = 42_i64.into();
        assert_eq!(v, GameValue::Int(42));
        assert_eq!(v.as_int(), Some(42));

        let v32: GameValue = 7_i32.into();
        assert_eq!(v32, GameValue::Int(7));
    }

    #[test]
    fn from_float() {
        let v: GameValue = 3.14_f64.into();
        assert_eq!(v.as_float(), Some(3.14));

        let v32: GameValue = 2.5_f32.into();
        assert_eq!(v32.as_float(), Some(2.5));
    }

    #[test]
    fn from_string() {
        let v: GameValue = "hello".into();
        assert_eq!(v.as_string(), Some("hello"));

        let v2: GameValue = String::from("world").into();
        assert_eq!(v2.as_string(), Some("world"));
    }

    #[test]
    fn from_vec3() {
        let v: GameValue = Vec3::new(1.0, 2.0, 3.0).into();
        assert_eq!(v.as_vec3(), Some(Vec3::new(1.0, 2.0, 3.0)));
        assert_eq!(v.as_float(), None);
    }

    #[test]
    fn from_world_position() {
        let wp = WorldPosition {
            chunk: IVec3::new(1, 0, 0),
            local: Vec3::new(5.0, 0.0, 0.0),
        };
        let v: GameValue = wp.clone().into();
        assert_eq!(v.as_world_position(), Some(&wp));
    }

    #[test]
    fn from_quat() {
        let q = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        let v: GameValue = q.into();
        let got = v.as_quat().unwrap();
        assert!((got.x - q.x).abs() < 1e-6);
        assert!((got.y - q.y).abs() < 1e-6);
    }

    #[test]
    fn color_value() {
        let c = [1.0, 0.5, 0.0, 1.0];
        let v = GameValue::Color(c);
        assert_eq!(v.as_color(), Some(c));
        assert_eq!(v.as_vec3(), None);
    }

    #[test]
    fn list_value() {
        let v = GameValue::List(vec![GameValue::Int(1), GameValue::Bool(true)]);
        let list = v.as_list().unwrap();
        assert_eq!(list.len(), 2);
        assert_eq!(list[0], GameValue::Int(1));
    }

    #[test]
    fn ron_value() {
        let v = GameValue::Ron("(x: 1, y: 2)".to_owned());
        assert_eq!(v.as_ron(), Some("(x: 1, y: 2)"));
        assert_eq!(v.as_string(), None);
    }

    #[test]
    fn try_from_roundtrip() {
        // Bool
        let v: GameValue = true.into();
        assert_eq!(bool::try_from(v).unwrap(), true);

        // Int
        let v: GameValue = 99_i64.into();
        assert_eq!(i64::try_from(v).unwrap(), 99);

        // i32 via Int
        let v: GameValue = 42_i32.into();
        assert_eq!(i32::try_from(v).unwrap(), 42);

        // Float
        let v: GameValue = 1.5_f64.into();
        assert!((f64::try_from(v).unwrap() - 1.5).abs() < 1e-12);

        // f32 via Float
        let v: GameValue = 2.5_f32.into();
        assert!((f32::try_from(v).unwrap() - 2.5).abs() < 1e-6);

        // String
        let v: GameValue = "test".into();
        assert_eq!(String::try_from(v).unwrap(), "test");

        // Vec3
        let v: GameValue = Vec3::X.into();
        assert_eq!(Vec3::try_from(v).unwrap(), Vec3::X);

        // Quat
        let q = Quat::IDENTITY;
        let v: GameValue = q.into();
        assert_eq!(Quat::try_from(v).unwrap(), q);
    }

    #[test]
    fn try_from_type_mismatch() {
        let v = GameValue::Bool(true);
        let err = i64::try_from(v).unwrap_err();
        assert_eq!(err.expected, "Int");
        assert_eq!(err.actual, "Bool");
    }

    #[test]
    fn from_color_array() {
        let c = [1.0_f32, 0.5, 0.0, 1.0];
        let v: GameValue = c.into();
        assert_eq!(v, GameValue::Color(c));
        assert_eq!(v.as_color(), Some(c));
    }

    #[test]
    fn from_list() {
        let items = vec![GameValue::Int(1), GameValue::Bool(false)];
        let v: GameValue = items.clone().into();
        assert_eq!(v, GameValue::List(items));
    }

    #[test]
    fn try_from_color() {
        let c = [0.2_f32, 0.4, 0.6, 0.8];
        let v: GameValue = c.into();
        let got = <[f32; 4]>::try_from(v).unwrap();
        assert_eq!(got, c);
    }

    #[test]
    fn try_from_list() {
        let items = vec![GameValue::Float(1.5), GameValue::String("x".into())];
        let v: GameValue = items.clone().into();
        let got = Vec::<GameValue>::try_from(v).unwrap();
        assert_eq!(got, items);
    }

    #[test]
    fn try_from_color_wrong_type() {
        let v = GameValue::Int(42);
        let err = <[f32; 4]>::try_from(v).unwrap_err();
        assert_eq!(err.expected, "Color");
        assert_eq!(err.actual, "Int");
    }

    #[test]
    fn try_from_list_wrong_type() {
        let v = GameValue::Bool(true);
        let err = Vec::<GameValue>::try_from(v).unwrap_err();
        assert_eq!(err.expected, "List");
        assert_eq!(err.actual, "Bool");
    }

    #[test]
    fn serialization_roundtrip() {
        let values = vec![
            GameValue::Bool(true),
            GameValue::Int(-42),
            GameValue::Float(3.14),
            GameValue::String("hello".into()),
            GameValue::Vec3(Vec3::new(1.0, 2.0, 3.0)),
            GameValue::Quat(Quat::IDENTITY),
            GameValue::Color([1.0, 0.5, 0.0, 1.0]),
            GameValue::List(vec![GameValue::Int(1)]),
            GameValue::Ron("(x: 1)".into()),
        ];

        for v in &values {
            let ron = ron::to_string(v).unwrap();
            let back: GameValue = ron::from_str(&ron).unwrap();
            assert_eq!(&back, v, "failed round-trip for {:?}", v);
        }
    }
}
