//! Property inspector data model for the RKIField editor.
//!
//! Provides a typed property system for inspecting and editing entity components.
//! Each entity gets a `PropertySheet` containing named `PropertyDef` entries with
//! typed `PropertyValue` variants. This is a pure data model independent of the GUI.

#![allow(dead_code)]

use glam::{Quat, Vec3};
use rkf_core::scene_node::SdfSource;

/// A typed property value that can appear in the inspector.
#[derive(Debug, Clone, PartialEq)]
pub enum PropertyValue {
    Float(f32),
    Vec3(Vec3),
    Quat(Quat),
    Color(Vec3),
    Bool(bool),
    String(String),
    U32(u32),
    MaterialId(u16),
}

impl PropertyValue {
    /// Returns a short type name for display and type-checking.
    fn type_name(&self) -> &'static str {
        match self {
            PropertyValue::Float(_) => "Float",
            PropertyValue::Vec3(_) => "Vec3",
            PropertyValue::Quat(_) => "Quat",
            PropertyValue::Color(_) => "Color",
            PropertyValue::Bool(_) => "Bool",
            PropertyValue::String(_) => "String",
            PropertyValue::U32(_) => "U32",
            PropertyValue::MaterialId(_) => "MaterialId",
        }
    }

    /// Check if two values have the same type variant (ignoring inner value).
    fn same_type(&self, other: &PropertyValue) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(other)
    }
}

/// A single property definition with name, value, optional bounds, and read-only flag.
#[derive(Debug, Clone)]
pub struct PropertyDef {
    /// Display name of this property.
    pub name: String,
    /// Current value.
    pub value: PropertyValue,
    /// Optional minimum bound (applies to Float, Vec3 components, etc.).
    pub min: Option<f32>,
    /// Optional maximum bound.
    pub max: Option<f32>,
    /// If true, the property cannot be modified through `set_property`.
    pub read_only: bool,
}

impl PropertyDef {
    /// Create a new property with no bounds and read-write access.
    pub fn new(name: impl Into<String>, value: PropertyValue) -> Self {
        Self {
            name: name.into(),
            value,
            min: None,
            max: None,
            read_only: false,
        }
    }

    /// Create a new property with min/max bounds.
    pub fn with_bounds(
        name: impl Into<String>,
        value: PropertyValue,
        min: f32,
        max: f32,
    ) -> Self {
        Self {
            name: name.into(),
            value,
            min: Some(min),
            max: Some(max),
            read_only: false,
        }
    }

    /// Create a read-only property.
    pub fn read_only(name: impl Into<String>, value: PropertyValue) -> Self {
        Self {
            name: name.into(),
            value,
            min: None,
            max: None,
            read_only: true,
        }
    }
}

/// A collection of properties for a single entity.
#[derive(Debug, Clone)]
pub struct PropertySheet {
    /// The entity this sheet describes.
    pub entity_id: u64,
    /// Ordered list of property definitions.
    pub properties: Vec<PropertyDef>,
}

impl PropertySheet {
    /// Create a new empty property sheet for the given entity.
    pub fn new(entity_id: u64) -> Self {
        Self {
            entity_id,
            properties: Vec::new(),
        }
    }

    /// Add a property to the sheet.
    pub fn add_property(&mut self, name: impl Into<String>, value: PropertyValue) {
        self.properties.push(PropertyDef::new(name, value));
    }

    /// Add a full property definition (with bounds, read-only, etc.).
    pub fn add_property_def(&mut self, def: PropertyDef) {
        self.properties.push(def);
    }

    /// Get a property value by name.
    pub fn get_property(&self, name: &str) -> Option<&PropertyValue> {
        self.properties
            .iter()
            .find(|p| p.name == name)
            .map(|p| &p.value)
    }

    /// Set a property value by name.
    ///
    /// Returns `false` if the property was not found, the new value has a different
    /// type than the existing one, or the property is read-only.
    pub fn set_property(&mut self, name: &str, value: PropertyValue) -> bool {
        if let Some(prop) = self.properties.iter_mut().find(|p| p.name == name) {
            if prop.read_only {
                return false;
            }
            if !prop.value.same_type(&value) {
                return false;
            }
            prop.value = value;
            true
        } else {
            false
        }
    }

    /// Remove a property by name. Returns `true` if it was found and removed.
    pub fn remove_property(&mut self, name: &str) -> bool {
        if let Some(idx) = self.properties.iter().position(|p| p.name == name) {
            self.properties.remove(idx);
            true
        } else {
            false
        }
    }

    /// Return all property names in order.
    pub fn property_names(&self) -> Vec<&str> {
        self.properties.iter().map(|p| p.name.as_str()).collect()
    }
}

/// Build a standard transform property sheet for an entity.
///
/// Creates properties for position (x/y/z), rotation as Euler angles (x/y/z in degrees),
/// and uniform scale.
pub fn build_transform_properties(entity_id: u64, pos: Vec3, rot: Quat, scale: Vec3) -> PropertySheet {
    let euler = rot.to_euler(glam::EulerRot::XYZ);

    let mut sheet = PropertySheet::new(entity_id);

    sheet.add_property("position.x", PropertyValue::Float(pos.x));
    sheet.add_property("position.y", PropertyValue::Float(pos.y));
    sheet.add_property("position.z", PropertyValue::Float(pos.z));

    sheet.add_property(
        "rotation.x",
        PropertyValue::Float(euler.0.to_degrees()),
    );
    sheet.add_property(
        "rotation.y",
        PropertyValue::Float(euler.1.to_degrees()),
    );
    sheet.add_property(
        "rotation.z",
        PropertyValue::Float(euler.2.to_degrees()),
    );

    sheet.add_property_def(PropertyDef::with_bounds(
        "scale.x",
        PropertyValue::Float(scale.x),
        0.001,
        1000.0,
    ));
    sheet.add_property_def(PropertyDef::with_bounds(
        "scale.y",
        PropertyValue::Float(scale.y),
        0.001,
        1000.0,
    ));
    sheet.add_property_def(PropertyDef::with_bounds(
        "scale.z",
        PropertyValue::Float(scale.z),
        0.001,
        1000.0,
    ));

    sheet
}

/// Build a full object property sheet including transform and SDF source info.
///
/// For voxelized objects, shows `voxel_size` as a read-only float.
/// For analytical objects, shows "Analytical (infinite resolution)" as a read-only string.
pub fn build_object_properties(
    entity_id: u64,
    pos: Vec3,
    rot: Quat,
    scale: Vec3,
    sdf_source: &SdfSource,
) -> PropertySheet {
    let mut sheet = build_transform_properties(entity_id, pos, rot, scale);

    match sdf_source {
        SdfSource::Analytical { .. } => {
            sheet.add_property_def(PropertyDef::read_only(
                "sdf_source",
                PropertyValue::String("Analytical (infinite resolution)".to_string()),
            ));
        }
        SdfSource::Voxelized { voxel_size, .. } => {
            sheet.add_property_def(PropertyDef::read_only(
                "sdf_source",
                PropertyValue::String("Voxelized".to_string()),
            ));
            sheet.add_property_def(PropertyDef::read_only(
                "voxel_size",
                PropertyValue::Float(*voxel_size),
            ));
        }
        SdfSource::None => {
            sheet.add_property_def(PropertyDef::read_only(
                "sdf_source",
                PropertyValue::String("None (group node)".to_string()),
            ));
        }
    }

    sheet
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_2;

    #[test]
    fn test_add_and_get_property() {
        let mut sheet = PropertySheet::new(1);
        sheet.add_property("health", PropertyValue::Float(100.0));
        assert_eq!(
            sheet.get_property("health"),
            Some(&PropertyValue::Float(100.0))
        );
    }

    #[test]
    fn test_get_nonexistent() {
        let sheet = PropertySheet::new(1);
        assert!(sheet.get_property("nope").is_none());
    }

    #[test]
    fn test_set_property() {
        let mut sheet = PropertySheet::new(1);
        sheet.add_property("speed", PropertyValue::Float(5.0));
        assert!(sheet.set_property("speed", PropertyValue::Float(10.0)));
        assert_eq!(
            sheet.get_property("speed"),
            Some(&PropertyValue::Float(10.0))
        );
    }

    #[test]
    fn test_set_wrong_type_returns_false() {
        let mut sheet = PropertySheet::new(1);
        sheet.add_property("name", PropertyValue::String("hello".into()));
        assert!(!sheet.set_property("name", PropertyValue::Float(1.0)));
        // Original value preserved
        assert_eq!(
            sheet.get_property("name"),
            Some(&PropertyValue::String("hello".into()))
        );
    }

    #[test]
    fn test_set_nonexistent_returns_false() {
        let mut sheet = PropertySheet::new(1);
        assert!(!sheet.set_property("ghost", PropertyValue::Bool(true)));
    }

    #[test]
    fn test_read_only_enforcement() {
        let mut sheet = PropertySheet::new(1);
        sheet.add_property_def(PropertyDef::read_only(
            "entity_id",
            PropertyValue::U32(42),
        ));
        assert!(!sheet.set_property("entity_id", PropertyValue::U32(99)));
        assert_eq!(
            sheet.get_property("entity_id"),
            Some(&PropertyValue::U32(42))
        );
    }

    #[test]
    fn test_remove_property() {
        let mut sheet = PropertySheet::new(1);
        sheet.add_property("temp", PropertyValue::Bool(false));
        assert!(sheet.remove_property("temp"));
        assert!(sheet.get_property("temp").is_none());
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut sheet = PropertySheet::new(1);
        assert!(!sheet.remove_property("nope"));
    }

    #[test]
    fn test_property_names() {
        let mut sheet = PropertySheet::new(1);
        sheet.add_property("alpha", PropertyValue::Float(1.0));
        sheet.add_property("beta", PropertyValue::Bool(true));
        sheet.add_property("gamma", PropertyValue::U32(3));
        assert_eq!(sheet.property_names(), vec!["alpha", "beta", "gamma"]);
    }

    #[test]
    fn test_build_transform_properties_identity() {
        let sheet = build_transform_properties(42, Vec3::ZERO, Quat::IDENTITY, Vec3::ONE);
        assert_eq!(sheet.entity_id, 42);

        assert_eq!(
            sheet.get_property("position.x"),
            Some(&PropertyValue::Float(0.0))
        );
        assert_eq!(
            sheet.get_property("position.y"),
            Some(&PropertyValue::Float(0.0))
        );
        assert_eq!(
            sheet.get_property("position.z"),
            Some(&PropertyValue::Float(0.0))
        );

        // Identity rotation should give ~0 degree Euler angles
        if let Some(PropertyValue::Float(rx)) = sheet.get_property("rotation.x") {
            assert!(rx.abs() < 0.01, "rotation.x should be ~0: {rx}");
        } else {
            panic!("rotation.x missing or wrong type");
        }

        assert_eq!(
            sheet.get_property("scale.x"),
            Some(&PropertyValue::Float(1.0))
        );
        assert_eq!(
            sheet.get_property("scale.y"),
            Some(&PropertyValue::Float(1.0))
        );
        assert_eq!(
            sheet.get_property("scale.z"),
            Some(&PropertyValue::Float(1.0))
        );
    }

    #[test]
    fn test_build_transform_properties_with_rotation() {
        let rot = Quat::from_rotation_y(FRAC_PI_2);
        let sheet = build_transform_properties(1, Vec3::new(1.0, 2.0, 3.0), rot, Vec3::splat(2.0));

        assert_eq!(
            sheet.get_property("position.x"),
            Some(&PropertyValue::Float(1.0))
        );
        assert_eq!(
            sheet.get_property("position.y"),
            Some(&PropertyValue::Float(2.0))
        );

        // rotation.y should be ~90 degrees
        if let Some(PropertyValue::Float(ry)) = sheet.get_property("rotation.y") {
            assert!(
                (ry - 90.0).abs() < 0.1,
                "rotation.y should be ~90 degrees: {ry}"
            );
        } else {
            panic!("rotation.y missing or wrong type");
        }

        assert_eq!(
            sheet.get_property("scale.x"),
            Some(&PropertyValue::Float(2.0))
        );
    }

    #[test]
    fn test_build_transform_has_nine_properties() {
        let sheet = build_transform_properties(1, Vec3::ZERO, Quat::IDENTITY, Vec3::ONE);
        assert_eq!(sheet.property_names().len(), 9);
        // pos x/y/z + rot x/y/z + scale x/y/z = 9
    }

    #[test]
    fn test_scale_has_bounds() {
        let sheet = build_transform_properties(1, Vec3::ZERO, Quat::IDENTITY, Vec3::ONE);
        let scale_prop = sheet.properties.iter().find(|p| p.name == "scale.x").unwrap();
        assert_eq!(scale_prop.min, Some(0.001));
        assert_eq!(scale_prop.max, Some(1000.0));
    }

    #[test]
    fn test_property_value_types() {
        // Verify all variants can be created and compared
        let vals = vec![
            PropertyValue::Float(1.0),
            PropertyValue::Vec3(Vec3::ONE),
            PropertyValue::Quat(Quat::IDENTITY),
            PropertyValue::Color(Vec3::new(1.0, 0.0, 0.0)),
            PropertyValue::Bool(true),
            PropertyValue::String("test".into()),
            PropertyValue::U32(42),
            PropertyValue::MaterialId(7),
        ];
        for (i, v) in vals.iter().enumerate() {
            assert!(v.same_type(v), "value should match its own type");
            // Each variant should differ from the next
            if i + 1 < vals.len() {
                assert!(
                    !v.same_type(&vals[i + 1]),
                    "different variants should not match: {} vs {}",
                    v.type_name(),
                    vals[i + 1].type_name()
                );
            }
        }
    }

    #[test]
    fn test_empty_sheet() {
        let sheet = PropertySheet::new(99);
        assert_eq!(sheet.entity_id, 99);
        assert!(sheet.properties.is_empty());
        assert!(sheet.property_names().is_empty());
    }

    #[test]
    fn test_build_object_properties_analytical() {
        use rkf_core::scene_node::SdfPrimitive;

        let sdf = SdfSource::Analytical {
            primitive: SdfPrimitive::Sphere { radius: 1.0 },
            material_id: 0,
        };
        let sheet = build_object_properties(1, Vec3::ZERO, Quat::IDENTITY, Vec3::ONE, &sdf);
        // 9 transform + sdf_source
        assert_eq!(sheet.properties.len(), 10);
        assert_eq!(
            sheet.get_property("sdf_source"),
            Some(&PropertyValue::String("Analytical (infinite resolution)".to_string())),
        );
        assert!(sheet.get_property("voxel_size").is_none());
    }

    #[test]
    fn test_build_object_properties_voxelized() {
        use rkf_core::scene_node::BrickMapHandle;

        let sdf = SdfSource::Voxelized {
            brick_map_handle: BrickMapHandle {
                offset: 0,
                dims: glam::UVec3::new(4, 4, 4),
            },
            voxel_size: 0.02,
            aabb: rkf_core::Aabb::new(Vec3::ZERO, Vec3::ONE),
        };
        let sheet = build_object_properties(1, Vec3::ZERO, Quat::IDENTITY, Vec3::ONE, &sdf);
        // 9 transform + sdf_source + voxel_size
        assert_eq!(sheet.properties.len(), 11);
        assert_eq!(
            sheet.get_property("sdf_source"),
            Some(&PropertyValue::String("Voxelized".to_string())),
        );
        assert_eq!(
            sheet.get_property("voxel_size"),
            Some(&PropertyValue::Float(0.02)),
        );
        // voxel_size should be read-only
        let vs = sheet.properties.iter().find(|p| p.name == "voxel_size").unwrap();
        assert!(vs.read_only);
    }

    #[test]
    fn test_build_object_properties_none() {
        let sdf = SdfSource::None;
        let sheet = build_object_properties(1, Vec3::ZERO, Quat::IDENTITY, Vec3::ONE, &sdf);
        assert_eq!(sheet.properties.len(), 10);
        assert_eq!(
            sheet.get_property("sdf_source"),
            Some(&PropertyValue::String("None (group node)".to_string())),
        );
    }
}
