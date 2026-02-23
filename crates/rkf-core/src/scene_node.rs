//! Scene hierarchy node — the core v2 data model.
//!
//! A [`SceneNode`] is a node in an object's SDF tree. Each node carries a
//! local [`Transform`], an [`SdfSource`] (geometry), a [`BlendMode`]
//! (combination rule), and zero or more children.
//!
//! # Design
//!
//! - Nodes form a tree hierarchy (no cross-references)
//! - Child transforms are relative to parent, not world space
//! - Blending is scoped to the tree — nodes in different root objects never blend
//! - Uniform scale only — non-uniform breaks SDF distances

use glam::{Mat4, Quat, Vec3};

use crate::aabb::Aabb;

/// Local transform: position, rotation, uniform scale.
///
/// Non-uniform scale is forbidden because it distorts SDF distances
/// differently along each axis, breaking ray marching.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform {
    /// Offset from parent, in parent's local space.
    pub position: Vec3,
    /// Rotation relative to parent.
    pub rotation: Quat,
    /// Uniform scale factor. Must be positive.
    pub scale: f32,
}

impl Transform {
    /// Create a new transform.
    #[inline]
    pub fn new(position: Vec3, rotation: Quat, scale: f32) -> Self {
        Self {
            position,
            rotation,
            scale,
        }
    }

    /// Convert to a 4×4 matrix (scale × rotation × translation).
    #[inline]
    pub fn to_matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(
            Vec3::splat(self.scale),
            self.rotation,
            self.position,
        )
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: 1.0,
        }
    }
}

/// How an SDF node contributes geometry.
#[derive(Debug, Clone)]
pub enum SdfSource {
    /// Pure transform node — no geometry. Used for grouping, skeleton joints,
    /// or hierarchy anchors. Zero memory cost.
    None,
    /// Analytical SDF primitive — evaluated as math during ray marching.
    /// Zero memory cost, infinite resolution.
    Analytical {
        /// The mathematical shape definition.
        primitive: SdfPrimitive,
        /// Material table index.
        material_id: u16,
    },
    /// Voxelized SDF — local-space brick data from imported mesh or sculpting.
    Voxelized {
        /// Handle to this node's brick map in the allocator.
        brick_map_handle: BrickMapHandle,
        /// World-space size of one voxel (e.g. 0.005, 0.02, 0.08).
        voxel_size: f32,
        /// Local-space bounding box.
        aabb: Aabb,
    },
}

impl Default for SdfSource {
    fn default() -> Self {
        Self::None
    }
}

/// Placeholder handle for per-object brick map storage.
///
/// In v2, each voxelized node owns a compact brick map. This handle
/// references its allocation in the `BrickMapAllocator` (Phase 2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BrickMapHandle {
    /// Offset into the packed brick map buffer.
    pub offset: u32,
    /// Dimensions of the 3D brick map grid.
    pub dims: glam::UVec3,
}

/// How this node's SDF combines with its siblings.
///
/// Blending is scoped to the tree — nodes in different root objects never blend.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BlendMode {
    /// Smooth-min blend with radius parameter. Creates organic joins.
    /// This is the default mode.
    SmoothUnion(f32),
    /// Hard union (min). Sharp intersection lines.
    Union,
    /// Removes this node's volume from the combined sibling field.
    Subtract,
    /// Only overlapping volume survives.
    Intersect,
}

impl Default for BlendMode {
    fn default() -> Self {
        Self::SmoothUnion(0.1)
    }
}

/// Analytical SDF primitive — evaluated as math, zero memory cost.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SdfPrimitive {
    /// Sphere centered at local origin.
    Sphere {
        /// Radius in metres.
        radius: f32,
    },
    /// Axis-aligned box centered at local origin.
    Box {
        /// Half-width, half-height, half-depth from center.
        half_extents: Vec3,
    },
    /// Capsule along the local Y axis.
    Capsule {
        /// Radius of the rounded caps and cylinder.
        radius: f32,
        /// Half-height of the cylindrical section.
        half_height: f32,
    },
    /// Torus in the XZ plane centered at local origin.
    Torus {
        /// Distance from center to middle of tube.
        major_radius: f32,
        /// Radius of the tube itself.
        minor_radius: f32,
    },
    /// Cylinder along the local Y axis.
    Cylinder {
        /// Radius of the circular cross-section.
        radius: f32,
        /// Half-height of the cylinder.
        half_height: f32,
    },
    /// Infinite plane.
    Plane {
        /// Normalized surface normal.
        normal: Vec3,
        /// Signed distance from origin along normal.
        distance: f32,
    },
}

/// Metadata for editor state (visibility, lock, selection).
///
/// These fields do not affect runtime rendering — they are editor-only state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeMetadata {
    /// Whether this node and all descendants are visible.
    pub visible: bool,
    /// Lock node from editing.
    pub locked: bool,
    /// Editor selection state.
    pub selected: bool,
    /// UI tree expansion state.
    pub expand_in_tree: bool,
}

impl Default for NodeMetadata {
    fn default() -> Self {
        Self {
            visible: true,
            locked: false,
            selected: false,
            expand_in_tree: true,
        }
    }
}

/// A node in an object's SDF tree.
///
/// Forms the core of the v2 scene hierarchy. Each node carries geometry
/// (via [`SdfSource`]), a local transform, a blend mode for sibling
/// combination, and zero or more children.
#[derive(Debug, Clone)]
pub struct SceneNode {
    /// Human-readable name.
    pub name: String,
    /// Local transform relative to parent.
    pub local_transform: Transform,
    /// How this node contributes geometry.
    pub sdf_source: SdfSource,
    /// How this node combines with siblings.
    pub blend_mode: BlendMode,
    /// Child nodes.
    pub children: Vec<SceneNode>,
    /// Editor metadata (visibility, lock, selection).
    pub metadata: NodeMetadata,
}

impl SceneNode {
    /// Create a new node with default transform, no geometry, and no children.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            local_transform: Transform::default(),
            sdf_source: SdfSource::None,
            blend_mode: BlendMode::default(),
            children: Vec::new(),
            metadata: NodeMetadata::default(),
        }
    }

    /// Create an analytical primitive node.
    pub fn analytical(
        name: impl Into<String>,
        primitive: SdfPrimitive,
        material_id: u16,
    ) -> Self {
        Self {
            name: name.into(),
            local_transform: Transform::default(),
            sdf_source: SdfSource::Analytical {
                primitive,
                material_id,
            },
            blend_mode: BlendMode::default(),
            children: Vec::new(),
            metadata: NodeMetadata::default(),
        }
    }

    /// Add a child node, returning `&mut Self` for chaining.
    pub fn add_child(&mut self, child: SceneNode) -> &mut Self {
        self.children.push(child);
        self
    }

    /// Set the local transform, returning `Self` for builder-style construction.
    pub fn with_transform(mut self, transform: Transform) -> Self {
        self.local_transform = transform;
        self
    }

    /// Set the blend mode, returning `Self` for builder-style construction.
    pub fn with_blend_mode(mut self, mode: BlendMode) -> Self {
        self.blend_mode = mode;
        self
    }

    /// Total number of nodes in this subtree (including self).
    pub fn node_count(&self) -> usize {
        1 + self.children.iter().map(|c| c.node_count()).sum::<usize>()
    }

    /// Find a node by name (depth-first search).
    pub fn find_by_name(&self, name: &str) -> Option<&SceneNode> {
        if self.name == name {
            return Some(self);
        }
        for child in &self.children {
            if let Some(found) = child.find_by_name(name) {
                return Some(found);
            }
        }
        None
    }

    /// Find a node by name (mutable, depth-first search).
    pub fn find_by_name_mut(&mut self, name: &str) -> Option<&mut SceneNode> {
        if self.name == name {
            return Some(self);
        }
        for child in &mut self.children {
            if let Some(found) = child.find_by_name_mut(name) {
                return Some(found);
            }
        }
        None
    }
}

impl std::fmt::Display for SceneNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SceneNode(\"{}\"", self.name)?;
        match &self.sdf_source {
            SdfSource::None => write!(f, ", None")?,
            SdfSource::Analytical { primitive, .. } => write!(f, ", {:?}", primitive)?,
            SdfSource::Voxelized { voxel_size, .. } => write!(f, ", Voxelized({voxel_size})")?,
        }
        if !self.children.is_empty() {
            write!(f, ", {} children", self.children.len())?;
        }
        write!(f, ")")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_2;

    #[test]
    fn transform_default_is_identity() {
        let t = Transform::default();
        assert_eq!(t.position, Vec3::ZERO);
        assert_eq!(t.rotation, Quat::IDENTITY);
        assert_eq!(t.scale, 1.0);
    }

    #[test]
    fn transform_to_matrix_identity() {
        let t = Transform::default();
        let m = t.to_matrix();
        let diff = (m - Mat4::IDENTITY).abs_diff_eq(Mat4::ZERO, 1e-6);
        assert!(diff, "identity transform should produce identity matrix");
    }

    #[test]
    fn transform_to_matrix_translation() {
        let t = Transform::new(Vec3::new(1.0, 2.0, 3.0), Quat::IDENTITY, 1.0);
        let m = t.to_matrix();
        let p = m.transform_point3(Vec3::ZERO);
        assert!((p - Vec3::new(1.0, 2.0, 3.0)).length() < 1e-5);
    }

    #[test]
    fn transform_to_matrix_scale() {
        let t = Transform::new(Vec3::ZERO, Quat::IDENTITY, 2.0);
        let m = t.to_matrix();
        let p = m.transform_point3(Vec3::ONE);
        assert!((p - Vec3::splat(2.0)).length() < 1e-5);
    }

    #[test]
    fn transform_to_matrix_rotation() {
        let t = Transform::new(Vec3::ZERO, Quat::from_rotation_z(FRAC_PI_2), 1.0);
        let m = t.to_matrix();
        let p = m.transform_point3(Vec3::X);
        // 90° Z rotation: X -> Y
        assert!((p - Vec3::Y).length() < 1e-4);
    }

    #[test]
    fn scene_node_new_defaults() {
        let node = SceneNode::new("test");
        assert_eq!(node.name, "test");
        assert!(matches!(node.sdf_source, SdfSource::None));
        assert!(matches!(node.blend_mode, BlendMode::SmoothUnion(_)));
        assert!(node.children.is_empty());
        assert!(node.metadata.visible);
        assert!(!node.metadata.locked);
    }

    #[test]
    fn scene_node_analytical() {
        let node = SceneNode::analytical(
            "sphere",
            SdfPrimitive::Sphere { radius: 0.5 },
            1,
        );
        assert_eq!(node.name, "sphere");
        match &node.sdf_source {
            SdfSource::Analytical {
                primitive,
                material_id,
            } => {
                assert!(matches!(primitive, SdfPrimitive::Sphere { radius } if (*radius - 0.5).abs() < 1e-6));
                assert_eq!(*material_id, 1);
            }
            _ => panic!("expected Analytical"),
        }
    }

    #[test]
    fn scene_node_builder_pattern() {
        let node = SceneNode::analytical(
            "box",
            SdfPrimitive::Box {
                half_extents: Vec3::ONE,
            },
            2,
        )
        .with_transform(Transform::new(Vec3::new(1.0, 0.0, 0.0), Quat::IDENTITY, 0.5))
        .with_blend_mode(BlendMode::Subtract);

        assert_eq!(node.local_transform.position, Vec3::new(1.0, 0.0, 0.0));
        assert_eq!(node.local_transform.scale, 0.5);
        assert!(matches!(node.blend_mode, BlendMode::Subtract));
    }

    #[test]
    fn scene_node_tree_building() {
        let mut root = SceneNode::new("root");
        let child_a = SceneNode::analytical(
            "arm_left",
            SdfPrimitive::Capsule {
                radius: 0.1,
                half_height: 0.3,
            },
            1,
        );
        let child_b = SceneNode::analytical(
            "arm_right",
            SdfPrimitive::Capsule {
                radius: 0.1,
                half_height: 0.3,
            },
            1,
        );
        root.add_child(child_a);
        root.add_child(child_b);

        assert_eq!(root.children.len(), 2);
        assert_eq!(root.node_count(), 3);
    }

    #[test]
    fn scene_node_deep_tree() {
        let mut root = SceneNode::new("hips");
        let mut spine = SceneNode::new("spine");
        let mut chest = SceneNode::new("chest");
        let head = SceneNode::analytical(
            "head",
            SdfPrimitive::Sphere { radius: 0.15 },
            1,
        );
        chest.add_child(head);
        spine.add_child(chest);
        root.add_child(spine);

        assert_eq!(root.node_count(), 4);
    }

    #[test]
    fn scene_node_find_by_name() {
        let mut root = SceneNode::new("root");
        let mut child = SceneNode::new("child");
        let grandchild = SceneNode::new("grandchild");
        child.add_child(grandchild);
        root.add_child(child);

        assert!(root.find_by_name("root").is_some());
        assert!(root.find_by_name("child").is_some());
        assert!(root.find_by_name("grandchild").is_some());
        assert!(root.find_by_name("missing").is_none());
    }

    #[test]
    fn scene_node_find_by_name_mut() {
        let mut root = SceneNode::new("root");
        let child = SceneNode::new("child");
        root.add_child(child);

        if let Some(node) = root.find_by_name_mut("child") {
            node.metadata.locked = true;
        }
        assert!(root.children[0].metadata.locked);
    }

    #[test]
    fn scene_node_display() {
        let node = SceneNode::analytical(
            "sphere",
            SdfPrimitive::Sphere { radius: 1.0 },
            0,
        );
        let s = format!("{node}");
        assert!(s.contains("sphere"));
        assert!(s.contains("Sphere"));
    }

    #[test]
    fn blend_mode_default_is_smooth_union() {
        let mode = BlendMode::default();
        assert!(matches!(mode, BlendMode::SmoothUnion(r) if r > 0.0));
    }

    #[test]
    fn node_metadata_default_is_visible_unlocked() {
        let m = NodeMetadata::default();
        assert!(m.visible);
        assert!(!m.locked);
        assert!(!m.selected);
        assert!(m.expand_in_tree);
    }

    #[test]
    fn all_sdf_primitives_constructable() {
        let _s = SdfPrimitive::Sphere { radius: 1.0 };
        let _b = SdfPrimitive::Box {
            half_extents: Vec3::ONE,
        };
        let _c = SdfPrimitive::Capsule {
            radius: 0.1,
            half_height: 0.5,
        };
        let _t = SdfPrimitive::Torus {
            major_radius: 1.0,
            minor_radius: 0.2,
        };
        let _cy = SdfPrimitive::Cylinder {
            radius: 0.5,
            half_height: 1.0,
        };
        let _p = SdfPrimitive::Plane {
            normal: Vec3::Y,
            distance: 0.0,
        };
    }

    #[test]
    fn all_blend_modes_constructable() {
        let _su = BlendMode::SmoothUnion(0.1);
        let _u = BlendMode::Union;
        let _s = BlendMode::Subtract;
        let _i = BlendMode::Intersect;
    }

    #[test]
    fn all_sdf_sources_constructable() {
        let _none = SdfSource::None;
        let _analytical = SdfSource::Analytical {
            primitive: SdfPrimitive::Sphere { radius: 1.0 },
            material_id: 0,
        };
        let _voxelized = SdfSource::Voxelized {
            brick_map_handle: BrickMapHandle {
                offset: 0,
                dims: glam::UVec3::new(4, 4, 4),
            },
            voxel_size: 0.02,
            aabb: Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0)),
        };
    }

    #[test]
    fn transform_combined_trs() {
        // Translation + rotation + scale combined
        let t = Transform::new(
            Vec3::new(5.0, 0.0, 0.0),
            Quat::from_rotation_z(FRAC_PI_2),
            2.0,
        );
        let m = t.to_matrix();
        // Point at (1, 0, 0) should be: scale(2) → (2,0,0), rotate 90°Z → (0,2,0), translate → (5,2,0)
        let p = m.transform_point3(Vec3::X);
        assert!((p.x - 5.0).abs() < 1e-4, "x: {}", p.x);
        assert!((p.y - 2.0).abs() < 1e-4, "y: {}", p.y);
        assert!(p.z.abs() < 1e-4, "z: {}", p.z);
    }
}
