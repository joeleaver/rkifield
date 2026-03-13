//! ECS component types for the RKIField scene graph.
//!
//! These components attach to hecs entities and drive the transform hierarchy,
//! rendering, lighting, and editor metadata systems.

use glam::{Mat4, Quat, Vec3};
use rkf_core::WorldPosition;
use serde::{Deserialize, Serialize};

/// Local transform relative to parent (or world if no parent).
///
/// Position uses [`WorldPosition`] for float-precision safety.
/// Scale is per-axis (non-uniform scale uses conservative `min(sx,sy,sz)` for SDF distances).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transform {
    /// World-space position (chunk + local).
    pub position: WorldPosition,
    /// Rotation quaternion.
    pub rotation: Quat,
    /// Per-axis scale factor (all components must be > 0).
    pub scale: Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            position: WorldPosition::default(),
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

/// Camera-relative transform matrix, recomputed each frame.
///
/// All GPU-visible transforms are camera-relative f32 to avoid precision loss.
/// The CPU computes this via f64 subtraction from [`WorldPosition`].
#[derive(Debug, Clone, Copy)]
pub struct WorldTransform {
    /// Camera-relative model matrix (translation + rotation + scale).
    pub matrix: Mat4,
}

impl Default for WorldTransform {
    fn default() -> Self {
        Self {
            matrix: Mat4::IDENTITY,
        }
    }
}

/// Parent entity reference for transform hierarchy.
///
/// If `bone_index` is `Some`, the child's transform is additionally
/// multiplied by the parent's bone matrix at that index.
#[derive(Debug, Clone)]
pub struct Parent {
    /// The parent entity.
    pub entity: hecs::Entity,
    /// Optional bone index for skeletal attachment.
    pub bone_index: Option<u32>,
}

/// Camera component for entity-based camera bookmarks.
///
/// Stores camera parameters on an ECS entity. The renderer's singleton
/// [`Camera`] is the "viewport camera" that actually renders — use
/// [`Renderer::snap_camera_to`] to copy an entity's camera state to
/// the viewport camera.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraComponent {
    /// Vertical field of view in degrees.
    pub fov_degrees: f32,
    /// Near clip distance.
    pub near: f32,
    /// Far clip distance.
    pub far: f32,
    /// Whether this is the active (primary) camera.
    pub active: bool,
    /// Display name for this camera.
    pub label: String,
    /// Yaw in degrees.
    pub yaw: f32,
    /// Pitch in degrees.
    pub pitch: f32,
}

impl Default for CameraComponent {
    fn default() -> Self {
        Self {
            fov_degrees: 60.0,
            near: 0.1,
            far: 1000.0,
            active: false,
            label: String::new(),
            yaw: 0.0,
            pitch: 0.0,
        }
    }
}

/// Fog volume component (links to volumetric fog system).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FogVolumeComponent {
    /// Density scale multiplier.
    pub density: f32,
    /// Scattering color (linear RGB).
    pub color: [f32; 3],
    /// Phase function asymmetry (HG g parameter).
    pub phase_g: f32,
    /// Half-extents of the fog volume AABB (local space).
    pub half_extents: Vec3,
}

impl Default for FogVolumeComponent {
    fn default() -> Self {
        Self {
            density: 0.3,
            color: [0.8, 0.85, 0.9],
            phase_g: 0.3,
            half_extents: Vec3::new(5.0, 5.0, 5.0),
        }
    }
}

/// Editor-only metadata (not used at runtime).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EditorMetadata {
    /// Human-readable entity name.
    pub name: String,
    /// Tags for filtering/grouping.
    pub tags: Vec<String>,
    /// Whether this entity is locked in the editor.
    pub locked: bool,
}

impl Default for EditorMetadata {
    fn default() -> Self {
        Self {
            name: String::from("Entity"),
            tags: Vec::new(),
            locked: false,
        }
    }
}

/// SDF blend tree — the shape definition for an SDF object entity.
///
/// Wraps a [`SceneNode`] tree that defines the object's local-space SDF.
/// Each node carries geometry (analytical primitives or voxelized brick data),
/// a local transform, and a blend mode for sibling combination.
///
/// # Serialization
///
/// Voxelized nodes reference runtime [`BrickMapHandle`]s that are not
/// serializable. For persistence, use [`SdfTree::asset_path`] to store
/// the `.rkf` asset path — the streaming system resolves handles on load.
#[derive(Debug, Clone)]
pub struct SdfTree {
    /// Root of the SDF blend tree.
    pub root: rkf_core::scene_node::SceneNode,
    /// Optional path to the `.rkf` asset file (for serialization).
    /// Runtime handles are resolved by the streaming system after load.
    pub asset_path: Option<String>,
    /// Local-space bounding box of the SDF. Recomputed after sculpt/import.
    pub aabb: rkf_core::aabb::Aabb,
}

impl Default for SdfTree {
    fn default() -> Self {
        Self {
            root: rkf_core::scene_node::SceneNode::new("root"),
            asset_path: None,
            aabb: rkf_core::aabb::Aabb::new(glam::Vec3::ZERO, glam::Vec3::ZERO),
        }
    }
}

/// Proxy type for SdfTree serialization — only persists asset_path + aabb.
/// SceneNode contains non-serializable runtime handles (BrickMapHandle),
/// so root is reconstructed as default on deserialize.
#[derive(serde::Serialize, serde::Deserialize)]
struct SdfTreeProxy {
    asset_path: Option<String>,
    aabb: rkf_core::aabb::Aabb,
}

impl serde::Serialize for SdfTree {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let proxy = SdfTreeProxy {
            asset_path: self.asset_path.clone(),
            aabb: self.aabb,
        };
        proxy.serialize(serializer)
    }
}

impl<'de> serde::Deserialize<'de> for SdfTree {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let proxy = SdfTreeProxy::deserialize(deserializer)?;
        Ok(SdfTree {
            root: rkf_core::scene_node::SceneNode::new("root"),
            asset_path: proxy.asset_path,
            aabb: proxy.aabb,
        })
    }
}

// ─── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transform_default() {
        let t = Transform::default();
        assert_eq!(t.position, WorldPosition::default());
        assert_eq!(t.rotation, Quat::IDENTITY);
        assert_eq!(t.scale, Vec3::ONE);
    }

    #[test]
    fn world_transform_default() {
        let wt = WorldTransform::default();
        assert_eq!(wt.matrix, Mat4::IDENTITY);
    }

    #[test]
    fn camera_component_default() {
        let c = CameraComponent::default();
        assert!((c.fov_degrees - 60.0).abs() < 1e-6);
        assert!((c.near - 0.1).abs() < 1e-6);
        assert!((c.far - 1000.0).abs() < 1e-6);
        assert!(!c.active);
        assert!(c.label.is_empty());
    }

    #[test]
    fn camera_component_fields_roundtrip() {
        let c = CameraComponent {
            fov_degrees: 90.0,
            near: 0.5,
            far: 500.0,
            active: true,
            label: "Main".to_string(),
            yaw: 45.0,
            pitch: -15.0,
        };
        assert!((c.fov_degrees - 90.0).abs() < 1e-6);
        assert!((c.yaw - 45.0).abs() < 1e-6);
        assert!((c.pitch - -15.0).abs() < 1e-6);
        assert_eq!(c.label, "Main");
    }

    #[test]
    fn camera_component_with_label() {
        let c = CameraComponent {
            label: "Cinematic".to_string(),
            ..Default::default()
        };
        assert_eq!(c.label, "Cinematic");
    }

    #[test]
    fn fog_volume_default() {
        let f = FogVolumeComponent::default();
        assert!(f.density > 0.0);
        assert!(f.phase_g >= 0.0 && f.phase_g <= 1.0);
        assert!(f.half_extents.x > 0.0);
    }

    #[test]
    fn editor_metadata_default() {
        let m = EditorMetadata::default();
        assert_eq!(m.name, "Entity");
        assert!(m.tags.is_empty());
        assert!(!m.locked);
    }

    #[test]
    fn sdf_tree_default() {
        let tree = SdfTree::default();
        assert_eq!(tree.root.name, "root");
        assert!(tree.asset_path.is_none());
        assert!(tree.root.children.is_empty());
    }

    #[test]
    fn sdf_tree_with_asset_path() {
        let tree = SdfTree {
            root: rkf_core::scene_node::SceneNode::new("guard"),
            asset_path: Some("assets/guard.rkf".to_string()),
            aabb: rkf_core::aabb::Aabb::new(glam::Vec3::ZERO, glam::Vec3::ZERO),
        };
        assert_eq!(tree.asset_path.as_deref(), Some("assets/guard.rkf"));
        assert_eq!(tree.root.name, "guard");
    }

    #[test]
    fn sdf_tree_with_analytical_node() {
        use rkf_core::scene_node::{SceneNode, SdfSource, SdfPrimitive, BlendMode};

        let mut root = SceneNode::new("sphere_group");
        let mut child = SceneNode::new("sphere");
        child.sdf_source = SdfSource::Analytical {
            primitive: SdfPrimitive::Sphere { radius: 1.0 },
            material_id: 0,
        };
        child.blend_mode = BlendMode::SmoothUnion(0.2);
        root.children.push(child);

        let tree = SdfTree { root, asset_path: None, aabb: rkf_core::aabb::Aabb::new(glam::Vec3::ZERO, glam::Vec3::ZERO) };
        assert_eq!(tree.root.children.len(), 1);
        assert_eq!(tree.root.children[0].name, "sphere");
    }

    #[test]
    fn sdf_tree_clone() {
        let tree = SdfTree {
            root: rkf_core::scene_node::SceneNode::new("test"),
            asset_path: Some("test.rkf".to_string()),
            aabb: rkf_core::aabb::Aabb::new(glam::Vec3::ZERO, glam::Vec3::ZERO),
        };
        let cloned = tree.clone();
        assert_eq!(cloned.root.name, "test");
        assert_eq!(cloned.asset_path, tree.asset_path);
    }
}
