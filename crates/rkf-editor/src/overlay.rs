//! Gizmo overlay line rendering data model.
//!
//! Provides `LineBatch` for accumulating wireframe lines used by editor overlays:
//! grid, selection outlines, light shapes, and volume wireframes. This is pure
//! geometry generation — no GPU rendering.

#![allow(dead_code)]

use glam::Vec3;
use std::f32::consts::PI;

/// A line segment with color and screen-space width.
#[derive(Debug, Clone, Copy)]
pub struct LineSeg {
    /// World-space start position.
    pub start: Vec3,
    /// World-space end position.
    pub end: Vec3,
    /// RGBA color.
    pub color: [f32; 4],
    /// Width in pixels (screen space).
    pub width: f32,
}

/// Accumulates line segments for overlay rendering.
///
/// Each segment is rendered as a camera-facing billboard quad with the
/// specified pixel width.
#[derive(Debug, Clone)]
pub struct LineBatch {
    pub segments: Vec<LineSeg>,
}

impl Default for LineBatch {
    fn default() -> Self {
        Self::new()
    }
}

impl LineBatch {
    /// Create an empty line batch.
    pub fn new() -> Self {
        Self {
            segments: Vec::new(),
        }
    }

    /// Add a single line segment with default width (1.5 px).
    pub fn add_line(&mut self, start: Vec3, end: Vec3, color: [f32; 4]) {
        self.segments.push(LineSeg {
            start,
            end,
            color,
            width: 1.5,
        });
    }

    /// Add a line segment with explicit pixel width.
    pub fn add_thick_line(
        &mut self,
        start: Vec3,
        end: Vec3,
        color: [f32; 4],
        width: f32,
    ) {
        self.segments.push(LineSeg {
            start,
            end,
            color,
            width,
        });
    }

    /// Add a circle (ring of line segments) in the plane defined by `normal`.
    pub fn add_circle(
        &mut self,
        center: Vec3,
        normal: Vec3,
        radius: f32,
        color: [f32; 4],
        segments: u32,
    ) {
        self.add_thick_circle(center, normal, radius, color, segments, 1.5);
    }

    /// Add a circle with a specified line width.
    pub fn add_thick_circle(
        &mut self,
        center: Vec3,
        normal: Vec3,
        radius: f32,
        color: [f32; 4],
        segments: u32,
        width: f32,
    ) {
        let normal = normal.normalize();
        // Build an orthonormal basis in the circle plane.
        let tangent = if normal.dot(Vec3::Y).abs() < 0.99 {
            normal.cross(Vec3::Y).normalize()
        } else {
            normal.cross(Vec3::X).normalize()
        };
        let bitangent = normal.cross(tangent);

        let step = 2.0 * PI / segments as f32;
        for i in 0..segments {
            let a0 = step * i as f32;
            let a1 = step * ((i + 1) % segments) as f32;
            let p0 = center + (tangent * a0.cos() + bitangent * a0.sin()) * radius;
            let p1 = center + (tangent * a1.cos() + bitangent * a1.sin()) * radius;
            self.add_thick_line(p0, p1, color, width);
        }
    }

    /// Add an axis-aligned box wireframe (12 edges).
    pub fn add_box_wireframe(&mut self, min: Vec3, max: Vec3, color: [f32; 4]) {
        // 8 corners
        let corners = [
            Vec3::new(min.x, min.y, min.z),
            Vec3::new(max.x, min.y, min.z),
            Vec3::new(max.x, max.y, min.z),
            Vec3::new(min.x, max.y, min.z),
            Vec3::new(min.x, min.y, max.z),
            Vec3::new(max.x, min.y, max.z),
            Vec3::new(max.x, max.y, max.z),
            Vec3::new(min.x, max.y, max.z),
        ];

        // 12 edges: 4 bottom, 4 top, 4 vertical
        let edges: [(usize, usize); 12] = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0), // bottom face
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4), // top face
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7), // verticals
        ];

        for (a, b) in &edges {
            self.add_line(corners[*a], corners[*b], color);
        }
    }

    /// Add a sphere wireframe using latitude/longitude rings.
    pub fn add_sphere_wireframe(
        &mut self,
        center: Vec3,
        radius: f32,
        color: [f32; 4],
        rings: u32,
    ) {
        // Three orthogonal great circles plus additional latitude rings.
        self.add_circle(center, Vec3::Y, radius, color, rings);
        self.add_circle(center, Vec3::X, radius, color, rings);
        self.add_circle(center, Vec3::Z, radius, color, rings);
    }

    /// Add a cone wireframe.
    ///
    /// The cone apex is at `apex`, extends along `direction` for `height`,
    /// with half-angle `angle` (radians).
    pub fn add_cone_wireframe(
        &mut self,
        apex: Vec3,
        direction: Vec3,
        height: f32,
        angle: f32,
        color: [f32; 4],
        segments: u32,
    ) {
        let dir = direction.normalize();
        let base_center = apex + dir * height;
        let base_radius = height * angle.tan();

        // Build orthonormal basis perpendicular to direction.
        let tangent = if dir.dot(Vec3::Y).abs() < 0.99 {
            dir.cross(Vec3::Y).normalize()
        } else {
            dir.cross(Vec3::X).normalize()
        };
        let bitangent = dir.cross(tangent);

        // Base circle
        self.add_circle(base_center, dir, base_radius, color, segments);

        // Lines from apex to base circle at cardinal points
        let step = 2.0 * PI / 4.0;
        for i in 0..4 {
            let a = step * i as f32;
            let p = base_center + (tangent * a.cos() + bitangent * a.sin()) * base_radius;
            self.add_line(apex, p, color);
        }
    }

    /// Add a grid in the plane defined by `normal`, centered at `center`.
    ///
    /// `size` is the total extent (half in each direction), `divisions` is the
    /// number of cells along each axis.
    pub fn add_grid(
        &mut self,
        center: Vec3,
        normal: Vec3,
        size: f32,
        divisions: u32,
        color: [f32; 4],
    ) {
        let normal = normal.normalize();
        let tangent = if normal.dot(Vec3::Y).abs() < 0.99 {
            normal.cross(Vec3::Y).normalize()
        } else {
            normal.cross(Vec3::X).normalize()
        };
        let bitangent = normal.cross(tangent);

        let half = size / 2.0;
        let step = size / divisions as f32;

        // Lines along tangent direction
        for i in 0..=divisions {
            let offset = -half + step * i as f32;
            let start = center + bitangent * offset - tangent * half;
            let end = center + bitangent * offset + tangent * half;
            self.add_line(start, end, color);
        }

        // Lines along bitangent direction
        for i in 0..=divisions {
            let offset = -half + step * i as f32;
            let start = center + tangent * offset - bitangent * half;
            let end = center + tangent * offset + bitangent * half;
            self.add_line(start, end, color);
        }
    }

    /// Remove all segments.
    pub fn clear(&mut self) {
        self.segments.clear();
    }

    /// Number of logical vertices (2 per segment).
    pub fn vertex_count(&self) -> usize {
        self.segments.len() * 2
    }

    /// Whether the batch has no segments.
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }
}

/// Configuration for which overlays are visible.
#[derive(Debug, Clone)]
pub struct OverlayConfig {
    pub show_grid: bool,
    pub show_selection_outlines: bool,
    pub show_light_shapes: bool,
    pub show_volume_wireframes: bool,
    pub grid_size: f32,
    pub grid_divisions: u32,
}

impl Default for OverlayConfig {
    fn default() -> Self {
        Self {
            show_grid: true,
            show_selection_outlines: true,
            show_light_shapes: true,
            show_volume_wireframes: false,
            grid_size: 10.0,
            grid_divisions: 10,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const WHITE: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
    const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];

    #[test]
    fn test_add_line_vertex_count() {
        let mut batch = LineBatch::new();
        assert!(batch.is_empty());
        batch.add_line(Vec3::ZERO, Vec3::X, WHITE);
        assert_eq!(batch.vertex_count(), 2);
        batch.add_line(Vec3::ZERO, Vec3::Y, WHITE);
        assert_eq!(batch.vertex_count(), 4);
    }

    #[test]
    fn test_add_thick_line() {
        let mut batch = LineBatch::new();
        batch.add_thick_line(Vec3::ZERO, Vec3::X, RED, 5.0);
        assert_eq!(batch.segments.len(), 1);
        assert_eq!(batch.segments[0].width, 5.0);
    }

    #[test]
    fn test_add_circle_vertex_count() {
        let mut batch = LineBatch::new();
        let segments = 16u32;
        batch.add_circle(Vec3::ZERO, Vec3::Y, 1.0, WHITE, segments);
        // Each segment produces 2 vertices (one line)
        assert_eq!(batch.vertex_count(), (segments * 2) as usize);
    }

    #[test]
    fn test_add_box_wireframe_vertex_count() {
        let mut batch = LineBatch::new();
        batch.add_box_wireframe(Vec3::ZERO, Vec3::ONE, RED);
        // 12 edges * 2 vertices each = 24
        assert_eq!(batch.vertex_count(), 24);
    }

    #[test]
    fn test_add_sphere_wireframe_vertex_count() {
        let mut batch = LineBatch::new();
        let rings = 16u32;
        batch.add_sphere_wireframe(Vec3::ZERO, 1.0, WHITE, rings);
        // 3 great circles, each with `rings` segments, each segment = 2 vertices
        assert_eq!(batch.vertex_count(), (3 * rings * 2) as usize);
    }

    #[test]
    fn test_add_cone_wireframe_vertex_count() {
        let mut batch = LineBatch::new();
        let segments = 12u32;
        batch.add_cone_wireframe(
            Vec3::ZERO,
            Vec3::Y,
            2.0,
            0.5, // ~28.6 degree half-angle
            WHITE,
            segments,
        );
        // Base circle: segments * 2 vertices + 4 apex-to-base lines: 4 * 2 vertices
        assert_eq!(batch.vertex_count(), (segments * 2 + 4 * 2) as usize);
    }

    #[test]
    fn test_add_grid_vertex_count() {
        let mut batch = LineBatch::new();
        let divisions = 4u32;
        batch.add_grid(Vec3::ZERO, Vec3::Y, 10.0, divisions, WHITE);
        // (divisions + 1) lines per axis direction, 2 directions, 2 vertices each
        let expected = 2 * (divisions + 1) * 2;
        assert_eq!(batch.vertex_count(), expected as usize);
    }

    #[test]
    fn test_clear() {
        let mut batch = LineBatch::new();
        batch.add_line(Vec3::ZERO, Vec3::ONE, WHITE);
        batch.add_box_wireframe(Vec3::ZERO, Vec3::ONE, RED);
        assert!(!batch.is_empty());
        batch.clear();
        assert!(batch.is_empty());
        assert_eq!(batch.vertex_count(), 0);
    }

    #[test]
    fn test_line_segment_data() {
        let mut batch = LineBatch::new();
        let start = Vec3::new(1.0, 2.0, 3.0);
        let end = Vec3::new(4.0, 5.0, 6.0);
        let color = [0.5, 0.6, 0.7, 0.8];
        batch.add_line(start, end, color);
        assert_eq!(batch.segments[0].start, start);
        assert_eq!(batch.segments[0].end, end);
        assert_eq!(batch.segments[0].color, color);
        assert_eq!(batch.segments[0].width, 1.5);
    }

    #[test]
    fn test_overlay_config_defaults() {
        let config = OverlayConfig::default();
        assert!(config.show_grid);
        assert!(config.show_selection_outlines);
        assert!(config.show_light_shapes);
        assert!(!config.show_volume_wireframes);
        assert_eq!(config.grid_size, 10.0);
        assert_eq!(config.grid_divisions, 10);
    }

    #[test]
    fn test_new_batch_is_empty() {
        let batch = LineBatch::new();
        assert!(batch.is_empty());
        assert_eq!(batch.vertex_count(), 0);
    }

    #[test]
    fn test_default_batch() {
        let batch = LineBatch::default();
        assert!(batch.is_empty());
    }
}
