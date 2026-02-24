//! Wireframe line rendering pass for selection highlights and gizmos.
//!
//! Uses wgpu `LineList` topology to draw 3D line segments onto the swapchain.
//! The pass blends additively with `LoadOp::Load` so engine output is preserved.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Quat, Vec3};
use wgpu::util::DeviceExt;

/// A single line vertex (position in camera-relative space + RGBA color).
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct LineVertex {
    pub position: [f32; 3],
    pub color: [f32; 4],
}

const LINE_WGSL: &str = "
struct Uniforms { view_proj: mat4x4<f32> }
@group(0) @binding(0) var<uniform> u: Uniforms;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec4<f32>,
}

@vertex fn vs(@location(0) position: vec3<f32>, @location(1) color: vec4<f32>) -> VsOut {
    var o: VsOut;
    o.pos = u.view_proj * vec4(position, 1.0);
    o.color = color;
    return o;
}

@fragment fn fs(in: VsOut) -> @location(0) vec4<f32> {
    return in.color;
}
";

/// GPU line rendering pass for editor wireframes.
pub struct WireframePass {
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
}

impl WireframePass {
    /// Create the wireframe rendering pipeline.
    pub fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("wireframe"),
            source: wgpu::ShaderSource::Wgsl(LINE_WGSL.into()),
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("wireframe bind group layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wireframe uniforms"),
            size: 64, // mat4x4<f32>
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("wireframe bind group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("wireframe pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<LineVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // position: vec3<f32>
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
                // color: vec4<f32>
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 12,
                    shader_location: 1,
                },
            ],
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("wireframe"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[vertex_layout],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview: None,
            cache: None,
        });

        Self {
            pipeline,
            uniform_buffer,
            bind_group_layout,
            bind_group,
        }
    }

    /// Draw line segments onto the target view.
    ///
    /// `vp_matrix` transforms from world space to clip space.
    /// `viewport` is `(x, y, width, height)` in physical pixels — restricts
    /// rendering to the engine viewport sub-region of the swapchain.
    /// `vertices` contains pairs of `LineVertex` (two per line segment).
    pub fn draw(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        vp_matrix: Mat4,
        viewport: (f32, f32, f32, f32),
        vertices: &[LineVertex],
    ) {
        if vertices.is_empty() {
            return;
        }

        // Upload VP matrix.
        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&vp_matrix.to_cols_array()),
        );

        // Create temporary vertex buffer.
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("wireframe vertices"),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("wireframe pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_viewport(viewport.0, viewport.1, viewport.2, viewport.3, 0.0, 1.0);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, vertex_buffer.slice(..));
        pass.draw(0..vertices.len() as u32, 0..1);
    }
}

/// Build wireframe vertices for an oriented bounding box (12 edges = 24 vertices).
///
/// `local_min`/`local_max` define the box in object-local space.
/// `position`, `rotation`, and `scale` transform corners into world space.
pub fn obb_wireframe(
    local_min: Vec3,
    local_max: Vec3,
    position: Vec3,
    rotation: Quat,
    scale: f32,
    color: [f32; 4],
) -> Vec<LineVertex> {
    // 8 local-space corners.
    let local_corners = [
        Vec3::new(local_min.x, local_min.y, local_min.z),
        Vec3::new(local_max.x, local_min.y, local_min.z),
        Vec3::new(local_max.x, local_max.y, local_min.z),
        Vec3::new(local_min.x, local_max.y, local_min.z),
        Vec3::new(local_min.x, local_min.y, local_max.z),
        Vec3::new(local_max.x, local_min.y, local_max.z),
        Vec3::new(local_max.x, local_max.y, local_max.z),
        Vec3::new(local_min.x, local_max.y, local_max.z),
    ];

    // Transform each corner: scale → rotate → translate.
    let corners: [Vec3; 8] = std::array::from_fn(|i| {
        position + rotation * (local_corners[i] * scale)
    });

    // 12 edges (pairs of corner indices).
    let edges: [(usize, usize); 12] = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ];

    let mut verts = Vec::with_capacity(24);
    for (a, b) in edges {
        verts.push(LineVertex {
            position: corners[a].to_array(),
            color,
        });
        verts.push(LineVertex {
            position: corners[b].to_array(),
            color,
        });
    }
    verts
}

/// Build wireframe vertices for an axis-aligned bounding box (12 edges = 24 vertices).
///
/// Positions are in world space — the VP matrix handles the camera transform.
/// Use this for voxelized objects and hierarchy roots where rotation isn't applicable.
pub fn aabb_wireframe(min: Vec3, max: Vec3, color: [f32; 4]) -> Vec<LineVertex> {
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

    let edges: [(usize, usize); 12] = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ];

    let mut verts = Vec::with_capacity(24);
    for (a, b) in edges {
        verts.push(LineVertex { position: corners[a].to_array(), color });
        verts.push(LineVertex { position: corners[b].to_array(), color });
    }
    verts
}

/// Build a small 3-axis crosshair (6 vertices = 3 line segments) centered at `center`.
///
/// Useful for debugging: visually marks an exact 3D position in world space.
pub fn crosshair(center: Vec3, size: f32, color: [f32; 4]) -> Vec<LineVertex> {
    let hs = size * 0.5;
    vec![
        LineVertex { position: (center - Vec3::X * hs).to_array(), color },
        LineVertex { position: (center + Vec3::X * hs).to_array(), color },
        LineVertex { position: (center - Vec3::Y * hs).to_array(), color },
        LineVertex { position: (center + Vec3::Y * hs).to_array(), color },
        LineVertex { position: (center - Vec3::Z * hs).to_array(), color },
        LineVertex { position: (center + Vec3::Z * hs).to_array(), color },
    ]
}

/// Build a circle of line segments in the plane with the given `normal`.
pub fn circle_wireframe(
    center: Vec3, normal: Vec3, radius: f32, color: [f32; 4], segments: u32,
) -> Vec<LineVertex> {
    let dir = normal.normalize();
    let tangent = if dir.dot(Vec3::Y).abs() < 0.99 {
        dir.cross(Vec3::Y).normalize()
    } else {
        dir.cross(Vec3::X).normalize()
    };
    let bitangent = dir.cross(tangent);

    let step = std::f32::consts::TAU / segments as f32;
    let mut verts = Vec::with_capacity(segments as usize * 2);
    for i in 0..segments {
        let a0 = step * i as f32;
        let a1 = step * ((i + 1) % segments) as f32;
        let p0 = center + (tangent * a0.cos() + bitangent * a0.sin()) * radius;
        let p1 = center + (tangent * a1.cos() + bitangent * a1.sin()) * radius;
        verts.push(LineVertex { position: p0.to_array(), color });
        verts.push(LineVertex { position: p1.to_array(), color });
    }
    verts
}

/// Build a point light gizmo: three orthogonal great circles showing range.
pub fn point_light_wireframe(center: Vec3, range: f32, color: [f32; 4]) -> Vec<LineVertex> {
    let segs = 32;
    let mut verts = circle_wireframe(center, Vec3::Y, range, color, segs);
    verts.extend(circle_wireframe(center, Vec3::X, range, color, segs));
    verts.extend(circle_wireframe(center, Vec3::Z, range, color, segs));
    // Small crosshair at center.
    verts.extend(crosshair(center, 0.2, color));
    verts
}

/// Build a spot light gizmo: cone showing outer angle + range, with a crosshair at apex.
pub fn spot_light_wireframe(
    apex: Vec3, direction: Vec3, range: f32, outer_angle: f32, color: [f32; 4],
) -> Vec<LineVertex> {
    let dir = direction.normalize();
    let base_center = apex + dir * range;
    let base_radius = range * outer_angle.tan();

    let tangent = if dir.dot(Vec3::Y).abs() < 0.99 {
        dir.cross(Vec3::Y).normalize()
    } else {
        dir.cross(Vec3::X).normalize()
    };
    let bitangent = dir.cross(tangent);

    // Base circle.
    let mut verts = circle_wireframe(base_center, dir, base_radius, color, 32);

    // 4 lines from apex to base.
    let step = std::f32::consts::TAU / 4.0;
    for i in 0..4 {
        let a = step * i as f32;
        let p = base_center + (tangent * a.cos() + bitangent * a.sin()) * base_radius;
        verts.push(LineVertex { position: apex.to_array(), color });
        verts.push(LineVertex { position: p.to_array(), color });
    }

    // Crosshair at apex.
    verts.extend(crosshair(apex, 0.2, color));
    verts
}

/// Build a directional light gizmo: arrow showing direction, placed at a reference position.
pub fn directional_light_wireframe(
    position: Vec3, direction: Vec3, color: [f32; 4],
) -> Vec<LineVertex> {
    let dir = direction.normalize();
    let arrow_len = 2.0;
    let head_len = 0.4;
    let head_radius = 0.15;
    let tip = position + dir * arrow_len;

    // Main shaft.
    let mut verts = vec![
        LineVertex { position: position.to_array(), color },
        LineVertex { position: tip.to_array(), color },
    ];

    // Arrowhead: 4 lines from tip back to a ring.
    let tangent = if dir.dot(Vec3::Y).abs() < 0.99 {
        dir.cross(Vec3::Y).normalize()
    } else {
        dir.cross(Vec3::X).normalize()
    };
    let bitangent = dir.cross(tangent);
    let base = tip - dir * head_len;

    let step = std::f32::consts::TAU / 4.0;
    for i in 0..4 {
        let a = step * i as f32;
        let p = base + (tangent * a.cos() + bitangent * a.sin()) * head_radius;
        verts.push(LineVertex { position: tip.to_array(), color });
        verts.push(LineVertex { position: p.to_array(), color });
    }

    // Sun disc at position.
    verts.extend(circle_wireframe(position, dir, 0.3, color, 16));
    // Cross through the disc.
    verts.extend(crosshair(position, 0.5, color));
    verts
}

// ── Transform gizmo wireframes ──────────────────────────────────────────────

/// Axis colors: R = X, G = Y, B = Z.
const GIZMO_X_COLOR: [f32; 4] = [1.0, 0.2, 0.2, 1.0];
const GIZMO_Y_COLOR: [f32; 4] = [0.2, 1.0, 0.2, 1.0];
const GIZMO_Z_COLOR: [f32; 4] = [0.3, 0.3, 1.0, 1.0];

/// Build a translate gizmo: 3 axis arrows from `center` with length `size`.
///
/// Each axis is a line + arrowhead in its respective color (R=X, G=Y, B=Z).
/// `size` should be proportional to camera distance for constant screen size.
pub fn translate_gizmo_wireframe(center: Vec3, size: f32) -> Vec<LineVertex> {
    let mut verts = Vec::new();
    let head_len = size * 0.2;
    let head_radius = size * 0.06;

    for (axis_dir, color) in [
        (Vec3::X, GIZMO_X_COLOR),
        (Vec3::Y, GIZMO_Y_COLOR),
        (Vec3::Z, GIZMO_Z_COLOR),
    ] {
        let tip = center + axis_dir * size;
        // Shaft.
        verts.push(LineVertex { position: center.to_array(), color });
        verts.push(LineVertex { position: tip.to_array(), color });

        // Arrowhead: 4 lines from tip to a base ring.
        let tangent = if axis_dir.dot(Vec3::Y).abs() < 0.99 {
            axis_dir.cross(Vec3::Y).normalize()
        } else {
            axis_dir.cross(Vec3::X).normalize()
        };
        let bitangent = axis_dir.cross(tangent);
        let base = tip - axis_dir * head_len;

        let step = std::f32::consts::TAU / 4.0;
        for i in 0..4 {
            let a = step * i as f32;
            let p = base + (tangent * a.cos() + bitangent * a.sin()) * head_radius;
            verts.push(LineVertex { position: tip.to_array(), color });
            verts.push(LineVertex { position: p.to_array(), color });
        }
    }
    verts
}

/// Build a rotate gizmo: 3 axis rings at `center` with radius `size`.
pub fn rotate_gizmo_wireframe(center: Vec3, size: f32) -> Vec<LineVertex> {
    let segs = 48;
    let mut verts = circle_wireframe(center, Vec3::X, size, GIZMO_X_COLOR, segs);
    verts.extend(circle_wireframe(center, Vec3::Y, size, GIZMO_Y_COLOR, segs));
    verts.extend(circle_wireframe(center, Vec3::Z, size, GIZMO_Z_COLOR, segs));
    verts
}

/// Build a scale gizmo: 3 axis lines with small cubes at the ends.
pub fn scale_gizmo_wireframe(center: Vec3, size: f32) -> Vec<LineVertex> {
    let cube_half = size * 0.06;
    let mut verts = Vec::new();

    for (axis_dir, color) in [
        (Vec3::X, GIZMO_X_COLOR),
        (Vec3::Y, GIZMO_Y_COLOR),
        (Vec3::Z, GIZMO_Z_COLOR),
    ] {
        let tip = center + axis_dir * size;
        // Shaft.
        verts.push(LineVertex { position: center.to_array(), color });
        verts.push(LineVertex { position: tip.to_array(), color });

        // Small cube at the tip.
        let min = tip - Vec3::splat(cube_half);
        let max = tip + Vec3::splat(cube_half);
        verts.extend(aabb_wireframe(min, max, color));
    }

    // Center cube for uniform scale.
    let cc = Vec3::splat(cube_half * 1.2);
    verts.extend(aabb_wireframe(center - cc, center + cc, [0.9, 0.9, 0.9, 1.0]));
    verts
}

/// Build a ground grid wireframe at Y=0 centered around the camera position.
///
/// Draws grid lines along X and Z axes with the given spacing. The grid
/// automatically follows the camera's XZ position (snapped to grid spacing).
pub fn ground_grid_wireframe(
    cam_pos: Vec3, extent: f32, spacing: f32, color: [f32; 4],
) -> Vec<LineVertex> {
    let half = extent * 0.5;
    // Snap center to grid.
    let cx = (cam_pos.x / spacing).round() * spacing;
    let cz = (cam_pos.z / spacing).round() * spacing;
    let count = (extent / spacing) as i32;

    let mut verts = Vec::with_capacity(count as usize * 4 + 4);
    for i in -count / 2..=count / 2 {
        let offset = i as f32 * spacing;
        // Line along X.
        verts.push(LineVertex { position: [cx - half, 0.0, cz + offset], color });
        verts.push(LineVertex { position: [cx + half, 0.0, cz + offset], color });
        // Line along Z.
        verts.push(LineVertex { position: [cx + offset, 0.0, cz - half], color });
        verts.push(LineVertex { position: [cx + offset, 0.0, cz + half], color });
    }
    verts
}

/// Build a wireframe sphere (3 orthogonal great circles) for brush preview.
///
/// `center` is the world-space hit point, `radius` is the brush radius.
/// `color` encodes the brush type (e.g., cyan for sculpt, magenta for paint).
pub fn sphere_wireframe(center: Vec3, radius: f32, color: [f32; 4]) -> Vec<LineVertex> {
    let segs = 32;
    let mut verts = circle_wireframe(center, Vec3::X, radius, color, segs);
    verts.extend(circle_wireframe(center, Vec3::Y, radius, color, segs));
    verts.extend(circle_wireframe(center, Vec3::Z, radius, color, segs));
    verts
}

/// Compute the world-space AABB of a scene node tree by recursing through all children.
///
/// `parent_world` is the accumulated world transform (translation + rotation + scale)
/// for the current node. For root calls, pass the object's world transform:
/// `Mat4::from_scale_rotation_translation(Vec3::splat(scale), rotation, world_pos)`.
pub fn compute_node_tree_aabb(
    node: &rkf_core::scene_node::SceneNode,
    parent_world: Mat4,
) -> Option<(Vec3, Vec3)> {
    use rkf_core::scene_node::SdfSource;

    let local = Mat4::from_scale_rotation_translation(
        Vec3::splat(node.local_transform.scale),
        node.local_transform.rotation,
        node.local_transform.position,
    );
    let world = parent_world * local;

    let mut aabb_min = Vec3::splat(f32::MAX);
    let mut aabb_max = Vec3::splat(f32::MIN);
    let mut has_bounds = false;

    // Get local bounds for this node's SDF.
    let local_bounds = match &node.sdf_source {
        SdfSource::Analytical { primitive, .. } => {
            Some(local_bounds_for_primitive(primitive))
        }
        SdfSource::Voxelized { brick_map_handle, voxel_size, .. } => {
            let brick_extent = voxel_size * 8.0;
            let grid_half = Vec3::new(
                brick_map_handle.dims.x as f32 * brick_extent * 0.5,
                brick_map_handle.dims.y as f32 * brick_extent * 0.5,
                brick_map_handle.dims.z as f32 * brick_extent * 0.5,
            );
            Some((-grid_half, grid_half))
        }
        SdfSource::None => None,
    };

    // Transform local bounds corners to world space and expand AABB.
    if let Some((lmin, lmax)) = local_bounds {
        let local_corners = [
            Vec3::new(lmin.x, lmin.y, lmin.z),
            Vec3::new(lmax.x, lmin.y, lmin.z),
            Vec3::new(lmax.x, lmax.y, lmin.z),
            Vec3::new(lmin.x, lmax.y, lmin.z),
            Vec3::new(lmin.x, lmin.y, lmax.z),
            Vec3::new(lmax.x, lmin.y, lmax.z),
            Vec3::new(lmax.x, lmax.y, lmax.z),
            Vec3::new(lmin.x, lmax.y, lmax.z),
        ];
        for c in &local_corners {
            let wc = world.transform_point3(*c);
            aabb_min = aabb_min.min(wc);
            aabb_max = aabb_max.max(wc);
        }
        has_bounds = true;
    }

    // Recurse into children.
    for child in &node.children {
        if let Some((cmin, cmax)) = compute_node_tree_aabb(child, world) {
            aabb_min = aabb_min.min(cmin);
            aabb_max = aabb_max.max(cmax);
            has_bounds = true;
        }
    }

    if has_bounds {
        Some((aabb_min, aabb_max))
    } else {
        None
    }
}

/// Given a scene tree entity ID, find the matching core SceneNode and the
/// world transform of its **parent**. Returns `None` if the entity doesn't
/// belong to this object.
///
/// The returned `parent_world` is suitable for passing directly to
/// [`compute_node_tree_aabb`], which applies the node's own `local_transform`.
///
/// Entity IDs follow the scheme: root = `obj.id`, children = `parent_id * 1000 + child_index`.
/// We decode by repeatedly dividing by 1000 to build a path of child indices,
/// then walk the core SceneNode tree along that path.
pub fn find_child_node_and_transform<'a>(
    entity_id: u64,
    obj_id: u64,
    root_node: &'a rkf_core::scene_node::SceneNode,
    root_world: Mat4,
) -> Option<(&'a rkf_core::scene_node::SceneNode, Mat4)> {
    // Decode the entity_id into a path of child indices.
    let mut path = Vec::new();
    let mut eid = entity_id;
    while eid != obj_id {
        if eid < 1000 {
            return None; // Not a child of this object.
        }
        let child_idx = (eid % 1000) as usize;
        eid /= 1000;
        path.push(child_idx);
    }
    // path is in reverse order (deepest child first), reverse it.
    path.reverse();

    // Walk the core SceneNode tree along the path, accumulating the
    // parent's world transform. We stop one step short so the returned
    // matrix does NOT include the target node's own local_transform
    // (compute_node_tree_aabb will apply it).
    let mut node = root_node;
    let mut parent_world = root_world;

    // First, include the root node's own transform (root_world is the object transform).
    let root_local = Mat4::from_scale_rotation_translation(
        Vec3::splat(node.local_transform.scale),
        node.local_transform.rotation,
        node.local_transform.position,
    );
    let mut world = root_world * root_local;

    for (i, &idx) in path.iter().enumerate() {
        if idx >= node.children.len() {
            return None;
        }
        parent_world = world; // world before the child's transform
        node = &node.children[idx];
        if i + 1 < path.len() {
            // Intermediate node: accumulate its transform.
            let local = Mat4::from_scale_rotation_translation(
                Vec3::splat(node.local_transform.scale),
                node.local_transform.rotation,
                node.local_transform.position,
            );
            world = parent_world * local;
        }
        // For the final node (target), don't apply its transform —
        // compute_node_tree_aabb will do that.
    }

    Some((node, parent_world))
}

/// Compute local-space bounding box for an SDF primitive.
pub fn local_bounds_for_primitive(primitive: &rkf_core::scene_node::SdfPrimitive) -> (Vec3, Vec3) {
    use rkf_core::scene_node::SdfPrimitive;
    match *primitive {
        SdfPrimitive::Sphere { radius } => {
            (Vec3::splat(-radius), Vec3::splat(radius))
        }
        SdfPrimitive::Box { half_extents } => {
            (-half_extents, half_extents)
        }
        SdfPrimitive::Capsule { radius, half_height } => {
            let h = half_height + radius;
            (Vec3::new(-radius, -h, -radius), Vec3::new(radius, h, radius))
        }
        SdfPrimitive::Torus { major_radius, minor_radius } => {
            let r = major_radius + minor_radius;
            (Vec3::new(-r, -minor_radius, -r), Vec3::new(r, minor_radius, r))
        }
        SdfPrimitive::Cylinder { radius, half_height } => {
            (Vec3::new(-radius, -half_height, -radius), Vec3::new(radius, half_height, radius))
        }
        SdfPrimitive::Plane { .. } => {
            // Infinite plane — use a large representative extent.
            let e = 100.0;
            (Vec3::new(-e, -0.01, -e), Vec3::new(e, 0.01, e))
        }
    }
}
