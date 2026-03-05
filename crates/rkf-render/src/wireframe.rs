//! Wireframe line rendering pass.
//!
//! Uses wgpu `LineList` topology to draw 3D line segments onto a render target.
//! The pass blends additively with `LoadOp::Load` so existing output is preserved.
//!
//! This is the only rasterization pass in the engine — all other rendering is
//! compute-only. Wireframes are used for selection highlights, gizmos, and debug
//! overlays.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Quat, Vec3};
use wgpu::util::DeviceExt;

/// A single line vertex (position in camera-relative space + RGBA color).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct LineVertex {
    /// Position in camera-relative world space.
    pub position: [f32; 3],
    /// RGBA color.
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

/// GPU line rendering pass for wireframe overlays.
pub struct WireframePass {
    pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    /// Persistent vertex buffer — reused across frames, grown when needed.
    vertex_buffer: wgpu::Buffer,
    /// Current capacity of the vertex buffer in vertices.
    vertex_buffer_capacity: usize,
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
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
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

        // Initial vertex buffer — 1024 vertices, grown on demand.
        let initial_cap = 1024;
        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("wireframe vertices"),
            size: (initial_cap * std::mem::size_of::<LineVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            uniform_buffer,
            bind_group_layout,
            bind_group,
            vertex_buffer,
            vertex_buffer_capacity: initial_cap,
        }
    }

    /// Draw line segments onto the target view.
    ///
    /// `vp_matrix` transforms from world space to clip space.
    /// `viewport` is `(x, y, width, height)` in physical pixels.
    /// `vertices` contains pairs of `LineVertex` (two per line segment).
    pub fn draw(
        &mut self,
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

        queue.write_buffer(
            &self.uniform_buffer,
            0,
            bytemuck::cast_slice(&vp_matrix.to_cols_array()),
        );

        // Grow vertex buffer if needed.
        if vertices.len() > self.vertex_buffer_capacity {
            let new_cap = vertices.len().next_power_of_two();
            self.vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("wireframe vertices"),
                size: (new_cap * std::mem::size_of::<LineVertex>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.vertex_buffer_capacity = new_cap;
        }

        queue.write_buffer(
            &self.vertex_buffer,
            0,
            bytemuck::cast_slice(vertices),
        );

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
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.draw(0..vertices.len() as u32, 0..1);
    }
}

// ── Wireframe geometry helpers ──────────────────────────────────────────────

/// Build wireframe vertices for an oriented bounding box (12 edges = 24 vertices).
pub fn obb_wireframe(
    local_min: Vec3,
    local_max: Vec3,
    position: Vec3,
    rotation: Quat,
    scale: Vec3,
    color: [f32; 4],
) -> Vec<LineVertex> {
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

    let corners: [Vec3; 8] = std::array::from_fn(|i| {
        position + rotation * (local_corners[i] * scale)
    });

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

/// Build wireframe vertices for an axis-aligned bounding box (12 edges = 24 vertices).
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

/// Build a small 3-axis crosshair (6 vertices = 3 line segments) at `center`.
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

/// Build a wireframe sphere (3 orthogonal great circles).
pub fn sphere_wireframe(center: Vec3, radius: f32, color: [f32; 4]) -> Vec<LineVertex> {
    let segs = 32;
    let mut verts = circle_wireframe(center, Vec3::X, radius, color, segs);
    verts.extend(circle_wireframe(center, Vec3::Y, radius, color, segs));
    verts.extend(circle_wireframe(center, Vec3::Z, radius, color, segs));
    verts
}

/// Build a point light gizmo: three orthogonal great circles showing range.
pub fn point_light_wireframe(center: Vec3, range: f32, color: [f32; 4]) -> Vec<LineVertex> {
    let segs = 32;
    let mut verts = circle_wireframe(center, Vec3::Y, range, color, segs);
    verts.extend(circle_wireframe(center, Vec3::X, range, color, segs));
    verts.extend(circle_wireframe(center, Vec3::Z, range, color, segs));
    verts.extend(crosshair(center, 0.2, color));
    verts
}

/// Build a spot light gizmo: cone showing outer angle + range.
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

    let mut verts = circle_wireframe(base_center, dir, base_radius, color, 32);
    let step = std::f32::consts::TAU / 4.0;
    for i in 0..4 {
        let a = step * i as f32;
        let p = base_center + (tangent * a.cos() + bitangent * a.sin()) * base_radius;
        verts.push(LineVertex { position: apex.to_array(), color });
        verts.push(LineVertex { position: p.to_array(), color });
    }
    verts.extend(crosshair(apex, 0.2, color));
    verts
}

/// Build a directional light gizmo: arrow showing direction.
pub fn directional_light_wireframe(
    position: Vec3, direction: Vec3, color: [f32; 4],
) -> Vec<LineVertex> {
    let dir = direction.normalize();
    let arrow_len = 2.0;
    let head_len = 0.4;
    let head_radius = 0.15;
    let tip = position + dir * arrow_len;

    let mut verts = vec![
        LineVertex { position: position.to_array(), color },
        LineVertex { position: tip.to_array(), color },
    ];

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

    verts.extend(circle_wireframe(position, dir, 0.3, color, 16));
    verts.extend(crosshair(position, 0.5, color));
    verts
}

/// Build a ground grid wireframe at Y=0 centered around the camera position.
pub fn ground_grid_wireframe(
    cam_pos: Vec3, extent: f32, spacing: f32, color: [f32; 4],
) -> Vec<LineVertex> {
    let half = extent * 0.5;
    let cx = (cam_pos.x / spacing).round() * spacing;
    let cz = (cam_pos.z / spacing).round() * spacing;
    let count = (extent / spacing) as i32;

    let mut verts = Vec::with_capacity(count as usize * 4 + 4);
    for i in -count / 2..=count / 2 {
        let offset = i as f32 * spacing;
        verts.push(LineVertex { position: [cx - half, 0.0, cz + offset], color });
        verts.push(LineVertex { position: [cx + half, 0.0, cz + offset], color });
        verts.push(LineVertex { position: [cx + offset, 0.0, cz - half], color });
        verts.push(LineVertex { position: [cx + offset, 0.0, cz + half], color });
    }
    verts
}
