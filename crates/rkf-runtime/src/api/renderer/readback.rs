//! GPU readback operations: screenshot, pick, brush_hit.

use glam::Vec3;

use super::{BrushHitResult, Renderer};

impl Renderer {
    // ── GPU readback ───────────────────────────────────────────────────────

    /// Capture a screenshot of the current frame as RGBA8 pixels.
    pub fn screenshot(&self) -> Vec<u8> {
        let (source_texture, w, h) = if let Some(ref tex) = self.offscreen_texture {
            (tex, self.display_width, self.display_height)
        } else {
            (&self.cosmetics.output_texture, self.internal_width, self.internal_height)
        };

        let bytes_per_pixel = 4u32;
        let unpadded_row = w * bytes_per_pixel;
        let padded_row = (unpadded_row + 255) & !255;

        let mut encoder = self.ctx.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("screenshot"),
            },
        );

        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: source_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.readback_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: Some(h),
                },
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );

        self.ctx.queue.submit(std::iter::once(encoder.finish()));

        let buffer_slice = self.readback_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
        let _ = self.ctx.device.poll(wgpu::PollType::wait_indefinitely());

        let pixel_count = (w * h) as usize;
        let mut rgba8 = vec![0u8; pixel_count * 4];

        if let Ok(Ok(())) = rx.recv() {
            let data = buffer_slice.get_mapped_range();
            for y in 0..h as usize {
                let src_offset = y * padded_row as usize;
                let dst_offset = y * w as usize * 4;
                let row_bytes = w as usize * 4;
                rgba8[dst_offset..dst_offset + row_bytes]
                    .copy_from_slice(&data[src_offset..src_offset + row_bytes]);
            }
            drop(data);
            self.readback_buffer.unmap();
        } else {
            self.readback_buffer.unmap();
        }

        rgba8
    }

    /// GPU pick readback — returns the object ID at the given pixel coordinate.
    pub fn pick(&self, x: u32, y: u32) -> Option<u32> {
        let px = x.min(self.internal_width.saturating_sub(1));
        let py = y.min(self.internal_height.saturating_sub(1));

        let mut encoder = self.ctx.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("pick_readback"),
            },
        );
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.gbuffer.material_texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: px, y: py, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.pick_readback_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(1),
                },
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        self.ctx.queue.submit(std::iter::once(encoder.finish()));

        let slice = self.pick_readback_buffer.slice(..4);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let _ = self.ctx.device.poll(wgpu::PollType::wait_indefinitely());

        if let Ok(Ok(())) = rx.recv() {
            let data = slice.get_mapped_range();
            let packed = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
            let object_id = packed >> 24;
            drop(data);
            self.pick_readback_buffer.unmap();
            if object_id > 0 { Some(object_id) } else { None }
        } else {
            self.pick_readback_buffer.unmap();
            None
        }
    }

    /// GPU brush hit readback — returns world position + object ID at pixel.
    pub fn brush_hit(&self, x: u32, y: u32) -> Option<BrushHitResult> {
        let bx = x.min(self.internal_width.saturating_sub(1));
        let by = y.min(self.internal_height.saturating_sub(1));

        let mut encoder = self.ctx.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("brush_readback"),
            },
        );

        // Copy 1 pixel from position G-buffer (Rgba32Float = 16 bytes).
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.gbuffer.position_texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: bx, y: by, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.brush_readback_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(1),
                },
            },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );

        // Also copy 1 pixel from material G-buffer for object_id.
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.gbuffer.material_texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x: bx, y: by, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.pick_readback_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(1),
                },
            },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );

        self.ctx.queue.submit(std::iter::once(encoder.finish()));

        // Read position (4xf32 = 16 bytes).
        let pos_slice = self.brush_readback_buffer.slice(..16);
        let (tx_pos, rx_pos) = std::sync::mpsc::channel();
        pos_slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx_pos.send(r);
        });
        let _ = self.ctx.device.poll(wgpu::PollType::wait_indefinitely());

        let mut hit_pos = [0.0f32; 4];
        let pos_ok = if let Ok(Ok(())) = rx_pos.recv() {
            let data = pos_slice.get_mapped_range();
            hit_pos = [
                f32::from_le_bytes([data[0], data[1], data[2], data[3]]),
                f32::from_le_bytes([data[4], data[5], data[6], data[7]]),
                f32::from_le_bytes([data[8], data[9], data[10], data[11]]),
                f32::from_le_bytes([data[12], data[13], data[14], data[15]]),
            ];
            drop(data);
            self.brush_readback_buffer.unmap();
            true
        } else {
            self.brush_readback_buffer.unmap();
            false
        };

        // Read object_id from material G-buffer (bits 24-31).
        let mat_slice = self.pick_readback_buffer.slice(..4);
        let (tx_mat, rx_mat) = std::sync::mpsc::channel();
        mat_slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx_mat.send(r);
        });
        let _ = self.ctx.device.poll(wgpu::PollType::wait_indefinitely());

        let mut object_id = 0u32;
        if let Ok(Ok(())) = rx_mat.recv() {
            let data = mat_slice.get_mapped_range();
            let packed = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
            object_id = packed >> 24;
            drop(data);
            self.pick_readback_buffer.unmap();
        } else {
            self.pick_readback_buffer.unmap();
        }

        if pos_ok && hit_pos[3] < 1e30 {
            Some(BrushHitResult {
                position: Vec3::new(hit_pos[0], hit_pos[1], hit_pos[2]),
                object_id,
            })
        } else {
            None
        }
    }
}
