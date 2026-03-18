#!/usr/bin/env python3
"""Generate test textures for the textured-cube OBJ test asset."""

import struct
import zlib
import os

def write_png(path, width, height, pixels):
    """Write an RGBA PNG file from raw pixel data. No dependencies needed."""
    def chunk(chunk_type, data):
        c = chunk_type + data
        crc = struct.pack('>I', zlib.crc32(c) & 0xffffffff)
        return struct.pack('>I', len(data)) + c + crc

    raw = b''
    for y in range(height):
        raw += b'\x00'  # filter byte
        for x in range(width):
            i = (y * width + x) * 4
            raw += bytes(pixels[i:i+4])

    sig = b'\x89PNG\r\n\x1a\n'
    ihdr = struct.pack('>IIBBBBB', width, height, 8, 6, 0, 0, 0)
    idat = zlib.compress(raw)

    with open(path, 'wb') as f:
        f.write(sig)
        f.write(chunk(b'IHDR', ihdr))
        f.write(chunk(b'IDAT', idat))
        f.write(chunk(b'IEND', b''))

def make_brick_texture(size=64):
    """Generate a simple brick pattern texture."""
    pixels = []
    brick_h = 8
    brick_w = 16
    mortar = (140, 135, 125, 255)

    for y in range(size):
        row = y // brick_h
        offset = (brick_w // 2) if (row % 2) else 0
        for x in range(size):
            bx = (x + offset) % brick_w
            # Mortar lines
            if y % brick_h < 1 or bx < 1:
                pixels.extend(mortar)
            else:
                # Brick color with slight variation
                v = ((x * 7 + y * 13) % 30)
                r = min(255, 180 + v)
                g = min(255, 80 + v // 2)
                b = min(255, 60 + v // 3)
                pixels.extend([r, g, b, 255])
    return pixels

def make_wood_texture(size=64):
    """Generate a simple wood grain texture."""
    pixels = []
    for y in range(size):
        for x in range(size):
            # Vertical grain lines
            grain = ((x * 3 + y) % 12)
            if grain < 2:
                r, g, b = 120, 85, 50
            elif grain < 4:
                r, g, b = 160, 120, 70
            else:
                r, g, b = 185, 145, 90
            # Add some noise
            n = ((x * 17 + y * 31) % 20) - 10
            r = max(0, min(255, r + n))
            g = max(0, min(255, g + n))
            b = max(0, min(255, b + n))
            pixels.extend([r, g, b, 255])
    return pixels

if __name__ == '__main__':
    out_dir = os.path.join(os.path.dirname(__file__), 'textured-cube')

    pixels = make_brick_texture(64)
    write_png(os.path.join(out_dir, 'brick.png'), 64, 64, pixels)
    print('wrote brick.png (64x64)')

    pixels = make_wood_texture(64)
    write_png(os.path.join(out_dir, 'wood.png'), 64, 64, pixels)
    print('wrote wood.png (64x64)')
