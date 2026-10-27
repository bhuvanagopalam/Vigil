#!/usr/bin/env python3
"""Generate marker.png: a high-contrast, non-repetitive, feature-rich image for ARKit image
detection. Standard library only (no PIL). Deterministic (fixed seed) so everyone has the same
marker.

Usage: python3 make_marker.py  -> writes marker.png (1200x800, 8-bit grayscale)
"""
import random
import struct
import zlib

W, H = 1200, 800
SEED = 20261024
random.seed(SEED)

img = bytearray([235]) * (W * H)  # light background


def fill_rect(x0, y0, x1, y1, g):
    x0, x1 = max(0, min(x0, x1)), min(W, max(x0, x1))
    y0, y1 = max(0, min(y0, y1)), min(H, max(y0, y1))
    for y in range(y0, y1):
        row = y * W
        img[row + x0:row + x1] = bytes([g]) * (x1 - x0)


def fill_circle(cx, cy, r, g):
    r2 = r * r
    for y in range(max(0, cy - r), min(H, cy + r + 1)):
        dy = y - cy
        half = int((r2 - dy * dy) ** 0.5)
        x0, x1 = max(0, cx - half), min(W, cx + half + 1)
        row = y * W
        img[row + x0:row + x1] = bytes([g]) * (x1 - x0)


def fill_triangle(p, g):
    (x0, y0), (x1, y1), (x2, y2) = p
    ymin, ymax = max(0, min(y0, y1, y2)), min(H - 1, max(y0, y1, y2))
    for y in range(ymin, ymax + 1):
        xs = []
        for (ax, ay), (bx, by) in (((x0, y0), (x1, y1)), ((x1, y1), (x2, y2)), ((x2, y2), (x0, y0))):
            if (ay <= y < by) or (by <= y < ay):
                xs.append(ax + (y - ay) * (bx - ax) / (by - ay))
        if len(xs) >= 2:
            xa, xb = int(max(0, min(xs))), int(min(W, max(xs)))
            row = y * W
            img[row + xa:row + xb] = bytes([g]) * max(0, xb - xa)


levels = [0, 25, 60, 110, 160, 200, 255]
for _ in range(220):
    g = random.choice(levels)
    kind = random.random()
    cx, cy = random.randrange(W), random.randrange(H)
    s = random.randrange(18, 110)
    if kind < 0.4:
        fill_rect(cx - s // 2, cy - random.randrange(10, 90) // 2, cx + s // 2, cy + random.randrange(10, 90) // 2, g)
    elif kind < 0.7:
        fill_circle(cx, cy, s // 2, g)
    else:
        pts = [(cx + random.randrange(-s, s), cy + random.randrange(-s, s)) for _ in range(3)]
        fill_triangle(pts, g)

# thick black border + corner squares (helps detection and gives a clear edge to measure)
fill_rect(0, 0, W, 14, 0); fill_rect(0, H - 14, W, H, 0); fill_rect(0, 0, 14, H, 0); fill_rect(W - 14, 0, W, H, 0)
for (x, y) in ((40, 40), (W - 120, 40), (40, H - 120), (W - 120, H - 120)):
    fill_rect(x, y, x + 80, y + 80, 0)
    fill_rect(x + 20, y + 20, x + 60, y + 60, 255)

# PNG encode (grayscale, 8-bit)
raw = b"".join(b"\x00" + bytes(img[y * W:(y + 1) * W]) for y in range(H))


def chunk(tag, data):
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 0, 0, 0, 0)) \
    + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
with open("marker.png", "wb") as f:
    f.write(png)
print("wrote marker.png", W, "x", H)
