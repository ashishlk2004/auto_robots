#!/usr/bin/env python3
#
# Generates an occupancy grid map (map2.pgm / map2.yaml) of the assessment world
# for use with Nav2/AMCL, by rasterising the collision meshes referenced by
# assessment/worlds/assessment.world.
#
# The map frame is identical to the Gazebo world frame, so poses estimated by
# AMCL can be compared directly against the zone coordinates used by the
# barrel_manager node.
#
# The four small obstacles (box1/2, cylinder1/2) in the west room are NOT baked
# into the static map because they can be disabled with the 'obstacles' launch
# parameter; the Nav2 costmap obstacle layer detects them from LiDAR instead.
#
# The two obstacle meshes (obstacle1/2) in the middle room are also left out:
# they are curved shells whose concave side is drivable, so a projected
# footprint in the static map would mark free space as lethal. The LiDAR
# obstacle layer observes their true shape at runtime instead.
#
# Usage: python3 generate_map.py [--mesh-dir DIR] [--out-dir DIR]

import argparse
import math
import os
import struct

import numpy as np

RESOLUTION = 0.05          # metres per cell
X_MIN, X_MAX = -18.0, 22.0  # map bounds (world frame, metres)
Y_MIN, Y_MAX = -4.0, 14.0
WALL_Z_MIN, WALL_Z_MAX = 0.15, 2.0  # z-slab that counts as a wall

FREE, OCCUPIED = 254, 0    # PGM pixel values (trinary map convention)


def load_stl(path):
    """Load a binary or ASCII STL file, returning an (N, 3, 3) triangle array."""
    with open(path, 'rb') as f:
        f.read(80)
        rest = f.read()
    if len(rest) >= 4:
        (n_tri,) = struct.unpack('<I', rest[:4])
        if len(rest) == 4 + n_tri * 50:
            tris = np.zeros((n_tri, 3, 3), dtype=np.float64)
            for i in range(n_tri):
                off = 4 + i * 50
                vals = struct.unpack('<12fH', rest[off:off + 50])
                tris[i, 0] = vals[3:6]
                tris[i, 1] = vals[6:9]
                tris[i, 2] = vals[9:12]
            return tris
    verts = []
    with open(path, 'r', errors='ignore') as f:
        for line in f:
            parts = line.split()
            if parts and parts[0] == 'vertex':
                verts.append([float(p) for p in parts[1:4]])
    return np.array(verts).reshape(-1, 3, 3)


def transform(tris, scale, pose_xyz, yaw):
    """Scale, rotate (about z) and translate mesh triangles into the world frame."""
    v = tris * scale
    c, s = math.cos(yaw), math.sin(yaw)
    x = v[:, :, 0] * c - v[:, :, 1] * s + pose_xyz[0]
    y = v[:, :, 0] * s + v[:, :, 1] * c + pose_xyz[1]
    z = v[:, :, 2] + pose_xyz[2]
    return np.stack([x, y, z], axis=2)


class Grid:
    def __init__(self):
        self.w = int(round((X_MAX - X_MIN) / RESOLUTION))
        self.h = int(round((Y_MAX - Y_MIN) / RESOLUTION))
        self.data = np.full((self.h, self.w), FREE, dtype=np.uint8)

    def world_to_px(self, x, y):
        # Row 0 of a PGM map image is the TOP of the map (maximum y).
        col = (x - X_MIN) / RESOLUTION
        row = (Y_MAX - y) / RESOLUTION
        return col, row

    def draw_segment(self, x0, y0, x1, y1):
        """Mark all cells along a world-frame segment as occupied."""
        c0, r0 = self.world_to_px(x0, y0)
        c1, r1 = self.world_to_px(x1, y1)
        n = int(max(abs(c1 - c0), abs(r1 - r0)) * 2) + 1
        cols = np.linspace(c0, c1, n).astype(int)
        rows = np.linspace(r0, r1, n).astype(int)
        ok = (cols >= 0) & (cols < self.w) & (rows >= 0) & (rows < self.h)
        self.data[rows[ok], cols[ok]] = OCCUPIED

    def draw_triangles(self, tris):
        """Rasterise the edges of wall triangles (vertical faces project to lines)."""
        zmin = tris[:, :, 2].min(axis=1)
        zmax = tris[:, :, 2].max(axis=1)
        keep = (zmin <= WALL_Z_MAX) & (zmax >= WALL_Z_MIN)
        for tri in tris[keep]:
            for a, b in ((0, 1), (1, 2), (2, 0)):
                self.draw_segment(tri[a, 0], tri[a, 1], tri[b, 0], tri[b, 1])

    def draw_rect(self, cx, cy, sx, sy):
        """Fill an axis-aligned rectangle (centre cx,cy, size sx,sy) as occupied."""
        c0, r1 = self.world_to_px(cx - sx / 2, cy - sy / 2)
        c1, r0 = self.world_to_px(cx + sx / 2, cy + sy / 2)
        c0, c1 = max(int(c0), 0), min(int(math.ceil(c1)), self.w)
        r0, r1 = max(int(r0), 0), min(int(math.ceil(r1)), self.h)
        self.data[r0:r1 + 1, c0:c1 + 1] = OCCUPIED

    def save_pgm(self, path):
        with open(path, 'wb') as f:
            f.write(b'P5\n%d %d\n255\n' % (self.w, self.h))
            f.write(self.data.tobytes())


def main():
    parser = argparse.ArgumentParser()
    default_mesh = os.path.join(os.path.dirname(__file__), '..', '..',
                                'assessment', 'models', 'meshes')
    default_out = os.path.join(os.path.dirname(__file__), '..', 'config')
    parser.add_argument('--mesh-dir', default=default_mesh)
    parser.add_argument('--out-dir', default=default_out)
    args = parser.parse_args()

    grid = Grid()

    # Static models taken from assessment/worlds/assessment.world
    building = load_stl(os.path.join(args.mesh_dir, 'building.stl'))
    grid.draw_triangles(transform(building, 0.01, (6.0, -6.0, 0.0), 0.0))

    # Walls belonging to the zone models (storage1/2, decontamination).
    # Each zone is a 3x3 m plate; 'back_wall' is the plate rotated upright at
    # local y=+1.5, and 'side_wall' at local x=+1.4 (storage2/decontamination
    # side walls are scaled to 1.8 m and centred at local y=+0.6).
    for zx, zy, side_full in ((19.5, 9.4, True), (13.5, 9.4, False), (7.5, 9.4, False)):
        grid.draw_rect(zx, zy + 1.5, 3.0, 0.1)          # back wall
        if side_full:
            grid.draw_rect(zx + 1.4, zy, 0.1, 3.0)      # full-length side wall
        else:
            grid.draw_rect(zx + 1.4, zy + 0.6, 0.1, 1.8)  # partial side wall

    os.makedirs(args.out_dir, exist_ok=True)
    pgm_path = os.path.join(args.out_dir, 'map2.pgm')
    grid.save_pgm(pgm_path)

    yaml_path = os.path.join(args.out_dir, 'map2.yaml')
    with open(yaml_path, 'w') as f:
        f.write('image: map2.pgm\n')
        f.write('mode: trinary\n')
        f.write(f'resolution: {RESOLUTION}\n')
        f.write(f'origin: [{X_MIN}, {Y_MIN}, 0.0]\n')
        f.write('negate: 0\n')
        f.write('occupied_thresh: 0.65\n')
        f.write('free_thresh: 0.196\n')
    print(f'Wrote {pgm_path} ({grid.w}x{grid.h}) and {yaml_path}')


if __name__ == '__main__':
    main()
