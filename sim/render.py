"""OpenGL rendering: world, car, camera and HUD."""

import math

import numpy as np
import pygame
from OpenGL.GL import *  # noqa: F401,F403
from OpenGL.GLU import gluLookAt, gluPerspective

from .track import ROAD_HALF_WIDTH

SKY = (0.62, 0.78, 0.93)


# primitives

def box(cx, cy, cz, sx, sy, sz, color, top_color=None):
    """Box centred at (cx, cy, cz) with full sizes (sx, sy, sz)."""
    x0, x1 = cx - sx / 2, cx + sx / 2
    y0, y1 = cy - sy / 2, cy + sy / 2
    z0, z1 = cz - sz / 2, cz + sz / 2
    glBegin(GL_QUADS)
    glColor3f(*color)
    for n, verts in (
        ((0, 0, 1), ((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1))),
        ((0, 0, -1), ((x1, y0, z0), (x0, y0, z0), (x0, y1, z0), (x1, y1, z0))),
        ((1, 0, 0), ((x1, y0, z1), (x1, y0, z0), (x1, y1, z0), (x1, y1, z1))),
        ((-1, 0, 0), ((x0, y0, z0), (x0, y0, z1), (x0, y1, z1), (x0, y1, z0))),
        ((0, -1, 0), ((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1))),
    ):
        glNormal3f(*n)
        for v in verts:
            glVertex3f(*v)
    glColor3f(*(top_color or color))
    glNormal3f(0, 1, 0)
    for v in ((x0, y1, z1), (x1, y1, z1), (x1, y1, z0), (x0, y1, z0)):
        glVertex3f(*v)
    glEnd()


def cone(cx, cy, cz, radius, height, color, segments=8):
    slope = radius / height
    glColor3f(*color)
    glBegin(GL_TRIANGLES)
    for i in range(segments):
        a0 = 2 * math.pi * i / segments
        a1 = 2 * math.pi * (i + 1) / segments
        am = (a0 + a1) / 2
        n = (math.cos(am), slope, math.sin(am))
        glNormal3f(*n)
        glVertex3f(cx, cy + height, cz)
        glVertex3f(cx + math.cos(a1) * radius, cy, cz + math.sin(a1) * radius)
        glVertex3f(cx + math.cos(a0) * radius, cy, cz + math.sin(a0) * radius)
    glEnd()


def cylinder_x(radius, width, color, segments=10):
    """Cylinder along X, centred at the origin."""
    hw = width / 2
    glColor3f(*color)
    glBegin(GL_QUAD_STRIP)
    for i in range(segments + 1):
        a = 2 * math.pi * i / segments
        y, z = math.cos(a), math.sin(a)
        glNormal3f(0, y, z)
        glVertex3f(-hw, y * radius, z * radius)
        glVertex3f(hw, y * radius, z * radius)
    glEnd()
    for side in (-1, 1):
        glBegin(GL_TRIANGLE_FAN)
        glNormal3f(side, 0, 0)
        glColor3f(0.55, 0.55, 0.58)
        glVertex3f(side * hw, 0, 0)
        glColor3f(*color)
        for i in range(segments + 1):
            a = 2 * math.pi * i / segments * side
            glVertex3f(side * hw, math.cos(a) * radius, math.sin(a) * radius)
        glEnd()


# setup

def init_gl():
    glClearColor(*SKY, 1)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)
    glFrontFace(GL_CCW)
    glShadeModel(GL_SMOOTH)

    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_NORMALIZE)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, (0.45, 0.45, 0.5, 1))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.75, 0.72, 0.65, 1))
    glLightfv(GL_LIGHT0, GL_SPECULAR, (0, 0, 0, 1))

    glEnable(GL_FOG)
    glFogi(GL_FOG_MODE, GL_LINEAR)
    glFogfv(GL_FOG_COLOR, (*SKY, 1))
    glFogf(GL_FOG_START, 180)
    glFogf(GL_FOG_END, 950)
    glHint(GL_FOG_HINT, GL_NICEST)


def build_world(track):
    """Compile all static geometry into one display list."""
    lst = glGenLists(1)
    glNewList(lst, GL_COMPILE)
    _ground()
    _road(track)
    for (x, z, s, v) in track.trees:
        _tree(x, z, s, v)
    for (x, z, s, v) in track.buildings:
        _building(x, z, s, v)
    for (x, z) in track.cones:
        box(x, 0.3, z, 0.45, 0.6, 0.45, (0.95, 0.45, 0.1), (1, 1, 1))
    _mountains(track)
    glEndList()
    return lst


def _ground():
    glNormal3f(0, 1, 0)
    size, step = 1400, 100
    glBegin(GL_QUADS)
    for i in range(-size, size, step):
        for j in range(-size, size, step):
            shade = 0.03 if ((i + j) // step) % 2 else 0.0
            glColor3f(0.33 + shade, 0.55 + shade, 0.25)
            glVertex3f(i, 0, j)
            glVertex3f(i, 0, j + step)
            glVertex3f(i + step, 0, j + step)
            glVertex3f(i + step, 0, j)
    glEnd()


def _strip(track, inner, outer, y, color_fn, every=1, only=None):
    """Quads between two lateral offsets along the track."""
    pts, nrm = track.points, track.normals
    n = len(pts)
    glNormal3f(0, 1, 0)
    glBegin(GL_QUADS)
    for i in range(0, n, every):
        if only and not only(i):
            continue
        j = (i + 1) % n
        glColor3f(*color_fn(i))
        (ax, az), (anx, anz) = pts[i], nrm[i]
        (bx, bz), (bnx, bnz) = pts[j], nrm[j]
        glVertex3f(ax + anx * inner, y, az + anz * inner)
        glVertex3f(ax + anx * outer, y, az + anz * outer)
        glVertex3f(bx + bnx * outer, y, bz + bnz * outer)
        glVertex3f(bx + bnx * inner, y, bz + bnz * inner)
    glEnd()


def _road(track):
    hw = ROAD_HALF_WIDTH
    glDisable(GL_CULL_FACE)  # tight bends flip winding
    _strip(track, -hw, hw, 0.02, lambda i: (0.27, 0.27, 0.29))
    _strip(track, -hw - 1.2, -hw, 0.03, lambda i: (0.85, 0.1, 0.1) if (i // 2) % 2 else (0.95, 0.95, 0.95))
    _strip(track, hw, hw + 1.2, 0.03, lambda i: (0.85, 0.1, 0.1) if (i // 2) % 2 else (0.95, 0.95, 0.95))
    white = lambda i: (0.92, 0.92, 0.9)  # noqa: E731
    _strip(track, -hw + 0.4, -hw + 0.65, 0.04, white)
    _strip(track, hw - 0.65, hw - 0.4, 0.04, white)
    _strip(track, -0.15, 0.15, 0.04, lambda i: (0.95, 0.8, 0.2), only=lambda i: (i // 2) % 3 == 0)
    # checkered start/finish line across sample 0..1
    (ax, az), (nx, nz) = track.points[0], track.normals[0]
    tx, tz = track.tangents[0]
    cells = 10
    for c in range(cells):
        for r in range(2):
            off0 = -hw + 2 * hw * c / cells
            off1 = -hw + 2 * hw * (c + 1) / cells
            f0, f1 = r * 1.0, (r + 1) * 1.0
            glColor3f(*((0.05,) * 3 if (c + r) % 2 else (0.97,) * 3))
            glBegin(GL_QUADS)
            glNormal3f(0, 1, 0)
            for off, f in ((off0, f0), (off1, f0), (off1, f1), (off0, f1)):
                glVertex3f(ax + nx * off + tx * f, 0.05, az + nz * off + tz * f)
            glEnd()
    # gantry over the start line
    for side in (-1, 1):
        px, pz = ax + nx * (hw + 2) * side, az + nz * (hw + 2) * side
        box(px, 3.5, pz, 0.5, 7, 0.5, (0.3, 0.3, 0.35))
    glPushMatrix()
    glTranslatef(ax, 7.2, az)
    glRotatef(math.degrees(math.atan2(tx, tz)), 0, 1, 0)
    box(0, 0, 0, 2 * hw + 4.5, 0.8, 0.5, (0.15, 0.15, 0.6), (0.2, 0.2, 0.7))
    glPopMatrix()
    glEnable(GL_CULL_FACE)


def _tree(x, z, s, v):
    trunk = (0.4 + v * 0.1, 0.28, 0.15)
    leaf = (0.13 + v * 0.1, 0.4 + v * 0.18, 0.15)
    box(x, 1.0 * s, z, 0.4 * s, 2.0 * s, 0.4 * s, trunk)
    cone(x, 1.4 * s, z, 2.2 * s, 3.5 * s, leaf)
    cone(x, 3.2 * s, z, 1.6 * s, 3.0 * s, tuple(c * 1.12 for c in leaf))


def _building(x, z, s, v):
    h = 8 + v * 30
    w = 12 * s
    base = (0.55 + v * 0.2, 0.55 + v * 0.15, 0.6 - v * 0.1)
    box(x, h / 2, z, w, h, w, base, (0.35, 0.35, 0.38))
    # window bands
    for k in range(int(h // 4)):
        y = 2.5 + k * 4
        if y > h - 1:
            break
        box(x, y, z, w + 0.06, 1.3, w + 0.06, (0.25, 0.35, 0.5))


def _mountains(track):
    xs = [p[0] for p in track.points]
    zs = [p[1] for p in track.points]
    cx, cz = (min(xs) + max(xs)) / 2, (min(zs) + max(zs)) / 2
    for i in range(28):
        a = 2 * math.pi * i / 28
        r = 850 + 60 * math.sin(i * 2.3)
        h = 90 + 70 * abs(math.sin(i * 1.7))
        cone(cx + math.cos(a) * r, 0, cz + math.sin(a) * r, 160, h,
             (0.42, 0.5, 0.45), segments=7)


# car (car space: +z forward, +y up, right is -x)

BODY = (0.55, 0.56, 0.58)
CARBON = (0.09, 0.09, 0.1)
GLASS = (0.1, 0.12, 0.16)
ACCENT = (0.95, 0.38, 0.06)
WHITE = (0.94, 0.94, 0.94)
SILVER = (0.72, 0.73, 0.76)
TIRE = (0.07, 0.07, 0.07)


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _avg(pts):
    n = len(pts)
    return tuple(sum(p[i] for p in pts) / n for i in range(3))


def _lerp(a, b, t):
    return tuple(x + (y - x) * t for x, y in zip(a, b))


class _Face:
    """Quad with an outward normal; (u, v) maps across it."""

    def __init__(self, p00, p10, p11, p01, inside):
        self.p = (p00, p10, p11, p01)
        n = _cross(_sub(p11, p00), _sub(p01, p10))
        if sum(a * b for a, b in zip(n, _sub(_avg(self.p), inside))) < 0:
            n = (-n[0], -n[1], -n[2])
        length = math.sqrt(sum(c * c for c in n)) or 1.0
        self.n = tuple(c / length for c in n)

    def at(self, u, v, lift=0.0):
        p00, p10, p11, p01 = self.p
        p = _lerp(_lerp(p00, p10, u), _lerp(p01, p11, u), v)
        return tuple(c + nc * lift for c, nc in zip(p, self.n))

    def draw(self, color):
        glColor3f(*color)
        glNormal3f(*self.n)
        for p in self.p:
            glVertex3f(*p)

    def decal(self, uvs, color, lift=0.004):
        glColor3f(*color)
        glBegin(GL_POLYGON)
        glNormal3f(*self.n)
        for u, v in uvs:
            glVertex3f(*self.at(u, v, lift))
        glEnd()

    def rect(self, u0, u1, v0, v1, color, lift=0.004):
        self.decal(((u0, v0), (u1, v0), (u1, v1), (u0, v1)), color, lift)


def prism(bottom, top, color, top_color=None, side_color=None, end_color=None):
    """Convex block from 4 bottom + 4 top corners (-x back, +x back, +x front, -x front)."""
    nb, pb, pf, nf = bottom
    nbt, pbt, pft, nft = top
    inside = _avg(bottom + top)
    faces = {
        "top": _Face(nbt, pbt, pft, nft, inside),
        "bottom": _Face(nb, pb, pf, nf, inside),
        "-x": _Face(nb, nf, nft, nbt, inside),
        "+x": _Face(pb, pf, pft, pbt, inside),
        "front": _Face(nf, pf, pft, nft, inside),
        "back": _Face(nb, pb, pbt, nbt, inside),
    }
    glBegin(GL_QUADS)
    faces["top"].draw(top_color or color)
    faces["bottom"].draw(color)
    faces["-x"].draw(side_color or color)
    faces["+x"].draw(side_color or color)
    faces["front"].draw(end_color or color)
    faces["back"].draw(end_color or color)
    glEnd()
    return faces


def _rect(zb, zf, wb, wf, yb, yf=None):
    """Symmetric quad: z range, widths and heights (back, front)."""
    yf = yb if yf is None else yf
    return ((-wb / 2, yb, zb), (wb / 2, yb, zb), (wf / 2, yf, zf), (-wf / 2, yf, zf))


def _block(x0, x1, z0, z1, y0, yt_back, yt_front):
    """Footprint with a sloped top."""
    return prism(((x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)),
                 ((x0, yt_back, z0), (x1, yt_back, z0), (x1, yt_front, z1), (x0, yt_front, z1)),
                 BODY)


def _disc_uv(uc, vc, ru, rv, segments=20):
    return [(uc + math.cos(2 * math.pi * i / segments) * ru,
             vc + math.sin(2 * math.pi * i / segments) * rv) for i in range(segments)]


def _stripes(face):
    face.rect(0.40, 0.47, 0, 1, ACCENT)
    face.rect(0.53, 0.60, 0, 1, ACCENT)


def _roundel(face, uc, vc, ru, rv, mirror):
    """Number '7' disc on a side face."""
    face.decal(_disc_uv(uc, vc, ru, rv), WHITE, 0.004)
    d = -1 if mirror else 1
    for glyph in (((-0.45, 0.55), (0.5, 0.55), (0.5, 0.33), (-0.45, 0.33)),
                  ((0.27, 0.33), (0.5, 0.33), (0.0, -0.62), (-0.23, -0.62))):
        pts = [(uc + d * s * ru, vc + t * rv) for s, t in glyph]
        if mirror:
            pts.reverse()
        face.decal(pts, CARBON, 0.008)


def _disc_x(x, radius, color, segments=18):
    glColor3f(*color)
    glBegin(GL_TRIANGLE_FAN)
    glNormal3f(1 if x > 0 else -1, 0, 0)
    glVertex3f(x, 0, 0)
    for i in range(segments + 1):
        a = 2 * math.pi * i / segments
        glVertex3f(x, math.cos(a) * radius, math.sin(a) * radius)
    glEnd()


def _tire(radius, width, segments=18):
    hw = width / 2
    glColor3f(*TIRE)
    glBegin(GL_QUAD_STRIP)
    for i in range(segments + 1):
        a = 2 * math.pi * i / segments
        y, z = math.cos(a), math.sin(a)
        glNormal3f(0, y, z)
        glVertex3f(-hw, y * radius, z * radius)
        glVertex3f(hw, y * radius, z * radius)
    glEnd()
    for side in (-1, 1):
        _disc_x(side * hw, radius, TIRE, segments)


def _wheel(car, wx, wz, front, spin):
    out = 1 if wx > 0 else -1
    radius, width = 0.34, 0.32
    glPushMatrix()
    glTranslatef(wx, radius, wz)
    if front:
        glRotatef(-math.degrees(car.steer), 0, 1, 0)
    face = out * width / 2
    # caliper doesn't spin
    glPushMatrix()
    glRotatef(-35 if front else 35, 1, 0, 0)
    box(face + out * 0.004, 0.17, 0, 0.012, 0.09, 0.16, ACCENT)
    glPopMatrix()
    glRotatef(spin, 1, 0, 0)
    _tire(radius, width)
    _disc_x(face + out * 0.002, 0.25, (0.2, 0.2, 0.22))     # rim barrel
    _disc_x(face + out * 0.003, 0.19, (0.32, 0.32, 0.34))   # brake disc
    for k in range(5):
        glPushMatrix()
        glRotatef(k * 72, 1, 0, 0)
        box(face + out * 0.012, 0.13, 0, 0.014, 0.24, 0.055, SILVER)
        glPopMatrix()
    _disc_x(face + out * 0.02, 0.06, SILVER)                 # hub
    _disc_x(face + out * 0.021, 0.025, CARBON)               # centre lock
    glPopMatrix()


def draw_car(car):
    glDisable(GL_CULL_FACE)
    glPushMatrix()
    glTranslatef(car.x, 0, car.z)
    glRotatef(math.degrees(car.heading), 0, 1, 0)

    # blob shadow
    glDisable(GL_LIGHTING)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glColor4f(0, 0, 0, 0.38)
    glBegin(GL_QUADS)
    for vx, vz in ((-1.15, -2.45), (-1.15, 2.45), (1.15, 2.45), (1.15, -2.45)):
        glVertex3f(vx, 0.06, vz)
    glEnd()
    glDisable(GL_BLEND)
    glEnable(GL_LIGHTING)

    # body roll
    roll = max(-4.0, min(4.0, -car.lateral_accel * 0.35))
    glPushMatrix()
    glRotatef(roll, 0, 0, 1)

    # main tub
    tub = prism(_rect(-2.15, 1.3, 1.80, 1.86, 0.22, 0.20),
                _rect(-2.15, 1.3, 1.70, 1.76, 0.68, 0.62), BODY)
    _stripes(tub["top"])
    for u0 in (0.12, 0.64):  # engine cover louvres
        for k in range(4):
            v = (0.2 + k * 0.14) / 3.45
            tub["top"].rect(u0, u0 + 0.24, v, v + 0.025, CARBON)
    tub["back"].rect(0.05, 0.3, 0.7, 0.88, (0.85, 0.05, 0.05))   # taillights
    tub["back"].rect(0.7, 0.95, 0.7, 0.88, (0.85, 0.05, 0.05))
    tub["back"].rect(0.3, 0.7, 0.77, 0.83, (0.45, 0.03, 0.03))
    _roundel(tub["-x"], 0.53, 0.5, 0.05, 0.38, mirror=False)
    _roundel(tub["+x"], 0.53, 0.5, 0.05, 0.38, mirror=True)

    # wedge nose
    nose = prism(_rect(1.3, 2.3, 1.86, 1.6, 0.20, 0.18),
                 _rect(1.3, 2.3, 1.76, 1.4, 0.62, 0.36), BODY)
    _stripes(nose["top"])
    for u0 in (0.1, 0.72):
        nose["top"].rect(u0, u0 + 0.18, 0.08, 0.34, CARBON)                 # hood vents
    nose["top"].rect(0.04, 0.24, 0.62, 0.9, (1, 1, 0.85))                   # headlights
    nose["top"].rect(0.76, 0.96, 0.62, 0.9, (1, 1, 0.85))
    nose["front"].rect(0.22, 0.78, 0.15, 0.8, CARBON)                       # grille

    # cabin
    cabin = prism(_rect(-1.3, 0.6, 1.62, 1.62, 0.6),
                  _rect(-0.6, -0.05, 1.18, 1.18, 1.1), GLASS,
                  top_color=BODY, side_color=BODY)
    _stripes(cabin["top"])
    for side in ("-x", "+x"):  # side windows
        cabin[side].decal(((0.2, 0.22), (0.8, 0.22), (0.86, 0.9), (0.16, 0.9)), GLASS)

    # wheel arches
    for sign in (-1, 1):
        x0, x1 = sorted((sign * 0.74, sign * 1.1))
        _block(x0, x1, 0.85, 1.85, 0.5, 0.76, 0.71)
        _block(x0, x1, -1.9, -0.8, 0.5, 0.75, 0.79)

    # splitter, canards, skirts, diffuser
    box(0, 0.14, 2.3, 1.94, 0.04, 0.36, CARBON)
    for sign in (-1, 1):
        box(sign * 0.86, 0.34, 2.05, 0.22, 0.02, 0.22, CARBON)
        box(sign * 0.92, 0.2, 0, 0.1, 0.1, 1.9, CARBON)
    box(0, 0.2, -2.25, 1.6, 0.16, 0.25, CARBON)
    for fx in (-0.45, -0.15, 0.15, 0.45):
        box(fx, 0.2, -2.3, 0.02, 0.2, 0.3, (0.16, 0.16, 0.17))

    # exhausts
    for ex in (-0.6, 0.6):
        glPushMatrix()
        glTranslatef(ex, 0.33, -2.2)
        glRotatef(90, 0, 1, 0)
        cylinder_x(0.055, 0.18, (0.35, 0.33, 0.3))
        glPopMatrix()

    # mirrors
    for sign in (-1, 1):
        box(sign * 0.85, 0.8, 0.35, 0.14, 0.03, 0.05, CARBON)
        box(sign * 0.97, 0.84, 0.35, 0.18, 0.11, 0.14, BODY)

    # rear wing
    for sx in (-0.4, 0.4):
        box(sx, 1.0, -1.98, 0.04, 0.66, 0.1, CARBON)
    wing = prism(_rect(-2.3, -1.72, 1.9, 1.9, 1.36, 1.26),
                 _rect(-2.3, -1.72, 1.9, 1.9, 1.41, 1.31), CARBON)
    _stripes(wing["top"])
    box(0, 1.44, -2.29, 1.9, 0.05, 0.02, ACCENT)                           # gurney flap
    for sign in (-1, 1):
        box(sign * 0.965, 1.32, -2.0, 0.025, 0.24, 0.62, CARBON)            # endplates
    glPopMatrix()

    spin = math.degrees(car.wheel_spin) % 360
    for wx, wz, front in ((0.92, 1.35, True), (-0.92, 1.35, True),
                          (0.92, -1.35, False), (-0.92, -1.35, False)):
        _wheel(car, wx, wz, front, spin)
    glPopMatrix()
    glEnable(GL_CULL_FACE)


# camera

class Camera:
    MODES = ("chase", "hood", "high")

    def __init__(self):
        self.mode = 0
        self.pos = None
        self.look = None

    def cycle(self):
        self.mode = (self.mode + 1) % len(self.MODES)
        self.pos = None

    def apply(self, car, dt, aspect):
        fx, fz = math.sin(car.heading), math.cos(car.heading)
        mode = self.MODES[self.mode]
        if mode == "chase":
            want_pos = (car.x - fx * 7.5, 2.9, car.z - fz * 7.5)
            want_look = (car.x + fx * 4, 1.0, car.z + fz * 4)
            lag = 6.0
        elif mode == "hood":
            want_pos = (car.x + fx * 0.3, 1.45, car.z + fz * 0.3)
            want_look = (car.x + fx * 20, 1.0, car.z + fz * 20)
            lag = 30.0
        else:
            want_pos = (car.x - fx * 18, 14, car.z - fz * 18)
            want_look = (car.x + fx * 6, 0, car.z + fz * 6)
            lag = 4.0
        if self.pos is None:
            self.pos, self.look = want_pos, want_look
        k = 1 - math.exp(-lag * dt)
        self.pos = tuple(p + (w - p) * k for p, w in zip(self.pos, want_pos))
        self.look = tuple(p + (w - p) * k for p, w in zip(self.look, want_look))
        if car.crashed > 0:
            shake = car.crashed * 0.25
            t = pygame.time.get_ticks() * 0.05
            self.pos = (self.pos[0] + math.sin(t * 7) * shake, self.pos[1] + math.sin(t * 9) * shake,
                        self.pos[2])

        set_view(self.pos, self.look, 62 + min(abs(car.speed), 60) * 0.22, aspect)


def set_view(pos, look, fov, aspect):
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(fov, aspect, 0.25, 1500)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    gluLookAt(*pos, *look, 0, 1, 0)
    glLightfv(GL_LIGHT0, GL_POSITION, (0.4, 1.0, 0.3, 0))  # after the view so the sun is world-fixed


def front_camera(car, aspect):
    """Bumper camera for the vision sim. Returns (modelview, projection)."""
    fx, fz = math.sin(car.heading), math.cos(car.heading)
    set_view((car.x + fx * 0.8, 1.25, car.z + fz * 0.8),
             (car.x + fx * 20, 1.0, car.z + fz * 20), 68, aspect)
    return glGetDoublev(GL_MODELVIEW_MATRIX), glGetDoublev(GL_PROJECTION_MATRIX)


# HUD

class Hud:
    def __init__(self):
        self.font = pygame.font.SysFont("dejavusansmono,monospace", 20, bold=True)
        self.big = pygame.font.SysFont("dejavusansmono,monospace", 44, bold=True)
        self.small = pygame.font.SysFont("dejavusansmono,monospace", 15)
        self.tiny = pygame.font.SysFont("dejavusansmono,monospace", 12, bold=True)

    def begin(self, w, h):
        self.w, self.h = w, h
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_FOG)
        glDisable(GL_CULL_FACE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(0, w, 0, h, -1, 1)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    def end(self):
        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_FOG)
        glEnable(GL_CULL_FACE)

    def text(self, s, x, y, font=None, color=(255, 255, 255), anchor="tl", shadow=True):
        """(x, y) from the top-left of the window."""
        font = font or self.font
        surf = font.render(s, True, color)
        if shadow:
            bg = font.render(s, True, (0, 0, 0))
            full = pygame.Surface((surf.get_width() + 2, surf.get_height() + 2), pygame.SRCALPHA)
            bg.set_alpha(160)
            full.blit(bg, (2, 2))
            full.blit(surf, (0, 0))
            surf = full
        w, h = surf.get_size()
        if "r" in anchor:
            x -= w
        if "c" in anchor:
            x -= w // 2
        data = pygame.image.tobytes(surf, "RGBA", True)
        tex = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)
        glEnable(GL_TEXTURE_2D)
        glColor4f(1, 1, 1, 1)
        y0 = self.h - y - h
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0); glVertex2f(x, y0)
        glTexCoord2f(1, 0); glVertex2f(x + w, y0)
        glTexCoord2f(1, 1); glVertex2f(x + w, y0 + h)
        glTexCoord2f(0, 1); glVertex2f(x, y0 + h)
        glEnd()
        glDisable(GL_TEXTURE_2D)
        glDeleteTextures([tex])
        return h

    def panel(self, x, y, w, h, alpha=0.45):
        y0 = self.h - y - h
        glColor4f(0, 0, 0, alpha)
        glBegin(GL_QUADS)
        glVertex2f(x, y0); glVertex2f(x + w, y0); glVertex2f(x + w, y0 + h); glVertex2f(x, y0 + h)
        glEnd()

    def bar(self, x, y, w, h, frac, color):
        self.panel(x, y, w, h, 0.5)
        y0 = self.h - y - h
        glColor4f(*color, 0.9)
        glBegin(GL_QUADS)
        glVertex2f(x, y0); glVertex2f(x + w * frac, y0)
        glVertex2f(x + w * frac, y0 + h); glVertex2f(x, y0 + h)
        glEnd()

    def steering_wheel(self, cx, cy, radius, angle_deg, active):
        """Mirrors the physical wheel. cy from top."""
        cy = self.h - cy
        glLineWidth(6)
        col = (1, 1, 1, 0.9) if active else (0.6, 0.6, 0.6, 0.6)
        glColor4f(*col)
        glBegin(GL_LINE_LOOP)
        for i in range(40):
            a = 2 * math.pi * i / 40
            glVertex2f(cx + math.cos(a) * radius, cy + math.sin(a) * radius)
        glEnd()
        a = math.radians(-angle_deg)
        glBegin(GL_LINES)
        for spoke in (0, math.pi, -math.pi / 2):
            glVertex2f(cx, cy)
            glVertex2f(cx + math.cos(a + spoke) * radius, cy + math.sin(a + spoke) * radius)
        glEnd()
        # top-centre marker
        glColor4f(1, 0.3, 0.2, 1)
        glPointSize(10)
        glBegin(GL_POINTS)
        glVertex2f(cx + math.cos(a + math.pi / 2) * radius, cy + math.sin(a + math.pi / 2) * radius)
        glEnd()
        glLineWidth(1)

    def minimap(self, track, car, x, y, size):
        xs = [p[0] for p in track.points]
        zs = [p[1] for p in track.points]
        x0, x1, z0, z1 = min(xs), max(xs), min(zs), max(zs)
        scale = (size - 20) / max(x1 - x0, z1 - z0)
        self.panel(x, y, size, size, 0.35)
        top = self.h - y

        def to_screen(px, pz):
            # mirror x: the car's right is -x
            return (x + size - 10 - (px - x0) * scale, top - size + 10 + (pz - z0) * scale)

        glLineWidth(4)
        glColor4f(0.9, 0.9, 0.9, 0.9)
        glBegin(GL_LINE_LOOP)
        for px, pz in track.points:
            glVertex2f(*to_screen(px, pz))
        glEnd()
        glLineWidth(1)
        glPointSize(9)
        glBegin(GL_POINTS)
        glColor4f(0.1, 0.1, 0.1, 1)
        glVertex2f(*to_screen(*track.points[0]))
        glColor4f(1, 0.2, 0.15, 1)
        glVertex2f(*to_screen(car.x, car.z))
        glEnd()

    def lines(self, pts, color, width=2, loop=False):
        if len(pts) < 2:
            return
        glLineWidth(width)
        glColor4f(*color)
        glBegin(GL_LINE_LOOP if loop else GL_LINE_STRIP)
        for x, y in pts:
            glVertex2f(x, self.h - y)
        glEnd()
        glLineWidth(1)

    def outline(self, x0, y0, x1, y1, color, width=2):
        self.lines(((x0, y0), (x1, y0), (x1, y1), (x0, y1)), color, width, loop=True)

    def circle(self, cx, cy, r, color, width=1, segments=48):
        pts = [(cx + math.cos(a) * r, cy + math.sin(a) * r)
               for a in np.linspace(0, 2 * math.pi, segments, endpoint=False)]
        self.lines(pts, color, width, loop=True)

    def cloud(self, xy, colors, size=2):
        """Points from top-left pixel coords (N, 2) with RGB or RGBA colours."""
        if not len(xy):
            return
        v = np.empty((len(xy), 2), np.float32)
        v[:, 0] = xy[:, 0]
        v[:, 1] = self.h - xy[:, 1]
        c = np.ascontiguousarray(colors, np.float32)
        glPointSize(size)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_COLOR_ARRAY)
        glVertexPointer(2, GL_FLOAT, 0, v)
        glColorPointer(c.shape[1], GL_FLOAT, 0, c)
        glDrawArrays(GL_POINTS, 0, len(v))
        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)

    def clip(self, x, y, w, h):
        glEnable(GL_SCISSOR_TEST)
        glScissor(int(x), int(self.h - y - h), int(w), int(h))

    def unclip(self):
        glDisable(GL_SCISSOR_TEST)
