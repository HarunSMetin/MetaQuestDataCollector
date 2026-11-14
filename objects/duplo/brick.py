"""Builder for real DUPLO bricks as plain object data (objects/duplo/brick.yaml calls it).

Hollow brick: top plate, four walls, studs with a chamfered convex-mesh top that centres a slightly misaligned
brick, tubes between the studs underneath (2-wide bricks), and an invisible solid core that only the robot hits.
Body frame = bottom centre. `size` is the outline without studs, so stacking height = size z.
"""

import numpy as np

PITCH = 0.016            # stud spacing
BRICK_H = 0.0192         # height of a 1-high brick without studs (DUPLO = 2x LEGO System)
STUD_R = 0.00468         # stud diameter 9.36 mm (estimate: ~2x System's 4.8 mm minus fit)
STUD_H = 0.0045          # matches BrickLink's 42.9 mm overall height for the 1x2x2 brick
STUD_CHAMFER = 0.0015    # 45 degree lead-in on the stud top: guides a brick that is up to ~1.8 mm off into place
PLAY = 0.0003            # stud clearance per side; larger = easier to seat, sloppier fit
TOP_T = 0.002            # top plate thickness
GAP = 0.0001             # brick outline is n * PITCH - 2 * GAP, like real bricks
WALL_T = PITCH / 2 - STUD_R - PLAY - GAP          # walls hug the studs of the brick below
TUBE_R = PITCH / np.sqrt(2) - STUD_R - PLAY       # underside tubes touch the 4 surrounding studs


def _stud_xy(n: int) -> np.ndarray:
    return (np.arange(n) - (n - 1) / 2) * PITCH


def _stud_top(n: int = 16) -> list[float]:
    """Frustum from the stud radius at z=0 to STUD_R - STUD_CHAMFER at z=STUD_CHAMFER (a convex mesh)."""
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = lambda r, z: np.stack([r * np.cos(a), r * np.sin(a), np.full(n, z)], 1)  # noqa: E731
    return np.concatenate([ring(STUD_R, 0.0), ring(STUD_R - STUD_CHAMFER, STUD_CHAMFER)]).ravel().tolist()


def brick(nx: int, ny: int, nz: int = 1) -> dict:
    hx, hy = nx * PITCH / 2 - GAP, ny * PITCH / 2 - GAP
    h = nz * BRICK_H
    hz = (h - TOP_T) / 2
    shaft = (STUD_H - STUD_CHAMFER) / 2
    studs = [(float(x), float(y)) for x in _stud_xy(nx) for y in _stud_xy(ny)]
    geoms = [{"type": "box", "size": [hx, hy, TOP_T / 2], "pos": [0, 0, h - TOP_T / 2]}]
    geoms += [{"type": "box", "size": [WALL_T / 2, hy, hz], "pos": [s * (hx - WALL_T / 2), 0, hz]} for s in (-1, 1)]
    geoms += [{"type": "box", "size": [hx - WALL_T, WALL_T / 2, hz], "pos": [0, s * (hy - WALL_T / 2), hz]}
              for s in (-1, 1)]
    geoms += [{"type": "cylinder", "size": [STUD_R, shaft], "pos": [x, y, h + shaft]} for x, y in studs]
    # Stud tops are placed by their base (z=0 of the vertices); the compiler re-centres the mesh and its geoms.
    geoms += [{"type": "mesh", "mesh": "stud_top", "pos": [x, y, h + 2 * shaft]} for x, y in studs]
    if ny == 2:
        geoms += [{"type": "cylinder", "size": [float(TUBE_R), hz], "pos": [float(x), 0, hz]}
                  for x in _stud_xy(nx - 1)]
    r1, r2 = STUD_R, STUD_R - STUD_CHAMFER
    return {
        "shape": "brick", "size": [2 * hx, 2 * hy, h], "studs": [nx, ny], "pitch": PITCH,
        "stud_r": STUD_R, "stud_h": STUD_H, "friction": 0.5,
        "geoms": geoms,
        "meshes": {"stud_top": {"vertices": _stud_top(),
                                "volume": float(np.pi * STUD_CHAMFER * (r1 * r1 + r1 * r2 + r2 * r2) / 3)}},
        "core": {"size": [hx - WALL_T, hy - WALL_T, hz], "pos": [0, 0, hz]},
    }
