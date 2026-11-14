"""Importing arbitrary meshes as object assets (`xrc import-mesh`).

MuJoCo collides convex shapes only: a concave mesh would be replaced by its convex hull (a mug becomes a solid
cylinder). So the mesh is split into convex pieces with CoACD; those are the (hidden) collision geoms, and the
original mesh, decimated for the headset, is the visual. The asset's body frame is moved to the bottom centre of
the bounding box, the platform's convention.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from xrcollect.config import ROOT

VISUAL_FACES = 1500


def import_mesh(mesh: Path, name: str, scale: float = 1.0, mass: float = 0.1, threshold: float = 0.05,
                max_hulls: int = 16) -> Path:
    try:
        import coacd
        import trimesh
    except ImportError as e:
        raise RuntimeError("mesh import needs coacd and trimesh: uv sync --extra mesh") from e
    import fast_simplification

    tm = trimesh.load(str(mesh), force="mesh")
    tm.apply_scale(scale)
    lo, hi = tm.bounds
    tm.apply_translation([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])  # body frame = bottom centre
    size = (hi - lo).tolist()

    out_dir = (ROOT / "objects" / name).parent
    stem = Path(name).name
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = coacd.run_coacd(coacd.Mesh(np.asarray(tm.vertices), np.asarray(tm.faces)), threshold=threshold,
                            max_convex_hull=max_hulls)
    meshes, geoms, volume = {}, [], 0.0
    v, f = np.asarray(tm.vertices, np.float32), np.asarray(tm.faces, np.int32)
    if len(f) > VISUAL_FACES:
        v, f = fast_simplification.simplify(v, f, target_reduction=1 - VISUAL_FACES / len(f))
    trimesh.Trimesh(v, f).export(out_dir / f"{stem}_visual.obj")
    meshes["visual"] = {"file": f"{stem}_visual.obj"}
    geoms.append({"type": "mesh", "mesh": "visual", "visual": True})
    for i, (pv, pf) in enumerate(parts):
        piece = trimesh.Trimesh(pv, pf)
        piece.export(out_dir / f"{stem}_col{i}.obj")
        vol = float(abs(piece.convex_hull.volume))
        volume += vol
        meshes[f"col{i}"] = {"file": f"{stem}_col{i}.obj", "volume": vol}
        geoms.append({"type": "mesh", "mesh": f"col{i}", "hidden": True})
    asset = {"label": stem.replace("_", " "), "shape": "mesh", "size": [round(s, 4) for s in size], "mass": mass,
             "friction": 0.8, "rgba": [0.75, 0.75, 0.78, 1], "meshes": meshes, "geoms": geoms}
    path = out_dir / f"{stem}.yaml"
    header = (f"# Imported from {Path(mesh).name} by `xrc import-mesh`: {len(parts)} convex collision pieces "
              f"(CoACD, threshold {threshold}), volume {volume * 1e6:.1f} cm^3.\n")
    path.write_text(header + yaml.safe_dump(asset, sort_keys=False))
    return path
