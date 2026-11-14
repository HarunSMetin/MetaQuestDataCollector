"""The built-in predicate library for declarative tasks.

A predicate in a task file is a one-key dict, `{name: args}`; `compile_predicate` turns it into a function
`(world, data, stages) -> bool`, where `stages` maps the names of earlier stages to their value this step. Every
predicate reads the simulator state only, so replay and audit can recompute it.

`CATALOGUE` describes each predicate's arguments for the task editor.
"""

from __future__ import annotations

import copy
from typing import Any, Callable

import mujoco
import numpy as np

from xrcollect.scene.constants import wrap
from xrcollect.task.base import World, default_held

Pred = Callable[[World, mujoco.MjData, dict], bool]

CATALOGUE: dict[str, dict] = {
    "inside": {"doc": "The object's whole footprint lies inside a region.",
               "args": {"object": "object", "region": "region", "margin": 0.0}},
    "on_top": {"doc": "The object rests on top of the base object, centred within tol_xy.",
               "args": {"object": "object", "base": "object", "tol_xy": 0.01, "tol_z": 0.004, "offset": [0.0, 0.0],
                        "max_tilt_deg": 10.0}},
    "pose_near": {"doc": "The object is at a world pose.",
                  "args": {"object": "object", "pos": [0.0, 0.0, 0.0], "yaw_deg": None, "tol_pos": 0.01,
                           "tol_rot_deg": 10.0, "yaw_period_deg": 360.0}},
    "relative_pose": {"doc": "The object is at a pose in the base object's frame.",
                      "args": {"object": "object", "base": "object", "pos": [0.0, 0.0, 0.0], "yaw_deg": 0.0,
                               "tol_pos": 0.005, "tol_rot_deg": 8.0, "yaw_period_deg": 360.0}},
    "contact": {"doc": "Two bodies touch (objects, `table`, `arm:<arm>` or `gripper:<arm>`).",
                "args": {"a": "body", "b": "body"}},
    "held": {"doc": "The object is held in a gripper (by: a given arm).", "args": {"object": "object", "by": None}},
    "not_held": {"doc": "The object (or `any` object) is not held.", "args": "any | object | [objects]"},
    "at_rest": {"doc": "The objects move slower than max_speed.", "args": {"objects": "all", "max_speed": 0.02}},
    "joint_at": {"doc": "A fixture joint (drawer, door) is at a value, or beyond min / max.",
                 "args": {"joint": "joint", "value": None, "tol": 0.01, "min": None, "max": None}},
    "upright": {"doc": "The object's z axis points up.", "args": {"object": "object", "max_tilt_deg": 10.0}},
    "snapped": {"doc": "The snap (press-fit) between two objects is engaged.", "args": {"a": "object", "b": "object"}},
    "count": {"doc": "At least `min` (and at most `max`) of the objects satisfy `when`, where `$` is the object.",
              "args": {"objects": ["object"], "when": "predicate", "min": 1, "max": None}},
    "all": {"doc": "Every listed predicate holds.", "args": ["predicate"]},
    "any": {"doc": "At least one listed predicate holds.", "args": ["predicate"]},
    "not": {"doc": "The predicate does not hold.", "args": "predicate"},
    "stage": {"doc": "An earlier stage holds.", "args": "stage name"},
}


def _pose(w: World, d: mujoco.MjData, name: str) -> tuple[np.ndarray, np.ndarray]:
    b = w.body_of(name)
    return d.xpos[b], d.xmat[b].reshape(3, 3)


def _yaw(R: np.ndarray) -> float:
    return float(np.arctan2(R[1, 0], R[0, 0]))


def _tilt_ok(R: np.ndarray, max_deg: float) -> bool:
    return bool(R[2, 2] >= np.cos(np.deg2rad(max_deg)))


def _region(w: World, ref) -> dict:
    return w.regions[ref] if isinstance(ref, str) else ref


def _bodies(w: World, name: str) -> np.ndarray:
    """Body ids for a contact argument."""
    b = w.bound
    if name.startswith(("arm:", "gripper:")):
        kind, arm = name.split(":", 1)
        i = b.arms.index(arm) + 1
        mask = (b.body_arm == i) & (b.body_tip if kind == "gripper" else True)
        return np.flatnonzero(mask)
    if name == "table":
        return np.asarray(w.surface_bodies)
    return np.array([w.body_of(name)])


def footprint_inside(w: World, d: mujoco.MjData, obj: str, region: dict, margin: float = 0.0) -> bool:
    p, R = _pose(w, d, obj)
    hx, hy = w.size[obj][0] / 2, w.size[obj][1] / 2
    corners = p[:2] + np.array([[sx * hx, sy * hy] for sx in (-1, 1) for sy in (-1, 1)]) @ R[:2, :2].T
    return bool(np.all(np.abs(corners - np.asarray(region["center"])) <= np.asarray(region["half"]) - margin))


# -- predicates: each returns a Pred from its (validated) arguments ------------------------------------
def _inside(a) -> Pred:
    obj, region, margin = a["object"], a["region"], float(a.get("margin", 0.0))
    return lambda w, d, s: footprint_inside(w, d, obj, _region(w, region), margin)


def _on_top(a) -> Pred:
    obj, base = a["object"], a["base"]
    tol_xy, tol_z = float(a.get("tol_xy", 0.01)), float(a.get("tol_z", 0.004))
    off, tilt = np.asarray(a.get("offset", [0.0, 0.0]), float), float(a.get("max_tilt_deg", 10.0))

    def f(w, d, s):
        (p, R), (pb, Rb) = _pose(w, d, obj), _pose(w, d, base)
        rel = Rb.T @ (p - pb)
        return bool(np.linalg.norm(rel[:2] - off) <= tol_xy and abs(rel[2] - w.size[base][2]) <= tol_z
                    and _tilt_ok(Rb.T @ R, tilt) and _tilt_ok(Rb, 45.0))
    return f


def _rot_ok(Rrel: np.ndarray, yaw: float | None, tol_deg: float, period_deg: float) -> bool:
    if not _tilt_ok(Rrel, tol_deg):
        return False
    if yaw is None:
        return True
    return abs(wrap(_yaw(Rrel) - yaw, np.deg2rad(period_deg))) <= np.deg2rad(tol_deg)


def _yaw_arg(a) -> float | None:
    if a.get("yaw") is not None:
        return float(a["yaw"])
    return None if a.get("yaw_deg") is None else float(np.deg2rad(a["yaw_deg"]))


def _pose_near(a) -> Pred:
    obj, pos, yaw = a["object"], np.asarray(a["pos"], float), _yaw_arg(a)
    tol_pos, tol_rot = float(a.get("tol_pos", 0.01)), float(a.get("tol_rot_deg", 10.0))
    period = float(a.get("yaw_period_deg", 360.0))

    def f(w, d, s):
        p, R = _pose(w, d, obj)
        return bool(np.linalg.norm(p - pos) <= tol_pos) and _rot_ok(R, yaw, tol_rot, period)
    return f


def _relative_pose(a) -> Pred:
    obj, base, pos = a["object"], a["base"], np.asarray(a.get("pos", [0, 0, 0]), float)
    yaw = _yaw_arg({"yaw_deg": 0.0, **a})
    tol_pos, tol_rot = float(a.get("tol_pos", 0.005)), float(a.get("tol_rot_deg", 8.0))
    period = float(a.get("yaw_period_deg", 360.0))

    def f(w, d, s):
        (p, R), (pb, Rb) = _pose(w, d, obj), _pose(w, d, base)
        return bool(np.linalg.norm(Rb.T @ (p - pb) - pos) <= tol_pos) and _rot_ok(Rb.T @ R, yaw, tol_rot, period)
    return f


def touching(w: World, d: mujoco.MjData, a: str, b: str) -> bool:
    A, B = _bodies(w, a), _bodies(w, b)
    gb = w.model.geom_bodyid
    b1, b2 = gb[d.contact.geom1], gb[d.contact.geom2]
    return bool(((np.isin(b1, A) & np.isin(b2, B)) | (np.isin(b1, B) & np.isin(b2, A))).any())


def _contact(a) -> Pred:
    return lambda w, d, s: touching(w, d, a["a"], a["b"])


def _held(a) -> Pred:
    if isinstance(a, str):
        a = {"object": a}
    obj, by = a["object"], a.get("by")
    return lambda w, d, s: any(o == obj and (by is None or arm == by) for arm, o in default_held(w, d).items())


def _not_held(a) -> Pred:
    names = None if a in ("any", "all", None) else ([a] if isinstance(a, str) else list(a))
    return lambda w, d, s: not any(o is not None and (names is None or o in names)
                                   for o in default_held(w, d).values())


def _at_rest(a) -> Pred:
    if not isinstance(a, dict):
        a = {"objects": a}
    which, vmax = a.get("objects", "all"), float(a.get("max_speed", 0.02))

    def f(w, d, s):
        names = w.objects if which in ("all", None) else ([which] if isinstance(which, str) else which)
        return all(np.linalg.norm(d.qvel[w.vadr[o]:w.vadr[o] + 3]) < vmax for o in names)
    return f


def _joint_at(a) -> Pred:
    joint = a["joint"]

    def f(w, d, s):
        q = float(d.joint(joint).qpos[0])
        if a.get("value") is not None and abs(q - float(a["value"])) > float(a.get("tol", 0.01)):
            return False
        if a.get("min") is not None and q < float(a["min"]):
            return False
        return not (a.get("max") is not None and q > float(a["max"]))
    return f


def _upright(a) -> Pred:
    if isinstance(a, str):
        a = {"object": a}
    obj, tilt = a["object"], float(a.get("max_tilt_deg", 10.0))
    return lambda w, d, s: _tilt_ok(_pose(w, d, obj)[1], tilt)


def _snapped(a) -> Pred:
    names = (f"snap__{a['a']}__{a['b']}", f"snap__{a['b']}__{a['a']}")

    def f(w, d, s):
        for n in names:
            eq = mujoco.mj_name2id(w.model, mujoco.mjtObj.mjOBJ_EQUALITY, n)
            if eq >= 0:
                return bool(d.eq_active[eq])
        return False
    return f


def _substitute(node, value: str):
    if isinstance(node, dict):
        return {k: _substitute(v, value) for k, v in node.items()}
    if isinstance(node, list):
        return [_substitute(v, value) for v in node]
    return value if node == "$" else node


def _count(a) -> Pred:
    preds = [compile_predicate(_substitute(copy.deepcopy(a["when"]), o)) for o in a["objects"]]
    lo, hi = int(a.get("min", 1)), a.get("max")

    def f(w, d, s):
        n = sum(p(w, d, s) for p in preds)
        return n >= lo and (hi is None or n <= int(hi))
    return f


def _all(a) -> Pred:
    preds = [compile_predicate(p) for p in a]
    return lambda w, d, s: all(p(w, d, s) for p in preds)


def _any(a) -> Pred:
    preds = [compile_predicate(p) for p in a]
    return lambda w, d, s: any(p(w, d, s) for p in preds)


def _not(a) -> Pred:
    p = compile_predicate(a)
    return lambda w, d, s: not p(w, d, s)


def _stage(a) -> Pred:
    name = a if isinstance(a, str) else a["name"]
    return lambda w, d, s: bool(s.get(name, False))


_BUILDERS: dict[str, Callable[[Any], Pred]] = {
    "inside": _inside, "on_top": _on_top, "pose_near": _pose_near, "relative_pose": _relative_pose,
    "contact": _contact, "held": _held, "not_held": _not_held, "at_rest": _at_rest, "joint_at": _joint_at,
    "upright": _upright, "snapped": _snapped, "count": _count, "all": _all, "any": _any, "not": _not, "stage": _stage,
}
assert set(_BUILDERS) == set(CATALOGUE)


def compile_predicate(node: dict) -> Pred:
    if not isinstance(node, dict) or len(node) != 1:
        raise ValueError(f"a predicate is a dict with exactly one key, got {node!r}")
    (name, args), = node.items()
    if name not in _BUILDERS:
        raise ValueError(f"unknown predicate {name!r}; known: {', '.join(sorted(_BUILDERS))}")
    return _BUILDERS[name](args)


def predicate_object(node) -> str | None:
    """The object a predicate is mainly about (its first `object` argument), for the next-object hint."""
    if isinstance(node, dict):
        if isinstance(node.get("object"), str):
            return node["object"]
        for v in node.values():
            found = predicate_object(v)
            if found:
                return found
    elif isinstance(node, list):
        for v in node:
            found = predicate_object(v)
            if found:
                return found
    return None
