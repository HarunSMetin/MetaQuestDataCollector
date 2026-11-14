"""Scene and task editing: every edit, from the dashboard or from VR, is a list of small operations applied to the
task YAML file (comments and layout preserved with ruamel.yaml). The runtime then hot-reloads the scene.

An edit changes the task version (the content hash stored in every episode); episodes recorded before stay
replayable because they carry their own model.

Operations (dicts with `op`):
  object_pose    {object, pos: [x, y(, z)], yaw_deg}       default pose; a spawn region of the object moves along
  object_set     {object, fields: {mass, rgba, friction, label, asset, builder_args, ...}}
  object_add     {name, asset, pos, fields}
  object_remove  {object}                                   also drops its spawn entry
  region_set     {name, center, half(, tape)}               region_remove {name}
  spawn_set      {object, center, half, yaw_deg}            spawn_remove {object}
  camera_set     {name, pos, lookat, fovy}                  camera_remove {name}
  stages_set     {stages: [{name, when, label, object}]}
  success_set    {success: predicate | null}
  capture_goal   {anchor, poses: {object: {pos, yaw_deg}}, tol_pos, tol_rot_deg}   stages + goal_view from poses
  set            {path: [key, ...], value}                  anything else (null value deletes)
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

from xrcollect.task.loader import fix_keys
from xrcollect.task.schema import validate_task


def _yaml() -> YAML:
    y = YAML()
    y.preserve_quotes = True
    y.width = 120
    y.indent(mapping=2, sequence=4, offset=2)
    return y


def _flow(value):
    """Short lists and small dicts stay on one line, like hand-written task files."""
    if isinstance(value, dict):
        m = CommentedMap((k, _flow(v)) for k, v in value.items())
        if all(not isinstance(v, (dict, list)) or (isinstance(v, list) and len(v) <= 4
                                                    and not any(isinstance(i, (dict, list)) for i in v))
               for v in value.values()):
            m.fa.set_flow_style()
        return m
    if isinstance(value, (list, tuple)):
        s = CommentedSeq(_flow(v) for v in value)
        if not any(isinstance(v, (dict, list, tuple)) for v in value):
            s.fa.set_flow_style()
        return s
    if isinstance(value, (np.floating, float)):
        return round(float(value), 5)
    if isinstance(value, np.integer):
        return int(value)
    return value


def _section(doc, key: str) -> CommentedMap:
    if doc.get(key) is None:
        doc[key] = CommentedMap()
    return doc[key]


def _apply_op(doc, op: dict) -> None:
    kind = op["op"]
    if kind == "object_pose":
        obj = _section(doc, "objects")[op["object"]]
        old = list(obj.get("pos", []))
        obj["pos"] = _flow([float(v) for v in op["pos"]])
        if "yaw_deg" in op:
            obj["yaw_deg"] = round(float(op["yaw_deg"]), 2)
        sp = (doc.get("spawn") or {}).get(op["object"])
        if sp is not None and isinstance(sp.get("region"), dict):  # the start region follows the object
            sp["region"]["center"] = _flow([float(op["pos"][0]), float(op["pos"][1])])
        del old
    elif kind == "object_set":
        obj = _section(doc, "objects")[op["object"]]
        for k, v in op["fields"].items():
            if v is None:
                obj.pop(k, None)
            else:
                obj[k] = _flow(v)
    elif kind == "object_add":
        entry = {"asset": op["asset"], "pos": [float(v) for v in op.get("pos", [0.45, 0.0])], **op.get("fields", {})}
        _section(doc, "objects")[op["name"]] = _flow(entry)
    elif kind == "object_remove":
        _section(doc, "objects").pop(op["object"], None)
        (doc.get("spawn") or {}).pop(op["object"], None)
    elif kind == "region_set":
        r = {"center": list(map(float, op["center"])), "half": list(map(float, op["half"]))}
        if "tape" in op:
            r["tape"] = bool(op["tape"])
        _section(doc, "regions")[op["name"]] = _flow(r)
    elif kind == "region_remove":
        _section(doc, "regions").pop(op["name"], None)
    elif kind == "spawn_set":
        _section(doc, "spawn")[op["object"]] = _flow({
            "region": {"center": list(map(float, op["center"])), "half": list(map(float, op["half"]))},
            "yaw_deg": list(map(float, op.get("yaw_deg", [0, 0])))})
    elif kind == "spawn_remove":
        _section(doc, "spawn").pop(op["object"], None)
    elif kind == "camera_set":
        _section(doc, "cameras")[op["name"]] = _flow({"pos": list(map(float, op["pos"])),
                                                      "lookat": list(map(float, op["lookat"])),
                                                      "fovy": float(op.get("fovy", 60))})
    elif kind == "camera_remove":
        _section(doc, "cameras").pop(op["name"], None)
    elif kind == "stages_set":
        doc["stages"] = CommentedSeq(_stage(s) for s in op["stages"])
    elif kind == "success_set":
        if op.get("success") is None:
            doc.pop("success", None)
        else:
            doc["success"] = _flow(op["success"])
    elif kind == "capture_goal":
        _capture_goal(doc, op)
    elif kind == "set":
        node = doc
        for k in op["path"][:-1]:
            if node.get(k) is None:
                node[k] = CommentedMap()
            node = node[k]
        if op.get("value") is None:
            node.pop(op["path"][-1], None)
        else:
            node[op["path"][-1]] = _flow(op["value"])
    else:
        raise ValueError(f"unknown edit operation {kind!r}")


def _stage(s: dict) -> CommentedMap:
    m = CommentedMap()
    for k in ("name", "label", "object"):
        if s.get(k):
            m[k] = s[k]
    m["when"] = _flow(s["when"])
    return m


def _capture_goal(doc, op: dict) -> None:
    """The goal as the operator arranged it: one `relative_pose` stage per object (pose in the anchor's frame,
    editable tolerances), in the order given, each requiring the one before; and the goal view the panel draws."""
    anchor, poses = op["anchor"], op["poses"]
    tol_pos, tol_rot = float(op.get("tol_pos", 0.01)), float(op.get("tol_rot_deg", 10.0))
    stages, show, prev = [], [], None
    for obj, pose in poses.items():
        pos = [round(float(v), 4) for v in pose["pos"]]
        yaw_deg = round(float(pose.get("yaw_deg", 0.0)), 1)
        rel = {"relative_pose": {"object": obj, "base": anchor, "pos": pos, "yaw_deg": yaw_deg, "tol_pos": tol_pos,
                                 "tol_rot_deg": tol_rot}}
        name = f"{obj}_placed"
        when = rel if prev is None else {"all": [{"stage": prev}, rel]}
        stages.append({"name": name, "object": obj, "when": when})
        show.append({"object": obj, "pose": {"pos": pos, "yaw": round(float(np.deg2rad(yaw_deg)), 4)}})
        prev = name
    keep = [s for s in (doc.get("stages") or []) if not str(s.get("name", "")).endswith("_placed")]
    doc["stages"] = CommentedSeq([*keep, *(_stage(s) for s in stages)])
    doc.pop("success", None)  # default: every stage done, nothing held, all at rest
    gv = CommentedMap()
    gv["anchor"] = anchor
    gv["show"] = CommentedSeq(_flow(s) for s in show)
    doc["goal_view"] = gv


def apply_edits(task_path: Path, ops: list[dict]) -> str:
    """Apply the operations to the task file. The result must still be a valid task; the previous text is returned
    so the caller can put it back if the scene fails to build."""
    task_path = Path(task_path)
    before = task_path.read_text()
    y = _yaml()
    doc = y.load(before)
    for op in ops:
        _apply_op(doc, op)
    buf = io.StringIO()
    y.dump(doc, buf)
    text = buf.getvalue()
    import yaml as pyyaml

    validate_task(fix_keys(pyyaml.safe_load(text)))
    task_path.write_text(text)
    return before
