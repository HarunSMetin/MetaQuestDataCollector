"""Loads a task into a `Bundle`: the four layers (embodiment, scene, task objects, task logic) fully resolved into
plain data. A bundle is stored with every episode (`task_yaml`) and rebuilds the exact scene and task later, without
the library files it came from; its content hash is the `task_version`.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from xrcollect.config import ROOT, rooted
from xrcollect.task.schema import validate_task

LIBRARY = {"embodiment": ROOT / "embodiments", "scene": ROOT / "scenes", "object": ROOT / "objects"}


def fix_keys(node):
    """YAML 1.1 reads the key `on` as the boolean True (`{on: stage_reached}` in the events table): undo that."""
    if isinstance(node, dict):
        return {("on" if k is True else "off" if k is False else k): fix_keys(v) for k, v in node.items()}
    if isinstance(node, list):
        return [fix_keys(v) for v in node]
    return node


def _read_yaml(path: Path) -> dict:
    with open(path) as f:
        return fix_keys(yaml.safe_load(f) or {})


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


def _call(ref: str, base_dir: Path | None, **kwargs):
    """Call `module:function` or `path/to/file.py:function` (relative to base_dir or the repository root)."""
    return load_object(ref, base_dir)(**kwargs)


def load_object(ref: str, base_dir: Path | None = None, source: str | None = None):
    """Resolve `module:name` or `file.py:name`. With `source`, the module is built from that text instead of the
    file (an episode's stored plugin when the file has changed since)."""
    target, name = ref.rsplit(":", 1)
    if target.endswith(".py"):
        path = next((p for p in ((base_dir / target) if base_dir else None, rooted(target)) if p and p.exists()), None)
        mod_name = "xrc_plugin_" + hashlib.sha1((str(path) + (source or "")).encode()).hexdigest()[:12]
        if mod_name in sys.modules:
            return getattr(sys.modules[mod_name], name)
        if source is not None:
            module = type(sys)(mod_name)
            module.__file__ = str(path) if path else f"<stored {target}>"
            sys.modules[mod_name] = module
            exec(compile(source, module.__file__, "exec"), module.__dict__)  # noqa: S102 - the user's own task plugin
            return getattr(module, name)
        if path is None:
            raise FileNotFoundError(f"plugin file {target} not found")
        spec = importlib.util.spec_from_file_location(mod_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
        return getattr(module, name)
    return getattr(importlib.import_module(target), name)


def resolve_object(name: str, entry: dict, base_dir: Path | None = None) -> dict:
    """One task object = its library asset (objects/<asset>.yaml, optionally made by a builder function) with the
    task's overrides on top."""
    entry = dict(entry)
    asset_ref = entry.pop("asset", None)
    asset: dict = {}
    asset_dir = None
    if asset_ref:
        path = next((p for p in ((base_dir / f"{asset_ref}.yaml") if base_dir else None,
                                 LIBRARY["object"] / f"{asset_ref}.yaml") if p and p.exists()), None)
        if path is None:
            raise FileNotFoundError(f"object asset {asset_ref!r} not found in {LIBRARY['object']}")
        asset, asset_dir = _read_yaml(path), path.parent
    out = _merge(asset, entry)
    if out.get("builder"):  # programmatic geometry (e.g. studded bricks): the result is plain data again
        built = _call(out.pop("builder"), asset_dir, **out.pop("builder_args", {}))
        out = _merge(built, out)
    out["asset"] = asset_ref
    out.setdefault("label", name.replace("_", " "))
    # File references become repository-relative, so the bundle can be rebuilt from anywhere.
    for mesh in out.get("meshes", {}).values():
        if mesh.get("file"):
            mesh["file"] = _rel(mesh["file"], asset_dir)
    if out.get("mjcf"):
        out["mjcf"] = _rel(out["mjcf"], asset_dir)
    return out


def _rel(file: str, base_dir: Path | None) -> str:
    p = Path(file)
    if not p.is_absolute():
        p = next((c for c in ((base_dir / p) if base_dir else None, ROOT / p) if c and c.exists()), ROOT / p)
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


@dataclass
class Bundle:
    task: dict
    embodiment: dict
    scene: dict
    objects: dict[str, dict]
    plugin: dict | None = None            # {"ref": "task.py:Class", "source": <file text>}
    path: Path | None = field(default=None, compare=False)   # the task YAML it was loaded from (edits go there)

    @property
    def name(self) -> str:
        return self.task["task"]

    @property
    def control_hz(self) -> int:
        return int(self.scene.get("physics", {}).get("control_hz", 50))

    @property
    def table_height(self) -> float:
        return float(self.scene["table"]["height"])

    def to_dict(self) -> dict:
        return {"task": self.task, "embodiment": self.embodiment, "scene": self.scene, "objects": self.objects,
                "plugin": self.plugin}

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    @classmethod
    def from_dict(cls, d: dict) -> Bundle:
        return cls(d["task"], d["embodiment"], d["scene"], d["objects"], d.get("plugin"))

    @classmethod
    def from_yaml(cls, text: str) -> Bundle:
        return cls.from_dict(yaml.safe_load(text))

    def files(self) -> list[Path]:
        """Binary/XML files the scene needs besides this data (meshes, MJCF fragments)."""
        out = []
        for o in self.objects.values():
            out += [rooted(m["file"]) for m in o.get("meshes", {}).values() if m.get("file")]
            if o.get("mjcf"):
                out.append(rooted(o["mjcf"]))
        return out

    @property
    def version(self) -> str:
        """task_version: content hash of all four layers (and the files they reference)."""
        h = hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, default=str).encode())
        for f in sorted(self.files()):
            if f.exists():
                h.update(f.read_bytes())
        return h.hexdigest()[:16]


def task_file(path: str | Path) -> Path:
    """A task folder (containing task.yaml) or a YAML file, relative to the repository root."""
    p = rooted(path)
    return p / "task.yaml" if p.is_dir() else p


def load_bundle(path: str | Path, overrides: dict[str, Any] | None = None) -> Bundle:
    """Resolve a task file into a Bundle. overrides: {"embodiment": name} etc. replace the task's own choice."""
    file = task_file(path)
    raw = _read_yaml(file)
    raw.update(overrides or {})
    task = validate_task(raw)
    base = file.parent
    embodiment = _read_yaml(LIBRARY["embodiment"] / f"{task['embodiment']}.yaml")
    scene = _read_yaml(LIBRARY["scene"] / f"{task['scene']}.yaml")
    scene.setdefault("cameras", {}).update(task.get("cameras") or {})
    objects = {name: resolve_object(name, entry, base) for name, entry in task["objects"].items()}
    plugin = None
    if task.get("plugin"):
        target = task["plugin"].rsplit(":", 1)[0]
        src = next((p for p in (base / target, rooted(target)) if p.exists()), None)
        if src is None:
            raise FileNotFoundError(f"task plugin {target} not found next to {file}")
        plugin = {"ref": task["plugin"], "source": src.read_text(), "dir": _rel(str(src.parent), None)}
    return Bundle(task, embodiment, scene, objects, plugin, path=file)


def list_tasks() -> list[dict]:
    out = []
    for f in sorted((ROOT / "tasks").glob("*/task.yaml")):
        try:
            raw = _read_yaml(f)
            out.append({"path": str(f.parent.relative_to(ROOT)), "task": raw.get("task", f.parent.name),
                        "instruction": raw.get("instruction", "")})
        except yaml.YAMLError:
            continue
    return out
