"""A task file compiled into a `Task`. Python plugins subclass DeclarativeTask and override what they need, so
regions, spawn distributions, goal view and snap bookkeeping come for free.
"""

from __future__ import annotations

import mujoco
import numpy as np

from xrcollect.scene.constants import yaw_quat
from xrcollect.task.base import TaskEvent, World, default_held
from xrcollect.task.loader import Bundle, load_object
from xrcollect.task.predicates import compile_predicate, predicate_object
from xrcollect.task.snap import SnapPair, SnapSet


class DeclarativeTask:
    def __init__(self, bundle: Bundle):
        self.bundle = bundle
        t = self.spec = bundle.task
        self.name = t["task"]
        self.instruction = t.get("instruction", "")
        self.stages = t.get("stages", [])
        self.stage_names = [s["name"] for s in self.stages]
        self._stage_preds = [compile_predicate(s["when"]) for s in self.stages]
        self._stage_objects = [s.get("object") or predicate_object(s["when"]) for s in self.stages]
        self._success = compile_predicate(t["success"]) if t.get("success") else None
        self.world: World | None = None
        self.snaps = SnapSet(self.snap_pairs())

    # -- overridable pieces ----------------------------------------------------------------------------
    def snap_pairs(self) -> list[SnapPair]:
        """Snap rules from the task file: aligned = the rule's `when` predicate."""
        pairs = []
        for rule in self.spec.get("snaps", []):
            pred = compile_predicate(rule["when"])
            pairs.append(SnapPair(rule["a"], rule["b"], lambda d, lo, up, pred=pred: pred(self.world, d, {}),
                                  rule["press_n"], rule["pull_n"], tuple(rule["solref"])))
        return pairs

    # -- Task protocol ---------------------------------------------------------------------------------
    def build(self, spec: mujoco.MjSpec, bundle: Bundle) -> None:
        self.snaps.build(spec)

    def bind(self, world: World) -> None:
        self.world = world
        self.snaps.bind(world.model)

    def reset(self, model, data, rng: np.random.Generator) -> None:
        """Each object with a `spawn` entry starts uniformly in its rectangle with a uniform yaw; the others stay
        where the scene put them."""
        w = self.world
        for name, sp in self.spec.get("spawn", {}).items():
            if name not in w.qadr:
                continue
            region = w.regions[sp["region"]] if isinstance(sp["region"], str) else sp["region"]
            c, h = np.asarray(region["center"], float), np.asarray(region["half"], float)
            xy = c + rng.uniform(-h, h)
            yaw = np.deg2rad(rng.uniform(*sp.get("yaw_deg", [0.0, 0.0])))
            z = w.table_height if sp.get("z") is None else float(sp["z"])
            a, v = w.qadr[name], w.vadr[name]
            data.qpos[a:a + 3] = [*xy, z]
            data.qpos[a + 3:a + 7] = yaw_quat(yaw)
            data.qvel[v:v + 6] = 0
        self.snaps.reset()

    def stage_flags(self, model, data) -> np.ndarray:
        done: dict[str, bool] = {}
        for name, pred in zip(self.stage_names, self._stage_preds):
            done[name] = bool(pred(self.world, data, done))
        return np.array([done[n] for n in self.stage_names], bool)

    def progress(self, model, data) -> int:
        """Stages done, counted in order: the first stage that doesn't hold ends the count."""
        flags = self.stage_flags(model, data)
        return int(len(flags) if flags.all() else np.argmin(flags))

    def success(self, model, data) -> bool:
        w = self.world
        if self._success is not None:
            flags = self.stage_flags(model, data)
            return bool(self._success(w, data, dict(zip(self.stage_names, flags.tolist()))))
        # Default: every stage done, nothing held, everything at rest.
        if self.stage_names and self.progress(model, data) < len(self.stage_names):
            return False
        if any(o is not None for o in self.held(model, data).values()):
            return False
        return all(np.linalg.norm(data.qvel[w.vadr[o]:w.vadr[o] + 3]) < 0.02 for o in w.objects)

    def after_step(self, model, data) -> list[TaskEvent]:
        return self.snaps.update(model, data)

    def held(self, model, data) -> dict[str, str | None]:
        return default_held(self.world, data)

    def next_object(self, model, data) -> str | None:
        flags = self.stage_flags(model, data)
        for done, obj in zip(flags, self._stage_objects):
            if not done:
                return obj
        return None

    def goal_view(self) -> dict | None:
        """{anchor, show: [{object, pose: {pos, yaw}}]}: where each object belongs relative to the anchor object."""
        return self.spec.get("goal_view")

    def next_goal(self, model, data) -> dict | None:
        """World pose where the next object belongs ({object, pos, quat wxyz}), if the goal view says so and its
        anchor is in place: drawn as an operator-only ghost with a distance readout."""
        gv, obj = self.goal_view(), self.next_object(model, data)
        if not gv or obj is None or obj == gv["anchor"] or gv["anchor"] not in self.world.body:
            return None
        item = next((i for i in gv["show"] if i["object"] == obj), None)
        if item is None:
            return None
        b = self.world.body[gv["anchor"]]
        R = data.xmat[b].reshape(3, 3)
        pos = data.xpos[b] + R @ np.asarray(item["pose"].get("pos", [0, 0, 0]), float)
        q = np.empty(4)
        mujoco.mju_mulQuat(q, data.xquat[b], np.array(yaw_quat(float(item["pose"].get("yaw", 0.0)))))
        return {"object": obj, "pos": pos.tolist(), "quat": q.tolist()}

    def goal_spec(self) -> dict:
        """The mission as data, for the operator UI: objects (size, colour, label), regions, ordered stages and the
        goal view. Everything the UI shows of the goal comes from here."""
        b = self.bundle
        objects = []
        for name, o in b.objects.items():
            objects.append({"name": name, "label": o.get("label", name), "size": list(o.get("size", [0.04] * 3)),
                            "rgba": list(o.get("rgba", [0.7, 0.7, 0.7, 1])), "shape": o.get("shape", "box"),
                            "studs": o.get("studs"), "fixed": bool(o.get("fixed") or o.get("mjcf")),
                            "pitch": o.get("pitch"), "stud_r": o.get("stud_r"), "stud_h": o.get("stud_h")})
        return {
            "task": self.name, "version": b.version, "instruction": self.instruction,
            "objects": objects,
            "regions": self.spec.get("regions", {}),
            "spawn": {k: (self.spec["regions"][v["region"]] if isinstance(v["region"], str) else v["region"])
                      for k, v in self.spec.get("spawn", {}).items()},
            "stages": [{"name": s["name"], "label": s.get("label") or s["name"].replace("_", " "), "object": o}
                       for s, o in zip(self.stages, self._stage_objects)],
            "goal_view": self.goal_view(),
            "tags": self.spec.get("tags", []),
            "arms": list(b.embodiment["arms"]),
            "table_height": b.table_height,
        }

    def extra_state(self, model) -> np.ndarray:
        return self.snaps.extra_state(model)

    def set_extra_state(self, model, x: np.ndarray) -> None:
        self.snaps.set_extra_state(model, x)

    def task_fields(self, model, data) -> dict:
        """Extra per-step arrays recorded under /task/."""
        return {"snaps": self.snaps.active(data)} if self.snaps.pairs else {}


def make_task(bundle: Bundle) -> DeclarativeTask:
    """The Task of a bundle: its plugin class, or the declarative task. The plugin is loaded from its file while
    that still has the stored text, else from the text stored in the bundle (an old episode's logic)."""
    if not bundle.plugin:
        return DeclarativeTask(bundle)
    from xrcollect.config import rooted

    ref, source = bundle.plugin["ref"], bundle.plugin["source"]
    base = rooted(bundle.plugin.get("dir", "."))
    path = base / ref.rsplit(":", 1)[0]
    current = path.read_text() if path.exists() else None
    cls = load_object(ref, base, source=None if current == source else source)
    return cls(bundle)
