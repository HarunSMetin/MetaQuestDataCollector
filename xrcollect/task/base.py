"""The task interface. The runtime, recorder, audit and UI only know `Task`; a declarative task file compiles into
one (declarative.py), and a Python plugin implements or subclasses it for anything predicates can't express.

Rule for every task: progress, success and stage flags are functions of the simulator state alone, and anything a
task changes at runtime outside MuJoCo's own state (weld poses, armed flags, ...) goes through extra_state /
set_extra_state. Then replay, audit and export can recompute everything from `sim/state`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import mujoco
import numpy as np

from xrcollect.embodiment.base import Bound


@dataclass
class TaskEvent:
    """Something the operator should feel/hear. name: grasp, release, stage_reached, stage_lost, success, snap_on,
    snap_off, region_enter, or anything a plugin emits; mapped to a cue by the task's `events` table."""
    name: str
    object: str | None = None
    stage: str | None = None
    arm: str | None = None
    text: str | None = None


@dataclass
class World:
    """What a task needs to know about a compiled scene."""
    model: mujoco.MjModel
    bound: Bound
    table_height: float
    objects: list[str]                       # free objects, in the order of obs/objects
    body: dict[str, int]                     # object / fixture name -> body id
    size: dict[str, np.ndarray]              # object name -> bounding box [x, y, z] (body frame at bottom centre)
    regions: dict[str, dict]                 # name -> {center, half}
    qadr: dict[str, int] = field(default_factory=dict)   # free object -> qpos address of its free joint
    vadr: dict[str, int] = field(default_factory=dict)   # free object -> dof address
    surface_bodies: list[int] = field(default_factory=list)  # table and other static support bodies
    fixture_joints: list[str] = field(default_factory=list)  # scalar joints of fixtures (drawers, doors)

    def body_of(self, name: str) -> int:
        """Body id for a predicate argument: an object, `table`, `arm:<arm>` is not a single body (see contact)."""
        if name in self.body:
            return self.body[name]
        return self.model.body(name).id


class Task(Protocol):
    name: str
    instruction: str
    stage_names: list[str]

    def build(self, spec: mujoco.MjSpec, bundle) -> None: ...            # add welds, sites, extra geoms
    def bind(self, world: World) -> None: ...                            # after compile: look up ids
    def reset(self, model, data, rng: np.random.Generator) -> None: ...  # start-state randomization
    def stage_flags(self, model, data) -> np.ndarray: ...                # (n_stages,) bool
    def progress(self, model, data) -> int: ...                          # stages done (need not be monotonic)
    def success(self, model, data) -> bool: ...
    def after_step(self, model, data) -> list[TaskEvent]: ...            # e.g. snap constraints
    def held(self, model, data) -> dict[str, str | None]: ...            # arm -> object
    def next_object(self, model, data) -> str | None: ...                # what the operator should handle next
    def goal_spec(self) -> dict: ...                                     # served at /api/task, drawn by the UI
    def extra_state(self, model) -> np.ndarray: ...                      # state outside mj_getState
    def set_extra_state(self, model, x: np.ndarray) -> None: ...
    def task_fields(self, model, data) -> dict: ...                      # extra per-step arrays for /task/*


def default_held(world: World, data: mujoco.MjData) -> dict[str, str | None]:
    """Per arm, the object every finger of its gripper touches (None if none): the default `held`."""
    m, b = world.model, world.bound
    obj_of = {world.body[o]: i for i, o in enumerate(world.objects)}
    finger_of = {bid: (a, f) for a, fingers in enumerate(b.finger_bodies) for f, bid in enumerate(fingers)}
    touch: dict[tuple[int, int], set[int]] = {k: set() for k in finger_of.values()}
    gb = m.geom_bodyid
    for g1, g2 in zip(data.contact.geom1, data.contact.geom2):
        b1, b2 = int(gb[g1]), int(gb[g2])
        for f, o in ((b1, b2), (b2, b1)):
            if f in finger_of and o in obj_of:
                touch[finger_of[f]].add(obj_of[o])
    out: dict[str, str | None] = {}
    for a, arm in enumerate(b.arms):
        sets = [touch[(a, f)] for f in range(len(b.finger_bodies[a]))]
        both = set.intersection(*sets) if sets else set()
        out[arm] = world.objects[min(both)] if both else None
    return out
