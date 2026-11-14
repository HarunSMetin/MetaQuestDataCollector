"""Snap / press-fit as an explicit constraint (LEGO-like clicks, lids, plugs).

Pure contact physics can't hold a press fit: MuJoCo bodies are rigid. So each pair of parts that can snap has one
(initially off) weld. It switches on, at the parts' current relative pose, when the pair touches, the pair's
alignment test passes and the contact force pressing them together exceeds `press_n`; and off when the weld has to
hold them together with more than `pull_n` (tension along the lower part's up axis plus shear; compression is the
geometry's job and doesn't count, otherwise pressing hard on a snapped part would pop it off). Peeling needs no
extra rule: the edge the part pivots on and the weld form a couple, so the moment shows up as weld force.

The weld must stay softer than the contacts (`solref`), or it takes the press and shear load and pops while pressing.

State: which welds are on is part of MuJoCo's state, but their relative poses live in `model.eq_data` and the
"armed" flags here; both go through extra_state / set_extra_state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import mujoco
import numpy as np

from xrcollect.task.base import TaskEvent


@dataclass
class SnapPair:
    a: str
    b: str
    # (data, lower object, upper object) -> may these two snap in their current relative pose?
    aligned: Callable[[mujoco.MjData, str, str], bool]
    press_n: float = 6.0
    pull_n: float = 10.0
    solref: tuple[float, float] = (0.01, 1.0)

    @property
    def eq_name(self) -> str:
        return f"snap__{self.a}__{self.b}"


class SnapSet:
    def __init__(self, pairs: list[SnapPair]):
        self.pairs = pairs
        self.eqs = np.zeros(0, int)
        self.armed = np.ones(len(pairs), bool)   # False after an unsnap until the pair has separated
        self._index: dict[tuple[int, int], int] = {}
        self._bodies: list[tuple[int, int]] = []

    def build(self, spec: mujoco.MjSpec) -> None:
        for p in self.pairs:
            spec.add_equality(name=p.eq_name, type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                              name1=p.a, name2=p.b, active=False, solref=list(p.solref),
                              data=[0] * 10 + [1] + [0] * (mujoco.mjNEQDATA - 11))  # anchor, relpose, torquescale

    def bind(self, model: mujoco.MjModel) -> None:
        self.eqs = np.array([model.equality(p.eq_name).id for p in self.pairs], int)
        self._bodies = [(model.body(p.a).id, model.body(p.b).id) for p in self.pairs]
        self._index = {(min(a, b), max(a, b)): k for k, (a, b) in enumerate(self._bodies)}

    def reset(self) -> None:
        self.armed[:] = True

    # -- state outside mj_getState ---------------------------------------------------------------------
    def extra_state(self, model: mujoco.MjModel) -> np.ndarray:
        if not self.pairs:
            return np.zeros(0)
        return np.concatenate([model.eq_data[self.eqs, 3:10].ravel(), self.armed.astype(float)])

    def set_extra_state(self, model: mujoco.MjModel, x: np.ndarray) -> None:
        n = len(self.pairs)
        if n:
            model.eq_data[self.eqs, 3:10] = np.asarray(x[:7 * n]).reshape(n, 7)
            self.armed[:] = np.asarray(x[7 * n:8 * n]) > 0.5

    def active(self, data: mujoco.MjData) -> np.ndarray:
        return data.eq_active[self.eqs].astype(bool) if len(self.eqs) else np.zeros(0, bool)

    def is_on(self, data: mujoco.MjData, a: str, b: str) -> bool:
        for k, p in enumerate(self.pairs):
            if {p.a, p.b} == {a, b}:
                return bool(data.eq_active[self.eqs[k]])
        return False

    # -- per control step ------------------------------------------------------------------------------
    def _press(self, model: mujoco.MjModel, data: mujoco.MjData) -> dict[int, float]:
        """Vertical contact force between each touching pair (pairs not in contact are absent)."""
        out: dict[int, float] = {}
        f6 = np.zeros(6)
        gb = model.geom_bodyid
        for c in range(data.ncon):
            con = data.contact[c]
            b1, b2 = int(gb[con.geom1]), int(gb[con.geom2])
            k = self._index.get((min(b1, b2), max(b1, b2)))
            if k is None:
                continue
            mujoco.mj_contactForce(model, data, c, f6)
            out[k] = out.get(k, 0.0) + f6[0] * abs(con.frame[2])  # normal force * the normal's z share
        return out

    def _lower_upper(self, data: mujoco.MjData, k: int) -> tuple[int, int]:
        """(index in the pair of the lower part, of the upper part): 0 = a, 1 = b."""
        a, b = self._bodies[k]
        return (0, 1) if data.xpos[b][2] > data.xpos[a][2] else (1, 0)

    def pull(self, data: mujoco.MjData, k: int) -> float:
        """Force (N) with which weld k holds its parts together: tension plus shear.

        The weld's 3 translational rows carry a world-frame force that points along the lower part's up axis when
        body b is pulled away (and the upper part is b)."""
        rows = np.flatnonzero((data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (data.efc_id == self.eqs[k]))
        f = data.efc_force[rows[:3]]
        lower, upper = self._lower_upper(data, k)
        up = data.xmat[self._bodies[k][lower]].reshape(3, 3)[:, 2]
        axial = f @ up
        tension = max(axial if upper == 1 else -axial, 0.0)
        return float(np.hypot(tension, np.linalg.norm(f - axial * up)))

    def update(self, model: mujoco.MjModel, data: mujoco.MjData) -> list[TaskEvent]:
        """Snap parts together when pressed in an aligned pose; pull (or peel) them apart past the limit."""
        events: list[TaskEvent] = []
        if not self.pairs:
            return events
        press = self._press(model, data)
        for k, p in enumerate(self.pairs):
            eq = self.eqs[k]
            names = (p.a, p.b)
            if data.eq_active[eq]:
                if self.pull(data, k) > p.pull_n:
                    data.eq_active[eq] = 0
                    self.armed[k] = False  # don't snap again until they have separated
                    lo, up = self._lower_upper(data, k)
                    events.append(TaskEvent("snap_off", object=names[up], text=names[lo]))
                continue
            if k not in press:
                self.armed[k] = True
                continue
            if not self.armed[k] or press[k] < p.press_n:
                continue
            seated = [(lo, up) for lo, up in ((0, 1), (1, 0)) if p.aligned(data, names[lo], names[up])]
            if not seated:
                continue
            lo, up = seated[0]
            # Lock exactly where the parts are: a weld that had to pull them anywhere would carry force from the
            # first step (and could pop right off).
            model.eq_data[eq, :11] = [0, 0, 0, *relpose(data, *self._bodies[k]), 1]
            data.eq_active[eq] = 1
            events.append(TaskEvent("snap_on", object=names[up], text=names[lo]))
        return events


def relpose(data: mujoco.MjData, body_a: int, body_b: int) -> list[float]:
    """Body b's current pose in body a's frame (a weld's relpose)."""
    pos = data.xmat[body_a].reshape(3, 3).T @ (data.xpos[body_b] - data.xpos[body_a])
    qi, q = np.empty(4), np.empty(4)
    mujoco.mju_negQuat(qi, data.xquat[body_a])
    mujoco.mju_mulQuat(q, qi, data.xquat[body_b])
    return [*pos, *q]
