"""A scripted expert on the real control stack (IK -> safety filter -> control_step): keyframed TCP moves, top-down
pick and place. `xrc validate-task` runs it on a task and requires success: an end-to-end check of the whole stack
and a validity check for new tasks (can the goal be reached at all with these tolerances?).

The generic plan handles stages that are about putting an object somewhere (inside a region, on top of / at a pose
relative to another object, or at the goal view's pose). A task whose stages need something else (opening a
drawer) ships `expert.py` next to its task file with `def script(expert): ...`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from xrcollect.config import Config
from xrcollect.env import Env
from xrcollect.frames import mat_to_quat_wxyz
from xrcollect.ik import ArmIK
from xrcollect.recorder import SOURCE_SCRIPT, control_step, episode_meta, snapshot, write_episode
from xrcollect.safety import SafetyFilter
from xrcollect.scene.constants import quat_yaw, rot_z
from xrcollect.task.loader import load_object

LIFT = 0.12


def grasp_quat(yaw: float) -> np.ndarray:
    """TCP pointing down with the fingers closing across the object's y axis (shortest wrist turn)."""
    return mat_to_quat_wxyz(rot_z((yaw + np.pi / 2) % np.pi - np.pi / 2) @ np.diag([1.0, -1.0, -1.0]))


def find_pred(node, name: str, obj: str | None = None) -> dict | None:
    """The first `{name: args}` inside a predicate tree (about `obj`, if given)."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k == name and isinstance(v, dict) and (obj is None or v.get("object") == obj):
                return v
            found = find_pred(v, name, obj)
            if found:
                return found
    elif isinstance(node, list):
        for v in node:
            found = find_pred(v, name, obj)
            if found:
                return found
    return None


class Expert:
    def __init__(self, env: Env, cfg: Config | None = None, record: bool = True):
        self.env, self.cfg, self.record = env, cfg or Config(), record
        self.ik = ArmIK(env.bundle, env.adapter, self.cfg.ik, env.control_dt)
        self.safety = SafetyFilter(env.bundle, env.adapter, self.cfg.ik, self.cfg.safety, env.control_dt)
        self.steps: list[dict] = []
        self.pose: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.grip: dict[str, float] = {}
        self.success_seen = False
        self.on_step = None  # optional callback(env) after every control step (rendering)
        self.sync()

    def sync(self) -> None:
        """Start from where the arms are now (after env.reset or env.set_state)."""
        q = self.env.arm_q
        self.ik.reset(q)
        self.safety.reset(q)
        self.home = {a: self.ik.tcp_pose(a) for a in self.env.arms}
        self.pose = dict(self.home)
        self.grip = dict(zip(self.env.arms, self.env.adapter.gripper_command(self.env.data, self.env.b)))

    def reset(self, seed: int | None = None) -> None:
        self.env.reset(seed)
        self.steps, self.success_seen = [], False
        self.sync()

    # -- one control step ----------------------------------------------------------------------------
    def tick(self) -> bool:
        env = self.env
        for a in env.arms:
            self.ik.set_target(a, *self.pose[a])
        q_measured = env.arm_q
        self.ik.follow(q_measured)
        q = self.ik.solve()
        q_safe, corr = self.safety.filter(q, q_measured, env.data.qpos[env.b.finger_qadr])
        grippers = np.array([self.grip[a] for a in env.arms])
        before = snapshot(env) if self.record else None
        step, success, _ = control_step(env, before, q_safe, grippers, self.safety.tcp_poses(q_safe), None,
                                        SOURCE_SCRIPT, correction=corr)
        if step is not None:
            self.steps.append(step)
        self.success_seen |= success
        if self.on_step:
            self.on_step(env)
        return success

    # -- motion primitives ---------------------------------------------------------------------------
    def move(self, arm: str, pos=None, quat=None, grip: float | None = None, seconds: float = 1.0) -> None:
        """Smoothstep from the current target to the new one."""
        p0, q0 = self.pose[arm]
        p1 = p0 if pos is None else np.asarray(pos, float)
        q1 = q0 if quat is None else np.asarray(quat, float)
        g0 = self.grip[arm]
        g1 = g0 if grip is None else float(grip)
        n = max(1, round(seconds / self.env.control_dt))
        for i in range(1, n + 1):
            s = i / n
            s = s * s * (3 - 2 * s)
            q = (1 - s) * (q0 if np.dot(q0, q1) >= 0 else -q0) + s * q1
            self.pose[arm] = (p0 + s * (p1 - p0), q / np.linalg.norm(q))
            self.grip[arm] = g0 + s * (g1 - g0) if grip is not None and seconds <= 0.2 else (g1 if grip is not None else g0)
            self.tick()

    def wait(self, seconds: float) -> None:
        for _ in range(max(1, round(seconds / self.env.control_dt))):
            self.tick()

    def object_pose(self, obj: str) -> tuple[np.ndarray, float]:
        b = self.env.world.body[obj]
        return self.env.data.xpos[b].copy(), quat_yaw(self.env.data.xquat[b])

    def arm_for(self, xy) -> str:
        """The arm whose base is nearest."""
        arms = self.env.bundle.embodiment["arms"]
        return min(self.env.arms, key=lambda a: np.linalg.norm(np.asarray(arms[a]["pos"][:2]) - np.asarray(xy[:2])))

    def grasp_height(self, obj: str) -> float:
        return float(self.env.world.size[obj][2]) / 2 + 0.002

    def pick(self, arm: str, obj: str) -> None:
        p, yaw = self.object_pose(obj)
        q = grasp_quat(yaw)
        self.move(arm, [p[0], p[1], p[2] + LIFT], q, seconds=1.2)
        self.move(arm, [p[0], p[1], p[2] + self.grasp_height(obj)], seconds=0.8)
        self.move(arm, grip=0.0, seconds=0.6)
        self.move(arm, [p[0], p[1], p[2] + LIFT], seconds=0.8)

    def place(self, arm: str, obj: str, pos, yaw: float = 0.0, press: bool = False) -> None:
        """Put the held object's bottom centre at pos with the given yaw (press: push it down to snap it on)."""
        gp, gz = np.asarray(pos, float), self.grasp_height(obj)
        self.move(arm, [gp[0], gp[1], gp[2] + LIFT], grasp_quat(yaw), seconds=1.2)
        self.move(arm, [gp[0], gp[1], gp[2] + gz + 0.004], seconds=0.9)
        self.move(arm, [gp[0], gp[1], gp[2] + gz], seconds=0.4)
        if press:
            self.move(arm, [gp[0], gp[1], gp[2] + gz - 0.004], seconds=0.4)
        self.move(arm, grip=1.0, seconds=0.6)
        self.move(arm, [gp[0], gp[1], gp[2] + LIFT], seconds=0.8)

    def go_home(self, arm: str) -> None:
        self.move(arm, *self.home[arm], seconds=1.0)

    # -- generic plan --------------------------------------------------------------------------------
    def target_for(self, stage: dict, obj: str) -> tuple[np.ndarray, float] | None:
        """Where the stage wants its object: (bottom-centre position, yaw), or None if this expert can't tell."""
        env, when = self.env, stage["when"]
        inside = find_pred(when, "inside", obj)
        if inside:
            r = inside["region"] if isinstance(inside["region"], dict) else env.world.regions[inside["region"]]
            return np.array([*r["center"], env.table_height]), 0.0
        goal = getattr(env.task, "next_goal", lambda *_: None)(env.model, env.data)
        if goal and goal["object"] == obj:
            return np.asarray(goal["pos"]), quat_yaw(goal["quat"])
        rel = find_pred(when, "relative_pose", obj) or find_pred(when, "on_top", obj)
        if rel:
            bp, byaw = self.object_pose(rel["base"])
            off = rel.get("pos") or [*rel.get("offset", [0.0, 0.0]), float(env.world.size[rel["base"]][2])]
            dyaw = np.deg2rad(rel.get("yaw_deg", 0.0)) if rel.get("yaw") is None else rel["yaw"]
            return bp + rot_z(byaw) @ np.asarray(off, float), byaw + float(dyaw)
        near = find_pred(when, "pose_near", obj)
        if near:
            return np.asarray(near["pos"], float), float(np.deg2rad(near.get("yaw_deg") or 0.0))
        return None

    def run_plan(self) -> bool:
        """Do the stages in order with pick and place. Returns whether the task's success check passed."""
        env = self.env
        press = bool(env.bundle.task.get("snaps")) or bool(getattr(env.task, "snaps", None) and env.task.snaps.pairs)
        spec = env.task.goal_spec()
        for i, stage in enumerate(env.bundle.task.get("stages", [])):
            if env.stage_flags()[i]:
                continue
            obj = spec["stages"][i]["object"]
            target = self.target_for(stage, obj) if obj in env.world.qadr else None
            if target is None:
                raise RuntimeError(f"stage {stage['name']!r}: the generic expert only places objects; add "
                                   "expert.py with `def script(expert)` next to the task file")
            arm = self.arm_for(self.object_pose(obj)[0])
            self.pick(arm, obj)
            self.place(arm, obj, *target, press=press and i > 0)
            self.go_home(arm)
        return self.settle()

    def settle(self, seconds: float = 2.0) -> bool:
        """Wait for the scene to come to rest; True if the success check holds at the end."""
        ok = False
        for _ in range(round(seconds / self.env.control_dt)):
            ok = self.tick()
        return ok

    def run(self) -> bool:
        """The task's own script (expert.py next to the task file) if there is one, else the generic plan."""
        path = self.env.bundle.path
        if path is not None and (path.parent / "expert.py").exists():
            load_object("expert.py:script", path.parent)(self)
            return self.settle()
        return self.run_plan()

    def save(self, path: Path, success: bool | None = None, extra: dict | None = None) -> Path:
        meta = episode_meta(self.cfg, self.env, len(self.steps), self.success_seen if success is None else success,
                            operator="expert", source="scripted")
        meta.update(extra or {})
        return write_episode(path, self.steps, meta)
