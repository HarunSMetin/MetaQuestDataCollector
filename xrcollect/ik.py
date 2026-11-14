"""Differential IK for every arm of the embodiment with mink: one QP, joint position + velocity limits."""

from __future__ import annotations

import mink
import mujoco
import numpy as np

from xrcollect.config import IKConfig
from xrcollect.scene.compose import build_spec
from xrcollect.task.loader import Bundle


def collision_geoms(model: mujoco.MjModel, bodies: list[str]) -> list[int]:
    """Colliding geoms of the named bodies (shared with safety.py)."""
    ids = []
    for b in bodies:
        bid = model.body(b).id
        ids += [g for g in mink.get_body_geom_ids(model, bid) if model.geom_contype[g] or model.geom_conaffinity[g]]
    return ids


def robot_model(bundle: Bundle, adapter) -> mujoco.MjModel:
    """Robots + static scene, no task objects: what the IK and the safety filter plan on."""
    return build_spec(bundle, adapter, with_objects=False).compile()


class ArmIK:
    """Keeps its own kinematic configuration (robots only) that tracks the end-effector targets smoothly."""

    def __init__(self, bundle: Bundle, adapter, cfg: IKConfig, dt: float):
        self.cfg, self.dt, self.adapter = cfg, dt, adapter
        self.arms = list(adapter.arms)
        self.model = robot_model(bundle, adapter)
        self.b = adapter.bind(self.model)
        self.configuration = mink.Configuration(self.model)
        m = self.model
        self.frame_tasks = {
            a: mink.FrameTask(adapter.tcp_site(a), "site", position_cost=cfg.position_cost,
                              orientation_cost=cfg.orientation_cost, lm_damping=1.0)
            for a in self.arms
        }
        self.posture = mink.PostureTask(m, cost=cfg.posture_cost)
        self.tasks = [*self.frame_tasks.values(), self.posture]
        if cfg.damping_cost:
            # Prefer the smallest joint motion. Without it the redundant joints of a strongly folded arm whip at
            # the velocity limit while the hand barely moves.
            self.tasks.append(mink.DampingTask(m, cost=cfg.damping_cost))
        vel = {j: adapter.max_joint_vel for a in self.arms for j in adapter.joint_names(a)}
        self.limits = [mink.ConfigurationLimit(m), mink.VelocityLimit(m, vel)]
        self.reset(adapter.home_q)

    def reset(self, arm_q: np.ndarray) -> None:
        q = self.model.qpos0.copy()
        q[self.b.qadr] = arm_q
        self.configuration.update(q)
        self.posture.set_target(q)
        for a in self.arms:
            self.frame_tasks[a].set_target(
                self.configuration.get_transform_frame_to_world(self.adapter.tcp_site(a), "site"))

    def follow(self, arm_q: np.ndarray) -> None:
        """Keep the IK within max_lead rad of the measured joint angles (call before solve).

        The IK has no obstacles; physics does. When an arm is blocked, the IK would otherwise run on through the
        obstacle while the arm stays put, then yank the arm across when it comes free. In free motion the arm
        trails its target by about speed x kd/kp (~0.1 s), so 0.15 rad leaves moves up to ~1.5 rad/s untouched.
        """
        q = self.configuration.q.copy()
        lead = q[self.b.qadr] - arm_q
        q[self.b.qadr] = arm_q + np.clip(lead, -self.cfg.max_lead, self.cfg.max_lead)
        self.configuration.update(q)

    def tcp_pose(self, arm: str) -> tuple[np.ndarray, np.ndarray]:
        T = self.configuration.get_transform_frame_to_world(self.adapter.tcp_site(arm), "site")
        return T.translation().copy(), T.rotation().wxyz.copy()

    def set_target(self, arm: str, pos: np.ndarray, wxyz: np.ndarray) -> None:
        T = mink.SE3.from_rotation_and_translation(mink.SO3(np.asarray(wxyz, float)), np.asarray(pos, float))
        self.frame_tasks[arm].set_target(T)

    def solve(self) -> np.ndarray:
        """Advance the IK configuration one control step; returns the joint targets of all arms."""
        sub_dt = self.dt / self.cfg.iterations
        for _ in range(self.cfg.iterations):
            v = mink.solve_ik(self.configuration, self.tasks, sub_dt, solver="daqp", damping=1e-3,
                              limits=self.limits)
            self.configuration.integrate_inplace(v, sub_dt)
        return self.configuration.q[self.b.qadr].copy()

    def errors(self) -> dict[str, tuple[float, float]]:
        """Per-arm (position error m, orientation error rad) to the current target."""
        out = {}
        for a, t in self.frame_tasks.items():
            e = t.compute_error(self.configuration)
            out[a] = (float(np.linalg.norm(e[:3])), float(np.linalg.norm(e[3:])))
        return out
