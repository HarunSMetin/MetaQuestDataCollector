"""The robot interface. Everything robot-specific (MJCF, joint names, gripper servo, collision bodies) is behind a
RobotAdapter, chosen by the embodiment YAML (`adapter: module:Class`). The rest of the platform only uses the
adapter and the `Bound` index tables it returns for a compiled model.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Protocol

import mujoco
import numpy as np


@dataclass
class Bound:
    """An adapter's indices into one compiled model (the scene, or a robots-only IK/safety model)."""
    arms: list[str]
    n_joints: int                 # arm joints per arm
    qadr: np.ndarray              # (n_arms * n_joints,) qpos addresses of the arm joints
    vadr: np.ndarray              # (n_arms * n_joints,) dof addresses
    finger_qadr: np.ndarray       # (n_arms, n_finger_joints) qpos addresses
    finger_dof: np.ndarray        # (n_arms * n_finger_joints,) dof addresses (frozen in the safety QP)
    finger_jid: np.ndarray        # (n_arms * n_finger_joints,) joint ids
    arm_act: np.ndarray           # (n_arms * n_joints,) actuator ids
    grip_act: np.ndarray          # (n_arms,) actuator ids
    tcp_sites: list[int]          # site id per arm
    body_arm: np.ndarray          # (nbody,) 0 = not a robot body, i + 1 = body of arm i
    body_tip: np.ndarray          # (nbody,) bool: hand and finger bodies
    hand_bodies: list[np.ndarray]  # per arm: body ids of hand + fingers (end-effector wrench)
    finger_bodies: list[list[int]]  # per arm: one body id per finger (held-object detection)

    @property
    def n_arms(self) -> int:
        return len(self.arms)

    def arm_slice(self, i: int) -> slice:
        return slice(i * self.n_joints, (i + 1) * self.n_joints)


class RobotAdapter(Protocol):
    name: str
    arms: list[str]               # one entry per teleoperated end-effector, e.g. ["l", "r"]
    n_joints: int
    home_q: np.ndarray            # (n_arms * n_joints,)
    max_joint_vel: float

    def attach(self, spec: mujoco.MjSpec) -> None: ...                 # add every arm at its mount pose
    def bind(self, model: mujoco.MjModel) -> Bound: ...
    def reset(self, data: mujoco.MjData, b: Bound) -> None: ...        # home pose, grippers open
    def apply(self, data: mujoco.MjData, b: Bound, joint_targets, grippers, control_dt: float) -> None: ...
    def gripper_width(self, data: mujoco.MjData, b: Bound) -> np.ndarray: ...      # (n_arms,) metres
    def gripper_command(self, data: mujoco.MjData, b: Bound) -> np.ndarray: ...    # (n_arms,) 0 closed .. 1 open
    def tcp_site(self, arm: str) -> str: ...
    def joint_names(self, arm: str) -> list[str]: ...
    def collision_bodies(self, arm: str, kind: str) -> list[str]: ...  # kind: "arm" | "link" | "finger"
    def action_layout(self) -> list[str]: ...                           # names of action/joint_pos entries
    def wrist_cameras(self) -> dict[str, str]: ...                      # arm -> camera name


def load_adapter(embodiment: dict) -> RobotAdapter:
    module, cls = embodiment["adapter"].split(":")
    return getattr(importlib.import_module(module), cls)(embodiment)


def join_action(q: np.ndarray, grippers: np.ndarray, n_joints: int) -> np.ndarray:
    """(n_arms * n_joints,) joint targets + (n_arms,) grippers -> the action layout [joints, grip] per arm."""
    q = np.asarray(q, float).reshape(-1, n_joints)
    return np.concatenate([q, np.asarray(grippers, float)[:, None]], axis=1).ravel()


def split_action(action: np.ndarray, n_joints: int) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of join_action, for (..., n_arms * (n_joints + 1)) arrays."""
    a = np.asarray(action, float)
    a = a.reshape(*a.shape[:-1], -1, n_joints + 1)
    return a[..., :n_joints].reshape(*a.shape[:-2], -1), a[..., n_joints]
