"""The policy hook: how something other than the operator can drive the arms (policy rollouts, shared control /
DAgger). The platform never imports an ML library; a runner is loaded lazily from `module:Class` or
`path/to/file.py:Class` (config control.policy, `xrc serve --policy`, or the `load_policy` command).

    class MyRunner:
        name = "my-policy"
        layout = {"action_names": [...]}          # optional: checked against the scene (FeatureLayout fields)
        def __init__(self, env_info: dict, **args): ...
        def reset(self, env) -> None: ...          # a new episode, or the policy takes over again
        def act(self, env) -> tuple[np.ndarray, np.ndarray]: ...   # (joint targets (n*J,), grippers (n,) 0..1)
        def close(self) -> None: ...

The runner renders its own cameras from the live state (scene.render.CameraRenderer on env.model / env.data): `act`
runs on the sim thread, the only thread that may touch them. Its targets go through the safety filter like the
operator's, and each step records who drove (`control/source`).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from xrcollect.task.loader import load_object


class PolicyRunner(Protocol):
    name: str

    def reset(self, env) -> None: ...
    def act(self, env) -> tuple[np.ndarray, np.ndarray]: ...
    def close(self) -> None: ...


class LayoutMismatch(ValueError):
    pass


def env_info(env) -> dict:
    return {"task": env.task.name, "task_version": env.bundle.version, "arms": list(env.arms),
            "action_names": env.adapter.action_layout(), "camera_names": env.camera_names,
            "control_hz": env.control_hz, "objects": list(env.objects)}


def load_policy(ref: str, env, **args) -> PolicyRunner:
    """Build a runner and refuse one whose declared layout doesn't match the scene."""
    info = env_info(env)
    runner = load_object(ref)(info, **args)
    layout = getattr(runner, "layout", None) or {}
    want = layout.get("action_names")
    if want is not None and list(want) != info["action_names"]:
        raise LayoutMismatch(f"the policy was trained for actions {list(want)}, the scene has {info['action_names']}")
    missing = [c for c in layout.get("cameras", []) if c not in info["camera_names"]]
    if missing:
        raise LayoutMismatch(f"the policy needs cameras {missing}; the scene has {info['camera_names']}")
    if not hasattr(runner, "name"):
        runner.name = ref.rsplit(":", 1)[-1]
    return runner


class HoldRunner:
    """A trivial runner for tests and for trying shared control without a model: it holds the arms where they were
    when it took over and keeps the grippers open (or follows a fixed joint-space sine if `wiggle` is set)."""

    def __init__(self, env_info: dict, wiggle: float = 0.0, **_):
        self.name = "hold"
        self.layout = {"action_names": env_info["action_names"]}
        self.wiggle = float(wiggle)
        self.q0: np.ndarray | None = None
        self.t = 0
        self.n_arms = len(env_info["arms"])
        self.hz = env_info["control_hz"]

    def reset(self, env) -> None:
        self.q0, self.t = env.arm_q, 0

    def act(self, env) -> tuple[np.ndarray, np.ndarray]:
        self.t += 1
        q = self.q0.copy()
        if self.wiggle:
            q[::env.b.n_joints] += self.wiggle * np.sin(2 * np.pi * self.t / self.hz / 4)
        return q, np.ones(self.n_arms)

    def close(self) -> None:
        pass
