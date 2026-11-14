"""The environment: one compiled scene, driven through a RobotAdapter, judged by a Task.

Everything that reads or writes simulator state goes through `get_state` / `set_state` here (never raw
mj_getState): the task's extra state (weld poses and the like) is appended to MuJoCo's integration state.
"""

from __future__ import annotations

import functools
import hashlib
import logging

import mujoco
import numpy as np

from xrcollect.embodiment.base import load_adapter
from xrcollect.scene.compose import compile_scene, load_model
from xrcollect.scene.constants import STATE_SPEC
from xrcollect.scene.fingerprint import scene_hash
from xrcollect.task.base import TaskEvent, World
from xrcollect.task.declarative import make_task
from xrcollect.task.loader import Bundle

log = logging.getLogger(__name__)

# The one contact definition (headset warnings, recorded `contacts`, audit, selection rules), in this order:
# an arm link on a support surface, two arms touching (hand-hand and finger-finger excepted), a hand or fingertip
# on a support surface.
CONTACT_KINDS = ("link_table", "arm_arm", "tip_table")
SEED_STREAMS = {"record": 0, "eval": 1, "validate": 2}


def stream_seed(stream: str, index: int) -> int:
    """Seed #index of a named stream. Recording draws fresh entropy; evaluation and validation use their own
    streams, so they never run on layouts that are in the recorded data."""
    h = hashlib.sha256(f"xrcollect/{stream}/{index}".encode()).digest()
    return int.from_bytes(h[:4], "little") % (2**31)


class Env:
    def __init__(self, bundle: Bundle, xml: str | None = None):
        """xml: an episode's saved model to run instead of composing the scene from the bundle."""
        self.bundle = bundle
        self.adapter = load_adapter(bundle.embodiment)
        self.task = make_task(bundle)
        if xml is None:
            self.model, self.xml = compile_scene(bundle, self.adapter, self.task)
        else:
            self.model, self.xml = load_model(xml, getattr(self.adapter, "asset_dirs", [])), xml
        self.data = mujoco.MjData(self.model)
        self.control_hz = bundle.control_hz
        self.control_dt = 1.0 / self.control_hz
        self.n_substeps = max(1, round(self.control_dt / self.model.opt.timestep))
        self.table_height = bundle.table_height
        m = self.model
        self.b = self.adapter.bind(m)
        self.arms = list(self.adapter.arms)

        free = [n for n, o in bundle.objects.items() if not o.get("fixed") and not o.get("mjcf")]
        self.objects = free
        body = {n: m.body(n).id for n in bundle.objects if not bundle.objects[n].get("mjcf")}
        robot_or_free = set(np.flatnonzero(self.b.body_arm > 0).tolist()) | {body[n] for n in free}
        fixture_joints = [m.joint(j).name for j in range(m.njnt)
                          if int(m.jnt_type[j]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
                          and int(m.jnt_bodyid[j]) not in robot_or_free]
        self.fixture_joints = fixture_joints
        self._fixture_qadr = np.array([m.joint(j).qposadr[0] for j in fixture_joints], int)
        self.world = World(
            model=m, bound=self.b, table_height=self.table_height, objects=free, body=body,
            size={n: np.asarray(o.get("size", [0.04] * 3), float) for n, o in bundle.objects.items()},
            regions=bundle.task.get("regions", {}),
            qadr={n: int(m.joint(f"{n}_free").qposadr[0]) for n in free},
            vadr={n: int(m.joint(f"{n}_free").dofadr[0]) for n in free},
            surface_bodies=[m.body("table").id], fixture_joints=fixture_joints)
        self._obj_qadr = np.array([[self.world.qadr[n] + k for k in range(7)] for n in free], int).reshape(-1, 7)
        self._surface = np.zeros(m.nbody, bool)
        self._surface[self.world.surface_bodies] = True
        self.task.bind(self.world)
        self.mj_state_size = mujoco.mj_stateSize(m, STATE_SPEC)
        self.state_size = self.mj_state_size + len(self.task.extra_state(m))
        self.seed: int | None = None
        self.events: list[TaskEvent] = []   # from the task's after_step of the last step
        self.scene_matches = True           # set by env_from_episode

    @functools.cached_property
    def scene_hash(self) -> str:
        return scene_hash(self.model)

    @property
    def camera_names(self) -> list[str]:
        return [self.model.camera(c).name for c in range(self.model.ncam)]

    # -- control -------------------------------------------------------------------------------------
    def reset(self, seed: int | None = None) -> dict:
        self.seed = int(seed) if seed is not None else int(np.random.SeedSequence().entropy % (2**31))
        rng = np.random.default_rng(self.seed)
        mujoco.mj_resetData(self.model, self.data)
        self.adapter.reset(self.data, self.b)
        self.task.reset(self.model, self.data, rng)
        self.events = []
        mujoco.mj_forward(self.model, self.data)
        return self.observe()

    def apply_action(self, joint_targets: np.ndarray, grippers: np.ndarray) -> None:
        """joint_targets: (n_arms * n_joints,) rad; grippers: (n_arms,) in [0 closed, 1 open]."""
        self.adapter.apply(self.data, self.b, joint_targets, grippers, self.control_dt)

    def step(self) -> None:
        mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
        self.events = self.task.after_step(self.model, self.data)

    # -- observation ---------------------------------------------------------------------------------
    @property
    def arm_q(self) -> np.ndarray:
        return self.data.qpos[self.b.qadr].copy()

    def tcp_poses(self) -> np.ndarray:
        """(n_arms, 7) [x y z qw qx qy qz] of each TCP."""
        d = self.data
        out = np.empty((self.b.n_arms, 7))
        for k, sid in enumerate(self.b.tcp_sites):
            out[k, :3] = d.site_xpos[sid]
            mujoco.mju_mat2Quat(out[k, 3:], d.site_xmat[sid])
        return out

    def observe(self) -> dict:
        d = self.data
        return {
            "joint_pos": d.qpos[self.b.qadr].copy(),
            "joint_vel": d.qvel[self.b.vadr].copy(),
            "gripper_width": self.adapter.gripper_width(d, self.b),
            "ee_pose": self.tcp_poses(),
            "ee_wrench": self.ee_wrench(),
            # (k, 7) [x y z qw qx qy qz] per free object, order = meta `objects` (privileged: sim only)
            "objects": d.qpos[self._obj_qadr].copy(),
            "fixture_joints": d.qpos[self._fixture_qadr].copy(),
        }

    def ee_wrench(self) -> np.ndarray:
        """(n_arms, 6) [fx fy fz tx ty tz]: the external wrench on each hand (hand + fingers) from everything that
        is not that arm, world axes, torque about the TCP, N and Nm. What a real arm estimates as its external
        end-effector wrench: a held object's weight and pressing it down show up, the squeeze cancels out."""
        m, d, b = self.model, self.data, self.b
        out = np.zeros((b.n_arms, 6))
        f6 = np.zeros(6)
        B1, B2 = m.geom_bodyid[d.contact.geom1], m.geom_bodyid[d.contact.geom2]
        for k, hands in enumerate(b.hand_bodies):
            in1, in2 = np.isin(B1, hands), np.isin(B2, hands)
            for i in np.flatnonzero((in1 & (b.body_arm[B2] != k + 1)) | (in2 & (b.body_arm[B1] != k + 1))):
                c, on1 = d.contact[i], bool(in1[i])
                mujoco.mj_contactForce(m, d, i, f6)
                f = c.frame.reshape(3, 3).T @ f6[:3]  # on geom2 (the normal points from geom1 to geom2)
                if on1:
                    f = -f
                out[k, :3] += f
                out[k, 3:] += np.cross(c.pos - d.site_xpos[b.tcp_sites[k]], f)
        return out

    def contact_flags(self) -> np.ndarray:
        """(3,) bool in the order of CONTACT_KINDS for the current contacts."""
        d, b = self.data, self.b
        b1, b2 = self.model.geom_bodyid[d.contact.geom1], self.model.geom_bodyid[d.contact.geom2]
        arm1, arm2 = b.body_arm[b1], b.body_arm[b2]
        tip1, tip2 = b.body_tip[b1], b.body_tip[b2]
        arm_arm = (arm1 > 0) & (arm2 > 0) & (arm1 != arm2) & ~(tip1 & tip2)
        s1, s2 = self._surface[b1], self._surface[b2]
        other = np.where(s1, b2, b1)
        on_surface = s1 | s2
        link = on_surface & (b.body_arm[other] > 0) & ~b.body_tip[other]
        tip = on_surface & b.body_tip[other]
        return np.array([link.any(), arm_arm.any(), tip.any()])

    # -- task ----------------------------------------------------------------------------------------
    def success(self) -> bool:
        return bool(self.task.success(self.model, self.data))

    def progress(self) -> int:
        return int(self.task.progress(self.model, self.data))

    def stage_flags(self) -> np.ndarray:
        return np.asarray(self.task.stage_flags(self.model, self.data), bool)

    def held(self) -> dict[str, str | None]:
        return self.task.held(self.model, self.data)

    # -- full state, for exact replay and offline re-rendering --------------------------------------------
    def get_state(self) -> np.ndarray:
        s = np.empty(self.mj_state_size)
        mujoco.mj_getState(self.model, self.data, s, STATE_SPEC)
        return np.concatenate([s, self.task.extra_state(self.model)])

    def set_state(self, state: np.ndarray) -> None:
        state = np.asarray(state, dtype=np.float64)
        self.task.set_extra_state(self.model, state[self.mj_state_size:])
        mujoco.mj_setState(self.model, self.data, state[:self.mj_state_size], STATE_SPEC)
        mujoco.mj_forward(self.model, self.data)
        self.events = []

    def executed_targets(self, states: np.ndarray) -> np.ndarray:
        """(T, n_arms * n_joints) arm joint targets actually sent at each recorded step: the ctrl in the next
        step's state (it is saved before that step's action). With DART noise these differ from action/joint_pos by
        the drift. The last step's target isn't saved: the one before is repeated."""
        ctrl_bit = int(mujoco.mjtState.mjSTATE_CTRL)  # elements are stored in bit order: ctrl follows the lower ones
        before = mujoco.mj_stateSize(self.model, int(STATE_SPEC) & (ctrl_bit - 1))
        ctrl = np.asarray(states)[:, before + self.b.arm_act]
        return np.concatenate([ctrl[1:], ctrl[-1:]])

    def tcp_poses_at(self, arm_q: np.ndarray) -> np.ndarray:
        """(T, n_arms, 7) TCP poses at the given (T, n_arms * n_joints) joint angles (forward kinematics only)."""
        d = mujoco.MjData(self.model)
        d.qpos[:] = self.model.qpos0
        arm_q = np.atleast_2d(arm_q)
        out = np.empty((len(arm_q), self.b.n_arms, 7))
        for t, q in enumerate(arm_q):
            d.qpos[self.b.qadr] = q
            mujoco.mj_kinematics(self.model, d)
            for k, sid in enumerate(self.b.tcp_sites):
                out[t, k, :3] = d.site_xpos[sid]
                mujoco.mju_mat2Quat(out[t, k, 3:], d.site_xmat[sid])
        return out


def env_from_episode(meta: dict) -> Env:
    """The environment an episode was recorded in, from what the episode itself stores: its bundle (`task_yaml`) and
    its compiled model (`model_xml`). Never built from today's task files. The saved fingerprint is checked;
    `env.scene_matches` is False (and a warning logged) when e.g. a mesh file changed since."""
    bundle = Bundle.from_yaml(meta["task_yaml"])
    env = Env(bundle, xml=meta["model_xml"])
    want = meta.get("scene_hash")
    env.scene_matches = want is None or want == env.scene_hash
    if not env.scene_matches:
        log.warning("episode recorded in a different scene (hash %s, now %s): replays may differ", want,
                    env.scene_hash)
    return env
