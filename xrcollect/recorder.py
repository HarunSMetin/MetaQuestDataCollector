"""Episode recording: state machine + HDF5 episode files.

States: IDLE -> COUNTDOWN -> RECORDING -> REVIEW -> (save | discard) -> IDLE

One HDF5 file per episode (<out_dir>/episode_XXXXXX.hdf5). Keys are dataset paths, time is axis 0. n = number of
arms, J = joints per arm, k = free objects.

meta (file attrs):
  schema_version, task, task_version (content hash of the four task layers), instruction, operator, session,
  input_mode, source (teleop | policy | shared | scripted | generated), success, fps, seed, created_at, duration_s,
  num_steps, model_xml (the compiled scene), task_yaml (the resolved bundle: embodiment, scene, objects, task
  logic, plugin source), config_yaml (platform settings), scene_hash, camera_names, arms, objects, action_names,
  stage_names, contact_kinds, fixture_joints
  optional: branched_from, branch_step (a take that took over from a replayed episode at that step; sim/state[0]
  is that episode's sim/state[branch_step]); generated_from, retimed_from, generation (derived episodes)

/time                         (T,)          sim time [s]
/wall_time                    (T,)          host monotonic time [s]
/obs/joint_pos                (T, n*J)      rad, arm order = meta arms
/obs/joint_vel                (T, n*J)
/obs/gripper_width            (T, n)        finger opening [m]
/obs/ee_pose                  (T, n, 7)     tcp [x y z qw qx qy qz] world
/obs/ee_wrench                (T, n, 6)     external [force N, torque Nm] on each hand, world axes, about the TCP
/obs/objects                  (T, k, 7)     object poses (body frame = bottom centre), order = meta objects
                                            (privileged, sim only)
/obs/fixture_joints           (T, f)        drawer / door joints, order = meta fixture_joints
/action/joint_pos             (T, n*(J+1))  per arm [J joint targets, gripper]; gripper in [0 closed, 1 open];
                                            names = meta action_names. The EXECUTED target: after the safety filter
                                            (minus the DART drift, if that is on: see Env.executed_targets)
/action/ee_pose               (T, n*7+n)    [tcp xyz qwxyz per arm, grippers]: the TCP pose of those joint targets
/sim/state                    (T, S)        Env.get_state() BEFORE the step: mj_getState(mjSTATE_INTEGRATION), then
                                            the task's extra state (weld poses...). Env.set_state restores both
/task/success                 (T,)          success predicate per step
/task/progress                (T,)          stages done so far, counted in order
/task/stages                  (T, s)        each stage's predicate, order = meta stage_names
/task/held                    (T, n)        index into meta objects of what each arm holds (-1: nothing)
/task/*                       ...           more from the task (Task.task_fields), e.g. snaps (T, pairs)
/contacts                     (T, 3)        Env.contact_flags, order = meta contact_kinds
/control/source               (T,)          who drove this step: 0 operator, 1 policy, 2 script
/safety/correction            (T,)          largest change the safety filter made to a joint target [rad]
/teleop/head                  (T, 7)        XR head pose [x y z qx qy qz qw]
/teleop/{left,right}/source   (T,)          0 none, 1 controller, 2 hand
/teleop/{left,right}/pose     (T, 7)        raw XR control pose
/teleop/{left,right}/buttons  (T, 8)
/teleop/{left,right}/joints   (T, 25, 7)    WebXR hand joints (zeros for controllers)
/teleop/{left,right}/pinch    (T, 2)
/teleop/engaged               (T, n)
/teleop/tracked               (T, n)
/teleop/rtt_ms                (T,)          round trip headset -> server -> headset
/teleop/input_hz              (T,)

Camera images are not stored: sim/state replays the episode exactly, and exporters re-render them.

Schema changelog:
  1  first xrcollect format (generalized from QuestCollect v7: per-arm arrays, task/* group, task_yaml,
     task_version, action_names, safety/correction, branched_from)
"""

from __future__ import annotations

import json
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import NamedTuple

import h5py
import numpy as np
import yaml

from xrcollect.embodiment.base import join_action
from xrcollect.env import CONTACT_KINDS
from xrcollect.protocol import N_BUTTONS, N_HAND_JOINTS, InputFrame

SCHEMA_VERSION = 1
SOURCE_OPERATOR, SOURCE_POLICY, SOURCE_SCRIPT = 0, 1, 2


# -- one step --------------------------------------------------------------------------------------
def teleop_fields(frame: InputFrame | None, engaged=(), tracked=(), rtt_ms: float = 0.0, input_hz: float = 0.0,
                  n_arms: int = 0) -> dict:
    """The raw operator input of one step (all zeros without a headset, e.g. a scripted take)."""
    n = n_arms or len(engaged)
    out = {"head": np.zeros(7) if frame is None else frame.head,
           "engaged": np.array(engaged, bool) if len(engaged) else np.zeros(n, bool),
           "tracked": np.array(tracked, bool) if len(tracked) else np.zeros(n, bool),
           "rtt_ms": float(rtt_ms), "input_hz": float(input_hz)}
    for side, s in (("left", "l"), ("right", "r")):
        h = None if frame is None else frame.hand(s)
        out[side] = {
            "source": 0 if h is None else h.source,
            "pose": np.zeros(7) if h is None else h.pose,
            "buttons": np.zeros(N_BUTTONS) if h is None else h.buttons,
            "joints": np.zeros((N_HAND_JOINTS, 7)) if h is None else h.joints,
            "pinch": np.zeros(2) if h is None else h.pinch,
        }
    return out


class Snapshot(NamedTuple):
    """What a recorded step keeps from before the action."""
    obs: dict
    state: np.ndarray
    contacts: np.ndarray


def snapshot(env) -> Snapshot:
    return Snapshot(env.observe(), env.get_state(), env.contact_flags())


def make_step(env, before: Snapshot, q_target: np.ndarray, grippers: np.ndarray, ee_targets: np.ndarray,
              success: bool, progress: int, teleop: dict, source: int = SOURCE_OPERATOR,
              correction: float = 0.0) -> dict:
    """One recorded step in the layout above: the same builder for every producer of episodes.

    q_target: (n * J,) joint targets; grippers: (n,) in [0 closed, 1 open]; ee_targets: (n, 7) TCP poses.
    """
    held = env.held()
    return {
        "time": env.data.time,
        "wall_time": time.monotonic(),
        "obs": before.obs,
        "action": {
            "joint_pos": join_action(q_target, grippers, env.b.n_joints),
            "ee_pose": np.concatenate([np.ravel(ee_targets), np.asarray(grippers, float)]),
        },
        "sim": {"state": before.state},
        "task": {
            "success": bool(success), "progress": int(progress), "stages": env.stage_flags(),
            "held": np.array([-1 if held[a] is None else env.objects.index(held[a]) for a in env.arms], np.int16),
            **env.task.task_fields(env.model, env.data),
        },
        "contacts": np.asarray(before.contacts, bool),
        "control": {"source": source},
        "safety": {"correction": float(correction)},
        "teleop": teleop,
    }


def control_step(env, before: Snapshot | None, q: np.ndarray, grippers: np.ndarray, ee_targets, teleop: dict | None,
                 source: int = SOURCE_OPERATOR, q_recorded: np.ndarray | None = None,
                 progress: int | None = None, correction: float = 0.0) -> tuple[dict | None, bool, int]:
    """One control step, the same for the live runtime, policy rollouts, scripted experts and generated takes:
    apply the joint targets and grippers, read success and progress, make the recorded step, step the physics.

    How the targets were made (IK, policy, safety filter, DART) stays with the caller. before: `snapshot(env)`,
    taken before the action (its state holds the previous ctrl, which `Env.executed_targets` relies on); None
    records nothing. q_recorded: the action to record if it is not the executed q (the runtime records the target
    without the DART drift). Returns (recorded step or None, success, progress)."""
    env.apply_action(q, grippers)
    success = env.success()
    progress = env.progress() if progress is None else progress
    step = None
    if before is not None:
        if teleop is None:
            teleop = teleop_fields(None, n_arms=env.b.n_arms)
        step = make_step(env, before, q if q_recorded is None else q_recorded, grippers, ee_targets, success,
                         progress, teleop, source, correction)
    env.step()
    return step, success, progress


# -- episode files ---------------------------------------------------------------------------------
def _flatten(d: dict, prefix: str = "") -> dict[str, object]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}/{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(v, key))
        else:
            out[key] = v
    return out


def write_episode(path: Path, steps: list[dict], meta: dict) -> Path:
    """steps: per-step nested dicts with identical structure; nesting becomes HDF5 groups."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    flat = [_flatten(s) for s in steps]
    with h5py.File(tmp, "w") as f:
        for k, v in meta.items():
            if v is None:
                continue
            f.attrs[k] = json.dumps(v) if isinstance(v, (list, dict)) else v
        for key in flat[0]:
            arr = np.stack([np.asarray(s[key]) for s in flat])
            f.create_dataset(key, data=arr, compression="lzf" if arr.size > 4096 else None)
    tmp.rename(path)
    return path


def _decode(v):
    if isinstance(v, bytes):
        v = v.decode()
    if isinstance(v, str) and v[:1] in "[{":
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return v
    if isinstance(v, np.generic):
        return v.item()
    return v


def read_meta(path: Path) -> dict:
    with h5py.File(path, "r") as f:
        return {k: _decode(v) for k, v in f.attrs.items()}


def read_episode(path: Path, keys: list[str] | None = None) -> tuple[dict, dict]:
    """Returns (meta, data) with data as a flat dict path -> ndarray. Reads every schema version so far."""
    data = {}
    with h5py.File(path, "r") as f:
        meta = {k: _decode(v) for k, v in f.attrs.items()}

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and (keys is None or name in keys):
                data[name] = obj[()]

        f.visititems(visit)
    if int(meta.get("schema_version", 0)) > SCHEMA_VERSION:
        raise ValueError(f"{path} has schema {meta['schema_version']}; this xrcollect reads up to {SCHEMA_VERSION}")
    return meta, data


def episode_files(out_dir: Path) -> list[Path]:
    return sorted(Path(out_dir).glob("episode_*.hdf5"))


def next_episode_path(out_dir: Path) -> Path:
    nums = [int(p.stem.split("_")[-1]) for p in episode_files(out_dir) if p.stem.split("_")[-1].isdigit()]
    return Path(out_dir) / f"episode_{(max(nums) + 1 if nums else 0):06d}.hdf5"


LIST_KEYS = ("success", "duration_s", "num_steps", "input_mode", "operator", "session", "created_at", "source",
             "task", "task_version", "seed", "branched_from", "branch_step", "generated_from", "retimed_from")


def list_episodes(out_dir: Path) -> list[dict]:
    rows = []
    for p in episode_files(out_dir):
        try:
            with h5py.File(p, "r") as f:  # only the small attributes: model_xml and task_yaml stay on disk
                m = {k: _decode(f.attrs[k]) for k in LIST_KEYS if k in f.attrs}
        except OSError:
            continue
        rows.append({"path": str(p), "file": p.name, **{k: m.get(k) for k in LIST_KEYS},
                     "success": bool(m.get("success")), "source": m.get("source", "teleop")})
    return rows


def episode_meta(cfg, env, num_steps: int, success: bool, input_mode: str = "none", operator: str | None = None,
                 source: str = "teleop") -> dict:
    """File attributes of an episode (see the module docstring)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "task": env.task.name,
        "task_version": env.bundle.version,
        "instruction": env.task.instruction,
        "operator": cfg.record.operator if operator is None else operator,
        "session": cfg.record.session,
        "input_mode": input_mode,
        "source": source,
        "success": bool(success),
        "fps": env.control_hz,
        "seed": env.seed if env.seed is not None else -1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "duration_s": num_steps / env.control_hz,
        "num_steps": num_steps,
        "model_xml": env.xml,
        "scene_hash": env.scene_hash,
        "task_yaml": env.bundle.to_yaml(),
        "config_yaml": yaml.safe_dump(cfg.model_dump(mode="json")),
        "camera_names": env.camera_names,
        "arms": list(env.arms),
        "objects": list(env.objects),
        "action_names": env.adapter.action_layout(),
        "stage_names": list(env.task.stage_names),
        "contact_kinds": list(CONTACT_KINDS),
        "fixture_joints": list(env.fixture_joints),
    }


# -- recorder state machine ------------------------------------------------------------------------
class RecState(str, Enum):
    IDLE = "idle"
    COUNTDOWN = "countdown"
    RECORDING = "recording"
    REVIEW = "review"


class Recorder:
    def __init__(self, cfg, out_dir: Path, fps: int):
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.fps = fps
        self.state = RecState.IDLE
        self.steps: list[dict] = []
        self.countdown_end = 0.0
        self.t_start = 0.0
        self.success_since: float | None = None
        self.success_seen = False
        self.auto_stop = True
        self.saved_count = 0
        self.discarded_count = 0
        self.session_start = time.monotonic()
        self._writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hdf5-writer")
        self._pending: Future | None = None

    # -- transitions ---------------------------------------------------------------------
    def start(self, now: float) -> bool:
        if self.state != RecState.IDLE:
            return False
        self.state = RecState.COUNTDOWN
        self.countdown_end = now + self.cfg.record.countdown_s
        return True

    def stop(self) -> bool:
        if self.state == RecState.COUNTDOWN:
            self.state = RecState.IDLE
            return True
        if self.state != RecState.RECORDING:
            return False
        self.state = RecState.REVIEW if self.steps else RecState.IDLE
        return True

    def toggle(self, now: float) -> None:
        if self.state == RecState.IDLE:
            self.start(now)
        else:
            self.stop()

    def discard(self) -> bool:
        """Drop the current take. Returns True if a take was dropped (the caller resets the scene)."""
        if self.state == RecState.IDLE:
            return False
        if self.steps:
            self.discarded_count += 1
        self._clear()
        return True

    def save(self, env, input_mode: str, success: bool | None = None, source: str = "teleop",
             extra: dict | None = None) -> str | None:
        """Write the take in the background (the control loop never waits on disk). success=None uses whether the
        task check ever passed; extra = more file attributes (branched_from, branch_step)."""
        if self.state == RecState.RECORDING:
            self.stop()
        if self.state != RecState.REVIEW or not self.steps:
            return None
        path = next_episode_path(self.out_dir)
        if self._pending is not None and not self._pending.done():
            self._pending.result(timeout=60)  # the previous file must exist before the next name is taken
            path = next_episode_path(self.out_dir)
        meta = episode_meta(self.cfg, env, len(self.steps), self.success_seen if success is None else bool(success),
                            input_mode=input_mode, source=source)
        meta.update(extra or {})
        self._pending = self._writer.submit(write_episode, path, self.steps, meta)
        self.saved_count += 1
        self._clear()
        return str(path)

    def _clear(self) -> None:
        self.state = RecState.IDLE
        self.steps = []
        self.success_since = None
        self.success_seen = False

    # -- per control tick ------------------------------------------------------------------
    def tick(self, now: float) -> None:
        """Advance the countdown; call once per control step before `append`."""
        if self.state == RecState.COUNTDOWN and now >= self.countdown_end:
            self.state = RecState.RECORDING
            self.t_start = now
            self.steps = []
            self.success_since = None
            self.success_seen = False

    def append(self, now: float, step: dict, success: bool) -> bool:
        """Store one step while recording. Returns True if auto-stop fired (success held long enough)."""
        if self.state != RecState.RECORDING:
            return False
        self.steps.append(step)
        if success:
            self.success_seen = True
            self.success_since = self.success_since or now
            if self.auto_stop and now - self.success_since >= self.cfg.record.auto_stop_tail_s:
                self.stop()
                return True
        else:
            self.success_since = None
        return False

    @property
    def recording(self) -> bool:
        return self.state == RecState.RECORDING

    def elapsed(self, now: float) -> float:
        if self.state == RecState.RECORDING:
            return now - self.t_start
        if self.state == RecState.REVIEW:
            return len(self.steps) / self.fps
        return 0.0

    def countdown_left(self, now: float) -> float:
        return max(0.0, self.countdown_end - now) if self.state == RecState.COUNTDOWN else 0.0

    def flush(self, timeout: float = 30.0) -> None:
        if self._pending is not None:
            self._pending.result(timeout=timeout)

    def close(self) -> None:
        self.flush()
        self._writer.shutdown(wait=True)
