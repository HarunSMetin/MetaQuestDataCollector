"""Physics, control and recording: scripted runs of every task, bit-exact replay, file format, state machine."""

import time

import mujoco
import numpy as np
import pytest

from conftest import TASKS, get_env
from xrcollect.config import Config
from xrcollect.embodiment.base import join_action, split_action
from xrcollect.env import CONTACT_KINDS, env_from_episode, stream_seed
from xrcollect.ik import ArmIK
from xrcollect.recorder import (SCHEMA_VERSION, Recorder, RecState, control_step, list_episodes, next_episode_path,
                                read_episode, read_meta, snapshot)
from xrcollect.scene.compose import load_model
from xrcollect.scene.fingerprint import scene_hash
from xrcollect.scene.render import CameraRenderer

CASES = [(t, None) for t in TASKS] + [("tasks/stack_two_blocks", "franka_single")]


@pytest.mark.parametrize("task,embodiment", CASES)
def test_scripted_run_succeeds_and_recorded_actions_replay_exactly(expert_episode, task, embodiment):
    """The task-validity check (an IK script must reach success) and the hidden-state check: from state[0], the
    recorded actions reproduce every recorded state bit for bit, in the scene rebuilt from the episode alone."""
    path, ok = expert_episode(task, embodiment)
    assert ok, "the scripted expert did not reach success"
    meta, data = read_episode(path)
    env = env_from_episode(meta)
    assert env.scene_matches and env.state_size == data["sim/state"].shape[1]
    q, grip = split_action(data["action/joint_pos"], env.b.n_joints)
    env.set_state(data["sim/state"][0])
    for i in range(len(q)):
        assert np.array_equal(env.get_state(), data["sim/state"][i]), f"diverged at step {i}"
        env.apply_action(q[i], grip[i])
        env.step()
    assert np.array_equal(env.executed_targets(data["sim/state"])[:-1], q[:-1])  # no drift: executed = recorded


def test_episode_file_layout(expert_episode):
    path, _ = expert_episode("tasks/stack_two_blocks")
    meta, data = read_episode(path)
    n = meta["num_steps"]
    assert meta["schema_version"] == SCHEMA_VERSION and meta["task"] == "stack_two_blocks"
    assert meta["arms"] == ["l", "r"] and meta["objects"] == ["red_block", "blue_block"]
    assert len(meta["action_names"]) == 16 and meta["action_names"][7] == "l_gripper"
    assert meta["contact_kinds"] == list(CONTACT_KINDS) and meta["source"] == "scripted"
    assert meta["stage_names"] == ["red_in_area", "blue_on_red"] and len(meta["task_version"]) == 16
    for key in ("model_xml", "task_yaml", "config_yaml", "scene_hash", "instruction", "created_at", "seed"):
        assert key in meta
    shapes = {"time": (n,), "wall_time": (n,), "obs/joint_pos": (n, 14), "obs/joint_vel": (n, 14),
              "obs/gripper_width": (n, 2), "obs/ee_pose": (n, 2, 7), "obs/ee_wrench": (n, 2, 6),
              "obs/objects": (n, 2, 7), "obs/fixture_joints": (n, 0), "action/joint_pos": (n, 16),
              "action/ee_pose": (n, 16), "task/success": (n,), "task/progress": (n,), "task/stages": (n, 2),
              "task/held": (n, 2), "contacts": (n, 3), "control/source": (n,), "safety/correction": (n,),
              "teleop/head": (n, 7), "teleop/left/joints": (n, 25, 7), "teleop/right/buttons": (n, 8),
              "teleop/engaged": (n, 2), "teleop/tracked": (n, 2), "teleop/rtt_ms": (n,)}
    for key, shape in shapes.items():
        assert data[key].shape == shape, key
    assert not any("image" in k for k in data)                      # images are never recorded
    assert set(np.unique(data["task/held"])) >= {-1, 0, 1}          # both blocks were held at some point
    assert data["task/progress"].max() == 2 and data["task/success"][-1]
    assert path.stat().st_size / (n / 50 / 60) < 8e6                # ~6 MB per minute
    assert list_episodes(path.parent)[0]["task"] == "stack_two_blocks"


def test_scene_from_episode_does_not_depend_on_todays_task_files(expert_episode, task_copy, tmp_path):
    """P4: an episode carries its own model and task; changing the task afterwards changes nothing for it."""
    import yaml

    from xrcollect.env import Env
    from xrcollect.expert import Expert
    from xrcollect.task.loader import load_bundle

    env = Env(load_bundle(task_copy))
    ex = Expert(env)
    ex.reset(stream_seed("validate", 1))
    ex.wait(0.4)
    path = ex.save(tmp_path / "episode_000000.hdf5")
    meta, data = read_episode(path)
    raw = yaml.safe_load((task_copy / "task.yaml").read_text())
    raw["objects"]["green_block"] = {"asset": "blocks/cube_4cm", "pos": [0.6, 0.1]}   # another state size
    (task_copy / "task.yaml").write_text(yaml.safe_dump(raw))
    assert Env(load_bundle(task_copy)).state_size != env.state_size
    old = env_from_episode(meta)
    assert old.scene_matches and old.state_size == env.state_size and old.objects == ["red_block", "blue_block"]
    old.set_state(data["sim/state"][5])
    assert scene_hash(load_model(meta["model_xml"], old.adapter.asset_dirs)) == meta["scene_hash"]
    meta["scene_hash"] = "0" * 16
    assert not env_from_episode(meta).scene_matches                  # a changed scene is reported, not hidden


def test_gripper_moves_at_the_real_hands_speed():
    env = get_env("tasks/stack_two_blocks")
    env.reset(0)
    q = env.adapter.home_q
    widths = []
    for _ in range(60):
        env.apply_action(q, np.zeros(2))
        env.step()
        widths.append(env.observe()["gripper_width"][0])
    speed = (widths[10] - widths[35]) / (25 * env.control_dt)       # opening closes at 2 fingers x 50 mm/s
    assert 0.08 < speed < 0.12 and widths[-1] < 0.01


def test_ik_converges_and_does_not_wind_up():
    env = get_env("tasks/stack_two_blocks")
    cfg = Config()
    ik = ArmIK(env.bundle, env.adapter, cfg.ik, env.control_dt)
    p0, q0 = ik.tcp_pose("l")
    ik.set_target("l", p0 + [0.1, -0.05, -0.1], q0)
    for _ in range(150):
        ik.solve()
    assert ik.errors()["l"][0] < 1e-3 and ik.errors()["r"][0] < 1e-3
    # A blocked arm: the measured joints stay at home, so the IK may lead them by max_lead at most.
    ik.reset(env.adapter.home_q)
    ik.set_target("l", p0 + [0.3, 0.0, 0.2], q0)
    for _ in range(100):
        ik.follow(env.adapter.home_q)
        q = ik.solve()
    step = env.adapter.max_joint_vel * env.control_dt
    assert np.abs(q - env.adapter.home_q).max() <= cfg.ik.max_lead + step + 1e-9


def test_recorder_state_machine(tmp_path):
    env = get_env("tasks/stack_two_blocks")
    env.reset(0)
    cfg = Config()
    cfg.record.countdown_s = 0.2
    rec = Recorder(cfg, tmp_path, env.control_hz)
    q, g = env.adapter.home_q, np.ones(2)
    ee = env.tcp_poses()

    def step(now, success=None):
        rec.tick(now)
        s, ok, _ = control_step(env, snapshot(env) if rec.recording else None, q, g, ee, None)
        return rec.append(now, s, ok if success is None else success) if s is not None else False

    assert rec.state == RecState.IDLE and rec.save(env, "none") is None
    rec.start(0.0)
    assert rec.state == RecState.COUNTDOWN and not step(0.1) and not rec.steps
    assert round(rec.countdown_left(0.1), 2) == 0.1
    rec.stop()                                                    # cancel in the countdown
    assert rec.state == RecState.IDLE
    rec.start(1.0)
    for k in range(20):
        step(1.3 + 0.02 * k)
    assert rec.recording and len(rec.steps) == 20
    assert not step(1.8, success=True) and rec.success_seen       # success must hold for the tail
    assert not step(1.9, success=False) and rec.success_since is None
    step(2.0, success=True)
    assert step(2.0 + cfg.record.auto_stop_tail_s, success=True)  # auto-stop
    assert rec.state == RecState.REVIEW
    path = rec.save(env, "controllers", success=False)            # a failure is a separate, explicit choice
    assert rec.state == RecState.IDLE and path.endswith("episode_000000.hdf5")
    rec.flush()
    assert read_meta(path)["success"] is False and read_meta(path)["num_steps"] == 24
    assert next_episode_path(tmp_path).name == "episode_000001.hdf5"
    rec.start(5.0)
    step(5.3)
    assert rec.discard() and rec.state == RecState.IDLE and rec.discarded_count == 1
    rec.close()


def test_save_never_blocks_on_disk(tmp_path, monkeypatch):
    import xrcollect.recorder as R

    env = get_env("tasks/stack_two_blocks")
    env.reset(0)
    cfg = Config()
    cfg.record.countdown_s = 0
    rec = Recorder(cfg, tmp_path, env.control_hz)
    real = R.write_episode
    monkeypatch.setattr(R, "write_episode", lambda *a: (time.sleep(0.4), real(*a))[1])
    rec.start(0.0)
    rec.tick(0.0)
    s, ok, _ = control_step(env, snapshot(env), env.adapter.home_q, np.ones(2), env.tcp_poses(), None)
    rec.append(0.0, s, ok)
    rec.stop()
    t = time.monotonic()
    assert rec.save(env, "none") and time.monotonic() - t < 0.2
    rec.close()
    assert len(list(tmp_path.glob("*.hdf5"))) == 1


def test_action_layout_helpers():
    q, g = np.arange(14.0), np.array([0.25, 0.75])
    a = join_action(q, g, 7)
    assert a[7] == 0.25 and a[15] == 0.75 and a[8] == 7.0
    q2, g2 = split_action(np.stack([a, a]), 7)
    assert np.array_equal(q2[1], q) and np.array_equal(g2[0], g)


def test_cameras_never_see_operator_only_geoms_and_renders_repeat(expert_episode):
    path, _ = expert_episode("tasks/stack_two_blocks")
    meta, data = read_episode(path)
    env = env_from_episode(meta)
    m = env.model
    marker = m.geom("l_tcp_marker").id
    assert m.geom_group[marker] == 5
    r = CameraRenderer(m, 120, 160, env.camera_names)
    env.set_state(data["sim/state"][200])
    a = r.render(env.data)
    m.geom_rgba[marker] = [1, 0, 1, 1]
    m.geom_size[marker] = [0.2, 0, 0]              # a huge magenta marker: still not in any camera image
    b = r.render(env.data)
    for cam in env.camera_names:
        assert np.abs(a[cam].astype(int) - b[cam].astype(int)).max() <= 1, cam
    assert a["cam_high"].std() > 5                 # ... and the images are not blank
    r.close()
    assert mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_INTEGRATION) == env.mj_state_size
