"""Safety filter: free motion is untouched; near the table and the other arm it stops at the margin."""

import mujoco
import numpy as np

from conftest import get_env
from xrcollect.config import Config
from xrcollect.ik import ArmIK
from xrcollect.safety import SafetyFilter


def make(task="tasks/stack_two_blocks", embodiment=None):
    env = get_env(task, embodiment)
    cfg = Config()
    return env, ArmIK(env.bundle, env.adapter, cfg.ik, env.control_dt), \
        SafetyFilter(env.bundle, env.adapter, cfg.ik, cfg.safety, env.control_dt)


def test_free_motion_is_unchanged():
    env, ik, sf = make()
    q = env.adapter.home_q.copy()
    q[1] += 0.01
    safe, change = sf.filter(q, env.adapter.home_q)
    assert np.allclose(safe, q, atol=1e-4) and change < 1e-4


def test_fingers_stop_above_the_table():
    env, ik, sf = make()
    p, quat = ik.tcp_pose("l")
    ik.set_target("l", [p[0], p[1], env.table_height - 0.05], quat)   # 5 cm below the table top
    q = env.adapter.home_q.copy()
    biggest = 0.0
    for _ in range(200):
        ik.follow(q)
        q, change = sf.filter(ik.solve(), q, np.full((2, 2), 0.04))
        biggest = max(biggest, change)
    z = sf.tcp_poses(q)[0][2]
    margin = env.bundle.embodiment["safety"]["finger_margin"]
    assert z > env.table_height - 0.002 and biggest > 0.01          # held back, and it says so
    d = mujoco.MjData(sf.model)
    d.qpos[:] = sf.configuration.q
    mujoco.mj_forward(sf.model, d)
    table = sf.model.geom("table_top").id
    gaps = [mujoco.mj_geomDistance(sf.model, d, g, table, 0.2, None)
            for g in range(sf.model.ngeom) if sf.model.geom_contype[g] and sf.b.body_tip[sf.model.geom_bodyid[g]]]
    assert min(gaps) > margin - 0.003


def test_arms_keep_apart():
    env, ik, sf = make()
    pl, ql = ik.tcp_pose("l")
    pr, _ = ik.tcp_pose("r")
    ik.set_target("l", pr, ql)                                       # the left hand goes for the right hand
    q = env.adapter.home_q.copy()
    for _ in range(200):
        ik.follow(q)
        q, _ = sf.filter(ik.solve(), q)
    poses = sf.tcp_poses(q)
    assert np.linalg.norm(poses[0][:3] - poses[1][:3]) > 0.03
    d = mujoco.MjData(sf.model)
    d.qpos[:] = sf.configuration.q
    mujoco.mj_forward(sf.model, d)
    assert d.ncon == 0 or not any(
        sf.b.body_arm[sf.model.geom_bodyid[c.geom1]] * sf.b.body_arm[sf.model.geom_bodyid[c.geom2]] == 2
        for c in d.contact)


def test_disabled_filter_passes_targets_through():
    env, _, _ = make()
    cfg = Config()
    cfg.safety.enabled = False
    sf = SafetyFilter(env.bundle, env.adapter, cfg.ik, cfg.safety, env.control_dt)
    q = env.adapter.home_q + 0.5
    safe, change = sf.filter(q, env.adapter.home_q)
    assert np.array_equal(safe, q) and change == 0.0


def test_single_arm_has_no_arm_pair_limit():
    env, ik, sf = make(embodiment="franka_single")
    assert sf.arms == ["arm"] and len(sf.limits) == 4               # configuration, velocity, links, fingers
    q = env.adapter.home_q.copy()
    safe, change = sf.filter(q, q)
    assert change < 1e-6 and sf.tcp_poses(safe).shape == (1, 7)
