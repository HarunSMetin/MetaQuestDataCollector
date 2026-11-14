"""Safety filter on every joint target, whoever made it (operator IK, policy, script): change the proposed targets
as little as possible so the arms keep apart from each other and from the support surfaces, within joint position
and speed limits. It only prevents collisions; it never helps with the task.

One QP per control step with mink, on the robots-only model: a posture task pulls towards the proposed joint
targets, and the limits keep it safe. In free motion the output equals the input; near an obstacle the arms slow
down and stop at the margin. Grippers are not filtered. Which bodies count as links / fingers comes from the robot
adapter, the margins from the embodiment file, the surfaces from the scene.
"""

from __future__ import annotations

import itertools

import mink
import numpy as np

from xrcollect.config import IKConfig, SafetyConfig
from xrcollect.ik import collision_geoms, robot_model
from xrcollect.task.loader import Bundle

DETECTION_EXTRA = 0.08  # m beyond the margin at which an approaching obstacle starts to slow the arm
# QP steps per control step. The collision limits are linearized: one 20 ms step let fingertips sink 7.6 mm into
# the table and arms close to 3 mm; 8 substeps keep them at ~0 mm and ~15 mm, for ~1.5 ms per control step.
SUBSTEPS = 8
SURFACE_GEOMS = ("table_top",)


class SafetyFilter:
    def __init__(self, bundle: Bundle, adapter, ik_cfg: IKConfig, cfg: SafetyConfig, dt: float):
        self.cfg, self.dt, self.max_lead, self.adapter = cfg, dt, ik_cfg.max_lead, adapter
        self.arms = list(adapter.arms)
        margins = bundle.embodiment.get("safety", {})
        arm_margin = float(margins.get("arm_margin", 0.02))
        link_margin = float(margins.get("link_margin", 0.01))
        finger_margin = float(margins.get("finger_margin", 0.005))
        m = self.model = robot_model(bundle, adapter)
        self.b = adapter.bind(m)
        self.configuration = mink.Configuration(m)
        self._fk = mink.Configuration(m)  # tcp_poses(q) for other joint angles than the filter's own
        # The fingers are only there to be in the right place: the QP may not move them to clear an obstacle.
        self._finger_range = m.jnt_range[self.b.finger_jid].T
        self.freeze = [mink.DofFreezingTask(m, [int(d) for d in self.b.finger_dof])] if len(self.b.finger_dof) else []
        self.posture = mink.PostureTask(m, cost=1.0)
        vel = {j: adapter.max_joint_vel for a in self.arms for j in adapter.joint_names(a)}
        surfaces = [m.geom(g).id for g in SURFACE_GEOMS]
        surfaces += [m.geom(p["name"]).id for p in (bundle.scene.get("props") or [])
                     if p.get("name") and p.get("collide", True)]

        def geoms(kind: str, arms=self.arms) -> list[int]:
            return [g for a in arms for g in collision_geoms(m, adapter.collision_bodies(a, kind))]

        self.limits = [mink.ConfigurationLimit(m), mink.VelocityLimit(m, vel)]
        pairs = [(geoms("arm", [a]), geoms("arm", [b])) for a, b in itertools.combinations(self.arms, 2)]
        if pairs:
            self.limits.append(mink.CollisionAvoidanceLimit(
                m, geom_pairs=pairs, minimum_distance_from_collisions=arm_margin,
                collision_detection_distance=arm_margin + DETECTION_EXTRA))
        self.limits += [
            mink.CollisionAvoidanceLimit(m, geom_pairs=[(geoms("link"), surfaces)],
                                         minimum_distance_from_collisions=link_margin,
                                         collision_detection_distance=link_margin + DETECTION_EXTRA),
            mink.CollisionAvoidanceLimit(m, geom_pairs=[(geoms("finger"), surfaces)],
                                         minimum_distance_from_collisions=finger_margin,
                                         collision_detection_distance=finger_margin + 0.02),
        ]
        self.reset(adapter.home_q)

    def reset(self, arm_q: np.ndarray) -> None:
        q = self.model.qpos0.copy()
        q[self.b.qadr] = arm_q
        self.configuration.update(q)

    def filter(self, q_proposed: np.ndarray, q_measured: np.ndarray,
               fingers: np.ndarray | None = None) -> tuple[np.ndarray, float]:
        """Proposed joint targets -> (safe targets, largest change in rad). Like ArmIK.follow, the filter's own
        configuration stays within ik.max_lead of the measured joints, so it can't wind up.
        fingers: the measured finger joints (n_arms, n_fingers) [m]. An open finger sits off the hand's centre, so
        with the hand tilted it hangs lower than a closed one; without them the fingers count as closed."""
        q = self.configuration.q.copy()
        qadr = self.b.qadr
        if fingers is not None and len(self.b.finger_dof):  # just inside the range: frozen at a limit is infeasible
            lo, hi = self._finger_range
            q[self.b.finger_qadr.ravel()] = np.clip(np.ravel(fingers), lo + 1e-6, hi - 1e-6)
        if not self.cfg.enabled:
            q[qadr] = q_proposed
            self.configuration.update(q)
            return np.asarray(q_proposed, float).copy(), 0.0
        lead = q[qadr] - q_measured
        q[qadr] = q_measured + np.clip(lead, -self.max_lead, self.max_lead)
        self.configuration.update(q)
        target = q.copy()
        target[qadr] = q_proposed
        self.posture.set_target(target)
        dt = self.dt / SUBSTEPS
        for _ in range(SUBSTEPS):
            v = mink.solve_ik(self.configuration, [self.posture], dt, solver="daqp", damping=1e-6,
                              limits=self.limits, constraints=self.freeze)
            self.configuration.integrate_inplace(v, dt)
        q_safe = self.configuration.q[qadr].copy()
        return q_safe, float(np.abs(q_safe - q_proposed).max())

    def tcp_poses(self, arm_q: np.ndarray | None = None) -> np.ndarray:
        """(n_arms, 7) TCP pose [x y z qw qx qy qz] at the last filtered targets, or at the joint angles given
        (recorded as action/ee_pose)."""
        conf = self.configuration
        if arm_q is not None:
            q = self.model.qpos0.copy()
            q[self.b.qadr] = arm_q
            self._fk.update(q)
            conf = self._fk
        out = []
        for a in self.arms:
            T = conf.get_transform_frame_to_world(self.adapter.tcp_site(a), "site")
            out.append(np.concatenate([T.translation(), T.rotation().wxyz]))
        return np.array(out)
