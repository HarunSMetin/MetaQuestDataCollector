"""The DUPLO tower as a plugin task: what the predicate library can't express.

- Snap test on the stud grid: a brick may click onto another wherever its sockets line up with studs (any offset,
  any multiple of 90 degrees), not only at the tower's poses.
- Stages: a level counts only when it is at its tower pose AND clicked in; the whole tower may stand turned by
  180 degrees on its base (the base looks the same both ways), but every level has to agree on which way.
"""

import numpy as np

from xrcollect.scene.constants import quat_yaw, rot_z, wrap
from xrcollect.task.declarative import DeclarativeTask
from xrcollect.task.predicates import footprint_inside
from xrcollect.task.snap import SnapPair


class TowerTask(DeclarativeTask):
    def __init__(self, bundle):
        self.params = bundle.task["params"]
        self.bricks = list(bundle.objects)                     # bottom to top
        self.info = bundle.objects
        self.pitch = float(self.info[self.bricks[0]]["pitch"])
        super().__init__(bundle)

    # -- snap: stud press fit --------------------------------------------------------------------------
    def snap_pairs(self):
        s = self.params["snap"]
        return [SnapPair(a, b, self.stud_seated, s["press_n"], s["pull_n"])
                for i, a in enumerate(self.bricks) for b in self.bricks[i + 1:]]

    def stud_seated(self, data, lower: str, upper: str) -> bool:
        """Whether the upper brick sits on the lower one's studs: right height, flat, yaw a multiple of 90 degrees
        and its sockets within align_xy of the studs (the stud lead-ins centre it that far)."""
        s, w = self.params["snap"], self.world
        lo, up = self.info[lower], self.info[upper]
        bl, bu = w.body[lower], w.body[upper]
        rl = data.xmat[bl].reshape(3, 3)
        rel = rl.T @ data.xmat[bu].reshape(3, 3)
        p = rl.T @ (data.xpos[bu] - data.xpos[bl])
        yaw = float(np.arctan2(rel[1, 0], rel[0, 0]))
        k = round(yaw / (np.pi / 2))
        tol_yaw = np.deg2rad(s["align_yaw_deg"])
        if abs(p[2] - lo["size"][2]) > s["align_z"] or rel[2, 2] < np.cos(tol_yaw) \
                or abs(yaw - k * np.pi / 2) > tol_yaw:
            return False
        (lnx, lny), (unx, uny) = lo["studs"], up["studs"]
        ux, uy = (unx, uny) if k % 2 == 0 else (uny, unx)
        # A brick with n studs has them at (i - (n-1)/2) * pitch, so sockets line up with studs when the centre
        # offset minus this half-stud parity term is a whole number of studs.
        parity = np.array([(ux - lnx) / 2, (uy - lny) / 2])
        g = p[:2] / self.pitch - parity
        return bool(np.all(np.abs(g - np.round(g)) * self.pitch <= s["align_xy"]))

    # -- the tower ---------------------------------------------------------------------------------------
    def tower_poses(self, base_pos, base_yaw: float) -> list[tuple[np.ndarray, float]]:
        """Goal (position, yaw) of every brick for a tower whose base brick is at base_pos / base_yaw."""
        pos, yaw = np.asarray(base_pos, float).copy(), float(base_yaw)
        out = [(pos.copy(), yaw)]
        for name, (dx, dy, dyaw) in zip(self.bricks, self.params["tower"]):
            pos = pos + rot_z(yaw) @ [self.pitch * dx, self.pitch * dy, self.info[name]["size"][2]]
            yaw += np.deg2rad(dyaw)
            out.append((pos.copy(), yaw))
        return out

    def goal_view(self) -> dict:
        poses = self.tower_poses([0.0, 0.0, 0.0], 0.0)
        return {"anchor": self.bricks[0],
                "show": [{"object": n, "pose": {"pos": p.tolist(), "yaw": float(y)}}
                         for n, (p, y) in list(zip(self.bricks, poses))[1:]]}

    def stage_flags(self, model, data) -> np.ndarray:
        w, P = self.world, self.params
        tol_yaw = np.deg2rad(P["tol_yaw_deg"])
        ids = [w.body[n] for n in self.bricks]
        up = [data.xmat[b][8] for b in ids]  # z axis . world z
        flags = np.zeros(len(self.bricks), bool)
        base = self.bricks[0]
        flags[0] = up[0] >= np.cos(tol_yaw) and footprint_inside(w, data, base, w.regions["build_area"])
        if not flags[0]:
            return flags
        best = 0
        for flip in (0.0, np.pi):
            goal = self.tower_poses(data.xpos[ids[0]], quat_yaw(data.xquat[ids[0]]) + flip)
            n = 0
            for k in range(1, len(self.bricks)):
                p, (gp, gyaw) = data.xpos[ids[k]], goal[k]
                nx, ny = self.info[self.bricks[k]]["studs"]
                period = np.pi / 2 if nx == ny else np.pi
                if not (up[k] > np.cos(tol_yaw) and np.linalg.norm(p[:2] - gp[:2]) < P["tol_xy"]
                        and abs(p[2] - gp[2]) < P["tol_z"]
                        and abs(wrap(quat_yaw(data.xquat[ids[k]]) - gyaw, period)) < tol_yaw
                        and self.snaps.is_on(data, self.bricks[k - 1], self.bricks[k])):
                    break
                n = k
            best = max(best, n)
        flags[1:best + 1] = True
        return flags
