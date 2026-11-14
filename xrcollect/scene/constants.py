"""Constants and small helpers shared by everything that builds or reads the scene."""

from __future__ import annotations

import mujoco
import numpy as np

STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION
# Geom groups: 0-2 are rendered by the policy cameras; 3 = invisible helper geoms; 4 = sites;
# 5 = operator-only visuals (markers, guides, ghosts): drawn in the headset and dashboard, never by a camera.
CAMERA_GROUPS = (0, 1, 2)
HIDDEN_GROUP = 3
OPERATOR_GROUP = 5
# Rigid contacts (objects, arm links, table top): timeconst 0.004 s (>= 2 * physics_dt for stability) and
# impedance 0.99-0.999, so a part pressed with 50+ N sinks < 0.4 mm into another and an arm pushing at full
# torque stays on the surface. MuJoCo's defaults let parts sink > 1 mm and arms sink centimetres.
RIGID_SOLREF = [0.004, 1]
RIGID_SOLIMP = [0.99, 0.999, 0.0005, 0.5, 2]
# Collision bits: everything collides on bit 0; bit 1 links robot geoms to object "cores" only. A core is an
# invisible solid inside a hollow object that only the robot hits, so a finger can't hook into the cavity.
ROBOT_CONTYPE = 0b11
CORE_CONAFFINITY = 0b10
TAPE_W = 0.006
TAPE_RGBA = (0.92, 0.92, 0.88, 1)

GEOM_TYPES = {
    "box": mujoco.mjtGeom.mjGEOM_BOX, "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
    "sphere": mujoco.mjtGeom.mjGEOM_SPHERE, "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
    "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID, "mesh": mujoco.mjtGeom.mjGEOM_MESH,
    "plane": mujoco.mjtGeom.mjGEOM_PLANE,
}


def lookat_xyaxes(pos, target, up=(0.0, 0.0, 1.0)) -> list[float]:
    """Camera xyaxes so that the camera at `pos` looks at `target`."""
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    right = np.cross(f, up)
    if np.linalg.norm(right) < 1e-6:  # looking straight up or down: any horizontal axis will do
        right = np.cross(f, (1.0, 0.0, 0.0))
    right /= np.linalg.norm(right)
    upv = np.cross(right, f)
    return [*right, *upv]


def yaw_quat(yaw: float) -> list[float]:
    return [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))]


def quat_yaw(q) -> float:
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def rot_z(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def wrap(a: float, period: float) -> float:
    return (a + period / 2) % period - period / 2
