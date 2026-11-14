"""Scene fingerprint: stored with every episode, so replaying or re-rendering it in a changed scene is caught."""

from __future__ import annotations

import hashlib

import mujoco
import numpy as np

from xrcollect.scene.constants import STATE_SPEC


def scene_hash(model: mujoco.MjModel) -> str:
    """Hash of what shapes the physics and the images (geoms, masses, materials, cameras, state layout). Always
    computed on a model compiled from the saved XML (compose.load_model): the in-memory compiled spec differs from
    it by float noise."""
    h = hashlib.sha1()
    for arr in (model.geom_type, model.geom_bodyid, model.geom_contype, model.geom_conaffinity,
                model.geom_size.round(6), model.geom_pos.round(6), model.body_mass.round(6),
                model.mat_rgba.round(4), model.cam_pos.round(6), model.eq_type, model.jnt_type):
        h.update(np.ascontiguousarray(arr).tobytes())
    h.update(str(mujoco.mj_stateSize(model, STATE_SPEC)).encode())
    return h.hexdigest()[:16]
