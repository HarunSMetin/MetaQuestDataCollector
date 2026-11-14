"""Offscreen camera rendering (previews, exporters, policy input) and per-episode visual randomization."""

from __future__ import annotations

import io

import mujoco
import numpy as np
from PIL import Image


class CameraRenderer:
    """Renders the named cameras as the policy sees them.

    Flat, matte light: no shadows (moving arm shadows are clutter the policy would have to learn to ignore), no
    specular highlights (glare spots that move with the camera) and no floor reflection. Sets the model's light /
    headlight specular to 0, which only rendering reads. Geom groups 0-2 only: operator aids (group 5) never show.
    """

    def __init__(self, model: mujoco.MjModel, height: int, width: int, cameras: list[str]):
        self.cameras = list(cameras)
        model.light_specular[:] = 0.0
        model.vis.headlight.specular[:] = 0.0
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        for flag in (mujoco.mjtRndFlag.mjRND_SHADOW, mujoco.mjtRndFlag.mjRND_REFLECTION):
            self.renderer.scene.flags[flag] = 0
        self.option = mujoco.MjvOption()
        self.option.geomgroup[:] = [1, 1, 1, 0, 0, 0]

    def render(self, data: mujoco.MjData) -> dict[str, np.ndarray]:
        out = {}
        for cam in self.cameras:
            self.renderer.update_scene(data, camera=cam, scene_option=self.option)
            out[cam] = self.renderer.render().copy()
        return out

    def render_jpeg(self, data: mujoco.MjData, quality: int = 70) -> dict[str, bytes]:
        out = {}
        for cam, img in self.render(data).items():
            buf = io.BytesIO()
            Image.fromarray(img).save(buf, format="JPEG", quality=quality)
            out[cam] = buf.getvalue()
        return out

    def close(self) -> None:
        self.renderer.close()


class AppearanceRandomizer:
    """Per-episode visual variation for exported images (domain randomization): lighting, surface colours, object
    brightness (not hue: instructions name colours) and the camera mounts a little. Rendering only: physics and the
    recorded data are untouched."""

    FIELDS = ("light_diffuse", "light_pos", "mat_rgba", "cam_pos")

    def __init__(self, model: mujoco.MjModel, object_names: list[str]):
        self.model = model
        self.base = {k: getattr(model, k).copy() for k in self.FIELDS}
        mats = {model.material(i).name: i for i in range(model.nmat)}
        self.surface_mats = [mats[n] for n in ("table_mat", "floor_mat") if n in mats]
        self.object_mats = [mats[f"{o}_mat"] for o in object_names if f"{o}_mat" in mats]
        self.world_cams = [c for c in range(model.ncam) if model.cam_bodyid[c] == 0]
        self.body_cams = [c for c in range(model.ncam) if model.cam_bodyid[c] != 0]

    def apply(self, rng: np.random.Generator) -> None:
        m = self.model
        self.restore()
        m.light_diffuse[:] *= rng.uniform(0.6, 1.4, (m.nlight, 1))
        m.light_pos[:, :2] += rng.uniform(-0.4, 0.4, (m.nlight, 2))
        m.mat_rgba[self.surface_mats, :3] *= rng.uniform(0.8, 1.2, (len(self.surface_mats), 3))
        m.mat_rgba[self.object_mats, :3] *= rng.uniform(0.85, 1.15, (len(self.object_mats), 1))
        np.clip(m.mat_rgba, 0.0, 1.0, out=m.mat_rgba)
        m.cam_pos[self.world_cams] += rng.uniform(-0.015, 0.015, (len(self.world_cams), 3))
        m.cam_pos[self.body_cams] += rng.uniform(-0.003, 0.003, (len(self.body_cams), 3))

    def restore(self) -> None:
        for k, v in self.base.items():
            getattr(self.model, k)[:] = v
