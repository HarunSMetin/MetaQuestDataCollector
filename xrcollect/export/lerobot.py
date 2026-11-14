"""LeRobot dataset export (needs `uv sync --extra lerobot`). All LeRobot API usage lives in this file.

Features: observation.state, action, observation.images.<camera> (video), optionally
observation.environment_state; the task string is the episode's instruction (or the annotation label).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class LeRobotExporter:
    def begin(self, out: Path, info: dict) -> None:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError as e:
            raise RuntimeError("the LeRobot exporter needs the lerobot package: uv sync --extra lerobot") from e

        names, (h, w) = info["names"], info["image_size"]
        features = {
            "observation.state": {"dtype": "float32", "shape": (len(names["state"]),), "names": names["state"]},
            "action": {"dtype": "float32", "shape": (len(names["action"]),), "names": names["action"]},
        }
        if info["layout"]["env_state"]:
            features["observation.environment_state"] = {"dtype": "float32", "shape": (len(names["env_state"]),),
                                                         "names": names["env_state"]}
        for cam in info["cameras"]:
            features[f"observation.images.{cam}"] = {"dtype": "video", "shape": (h, w, 3),
                                                     "names": ["height", "width", "channels"]}
        self.env_state = info["layout"]["env_state"]
        out = Path(out)
        if out.exists():  # LeRobot wants to create the folder itself
            out.rmdir()
        self.ds = LeRobotDataset.create(repo_id=f"xrcollect/{out.name}", fps=info["fps"], features=features,
                                        root=out, robot_type=info.get("robot_type") or "robot",
                                        use_videos=bool(info["cameras"]), image_writer_threads=4)

    def add_episode(self, frames: dict, info: dict) -> None:
        for t in range(len(frames["action"])):
            frame = {"observation.state": frames["state"][t].astype(np.float32),
                     "action": frames["action"][t].astype(np.float32), "task": info["instruction"]}
            if self.env_state:
                frame["observation.environment_state"] = frames["env_state"][t].astype(np.float32)
            for cam, imgs in frames.get("images", {}).items():
                frame[f"observation.images.{cam}"] = imgs[t]
            self.ds.add_frame(frame)
        self.ds.save_episode()

    def finish(self) -> dict:
        if hasattr(self.ds, "finalize"):
            self.ds.finalize()
        return {"format": "LeRobot dataset", "repo_id": self.ds.repo_id}
