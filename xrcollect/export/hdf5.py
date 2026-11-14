"""Raw HDF5 export: one file per exported episode, no dependencies beyond h5py.

<out>/episode_000000.hdf5:
  attrs: instruction, fps, source_file, source_start, source_stop, success, state_names, action_names
  /observation/state            (T, S) float32
  /observation/environment_state (T, E) float32   (only with layout.env_state)
  /observation/images/<camera>  (T, H, W, 3) uint8, gzip
  /action                       (T, A) float32
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py


class Hdf5Exporter:
    def begin(self, out: Path, info: dict) -> None:
        self.out, self.info, self.n = Path(out), info, 0
        self.out.mkdir(parents=True, exist_ok=True)

    def add_episode(self, frames: dict, info: dict) -> None:
        path = self.out / f"episode_{self.n:06d}.hdf5"
        with h5py.File(path, "w") as f:
            f.attrs.update(instruction=info["instruction"], fps=self.info["fps"], source_file=info["file"],
                           source_start=info["start"], source_stop=info["stop"], success=info["success"],
                           state_names=json.dumps(self.info["names"]["state"]),
                           action_names=json.dumps(self.info["names"]["action"]))
            f.create_dataset("observation/state", data=frames["state"])
            if "env_state" in frames:
                f.create_dataset("observation/environment_state", data=frames["env_state"])
            for cam, imgs in frames.get("images", {}).items():
                f.create_dataset(f"observation/images/{cam}", data=imgs, compression="gzip", compression_opts=4,
                                 chunks=(1, *imgs.shape[1:]))
            f.create_dataset("action", data=frames["action"])
        self.n += 1

    def finish(self) -> dict:
        return {"format": "one HDF5 file per episode (see xrcollect/export/hdf5.py)"}
