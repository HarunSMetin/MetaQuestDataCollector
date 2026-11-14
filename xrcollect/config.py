"""Platform settings loaded from config.yaml.

Only what is independent of the task lives here: server, teleop, IK, recording, curation, feedback cues. Everything
that defines the world (robots, scene, objects, task logic) is in the four task layers (xrcollect/task/loader.py)
and is stored with every episode.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config.yaml"


class TeleopConfig(BaseModel):
    mode: Literal["controllers", "hands"] = "controllers"
    # clutched: target = arm pose at engage + scaled hand motion since (default); absolute: the arm goes to the hand
    # pose (plus the offset at engage in rotation only); rate: hand offset from the engage point = TCP velocity.
    control: Literal["clutched", "absolute", "rate"] = "clutched"
    rate_gain: float = 2.0             # rate control: (m/s) per metre of hand offset
    scale: float = 1.0
    # Which hand drives which arm: {"l": <arm>, "r": <arm>}. None = by embodiment: arms named l/r map to the same
    # hand, a single arm goes to the right hand. Swap the entries for left-handed use; give both hands the same arm
    # name for one-handed switching (the hand that engaged last drives).
    mapping: dict[str, str] | None = None
    pinch_open_m: float = 0.08
    pinch_closed_m: float = 0.015
    # Hands clutch: close middle, ring and little finger (a loose fist, index and thumb free) and open them again
    # to toggle. Per finger, curl = |tip - wrist| / |knuckle - wrist| (hand-size free): ~1.9 open, ~0.8 curled.
    curl_on: float = 1.25              # all three below this: gesture made (toggles once)
    curl_off: float = 1.5              # their mean back above this: hand open again, ready for the next toggle
    filter_min_cutoff: float = 1.5
    filter_beta: float = 0.5


class IKConfig(BaseModel):
    position_cost: float = 1.0
    orientation_cost: float = 0.6
    posture_cost: float = 0.01
    max_lead: float = 0.15             # rad the IK may run ahead of the measured joints
    damping_cost: float = 0.1
    iterations: int = 2
    warn_m: float = 0.03               # target farther than this from what the IK reaches = out of reach


class RecordConfig(BaseModel):
    out_dir: str = "data/episodes"
    countdown_s: float = 3
    auto_stop_tail_s: float = 1.0      # success has to hold this long before a take stops by itself
    cameras: list[str] | None = None   # dashboard/headset previews; None = every camera of the scene
    operator: str = "operator"
    session: str = ""                  # free text; stored per episode for multi-operator statistics
    target_takes: int = 0              # session panel quota (0 = none)
    action_noise: float = 0.0          # DART: std (rad) of a slow drift on the executed joint targets


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    https_port: int = 8443
    cert_dir: str = "certs"


class CurateConfig(BaseModel):
    """Which recorded steps go into a dataset (audit.select). Contact kinds: env.CONTACT_KINDS."""
    include_failures: bool = False
    max_contact_frac: dict[str, float] = Field(
        default_factory=lambda: {"link_table": 0.01, "arm_arm": 0.0, "tip_table": 0.3})
    # Contacts of these kinds are cut out of the take (plus a margin) instead of rejecting it.
    cut_contacts: list[str] = Field(default_factory=lambda: ["link_table", "arm_arm"])
    contact_margin_s: float = 0.2
    trim_idle: bool = True
    shared_frames: Literal["operator", "all"] = "operator"  # shared-control takes: only the operator's steps
    until_stage: int | None = None     # sub-task dataset: end each take this long after stage k (1-based) was reached
    until_stage_tail_s: float = 1.0
    cut_labels: list[str] = Field(default_factory=lambda: ["bad"])  # time annotations with these labels are cut out


class SafetyConfig(BaseModel):
    enabled: bool = True
    warn_rad: float = 0.05             # buzz the operator when the filter changes a joint target this much


class ControlConfig(BaseModel):
    """Who drives the arms: the operator, a policy, or the policy until the operator takes over (shared)."""
    mode: Literal["teleop", "policy", "shared"] = "teleop"
    policy: str | None = None          # "module:Class" or "path/to/file.py:Class" of a PolicyRunner (policy.py)
    policy_args: dict[str, Any] = Field(default_factory=dict)


class Cue(BaseModel):
    """A feedback pattern: haptic [intensity 0..1, ms, repeat, gap_ms] and a sound name (web/src/xr/sounds.ts)."""
    haptic: list[float] | None = None
    sound: str | None = None


def _default_cues() -> dict[str, Cue]:
    c = {
        "grasp": ([0.5, 40, 1, 0], "grasp"), "hold": ([0.15, 20, 1, 0], None), "release": ([0.2, 20, 1, 0], "release"),
        "click": ([0.8, 15, 1, 0], "click"), "unclick": ([0.4, 30, 1, 0], "unclick"),
        "placed": ([0.9, 60, 2, 70], "placed"), "lost": ([0.3, 40, 1, 0], "unclick"),
        "complete": ([1.0, 120, 3, 80], "complete"), "area": (None, "area"),
        "engage": ([0.6, 40, 1, 0], "engage"), "disengage": ([0.25, 25, 1, 0], "disengage"),
        "glitch": ([0.8, 60, 1, 0], None), "warn": ([0.3, 30, 1, 0], None),
        "collision": (None, "collision"), "table": (None, "table"), "out_of_reach": (None, "out_of_reach"),
        "rec_start": ([1.0, 120, 1, 0], "rec_start"), "saved": ([0.8, 80, 1, 0], "saved"),
    }
    return {k: Cue(haptic=h, sound=s) for k, (h, s) in c.items()}


class FeedbackConfig(BaseModel):
    cues: dict[str, Cue] = Field(default_factory=_default_cues)
    hold_tick_s: float = 0.25          # a soft tick this often while holding an object
    quiet_s: float = 0.4               # no hold ticks right after another pattern
    warn_sound_gap_s: float = 1.0      # at most one sound per warning kind and arm per second
    guidance: bool = True              # operator-only ghost of the next object's goal pose + distance readout


class Config(BaseModel):
    task: str = "tasks/stack_two_blocks"   # task folder or YAML file (relative to the repository root)
    teleop: TeleopConfig = Field(default_factory=TeleopConfig)
    ik: IKConfig = Field(default_factory=IKConfig)
    record: RecordConfig = Field(default_factory=RecordConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    curate: CurateConfig = Field(default_factory=CurateConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    control: ControlConfig = Field(default_factory=ControlConfig)
    feedback: FeedbackConfig = Field(default_factory=FeedbackConfig)

    @property
    def out_dir(self) -> Path:
        return rooted(self.record.out_dir)


def rooted(p: str | Path) -> Path:
    """Config paths are relative to the repository root."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    if not path.exists() and path == DEFAULT_CONFIG:
        return Config()
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    cfg = Config.model_validate(raw)
    # Cues given in the file are added to the defaults instead of replacing the whole table.
    cues = _default_cues()
    cues.update(cfg.feedback.cues)
    cfg.feedback.cues = cues
    return cfg
