"""Operator input -> per-arm end-effector targets.

Pipeline per hand and control tick:
  XR pose -> MuJoCo world (frames.Placement) -> jump rejection -> One-Euro filter -> clutch -> workspace clamp

Controllers: a Grip press toggles the clutch, Trigger closes the gripper, the buttons in BUTTONS send commands.
Hands:       thumb-middle pinch toggles the clutch, thumb-index distance sets the gripper.

Which hand drives which arm is a mapping (config teleop.mapping, default by embodiment): two hands on two arms,
one hand on a single arm, swapped for left-handed use, or both hands on one arm (the hand that engages drives).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from xrcollect.config import TeleopConfig
from xrcollect.frames import Placement
from xrcollect.protocol import (
    B_MENU, B_PRIMARY, B_SECONDARY, B_SQUEEZE, B_STICK_PRESS, B_TRIGGER, HANDS, J_MIDDLE_META, J_WRIST,
    SRC_CONTROLLER, SRC_HAND, HandInput, InputFrame,
)

MODES = ("controllers", "hands")
INPUT_TIMEOUT_S = 0.25
MAX_JUMP_M = 0.12  # hand/controller jump between control ticks treated as a tracking glitch


# -- One-Euro filtering for positions and orientations (Casiez et al., 2012) --------------------------
def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuro:
    def __init__(self, min_cutoff: float = 1.5, beta: float = 0.5, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = None
        self.dx = None

    def reset(self) -> None:
        self.x = self.dx = None

    def __call__(self, x, dt: float) -> np.ndarray:
        x = np.asarray(x, float)
        if self.x is None or dt <= 0:
            self.x, self.dx = x.copy(), np.zeros_like(x)
            return self.x.copy()
        dx = (x - self.x) / dt
        a_d = _alpha(self.d_cutoff, dt)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * np.linalg.norm(self.dx)
        a = _alpha(cutoff, dt)
        self.x = a * x + (1 - a) * self.x
        return self.x.copy()


def _slerp_mat(Ra: np.ndarray, Rb: np.ndarray, Rrel: np.ndarray, angle: float, a: float) -> np.ndarray:
    """Interpolate from Ra towards Rb (Rrel = Ra^T Rb, rotating by `angle`) by fraction a."""
    if angle < 1e-9:
        return Rb.copy()
    axis = np.array([Rrel[2, 1] - Rrel[1, 2], Rrel[0, 2] - Rrel[2, 0], Rrel[1, 0] - Rrel[0, 1]])
    n = np.linalg.norm(axis)
    if n < 1e-9:  # angle ~ pi: no unique axis, snap
        return Rb.copy()
    axis /= n
    th = a * angle
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return Ra @ (np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K)


class OneEuroRotation:
    """One-Euro style adaptive smoothing on SO(3) using angular speed."""

    def __init__(self, min_cutoff: float = 1.5, beta: float = 0.3, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.R = None
        self.speed = 0.0

    def reset(self) -> None:
        self.R = None
        self.speed = 0.0

    def __call__(self, R, dt: float) -> np.ndarray:
        R = np.asarray(R, float)
        if self.R is None or dt <= 0:
            self.R = R.copy()
            return self.R.copy()
        Rrel = self.R.T @ R
        angle = float(np.arccos(np.clip((np.trace(Rrel) - 1) / 2, -1.0, 1.0)))
        a_d = _alpha(self.d_cutoff, dt)
        self.speed = a_d * angle / dt + (1 - a_d) * self.speed
        a = _alpha(self.min_cutoff + self.beta * self.speed, dt)
        self.R = _slerp_mat(self.R, R, Rrel, angle, a)
        return self.R.copy()


# -- clutch ------------------------------------------------------------------------------------------
CONTROL_MODES = ("clutched", "absolute", "rate")
RATE_DEADZONE_M = 0.01


class ArmClutch:
    """Maps the hand to an end-effector target while engaged. Releasing and re-engaging never makes the
    orientation jump; the position behaves per control mode:

    clutched (default): p = p_E0 + s * (p_H - p_H0)        relative, indexing like a mouse
    absolute:           p = p_H                            the gripper goes to the hand
    rate:               dp/dt = gain * s * (p_H - p_H0)    hand offset = velocity, for large workspaces
    Orientation, all modes: R = (R_H @ R_H0^T) @ R_E0      (hand rotation applied in the world frame)
    """

    def __init__(self):
        self.engaged = False
        self._hp0 = self._hR0 = self._ep0 = self._eR0 = None

    def engage(self, hand_p, hand_R, ee_p, ee_R) -> None:
        self._hp0, self._hR0 = np.array(hand_p, float), np.array(hand_R, float)
        self._ep0, self._eR0 = np.array(ee_p, float), np.array(ee_R, float)
        self.engaged = True

    def release(self) -> None:
        self.engaged = False

    def target(self, hand_p, hand_R, scale: float = 1.0, mode: str = "clutched", dt: float = 0.0,
               rate_gain: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
        assert self.engaged, "target() requires an engaged clutch"
        d = np.asarray(hand_p) - self._hp0
        if mode == "absolute":
            p = np.asarray(hand_p, float).copy()
        elif mode == "rate":
            n = np.linalg.norm(d)
            v = np.zeros(3) if n < RATE_DEADZONE_M else rate_gain * scale * d * (1 - RATE_DEADZONE_M / n)
            self._ep0 = self._ep0 + v * dt
            p = self._ep0.copy()
        else:
            p = self._ep0 + scale * d
        R = np.asarray(hand_R) @ self._hR0.T @ self._eR0
        return p, R

    def clamp(self, p: np.ndarray) -> None:
        """Rate mode integrates its own position: keep it inside the workspace too."""
        self._ep0 = np.where(np.isfinite(p), p, self._ep0) if self._ep0 is not None else self._ep0


# -- buttons ---------------------------------------------------------------------------------------
# The ONE button table. Rising-edge buttons with a `command` are sent to the runtime by ControllerMapper; the rest
# is handled where noted. The headset's button labels are generated from this table (GET /api/commands), so they
# can't drift from what the buttons do. `when` names the recorder states in which the hint is lit.
BUTTON_NAMES = {B_TRIGGER: "trigger", B_SQUEEZE: "squeeze", B_PRIMARY: "primary", B_SECONDARY: "secondary",
                B_STICK_PRESS: "stick", B_MENU: "menu"}
BUTTONS = [
    {"hand": "r", "button": B_PRIMARY, "command": "record_toggle", "label": "Record / Stop",
     "labels": {"idle": "Record", "countdown": "Cancel", "recording": "Stop", "review": ""}},
    {"hand": "r", "button": B_SECONDARY, "command": "discard", "label": "Discard", "when": ["recording", "review"]},
    {"hand": "l", "button": B_PRIMARY, "command": "save", "label": "Save", "when": ["recording", "review"]},
    {"hand": "l", "button": B_SECONDARY, "command": None, "label": "Menu", "client": "menu"},
    {"hand": "l", "button": B_STICK_PRESS, "command": "recenter", "label": "Recenter"},
    {"hand": "l", "button": B_SQUEEZE, "command": None, "label": "Move arm", "client": "clutch"},
    {"hand": "r", "button": B_SQUEEZE, "command": None, "label": "Move arm", "client": "clutch"},
    {"hand": "l", "button": B_TRIGGER, "command": None, "label": "Gripper", "client": "gripper"},
    {"hand": "r", "button": B_TRIGGER, "command": None, "label": "Gripper", "client": "gripper"},
]
CONTROLLER_COMMANDS = {(b["hand"], b["button"]): b["command"] for b in BUTTONS if b["command"]}


def command_table() -> list[dict]:
    return [{**b, "button": BUTTON_NAMES[b["button"]]} for b in BUTTONS]


class ControllerMapper:
    """Grip press toggles the clutch (no need to keep squeezing), Trigger sets the gripper."""

    GRIP_ON, GRIP_OFF = 0.6, 0.4  # press hysteresis, so one press = one toggle

    def __init__(self, side: str):
        self.side = side
        self.engaged = False
        self.grip_down = False
        self.prev_buttons: dict[int, bool] = {}

    def reset(self) -> None:
        """Switch the clutch off. A grip still held down doesn't count as a new press."""
        self.engaged = False

    def update(self, h: HandInput) -> tuple[bool, float, list[str]]:
        """Returns (engaged, gripper open fraction, commands)."""
        down = h.buttons[B_SQUEEZE] > (self.GRIP_OFF if self.grip_down else self.GRIP_ON)
        if down and not self.grip_down:
            self.engaged = not self.engaged
        self.grip_down = down
        gripper = 1.0 - float(h.buttons[B_TRIGGER])
        commands = []
        for (side, b), cmd in CONTROLLER_COMMANDS.items():
            if side != self.side:
                continue
            pressed = h.buttons[b] > 0.5
            if pressed and not self.prev_buttons.get(b, False):
                commands.append(cmd)
            self.prev_buttons[b] = pressed
        return self.engaged, gripper, commands


class HandMapper:
    """Control pose = middle-metacarpal position + wrist orientation (stable while pinching).

    Gripper:  thumb-index distance, mapped linearly between closed and open distances.
    Clutch:   thumb-middle pinch toggles engagement (with hysteresis so one pinch = one toggle).
    """

    CLUTCH_ON, CLUTCH_OFF = 0.018, 0.035
    MIDDLE_MARGIN = 0.01

    def __init__(self, cfg: TeleopConfig):
        self.cfg = cfg
        self.engaged = False
        self._latched = False

    def reset(self) -> None:
        self.engaged = False
        self._latched = False

    def control_pose(self, h: HandInput) -> np.ndarray:
        pose = h.joints[J_WRIST].copy()
        pose[:3] = h.joints[J_MIDDLE_META, :3]
        return pose

    def gripper(self, h: HandInput) -> float:
        lo, hi = self.cfg.pinch_closed_m, self.cfg.pinch_open_m
        return float(np.clip((h.pinch[0] - lo) / (hi - lo), 0.0, 1.0))

    def update(self, h: HandInput) -> tuple[bool, float]:
        """Returns (engaged, gripper open fraction)."""
        d, d_index = h.pinch[1], h.pinch[0]
        # An index pinch (closing the gripper) also brings the thumb near the middle fingertip;
        # only count it as a clutch gesture when the thumb is clearly closer to the middle tip.
        is_middle_pinch = d < self.CLUTCH_ON and d < d_index - self.MIDDLE_MARGIN
        if not self._latched and is_middle_pinch:
            self._latched = True
            self.engaged = not self.engaged
        elif self._latched and d > self.CLUTCH_OFF:
            self._latched = False
        return self.engaged, self.gripper(h)


# -- teleop ----------------------------------------------------------------------------------------
@dataclass
class TeleopOutput:
    targets: dict[str, tuple[np.ndarray, np.ndarray]]   # arm -> (position, rotation matrix)
    grippers: np.ndarray                                 # (n_arms,) 0 closed .. 1 open
    engaged: dict[str, bool]
    tracked: dict[str, bool]
    commands: list[str] = field(default_factory=list)
    cues: list[tuple[str, str]] = field(default_factory=list)  # (arm, cue name): engage, disengage, glitch
    mode_mismatch: bool = False
    hand_of: dict[str, str | None] = field(default_factory=dict)  # arm -> the hand ("l"/"r") that drives it


class _HandState:
    def __init__(self, side: str, cfg: TeleopConfig):
        self.pos_filter = OneEuro(cfg.filter_min_cutoff, cfg.filter_beta)
        self.rot_filter = OneEuroRotation(cfg.filter_min_cutoff, cfg.filter_beta * 0.6)
        self.ctrl = ControllerMapper(side)
        self.hand = HandMapper(cfg)
        self.last_raw: np.ndarray | None = None

    def drop(self) -> None:
        """Tracking lost or glitched: disengage until the operator engages again."""
        self.ctrl.reset()
        self.hand.reset()
        self.pos_filter.reset()
        self.rot_filter.reset()


class _ArmState:
    def __init__(self):
        self.clutch = ArmClutch()
        self.gripper = 1.0
        self.hold_p = np.zeros(3)
        self.hold_R = np.eye(3)
        self.driver: str | None = None  # the hand whose clutch holds this arm


def default_mapping(arms: list[str]) -> dict[str, str]:
    """Arms named like the hands map to them; otherwise the right hand takes the first arm, the left the second."""
    if all(h in arms for h in HANDS):
        return {h: h for h in HANDS}
    out = {"r": arms[0]}
    if len(arms) > 1:
        out["l"] = arms[1]
    return out


class Teleop:
    def __init__(self, cfg: TeleopConfig, arms: list[str], workspace: dict | None = None, operator_x: float = -0.5):
        self.cfg = cfg
        self.arm_names = list(arms)
        self.mode = cfg.mode if cfg.mode in MODES else "controllers"
        self.control = cfg.control if cfg.control in CONTROL_MODES else "clutched"
        self.placement = Placement(operator_x=operator_x)
        self.scale = cfg.scale
        ws = workspace or {}
        self.ws_min = np.array(ws.get("min", [-np.inf] * 3), float)
        self.ws_max = np.array(ws.get("max", [np.inf] * 3), float)
        self.arms = {a: _ArmState() for a in arms}
        self.hands = {s: _HandState(s, cfg) for s in HANDS}
        self.mapping: dict[str, str] = {}
        self.set_mapping(cfg.mapping)

    # -- settings ------------------------------------------------------------------------------------
    def set_mapping(self, mapping: dict[str, str] | None) -> None:
        m = {h: a for h, a in (mapping or default_mapping(self.arm_names)).items()
             if h in HANDS and a in self.arms}
        self.mapping = m or default_mapping(self.arm_names)
        self.release_all()

    def swap_hands(self) -> None:
        """Left-handed / mirrored use: the two hands exchange their arms."""
        self.set_mapping({"l": self.mapping.get("r"), "r": self.mapping.get("l")}
                         if len(self.mapping) > 1 else {("l" if "r" in self.mapping else "r"):
                                                        next(iter(self.mapping.values()))})

    def hands_of(self, arm: str) -> list[str]:
        return [h for h, a in self.mapping.items() if a == arm]

    def reset(self, ee_poses: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
        """Disengage everything and hold the given EE poses."""
        for h in self.hands.values():
            h.drop()
            h.last_raw = None
        for a, st in self.arms.items():
            st.clutch.release()
            st.driver = None
            st.gripper = 1.0
            st.hold_p, st.hold_R = ee_poses[a][0].copy(), ee_poses[a][1].copy()

    def sync_hold(self, ee_poses: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
        """Something else (a policy) moved the arms: hold where they are now, so taking control starts there."""
        for a, st in self.arms.items():
            if not st.clutch.engaged:
                st.hold_p, st.hold_R = ee_poses[a][0].copy(), ee_poses[a][1].copy()

    def release_all(self) -> None:
        """Drop every clutch, e.g. after the mapping changed, so re-engaging starts from the held target."""
        for st in self.arms.values():
            st.clutch.release()
            st.driver = None
        for h in self.hands.values():
            h.ctrl.reset()
            h.hand.reset()

    def set_mode(self, mode: str) -> None:
        if mode in MODES and mode != self.mode:
            self.mode = mode
            self.release_all()

    def set_control(self, control: str) -> None:
        if control in CONTROL_MODES and control != self.control:
            self.control = control
            self.release_all()

    def set_scale(self, scale: float) -> None:
        self.scale = float(np.clip(scale, 0.2, 3.0))
        self.release_all()

    # -- per control tick ----------------------------------------------------------------------------
    def update(self, frame: InputFrame | None, now: float, dt: float) -> TeleopOutput:
        out = TeleopOutput(targets={}, grippers=np.ones(len(self.arms)), engaged={}, tracked={})
        stale = frame is None or (now - frame.recv_time) > INPUT_TIMEOUT_S
        want_src = SRC_CONTROLLER if self.mode == "controllers" else SRC_HAND
        tracked_arm = {a: False for a in self.arms}
        for side, hs in self.hands.items():
            h = None if stale else frame.hand(side)
            tracked = h is not None and h.source == want_src
            if h is not None and h.source not in (want_src, 0):
                out.mode_mismatch = True
            arm = self.mapping.get(side)
            st = self.arms.get(arm) if arm else None
            if not tracked:
                # Lost tracking: freeze the arm where it is commanded to be.
                hs.drop()
                hs.last_raw = None
                if st is not None and st.driver == side:
                    st.clutch.release()
                    st.driver = None
                continue
            if self.mode == "controllers":
                engage, gripper, cmds = hs.ctrl.update(h)
                out.commands += cmds  # buttons work whether or not this hand drives an arm
                pose = h.pose
            else:
                engage, gripper = hs.hand.update(h)
                pose = hs.hand.control_pose(h)
            if st is None:
                continue
            tracked_arm[arm] = True
            p_w, R_w = self.placement.world_from_xr(pose[:3], pose[3:])
            jumped = hs.last_raw is not None and np.linalg.norm(p_w - hs.last_raw) > MAX_JUMP_M
            hs.last_raw = p_w
            if jumped:
                # Tracking glitch: drop the clutch rather than yank the arm across the table.
                if st.driver == side:
                    out.cues.append((arm, "glitch"))
                    st.clutch.release()
                    st.driver = None
                hs.drop()
                engage = False
            p_f, R_f = hs.pos_filter(p_w, dt), hs.rot_filter(R_w, dt)

            if st.driver not in (None, side):  # the other hand holds this arm: this one waits
                if engage:
                    hs.ctrl.reset()
                    hs.hand.reset()
                continue
            if len(self.hands_of(arm)) == 1 or st.driver == side:
                st.gripper = gripper
            if engage and not st.clutch.engaged:
                st.clutch.engage(p_f, R_f, st.hold_p, st.hold_R)
                st.driver = side
                out.cues.append((arm, "engage"))
            elif not engage and st.clutch.engaged:
                st.clutch.release()
                st.driver = None
                out.cues.append((arm, "disengage"))
            if st.clutch.engaged:
                p, R = st.clutch.target(p_f, R_f, self.scale, self.control, dt, self.cfg.rate_gain)
                st.hold_p = np.clip(p, self.ws_min, self.ws_max)
                st.hold_R = R
                if self.control == "rate":
                    st.clutch.clamp(st.hold_p)

        for i, (a, st) in enumerate(self.arms.items()):
            out.targets[a] = (st.hold_p.copy(), st.hold_R.copy())
            out.grippers[i] = st.gripper
            out.engaged[a] = st.clutch.engaged
            out.tracked[a] = tracked_arm[a]
            hands = self.hands_of(a)
            out.hand_of[a] = st.driver or (hands[0] if hands else None)
        return out
