"""Binary wire protocol between the WebXR client and the server.

Every constant, size and offset comes from protocol_gen.py (generated from wire/schema.json, like
web/src/protocol.gen.ts). Layout, little endian:

Input (client -> server, every XR frame):
  u8 type=1 | u8 src_left | u8 src_right | u8 version | u32 seq | f64 t_echo_ms (last pose frame's t_server)
  f32[7]  head pose            (pos xyz, quat xyzw; XR local-floor frame)
  per hand (left, right):
    f32[7]   control pose      (controller grip pose, or hand wrist pose)
    f32[8]   buttons           (trigger, squeeze, stick_x, stick_y, primary, secondary, stick_press, menu)
    f32[175] 25 hand joints    (pos xyz, quat xyzw each; zeros for controllers)
    f32[2]   pinch distances   (thumb-index, thumb-middle) in metres; recorded, the clutch uses the joints

Pose frame (server -> client), type=2 (live scene) or 3 (ghost: a reference take's bodies, operator-only):
  u8 type | u8 flags | u16 nbody | u32 seq | f64 t_server_ms
  f32[nbody*7] body poses (MuJoCo world, pos xyz + quat xyzw)
  flags: bit i = arm i engaged (embodiment arm order), FLAG_RECORDING

The server measures round-trip latency as (now - t_echo).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from xrcollect.protocol_gen import *  # noqa: F401,F403 - the generated constants are this module's API
from xrcollect.protocol_gen import (BODY_FIELDS, FLAG_RECORDING, HAND_FIELDS, HAND_FLOATS, HANDS_OFFSET, INPUT_BYTES,
                                    INPUT_HEADER, MAX_FLAG_ARMS, MSG_GHOST, MSG_INPUT, MSG_POSE, N_BUTTONS,
                                    N_HAND_JOINTS, POSE_HEADER, SRC_NONE, VERSION)

HANDS = ("l", "r")  # the operator's hands, in wire order


class ProtocolError(ValueError):
    """A frame the server can't use; the message is shown to the operator."""


@dataclass
class HandInput:
    source: int = SRC_NONE
    pose: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0, 0, 0, 0, 1.0]))
    buttons: np.ndarray = field(default_factory=lambda: np.zeros(N_BUTTONS))
    joints: np.ndarray = field(default_factory=lambda: np.zeros((N_HAND_JOINTS, 7)))
    pinch: np.ndarray = field(default_factory=lambda: np.full(2, 0.1))


@dataclass
class InputFrame:
    seq: int = 0
    t_echo: float = 0.0
    head: np.ndarray = field(default_factory=lambda: np.array([0, 1.6, 0, 0, 0, 0, 1.0]))
    left: HandInput = field(default_factory=HandInput)
    right: HandInput = field(default_factory=HandInput)
    recv_time: float = field(default_factory=time.monotonic)

    def hand(self, side: str) -> HandInput:
        return self.left if side == "l" else self.right


def _cut(f: np.ndarray, table: dict, name: str) -> np.ndarray:
    o, n = table[name]
    return f[o:o + n].astype(float)


def _parse_hand(src: int, f: np.ndarray) -> HandInput:
    return HandInput(source=src, pose=_cut(f, HAND_FIELDS, "pose"), buttons=_cut(f, HAND_FIELDS, "buttons"),
                     joints=_cut(f, HAND_FIELDS, "joints").reshape(N_HAND_JOINTS, 7),
                     pinch=_cut(f, HAND_FIELDS, "pinch"))


def parse_input(buf: bytes) -> InputFrame:
    if len(buf) < INPUT_HEADER.size:
        raise ProtocolError(f"input message has {len(buf)} bytes")
    mtype, src_l, src_r, version, seq, t_echo = INPUT_HEADER.unpack_from(buf, 0)
    if mtype != MSG_INPUT:
        raise ProtocolError(f"unexpected message type {mtype}")
    if version != VERSION:
        raise ProtocolError(f"The page speaks protocol version {version}, the server version {VERSION}. "
                            "Rebuild the web client (cd web && npm run build) and reload the page.")
    if len(buf) != INPUT_BYTES:
        raise ProtocolError(f"input message has {len(buf)} bytes, expected {INPUT_BYTES}")
    f = np.frombuffer(buf, dtype="<f4", offset=INPUT_HEADER.size)
    o = HANDS_OFFSET
    left = _parse_hand(src_l, f[o:o + HAND_FLOATS])
    right = _parse_hand(src_r, f[o + HAND_FLOATS:o + 2 * HAND_FLOATS])
    return InputFrame(seq, t_echo, _cut(f, BODY_FIELDS, "head"), left, right)


def pack_input(frame: InputFrame, version: int = VERSION) -> bytes:
    """Python-side encoder (the browser uses InputPacker in protocol.ts); used by tests and scripted clients."""
    parts = [frame.head]
    for h in (frame.left, frame.right):
        parts += [h.pose, h.buttons, h.joints.reshape(-1), h.pinch]
    body = np.concatenate(parts).astype("<f4").tobytes()
    head = INPUT_HEADER.pack(MSG_INPUT, frame.left.source, frame.right.source, version, frame.seq, frame.t_echo)
    return head + body


def pose_flags(engaged: list[bool], recording: bool) -> int:
    flags = FLAG_RECORDING if recording else 0
    for i, e in enumerate(engaged[:MAX_FLAG_ARMS]):
        flags |= (1 << i) if e else 0
    return flags


def pack_pose_frame(seq: int, t_server: float, body_pose: np.ndarray, flags: int, ghost: bool = False) -> bytes:
    head = POSE_HEADER.pack(MSG_GHOST if ghost else MSG_POSE, flags, body_pose.shape[0], seq, t_server)
    return head + body_pose.astype("<f4").tobytes()


def parse_pose_frame(buf: bytes) -> tuple[int, int, int, float, np.ndarray]:
    """(type, flags, seq, t_server, (nbody, 7) poses); the browser's parser is in protocol.ts."""
    mtype, flags, nbody, seq, t_server = POSE_HEADER.unpack_from(buf, 0)
    poses = np.frombuffer(buf, dtype="<f4", offset=POSE_HEADER.size, count=nbody * 7).reshape(nbody, 7)
    return mtype, flags, seq, t_server, poses
