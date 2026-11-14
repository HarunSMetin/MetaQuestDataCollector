"""Wire protocol: round trip, generated files in sync with the schema, version check."""

import numpy as np
import pytest

from xrcollect import protocol as P
from xrcollect.wire import codegen


def test_generated_files_match_the_schema():
    assert codegen.generate(check=True), "run `xrc codegen`"


def test_layout_constants():
    assert P.INPUT_BYTES == 1580 and P.INPUT_HEADER.size == 16 and P.POSE_HEADER.size == 16
    assert P.HAND_FLOATS == 7 + 8 + 25 * 7 + 2
    assert P.HAND_JOINTS[P.J_THUMB_TIP] == "thumb-tip" and P.HAND_JOINTS[P.J_MIDDLE_META] == "middle-finger-metacarpal"


def test_protocol_roundtrip():
    rng = np.random.default_rng(0)
    f = P.InputFrame(seq=7, t_echo=1234.5, head=rng.normal(size=7))
    for h in (f.left, f.right):
        h.source = P.SRC_HAND
        h.pose, h.buttons = rng.normal(size=7), rng.uniform(size=8)
        h.joints, h.pinch = rng.normal(size=(25, 7)), rng.uniform(size=2)
    g = P.parse_input(P.pack_input(f))
    assert g.seq == 7 and g.t_echo == 1234.5 and g.left.source == P.SRC_HAND
    for a, b in ((f.head, g.head), (f.left.joints, g.left.joints), (f.right.buttons, g.right.buttons),
                 (f.right.pinch, g.right.pinch), (f.left.pose, g.left.pose)):
        assert np.allclose(a, b, atol=1e-6)


def test_other_protocol_version_is_refused_with_a_readable_message():
    with pytest.raises(P.ProtocolError, match="protocol version"):
        P.parse_input(P.pack_input(P.InputFrame(), version=P.VERSION + 1))
    with pytest.raises(P.ProtocolError):
        P.parse_input(b"\x01\x00")


def test_pose_frame_roundtrip_and_flags():
    poses = np.arange(21, dtype=float).reshape(3, 7)
    flags = P.pose_flags([True, False, True], recording=True)
    mtype, f, seq, t, back = P.parse_pose_frame(P.pack_pose_frame(5, 99.0, poses, flags))
    assert (mtype, seq, t) == (P.MSG_POSE, 5, 99.0) and np.allclose(back, poses)
    assert f & 1 and not f & 2 and f & 4 and f & P.FLAG_RECORDING
    assert P.parse_pose_frame(P.pack_pose_frame(1, 0.0, poses, 0, ghost=True))[0] == P.MSG_GHOST
