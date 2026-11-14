"""Hands-mode gestures: the three-finger curl clutch and the thumb-index gripper."""

import numpy as np

from xrcollect.config import TeleopConfig
from xrcollect.protocol import HAND_JOINTS, J_WRIST, HandInput
from xrcollect.teleop import HandMapper, finger_curl

J = HAND_JOINTS.index


def hand(curl: float = 1.9, pinch_m: float = 0.08, fingers=("middle", "ring", "pinky")) -> HandInput:
    """A flat hand pointing along -z: knuckles 9.6 cm from the wrist, the given fingers' tips at `curl` times that
    distance (1.9 open, 0.8 a fist), the rest open. `pinch_m` is the thumb-index distance."""
    h = HandInput()
    h.joints[:, 6] = 1.0  # identity quaternions
    h.joints[J_WRIST, :3] = 0.0
    for i, f in enumerate(("index", "middle", "ring", "pinky")):
        x = 0.02 * (i - 1.5)
        h.joints[J(f"{f}-finger-phalanx-proximal"), :3] = [x, 0, -0.096]
        c = curl if f in fingers else 1.9
        h.joints[J(f"{f}-finger-tip"), :3] = [x, 0, -0.096 * c]
    h.pinch[:] = [pinch_m, 0.1]
    return h


def test_finger_curl_reads_open_and_curled_fingers():
    assert np.all(finger_curl(hand(1.9)) > 1.8)
    assert np.all(finger_curl(hand(0.8)) < 0.9)


def test_curling_three_fingers_toggles_once_per_gesture():
    m = HandMapper(TeleopConfig())
    assert m.update(hand(1.9))[0] is False
    assert m.update(hand(0.8))[0] is True       # curl: engage
    assert m.update(hand(0.8))[0] is True       # held curled: no second toggle
    assert m.update(hand(1.3))[0] is True       # half open: still latched
    assert m.update(hand(1.9))[0] is True       # open: ready again
    assert m.update(hand(0.8))[0] is False      # curl again: release


def test_index_pinch_and_partial_curls_do_not_toggle():
    m = HandMapper(TeleopConfig())
    assert m.update(hand(1.9, pinch_m=0.01))[0] is False                       # closing the gripper
    assert m.update(hand(0.8, fingers=("middle", "ring")))[0] is False          # little finger still open
    assert m.update(hand(1.4))[0] is False                                      # a relaxed, half-curled hand


def test_gripper_holds_while_the_fingers_are_curled():
    m = HandMapper(TeleopConfig())
    _, g_open = m.update(hand(1.9, pinch_m=0.08))
    assert g_open == 1.0
    _, g = m.update(hand(0.8, pinch_m=0.02))     # the thumb drifts toward the index while making the fist
    assert g == g_open
    _, g = m.update(hand(1.9, pinch_m=0.02))     # hand open again: the gripper follows the pinch
    assert g < 0.1
