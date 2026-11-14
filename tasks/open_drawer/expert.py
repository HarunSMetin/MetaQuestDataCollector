"""Scripted run for `xrc validate-task tasks/open_drawer`: grasp the handle tab from above and pull."""

import numpy as np

from xrcollect.expert import grasp_quat


def script(ex) -> None:
    env = ex.env
    arm = env.arms[0]
    grip = env.data.site("drawer_grip").xpos.copy()
    q = grasp_quat(0.0)                      # fingers along y, closing on the tab
    ex.move(arm, grip + [0, 0, 0.12], q, seconds=1.5)
    ex.move(arm, grip, seconds=1.0)
    ex.move(arm, grip=0.0, seconds=0.6)
    ex.move(arm, grip + np.array([-0.15, 0, 0]), seconds=2.5)
    ex.move(arm, grip=1.0, seconds=0.6)
    ex.move(arm, grip + np.array([-0.15, 0, 0.12]), seconds=0.8)
    ex.go_home(arm)
