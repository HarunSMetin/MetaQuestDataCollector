"""The task file format as pydantic models. They validate task YAML on load and give the dashboard's editor its
forms (`GET /api/schema` = TaskModel's JSON Schema plus the predicate catalogue from predicates.py)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Region(BaseModel):
    """A named rectangle on the table, drawn as tape (visible to the cameras)."""
    model_config = ConfigDict(extra="forbid")
    center: list[float] = Field(min_length=2, max_length=2)
    half: list[float] = Field(min_length=2, max_length=2)
    tape: bool = True


class Spawn(BaseModel):
    """Start-state distribution of one object: uniform in a rectangle, uniform yaw."""
    model_config = ConfigDict(extra="forbid")
    region: Region | str                      # inline rectangle, or the name of a region
    yaw_deg: list[float] = Field(default_factory=lambda: [0.0, 0.0], min_length=2, max_length=2)
    z: float | None = None                    # height of the body frame; default: on the table


class Stage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    when: dict[str, Any]                      # a predicate (predicates.py)
    label: str | None = None                  # shown to the operator; default: the name
    object: str | None = None                 # the object this stage is about (next-object hint); default: guessed


class EventCue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    on: str                                   # grasp | release | stage_reached | stage_lost | success | snap_on | ...
    cue: str                                  # a cue name from config.yaml feedback.cues
    stage: str | None = None                  # only for this stage
    toast: str | None = None                  # text shown to the operator; {stage} {object} are filled in


class GoalItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    object: str
    pose: dict[str, Any] = Field(default_factory=dict)   # {pos: [x, y, z], yaw: rad} relative to the anchor


class GoalView(BaseModel):
    """What the goal panel draws: objects at poses relative to an anchor object's frame."""
    model_config = ConfigDict(extra="forbid")
    anchor: str
    show: list[GoalItem] = Field(default_factory=list)


class SnapRule(BaseModel):
    """Press-fit between two objects: a weld that switches on when they touch, `when` holds and they are pressed
    together with more than press_n; off when the weld holds more than pull_n of tension + shear."""
    model_config = ConfigDict(extra="forbid")
    a: str
    b: str
    when: dict[str, Any]
    press_n: float = 6.0
    pull_n: float = 10.0
    solref: list[float] = Field(default_factory=lambda: [0.01, 1.0])  # softer than the contacts


class CameraSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pos: list[float] = Field(min_length=3, max_length=3)
    lookat: list[float] = Field(min_length=3, max_length=3)
    fovy: float = 60.0


class TaskModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task: str
    instruction: str = ""
    embodiment: str = "franka_bimanual"       # embodiments/<name>.yaml
    scene: str = "table"                      # scenes/<name>.yaml
    plugin: str | None = None                 # "task.py:ClassName" next to the task file, for logic beyond predicates
    objects: dict[str, dict[str, Any]] = Field(default_factory=dict)   # name -> {asset: <library path>, overrides}
    regions: dict[str, Region] = Field(default_factory=dict)
    spawn: dict[str, Spawn] = Field(default_factory=dict)
    stages: list[Stage] = Field(default_factory=list)
    success: dict[str, Any] | None = None     # a predicate; default: the last stage, nothing held, all at rest
    snaps: list[SnapRule] = Field(default_factory=list)
    events: list[EventCue] = Field(default_factory=list)
    goal_view: GoalView | None = None
    cameras: dict[str, CameraSpec] = Field(default_factory=dict)       # added to / replacing the scene's cameras
    tags: list[str] = Field(default_factory=lambda: ["recovery", "slow", "odd grasp"])   # review tag chips
    params: dict[str, Any] = Field(default_factory=dict)               # free-form, for plugins


def validate_task(raw: dict) -> dict:
    """Validated task dict with defaults filled in (plain data, safe to store and hash)."""
    return TaskModel.model_validate(raw).model_dump(mode="json")
