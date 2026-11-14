"""REST API over a runtime that is ticked by hand (no sim thread, no headset)."""

import shutil
import time

import pytest
from fastapi.testclient import TestClient

from conftest import controllers, ticks
from xrcollect.protocol import VERSION, InputFrame, pack_input
from xrcollect.runtime import Runtime
from xrcollect.server import create_app


@pytest.fixture
def app(cfg, expert_episode):
    cfg.out_dir.mkdir(parents=True)
    for i in range(2):
        path, _ = expert_episode("tasks/stack_two_blocks", None, i)
        shutil.copy(path, cfg.out_dir / f"episode_{i:06d}.hdf5")
    rt = Runtime(cfg)
    ticks(rt, 6, controllers())
    client = TestClient(create_app(rt))
    yield rt, client
    rt.recorder.close()


def test_scene_task_and_tables(app):
    rt, c = app
    scene = c.get("/api/scene").json()
    assert scene["nbody"] == rt.env.model.nbody and scene["arms"] == ["l", "r"]
    assert {"cam_high", "cam_left_wrist"} <= {cam["name"] for cam in scene["cameras"]}
    assert any(g["group"] == 5 for g in scene["geoms"]) and scene["objects"]["red_block"] > 0
    assert len(c.get("/api/scene.bin").content) == scene["blobSize"]
    # One mesh per shape even with two identical arms.
    assert len(scene["meshes"]) == len({m["name"].split("_", 1)[-1] for m in scene["meshes"]})
    task = c.get("/api/task").json()
    assert task["task"] == "stack_two_blocks" and [s["name"] for s in task["stages"]] == ["red_in_area", "blue_on_red"]
    assert task["goal_view"]["anchor"] == "red_block" and task["regions"]["build_area"]["half"] == [0.06, 0.06]
    assert c.get("/api/protocol").json() == {"version": VERSION}
    cmds = c.get("/api/commands").json()
    assert {"record_toggle", "save", "discard", "recenter"} <= {b["command"] for b in cmds}
    schema = c.get("/api/schema").json()
    assert "stages" in schema["task"]["properties"] and "on_top" in schema["predicates"]
    lib = c.get("/api/library").json()
    assert "blocks/cube_4cm" in lib["objects"] and "franka_single" in lib["embodiments"]
    assert any(t["task"] == "duplo_tower" for t in c.get("/api/tasks").json()["tasks"])
    assert "stack_two_blocks" in c.get("/api/taskfile").json()["text"]
    assert c.get("/api/status").json()["rec_state"] == "idle"


def test_episodes_labels_audit_health(app):
    rt, c = app
    assert c.get("/api/datasets").json()[0] == {"name": "episodes", "recording": True, "count": 2,
                                                "title": "Recorded demos", "description": ""}
    rows = c.get("/api/episodes").json()
    assert len(rows) == 2 and rows[0]["success"] and rows[0]["task"] == "stack_two_blocks" and rows[0]["tags"] == []
    f = rows[0]["file"]
    r = c.patch(f"/api/episodes/{f}", json={"tags": ["slow"], "success": False, "quality": 3,
                                             "annotations": [{"start": 10, "stop": 90, "label": "reach"}]})
    assert r.status_code == 200 and r.json()["quality"] == 3
    row = c.get("/api/episodes").json()[0]
    assert not row["success"] and row["recorded_success"] and row["tags"] == ["slow"] and row["annotations"]
    audit = c.get("/api/audit").json()
    assert audit[f]["verdict"] == "failed take" and audit[f]["spans"] == []
    other = rows[1]["file"]
    assert audit[other]["verdict"] is None and audit[other]["spans"] and audit[other]["phases"]
    health = c.get("/api/health").json()
    assert health["count"] == 2 and health["used"] == 1
    assert c.patch("/api/episodes/nope.hdf5", json={}).status_code == 404
    assert c.get("/api/episodes", params={"dataset": "../.."}).status_code == 404
    assert c.delete(f"/api/episodes/{f}").json() == {"deleted": f}
    assert len(c.get("/api/episodes").json()) == 1 and f not in c.get("/api/audit").json()


def test_commands_go_through_the_queue_and_replay_runs(app):
    rt, c = app
    path = c.get("/api/episodes").json()[0]["path"]
    assert c.post("/api/command", json={"name": "replay", "args": {"path": path, "paused": True, "i": 12}}).json()["ok"]
    assert rt._replay is None                              # nothing happens until the sim thread's next tick
    ticks(rt, 6)
    assert c.get("/api/status").json()["replay"]["i"] == 12
    assert c.post("/api/command", json={}).status_code == 400
    c.post("/api/command", json={"name": "replay_stop"})
    ticks(rt, 6)
    assert c.get("/api/status").json()["replay"] is None
    ticks(rt, 12)
    assert c.get("/api/preview/cam_high.jpg").status_code in (200, 404)   # rendered only once someone asked
    ticks(rt, 12)
    img = c.get("/api/preview/cam_high.jpg")
    assert img.status_code == 200 and img.content[:2] == b"\xff\xd8"


def test_export_preview_and_job(app):
    rt, c = app
    assert set(c.get("/api/exporters").json()["exporters"]) == {"hdf5", "lerobot"}
    p = c.post("/api/export/preview", json={"rules": {"trim_idle": False}}).json()
    assert p["segments"] == 2 and p["steps"] == sum(e["num_steps"] for e in p["episodes"].values())
    assert c.post("/api/export/preview", json={}).json()["steps"] < p["steps"]
    r = c.post("/api/export", json={"name": "api_test", "exporter": "hdf5", "layout": {"cameras": []}})
    assert r.json() == {"name": "api_test"}
    for _ in range(100):
        job = c.get("/api/exporters").json()["jobs"]["api_test"]
        if job["state"] != "running":
            break
        time.sleep(0.1)
    assert job["state"] == "done" and job["manifest"]["episodes"] == 2
    assert (rt.cfg.out_dir.parent / "exports" / "api_test" / "episode_000001.hdf5").exists()


def test_websocket_streams_poses_and_refuses_another_protocol_version(app):
    rt, c = app
    with c.websocket_connect("/ws/xr") as ws:
        ws.send_bytes(pack_input(controllers()))
        got = {"bytes": 0, "status": 0}
        for _ in range(6):
            msg = ws.receive()
            if msg.get("bytes"):
                got["bytes"] += 1
            elif '"type": "status"' in (msg.get("text") or ""):
                got["status"] += 1
        assert got["bytes"] >= 1 and got["status"] >= 1 and rt._input is not None
        ws.send_text('{"type": "command", "name": "set_scale", "args": {"scale": 0.5}}')
        time.sleep(0.2)
        ticks(rt, 1)
        assert rt.teleop.scale == 0.5
    with c.websocket_connect("/ws/xr") as ws:
        ws.send_bytes(pack_input(InputFrame(), version=VERSION + 1))
        texts = []
        try:
            for _ in range(8):
                msg = ws.receive()
                if msg.get("text"):
                    texts.append(msg["text"])
                if msg["type"] == "websocket.close":
                    break
        except Exception:  # noqa: BLE001 - the server hung up
            pass
        assert any('"type": "error"' in t and "protocol version" in t for t in texts)
