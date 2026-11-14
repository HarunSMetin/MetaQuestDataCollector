"""FastAPI server: WebXR + dashboard websockets, REST API, static web client.

It never touches simulation state: actions go into the runtime's command queue (`POST /api/command` or a websocket
`{"type": "command"}` message), and everything it sends comes from the runtime's lock-protected snapshots.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from xrcollect.audit import (CURATION_FIELDS, audit_dir, dataset_health, forget_curation, kept_spans,
                             labelled_success, load_curation, update_curation, why_not)
from xrcollect.config import ROOT, CurateConfig
from xrcollect.protocol import VERSION, ProtocolError, parse_input
from xrcollect.recorder import episode_files, list_episodes
from xrcollect.runtime import Runtime
from xrcollect.task.loader import LIBRARY, list_tasks
from xrcollect.task.predicates import CATALOGUE
from xrcollect.task.schema import TaskModel

log = logging.getLogger(__name__)
WEB_DIST = ROOT / "web" / "dist"
SEND_PERIOD_S = 0.01
STATUS_PERIOD_S = 0.1
NO_STORE = {"Cache-Control": "no-store"}


class Hub:
    """Fans runtime events (from the sim thread) out to every connected websocket."""

    def __init__(self):
        self.queues: set[asyncio.Queue] = set()
        self.loop: asyncio.AbstractEventLoop | None = None

    def push_threadsafe(self, event: dict) -> None:
        if self.loop is not None:
            self.loop.call_soon_threadsafe(self._broadcast, event)

    def _broadcast(self, event: dict) -> None:
        for q in self.queues:
            if q.qsize() < 256:
                q.put_nowait(event)


def create_app(runtime: Runtime) -> FastAPI:
    app = FastAPI(title="xrcollect")
    app.add_middleware(GZipMiddleware, minimum_size=1024)
    hub = Hub()
    runtime.add_listener(hub.push_threadsafe)
    exports: dict[str, dict] = {}   # export jobs by name: {state, done, total, error, manifest}

    # -- scene, task, protocol ----------------------------------------------------------------------
    @app.get("/")
    async def index():
        return RedirectResponse("/dashboard/")

    @app.get("/api/scene")
    async def scene():
        return JSONResponse(runtime.scene()[0], headers=NO_STORE)

    @app.get("/api/scene.bin")
    async def scene_bin():
        return Response(runtime.scene()[1], media_type="application/octet-stream", headers=NO_STORE)

    @app.get("/api/task")
    async def task():
        return JSONResponse(runtime.task(), headers=NO_STORE)

    @app.get("/api/commands")
    async def commands():
        """The button table (teleop.BUTTONS): the headset draws its button labels from this."""
        return JSONResponse(runtime.commands(), headers=NO_STORE)

    @app.get("/api/protocol")
    async def protocol():
        return {"version": VERSION}

    @app.get("/api/status")
    async def status():
        return runtime.status()

    # -- task authoring -----------------------------------------------------------------------------
    @app.get("/api/schema")
    async def schema():
        """What the task editor builds its forms from: the task file's JSON Schema and the predicate catalogue."""
        return {"task": TaskModel.model_json_schema(), "predicates": CATALOGUE}

    @app.get("/api/tasks")
    async def tasks():
        return {"current": runtime.cfg.task, "tasks": list_tasks()}

    @app.get("/api/library")
    async def library():
        """Object assets (objects/**.yaml) and embodiments / scenes a task can name."""
        objs = sorted(str(p.relative_to(LIBRARY["object"]).with_suffix("")) for p in LIBRARY["object"].rglob("*.yaml"))
        return {"objects": objs,
                "embodiments": sorted(p.stem for p in LIBRARY["embodiment"].glob("*.yaml")),
                "scenes": sorted(p.stem for p in LIBRARY["scene"].glob("*.yaml"))}

    @app.get("/api/taskfile")
    async def taskfile():
        """The task file being edited: its text and its parsed content."""
        import yaml

        from xrcollect.task.loader import fix_keys

        path = runtime.bundle.path
        if path is None or not Path(path).exists():
            raise HTTPException(404, "the current task has no file")
        text = Path(path).read_text()
        return {"path": str(Path(path).relative_to(ROOT)) if Path(path).is_relative_to(ROOT) else str(path),
                "text": text, "data": fix_keys(yaml.safe_load(text)), "version": runtime.bundle.version}

    @app.post("/api/scene/reload")
    async def reload_scene(body: dict | None = None):
        """Hot reload after editing files by hand: rebuild, send the new scene to the clients, reset."""
        runtime.command("reload_scene", **({"task": body["task"]} if body and body.get("task") else {}))
        return {"ok": True}

    # -- datasets and episodes ------------------------------------------------------------------------
    def _dataset_dir(dataset: str | None) -> Path:
        """An episode folder by name: the recording folder (record.out_dir) by default, else a sibling of it."""
        if not dataset or dataset == runtime.cfg.out_dir.name:
            return runtime.cfg.out_dir
        name = Path(dataset).name
        d = runtime.cfg.out_dir.parent / name
        if not name or name.startswith(".") or not d.is_dir():
            raise HTTPException(404, "unknown dataset")
        return d

    def _episode_file(file: str, dataset: str | None = None) -> Path:
        p = _dataset_dir(dataset) / Path(file).name
        if p.suffix != ".hdf5" or not p.exists():
            raise HTTPException(404, "unknown episode")
        return p

    @app.get("/api/datasets")
    async def datasets():
        """Every episode folder next to the recording folder, recording folder first. Title and description come
        from an optional dataset.json in the folder."""
        def run():
            rec = runtime.cfg.out_dir
            rec.mkdir(parents=True, exist_ok=True)
            folders = [rec] + sorted(d for d in rec.parent.iterdir()
                                     if d.is_dir() and d != rec and not d.name.startswith("."))
            out = []
            for d in folders:
                files = episode_files(d)
                if not files and d != rec:
                    continue
                info = {}
                if (d / "dataset.json").exists():
                    try:
                        info = json.loads((d / "dataset.json").read_text())
                    except ValueError:
                        pass
                out.append({"name": d.name, "recording": d == rec, "count": len(files),
                            "title": info.get("title", "Recorded demos" if d == rec else d.name),
                            "description": info.get("description", "")})
            return out
        return await asyncio.to_thread(run)

    @app.get("/api/episodes")
    async def episodes(dataset: str | None = None):
        """Saved episodes with their manual labels: success = the label, recorded_success = as saved."""
        folder = _dataset_dir(dataset)
        rows = await asyncio.to_thread(list_episodes, folder)
        curation = load_curation(folder)
        for r in rows:
            c = curation.get(r["file"], {})
            r.update(recorded_success=r["success"], success=bool(c.get("success", r["success"])),
                     tags=c.get("tags", []), note=c.get("note", ""), excluded=bool(c.get("exclude")),
                     quality=c.get("quality", 0), annotations=c.get("annotations", []))
        return rows

    @app.patch("/api/episodes/{file}")
    async def edit_episode(file: str, body: dict, dataset: str | None = None):
        """Change an episode's manual labels: any of exclude, note, tags, success (null = as recorded), quality,
        annotations ([{start, stop, label}], steps)."""
        p = _episode_file(file, dataset)
        fields = {k: v for k, v in body.items() if k in CURATION_FIELDS}
        try:
            return update_curation(p.parent, p.name, **fields)
        except (ValueError, KeyError, TypeError) as e:
            raise HTTPException(400, str(e)) from e

    @app.delete("/api/episodes/{file}")
    async def delete_episode(file: str, dataset: str | None = None):
        if _dataset_dir(dataset) != runtime.cfg.out_dir:  # recorded takes only: derived sets are rebuilt by tools
            raise HTTPException(403, "episodes can only be deleted from the recording folder")
        p = _episode_file(file)
        st = runtime.status()
        for key in ("replay", "ghost"):
            if Path((st.get(key) or {}).get("path", "")).name == p.name:
                runtime.command("replay_stop" if key == "replay" else "ghost")
        p.unlink()
        forget_curation(runtime.cfg.out_dir, p.name)
        return {"deleted": p.name}

    def _rules(body: dict | None) -> CurateConfig:
        if not body or not body.get("rules"):
            return runtime.cfg.curate
        return CurateConfig.model_validate({**runtime.cfg.curate.model_dump(), **body["rules"]})

    def _audit_rows(folder: Path, rules: CurateConfig) -> dict:
        audits = audit_dir(folder)
        curation = load_curation(folder)
        out = {}
        for name, a in audits.items():
            why = why_not(a, rules, curation)
            out[name] = {"contact_frac": a.contact_frac, "contact_events": a.contact_events,
                         "idle_frac": a.idle_frac, "stage_times": a.stage_times, "fps": a.fps,
                         "num_steps": a.num_steps, "safety_events": a.safety_events, "phases": a.phases,
                         "success": labelled_success(a, curation), "verdict": why,
                         "spans": [] if why else kept_spans(a, rules, curation),
                         "task_version": (a.info or {}).get("task_version")}
        return out

    @app.get("/api/audit")
    async def audit(dataset: str | None = None):
        """Per episode: contact shares, idle share, stage times, automatic phases, the steps a dataset would use
        and why it would leave the episode out (or null). The same rules as the headset's review and the export."""
        return await asyncio.to_thread(_audit_rows, _dataset_dir(dataset), runtime.cfg.curate)

    @app.get("/api/health")
    async def health(dataset: str | None = None):
        """Dataset health: start-state coverage, durations, success over time, contacts, per-operator numbers."""
        return await asyncio.to_thread(dataset_health, _dataset_dir(dataset), runtime.cfg.curate)

    # -- export -------------------------------------------------------------------------------------
    @app.get("/api/exporters")
    async def exporters():
        from xrcollect.export.base import EXPORTERS, FeatureLayout

        return {"exporters": list(EXPORTERS), "layout": FeatureLayout().model_dump(),
                "rules": runtime.cfg.curate.model_dump(), "jobs": exports}

    @app.post("/api/export/preview")
    async def export_preview(body: dict):
        """The segments an export with these rules would write, per episode, without writing anything."""
        folder = _dataset_dir(body.get("dataset"))
        rows = await asyncio.to_thread(_audit_rows, folder, _rules(body))
        used = {k: v for k, v in rows.items() if v["verdict"] is None}
        return {"episodes": rows, "segments": sum(len(v["spans"]) for v in used.values()),
                "steps": sum(e - s for v in used.values() for s, e in v["spans"]),
                "task_versions": sorted({v["task_version"] or "?" for v in used.values()})}

    @app.post("/api/export")
    async def export_start(body: dict):
        """Start an export in the background: {dataset, name, exporter, rules, layout, allow_mixed}."""
        from xrcollect.export.base import FeatureLayout, run_export

        folder = _dataset_dir(body.get("dataset"))
        name = Path(str(body.get("name") or f"{folder.name}_{body.get('exporter', 'hdf5')}")).name
        if exports.get(name, {}).get("state") == "running":
            raise HTTPException(409, "this export is already running")
        job = exports[name] = {"state": "running", "done": 0, "total": 0, "error": None, "manifest": None}
        layout = FeatureLayout.model_validate(body.get("layout") or {})
        rules = _rules(body)

        def progress(done: int, total: int) -> None:
            job.update(done=done, total=total)

        def work():
            try:
                job["manifest"] = run_export(folder, runtime.cfg.out_dir.parent / "exports" / name,
                                             body.get("exporter", "hdf5"), rules, layout,
                                             allow_mixed=bool(body.get("allow_mixed")), progress=progress)
                job["state"] = "done"
            except Exception as e:  # noqa: BLE001 - shown on the export page
                log.exception("export %s failed", name)
                job.update(state="failed", error=f"{type(e).__name__}: {e}")

        asyncio.get_running_loop().run_in_executor(None, work)
        return {"name": name}

    # -- commands, previews ---------------------------------------------------------------------------
    @app.post("/api/command")
    async def command(body: dict):
        name = body.get("name")
        if not name:
            raise HTTPException(400, "missing command name")
        runtime.command(name, **(body.get("args") or {}))
        return {"ok": True}

    @app.get("/api/preview/{cam}.jpg")
    async def preview(cam: str):
        img = runtime.previews().get(cam)
        if img is None:
            raise HTTPException(404, "no preview yet")
        return Response(img, media_type="image/jpeg", headers=NO_STORE)

    @app.get("/api/hint")
    async def hint():
        """Connection help shown on the headset's landing page."""
        return PlainTextResponse("USB: run `xrc setup-quest` again after replugging the headset "
                                 f"(adb reverse tcp:{runtime.cfg.server.port}).", headers=NO_STORE)

    # -- websockets ---------------------------------------------------------------------------------
    async def _sender(ws: WebSocket, q: asyncio.Queue, pose_period: float):
        last_seq, last_ghost, last_pose_t, last_status_t = -1, -1, 0.0, 0.0
        while True:
            while not q.empty():
                await ws.send_text(json.dumps(q.get_nowait()))
            now = time.monotonic()
            if now - last_pose_t >= pose_period:
                seq, frame = runtime.latest_pose_frame()
                if seq != last_seq and frame:
                    await ws.send_bytes(frame)
                    last_seq, last_pose_t = seq, now
                gseq, gframe = runtime.latest_ghost_frame()
                if gseq != last_ghost and gframe:
                    await ws.send_bytes(gframe)
                    last_ghost = gseq
            if now - last_status_t >= STATUS_PERIOD_S:
                st = runtime.status()
                if st:
                    await ws.send_text(json.dumps(st))
                last_status_t = now
            await asyncio.sleep(SEND_PERIOD_S)

    async def _handle_text(text: str):
        try:
            msg = json.loads(text)
        except json.JSONDecodeError:
            return
        if msg.get("type") == "command" and msg.get("name"):
            runtime.command(msg["name"], **(msg.get("args") or {}))

    async def _session(ws: WebSocket, accept_input: bool, pose_period: float):
        await ws.accept()
        hub.loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()
        hub.queues.add(q)
        sender = asyncio.create_task(_sender(ws, q, pose_period))
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break
                if msg.get("bytes") is not None and accept_input:
                    try:
                        runtime.submit_input(parse_input(msg["bytes"]))
                    except ProtocolError as e:
                        # A client built for another protocol version: tell it in words, then hang up.
                        log.warning("refusing client: %s", e)
                        await ws.send_text(json.dumps({"type": "error", "text": str(e)}))
                        await ws.close(code=4000)
                        break
                elif msg.get("text") is not None:
                    await _handle_text(msg["text"])
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            sender.cancel()
            hub.queues.discard(q)

    @app.websocket("/ws/xr")
    async def ws_xr(ws: WebSocket):
        await _session(ws, accept_input=True, pose_period=0.0)

    @app.websocket("/ws/dash")
    async def ws_dash(ws: WebSocket):
        await _session(ws, accept_input=False, pose_period=1 / 30)

    if WEB_DIST.exists():
        app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
    else:
        @app.get("/dashboard/")
        async def no_web():
            return Response("Web client not built. Run: cd web && npm ci && npm run build", media_type="text/plain")

    return app
