"""Episode quality audit, labels and dataset selection (the stage between recording and export).

Every episode gets an `EpisodeAudit`: how often the arms touched the table or each other, how much of it is still,
when each stage was reached, who drove, where the safety filter stepped in, automatic phases (grasp / hold / stage
reached), and where the objects started. `select` turns the audits, the `curate` rules from config.yaml and the
manual labels into the segments a dataset is built from, with a reason for every rejected episode. The same
functions give the verdict in the headset's review card, the dashboard's episode list and the exporter (one
definition).

Raw episodes are never modified: audits are cached in <episodes>/audit.json (keyed by file size + mtime), manual
labels live in <episodes>/curation.json, written only through `update_curation`:
  {"episode_000001.hdf5": {"exclude": true, "note": "...", "tags": ["slow"], "success": false, "quality": 4,
                           "annotations": [{"start": 120, "stop": 180, "label": "bad"}]}}
Every field is optional; `success` overrides the label saved with the take; annotation ranges are steps
[start, stop) with a free label (phase names, language sub-instructions, or a rules.cut_labels entry to cut).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from xrcollect.config import CurateConfig
from xrcollect.env import CONTACT_KINDS
from xrcollect.recorder import SOURCE_OPERATOR, SOURCE_POLICY, episode_files, read_episode

AUDIT_VERSION = 1          # bump whenever a definition below changes: caches are redone
CACHE_NAME = "audit.json"
CURATION_NAME = "curation.json"
IDLE_SPEED = 0.02          # rad/s: every joint slower than this, and nobody engaged = still
MIN_SEGMENT_S = 1.0        # shorter pieces are not worth a training episode
SAFETY_EVENT_RAD = 0.05    # a safety intervention: the filter changed a joint target by more than this
AUDIT_KEYS = ["obs/joint_vel", "obs/objects", "teleop/engaged", "task/progress", "task/held", "contacts",
              "control/source", "safety/correction"]
CONTACT_TEXT = {"link_table": "arm link on the table", "arm_arm": "arms touching",
                "tip_table": "hand dragging on the table"}


@dataclass
class EpisodeAudit:
    file: str
    num_steps: int
    fps: int
    success: bool
    contact_frac: dict[str, float]           # share of steps with each contact kind
    contact_events: dict[str, int]           # separate times each contact kind started
    idle_frac: float
    active: tuple[int, int]                  # [start, stop): first to last step with motion
    stage_times: list[float | None]          # seconds until progress first reached 1..n_stages
    operator_runs: list[tuple[int, int]] = field(default_factory=list)  # [start, stop) not driven by a policy
    contact_runs: dict[str, list[tuple[int, int]]] = field(default_factory=dict)  # [start, stop) per contact kind
    safety_events: int = 0                   # times the safety filter stepped in
    phases: list[dict] = field(default_factory=list)   # automatic {start, stop, label}
    start_poses: dict[str, list[float]] = field(default_factory=dict)  # object -> [x, y, yaw] at step 0
    info: dict = field(default_factory=dict)  # task, task_version, operator, session, created_at, source

    @property
    def duration_s(self) -> float:
        return self.num_steps / self.fps


@dataclass
class Segment:
    """A contiguous piece of an episode that becomes one exported episode."""
    path: Path
    start: int
    stop: int
    label: str | None = None

    @property
    def num_steps(self) -> int:
        return self.stop - self.start


# -- audit ------------------------------------------------------------------------------------------
def true_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, stop) of every run of True."""
    edges = np.diff(np.concatenate([[0], np.asarray(mask).astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def auto_phases(held: np.ndarray, progress: np.ndarray, arms: list[str], objects: list[str],
                stages: list[str]) -> list[dict]:
    """Phases from task events: every hold of an object (grasp to release) and every stage reached. Pre-filled
    annotations for the timeline; the operator's own annotations live in curation.json."""
    out = []
    held = np.asarray(held).reshape(len(progress), -1)
    for a, arm in enumerate(arms[:held.shape[1]]):
        col = held[:, a]
        for obj in np.unique(col[col >= 0]):
            name = objects[int(obj)] if int(obj) < len(objects) else str(obj)
            out += [{"start": s, "stop": e, "label": f"{arm}: hold {name}", "kind": "hold"}
                    for s, e in true_runs(col == obj)]
    for k in range(1, int(np.max(progress, initial=0)) + 1):
        t = int(np.argmax(progress >= k))
        name = stages[k - 1] if k - 1 < len(stages) else f"stage {k}"
        out.append({"start": t, "stop": t + 1, "label": name, "kind": "stage"})
    return sorted(out, key=lambda p: (p["start"], p["stop"]))


def audit_arrays(file: str, fps: int, success: bool, data: dict, meta: dict | None = None) -> EpisodeAudit:
    """The audit of one take from its AUDIT_KEYS arrays (an episode file, or a take still in memory)."""
    meta = meta or {}
    contacts, progress = np.asarray(data["contacts"], bool), np.asarray(data["task/progress"])
    n = len(progress)
    stages = list(meta.get("stage_names") or [])
    n_stages = max(len(stages), int(np.max(progress, initial=0)))
    moving = (np.abs(data["obs/joint_vel"]).max(axis=1) > IDLE_SPEED) | np.asarray(data["teleop/engaged"]).any(axis=1)
    active = (int(np.argmax(moving)), int(n - np.argmax(moving[::-1]))) if moving.any() else (0, 0)
    source = np.asarray(data.get("control/source", np.full(n, SOURCE_OPERATOR)))
    corr = np.asarray(data.get("safety/correction", np.zeros(n)))
    objects = list(meta.get("objects") or [])
    start_poses = {}
    obs_objects = data.get("obs/objects")
    if obs_objects is not None and len(obs_objects):
        for name, p in zip(objects, np.asarray(obs_objects)[0]):
            w, x, y, z = p[3:7]
            start_poses[name] = [float(p[0]), float(p[1]), float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))]
    held = data.get("task/held")
    return EpisodeAudit(
        file=file, num_steps=n, fps=fps, success=success,
        contact_frac={k: float(contacts[:, i].mean()) if n else 0.0 for i, k in enumerate(CONTACT_KINDS)},
        contact_events={k: len(true_runs(contacts[:, i])) for i, k in enumerate(CONTACT_KINDS)},
        idle_frac=float(1.0 - moving.mean()) if n else 0.0, active=active,
        stage_times=[float(np.argmax(progress >= k)) / fps if (progress >= k).any() else None
                     for k in range(1, n_stages + 1)],
        operator_runs=true_runs(source != SOURCE_POLICY),
        contact_runs={k: true_runs(contacts[:, i]) for i, k in enumerate(CONTACT_KINDS)},
        safety_events=len(true_runs(corr > SAFETY_EVENT_RAD)),
        phases=[] if held is None else auto_phases(held, progress, list(meta.get("arms") or []), objects, stages),
        start_poses=start_poses,
        info={k: meta.get(k) for k in ("task", "task_version", "operator", "session", "created_at", "source")},
    )


def audit_episode(path: Path) -> EpisodeAudit:
    meta, data = read_episode(path, keys=AUDIT_KEYS)
    return audit_arrays(Path(path).name, int(meta["fps"]), bool(meta.get("success")), data, meta)


def _from_cache(d: dict) -> EpisodeAudit:
    a = EpisodeAudit(**d)
    a.active = tuple(a.active)
    a.operator_runs = [tuple(r) for r in a.operator_runs]
    a.contact_runs = {k: [tuple(r) for r in v] for k, v in a.contact_runs.items()}
    return a


def audit_dir(episodes_dir: Path, progress=None, cache: bool = True) -> dict[str, EpisodeAudit]:
    """Audits of every episode in a folder, from the cache where the file hasn't changed since."""
    episodes_dir = Path(episodes_dir)
    cache_path = episodes_dir / CACHE_NAME
    cached = {}
    if cache_path.exists():
        try:
            raw = json.loads(cache_path.read_text())
            if raw.get("version") == AUDIT_VERSION:
                cached = raw["episodes"]
        except ValueError:
            pass
    out, fresh = {}, {}
    for path in episode_files(episodes_dir):
        st = path.stat()
        key = [st.st_size, st.st_mtime]
        entry = cached.get(path.name)
        if entry and entry["key"] == key:
            a = _from_cache(entry["audit"])
        else:
            if progress:
                progress(path.name)
            a = audit_episode(path)
        out[path.name] = a
        fresh[path.name] = {"key": key, "audit": asdict(a)}
    if cache and json.loads(json.dumps(fresh)) != cached:  # as JSON: tuples come back from the file as lists
        cache_path.write_text(json.dumps({"version": AUDIT_VERSION, "episodes": fresh}, indent=1))
    return out


# -- curation (manual labels) -------------------------------------------------------------------------
CURATION_FIELDS = {"exclude": bool, "note": str, "tags": list, "success": bool, "quality": int, "annotations": list}


def load_curation(episodes_dir: Path) -> dict[str, dict]:
    p = Path(episodes_dir) / CURATION_NAME
    return json.loads(p.read_text()) if p.exists() else {}


def _clean_annotations(v) -> list[dict]:
    out = []
    for a in v or []:
        start, stop = int(a["start"]), int(a["stop"])
        label = str(a.get("label", "")).strip()[:120]
        if stop > start >= 0 and label:
            out.append({"start": start, "stop": stop, "label": label})
    return sorted(out, key=lambda a: (a["start"], a["stop"]))


def update_curation(episodes_dir: Path, file: str, **fields) -> dict:
    """Change the manual labels of one episode (CURATION_FIELDS); None, False for exclude, "", [] and 0 clear a
    field. The only writer of curation.json. Returns the episode's entry."""
    cur = load_curation(episodes_dir)
    entry = cur.get(file, {})
    for k, v in fields.items():
        if k not in CURATION_FIELDS:
            raise ValueError(f"unknown curation field {k!r}")
        if k == "tags" and v is not None:
            v = sorted({str(t).strip()[:32] for t in v if str(t).strip()})
        if k == "annotations" and v is not None:
            v = _clean_annotations(v)
        if k == "quality" and v is not None:
            v = int(np.clip(int(v), 0, 5))
        if k == "note" and v is not None:
            v = str(v)[:500]
        if v is None or v == "" or v == [] or (k in ("exclude", "quality") and not v):
            entry.pop(k, None)
        else:
            entry[k] = CURATION_FIELDS[k](v)
    if entry:
        cur[file] = entry
    else:
        cur.pop(file, None)
    Path(episodes_dir).mkdir(parents=True, exist_ok=True)
    (Path(episodes_dir) / CURATION_NAME).write_text(json.dumps(cur, indent=1, sort_keys=True))
    return entry


def forget_curation(episodes_dir: Path, file: str) -> None:
    """Drop the manual labels of a deleted episode."""
    cur = load_curation(episodes_dir)
    if cur.pop(file, None) is not None:
        (Path(episodes_dir) / CURATION_NAME).write_text(json.dumps(cur, indent=1, sort_keys=True))


def labelled_success(a: EpisodeAudit, curation: dict) -> bool:
    """The take's success label: the manual override if there is one, else the one saved with the take."""
    return bool(curation.get(a.file, {}).get("success", a.success))


# -- selection ----------------------------------------------------------------------------------------
def verdict(a: EpisodeAudit, rules: CurateConfig, curation: dict) -> str | None:
    """Why an episode is left out of datasets, or None if it is used."""
    c = curation.get(a.file, {})
    if c.get("exclude"):
        return "excluded by hand" + (f": {c['note']}" if c.get("note") else "")
    reached = rules.until_stage is not None and rules.until_stage <= len(a.stage_times) \
        and a.stage_times[rules.until_stage - 1] is not None  # a sub-task dataset: that stage is the success
    if not labelled_success(a, curation) and not rules.include_failures and not reached:
        return "failed take"
    for kind, limit in rules.max_contact_frac.items():
        if kind in rules.cut_contacts or kind not in a.contact_frac:  # cut out in kept_spans instead
            continue
        if a.contact_frac[kind] > limit:
            return f"{CONTACT_TEXT.get(kind, kind)} {a.contact_frac[kind]:.1%} of the time (limit {limit:.1%})"
    return None


def _without(span: tuple[int, int], cuts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """The parts of [start, stop) that no cut covers, in order."""
    start, stop = span
    out = []
    for cut_start, cut_stop in sorted(cuts):
        if cut_stop <= start or cut_start >= stop:
            continue
        if cut_start > start:
            out.append((start, cut_start))
        start = max(start, cut_stop)
    if start < stop:
        out.append((start, stop))
    return out


def kept_spans(a: EpisodeAudit, rules: CurateConfig, curation: dict | None = None) -> list[tuple[int, int]]:
    """[start, stop) of the steps of an accepted take that go into a dataset: idle ends trimmed, only the
    operator's steps of a shared-control take, cut contacts and "bad" annotations removed (plus a margin), pieces
    shorter than MIN_SEGMENT_S dropped; with rules.until_stage, only up to just after that stage was reached."""
    lo, hi = a.active if rules.trim_idle else (0, a.num_steps)
    if rules.until_stage is not None:
        t = a.stage_times[rules.until_stage - 1] if rules.until_stage <= len(a.stage_times) else None
        if t is None:
            return []
        hi = min(hi, round((t + rules.until_stage_tail_s) * a.fps))
    runs = a.operator_runs if rules.shared_frames == "operator" else [(0, a.num_steps)]
    margin = round(rules.contact_margin_s * a.fps)
    cuts = [(max(0, s - margin), e + margin) for kind in rules.cut_contacts for s, e in a.contact_runs.get(kind, [])]
    notes = (curation or {}).get(a.file, {}).get("annotations", [])
    cuts += [(n["start"], n["stop"]) for n in notes if n["label"] in rules.cut_labels]
    pieces = [part for s, e in runs for part in _without((max(s, lo), min(e, hi)), cuts)]
    return [(s, e) for s, e in pieces if e - s >= MIN_SEGMENT_S * a.fps]


def why_not(a: EpisodeAudit, rules: CurateConfig, curation: dict) -> str | None:
    """verdict + the case that trimming leaves nothing: the one answer shown everywhere."""
    return verdict(a, rules, curation) or (None if kept_spans(a, rules, curation) else "nothing left after trimming")


def select(episodes_dir: Path, audits: dict[str, EpisodeAudit], rules: CurateConfig,
           curation: dict | None = None) -> tuple[list[Segment], dict[str, str]]:
    """Segments to export, and the reason each rejected episode was left out."""
    curation = load_curation(episodes_dir) if curation is None else curation
    segments, rejected = [], {}
    for name, a in sorted(audits.items()):
        why = why_not(a, rules, curation)
        if why:
            rejected[name] = why
            continue
        segments += [Segment(Path(episodes_dir) / name, s, e) for s, e in kept_spans(a, rules, curation)]
    return segments, rejected


def annotation_segments(episodes_dir: Path, audits: dict[str, EpisodeAudit], rules: CurateConfig,
                        curation: dict | None = None) -> list[Segment]:
    """Language-annotated sub-tasks: one labelled segment per manual annotation of an accepted take (cut labels
    excepted). An exporter can use the label as the sub-instruction."""
    curation = load_curation(episodes_dir) if curation is None else curation
    out = []
    for name, a in sorted(audits.items()):
        if why_not(a, rules, curation):
            continue
        for n in curation.get(name, {}).get("annotations", []):
            if n["label"] not in rules.cut_labels and n["stop"] <= a.num_steps:
                out.append(Segment(Path(episodes_dir) / name, n["start"], n["stop"], n["label"]))
    return out


# -- dataset health -----------------------------------------------------------------------------------
def dataset_health(episodes_dir: Path, rules: CurateConfig) -> dict:
    """Numbers for the dataset health page: start-state coverage, durations, success over time, contact events,
    per-operator statistics, task versions."""
    audits = audit_dir(episodes_dir)
    curation = load_curation(episodes_dir)
    rows, operators, versions = [], {}, {}
    for name, a in sorted(audits.items()):
        ok = labelled_success(a, curation)
        why = why_not(a, rules, curation)
        info = a.info or {}
        rows.append({"file": name, "success": ok, "duration_s": round(a.duration_s, 2), "used": why is None,
                     "verdict": why, "start_poses": a.start_poses, "created_at": info.get("created_at"),
                     "operator": info.get("operator"), "contact_events": a.contact_events,
                     "safety_events": a.safety_events, "stage_times": a.stage_times,
                     "task_version": info.get("task_version"), "quality": curation.get(name, {}).get("quality")})
        op = operators.setdefault(info.get("operator") or "?", {"takes": 0, "success": 0, "duration_s": 0.0,
                                                               "first": None, "last": None})
        op["takes"] += 1
        op["success"] += ok
        op["duration_s"] += a.duration_s
        t = info.get("created_at")
        if t:
            op["first"] = min(op["first"] or t, t)
            op["last"] = max(op["last"] or t, t)
        v = versions.setdefault(info.get("task_version") or "?", {"task": info.get("task"), "count": 0})
        v["count"] += 1
    for op in operators.values():
        op["success_rate"] = op["success"] / op["takes"]
        op["mean_duration_s"] = op["duration_s"] / op["takes"]
        hours = None
        if op["first"] and op["last"] and op["first"] != op["last"]:
            from datetime import datetime

            hours = (datetime.fromisoformat(op["last"]) - datetime.fromisoformat(op["first"])).total_seconds() / 3600
        op["takes_per_hour"] = round(op["takes"] / hours, 1) if hours and hours > 0.02 else None
    used = [r for r in rows if r["used"]]
    return {"episodes": rows, "operators": operators, "task_versions": versions,
            "count": len(rows), "used": len(used), "success_rate": float(np.mean([r["success"] for r in rows])) if rows else 0.0,
            "total_s": float(sum(r["duration_s"] for r in rows)), "contact_kinds": list(CONTACT_KINDS)}
