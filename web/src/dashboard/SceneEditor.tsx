import { useCallback, useEffect, useMemo, useState } from "react";
import type { Status, TaskSpec } from "../protocol";
import type { Cmd } from "./util";

/** Scene tab: edit the task file from the dashboard. Every change is an edit operation (xrcollect/editor.py) the
 *  server writes into the task YAML (comments kept) and then hot-reloads; nothing here knows any particular task.
 *  Region, start-region and camera forms are built from the task file's JSON Schema (GET /api/schema). */

type Json = null | boolean | number | string | Json[] | { [k: string]: Json };
type Obj = { [k: string]: Json };
interface SchemaNode { type?: string; title?: string; description?: string; items?: SchemaNode; minItems?: number;
  maxItems?: number; properties?: Record<string, SchemaNode>; required?: string[]; default?: Json; anyOf?: SchemaNode[];
  $ref?: string }
interface Schema { task: SchemaNode & { $defs: Record<string, SchemaNode> }; predicates: Record<string, { doc: string; args: Obj }> }
interface TaskFile { path: string; text: string; data: Obj; version: string }
interface Library { objects: string[]; embodiments: string[]; scenes: string[] }
interface TaskList { current: string; tasks: { path: string; task: string; instruction: string }[] }

const getJSON = <T,>(url: string): Promise<T> => fetch(url, { cache: "no-store" }).then((r) => {
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json() as Promise<T>;
});
const num = (v: Json | undefined, d = 0) => (typeof v === "number" ? v : d);
const arr = (v: Json | undefined): number[] => (Array.isArray(v) ? v.map((x) => num(x)) : []);
const round = (v: number, k = 4) => Math.round(v * 10 ** k) / 10 ** k;
const hex = (rgba: number[]) => `#${rgba.slice(0, 3).map((c) => Math.round(Math.min(1, Math.max(0, c)) * 255).toString(16).padStart(2, "0")).join("")}`;
const fromHex = (h: string, a = 1) => [1, 3, 5].map((i) => round(parseInt(h.slice(i, i + 2), 16) / 255, 3)).concat(a);
/** JSON like a hand-written task file: short arrays and small objects on one line. */
function inline(v: Json): string {
  if (v === null || typeof v !== "object") return JSON.stringify(v);
  if (Array.isArray(v)) return `[${v.map(inline).join(", ")}]`;
  return `{${Object.entries(v).map(([k, x]) => `${JSON.stringify(k)}: ${inline(x)}`).join(", ")}}`;
}
function pretty(v: Json, ind = ""): string {
  const one = inline(v);
  if (v === null || typeof v !== "object" || one.length <= 64) return one;
  const next = ind + "  ";
  if (Array.isArray(v)) return `[\n${v.map((x) => next + pretty(x, next)).join(",\n")}\n${ind}]`;
  return `{\n${Object.entries(v).map(([k, x]) => `${next}${JSON.stringify(k)}: ${pretty(x, next)}`).join(",\n")}\n${ind}}`;
}
const resolve = (n: SchemaNode, defs: Record<string, SchemaNode>): SchemaNode => {
  if (n.$ref) return resolve(defs[n.$ref.split("/").pop()!], defs);
  const plain = n.anyOf?.find((x) => x.type !== "null" && x.type !== "string");
  return plain && !n.type ? resolve(plain, defs) : n;
};

/** One sub-schema (Region, CameraSpec, ...) as a form: number vectors, numbers, booleans and strings. */
function SchemaForm({ node, defs, value, onChange, skip = [] }: {
  node: SchemaNode; defs: Record<string, SchemaNode>; value: Obj; onChange: (v: Obj) => void; skip?: string[];
}) {
  const props = Object.entries(resolve(node, defs).properties ?? {}).filter(([k]) => !skip.includes(k));
  return (
    <div className="sform">
      {props.map(([key, raw]) => {
        const p = resolve(raw, defs);
        const v = value[key] ?? p.default ?? null;
        const set = (x: Json) => onChange({ ...value, [key]: x });
        const label = <span className="muted small" title={p.description}>{(p.title ?? key).toLowerCase()}</span>;
        if (p.type === "array" && p.items?.type === "number") {
          const n = p.maxItems ?? arr(v).length;
          const xs = Array.from({ length: n }, (_, i) => arr(v)[i] ?? 0);
          return (
            <label key={key}>{label}
              <span className="vec">{xs.map((x, i) => (
                <NumberInput key={i} value={x} onChange={(y) => set(xs.map((o, j) => (j === i ? y : o)))} />
              ))}</span>
            </label>
          );
        }
        if (p.type === "number" || p.type === "integer") {
          return <label key={key}>{label}<NumberInput value={num(v)} onChange={set} /></label>;
        }
        if (p.type === "boolean") {
          return <label key={key} className="check"><input type="checkbox" checked={!!v} onChange={(e) => set(e.target.checked)} />{label}</label>;
        }
        if (p.type === "string") {
          return <label key={key}>{label}<input type="text" value={typeof v === "string" ? v : ""} onChange={(e) => set(e.target.value)} /></label>;
        }
        return null;
      })}
    </div>
  );
}

/** A number field that keeps what is being typed ("0.", "-") and reports only valid numbers. */
function NumberInput({ value, onChange, step = 0.01, title, placeholder }: {
  value: number | null; onChange: (v: number) => void; step?: number; title?: string; placeholder?: string;
}) {
  const shown = (v: number | null) => (v == null || !Number.isFinite(v) ? "" : String(v));
  const [text, setText] = useState(shown(value));
  useEffect(() => { if (text === "" ? value != null : +text !== value) setText(shown(value)); }, [value]); // eslint-disable-line react-hooks/exhaustive-deps
  return <input type="number" step={step} value={text} title={title} placeholder={placeholder}
    onChange={(e) => { setText(e.target.value); if (e.target.value !== "" && Number.isFinite(+e.target.value)) onChange(+e.target.value); }} />;
}

export function SceneEditor({ status, task, cmd, selected, onSelect }: {
  status: Status | null; task: TaskSpec | null; cmd: Cmd; selected: string | null; onSelect: (n: string | null) => void;
}) {
  const [file, setFile] = useState<TaskFile | null>(null);
  const [fileErr, setFileErr] = useState<string | null>(null);
  const [schema, setSchema] = useState<Schema | null>(null);
  const [lib, setLib] = useState<Library | null>(null);
  const [tasks, setTasks] = useState<TaskList | null>(null);
  const version = status?.task_version;
  const sceneVersion = status?.scene_version;
  const refresh = useCallback(() => {
    getJSON<TaskFile>("/api/taskfile").then((f) => { setFile(f); setFileErr(null); })
      .catch((e: Error) => { setFile(null); setFileErr(e.message.endsWith("404") ? "This task has no file to edit." : e.message); });
    getJSON<TaskList>("/api/tasks").then(setTasks).catch(() => {});
  }, []);
  useEffect(refresh, [refresh, version, sceneVersion]);
  useEffect(() => {
    getJSON<Schema>("/api/schema").then(setSchema).catch(() => {});
    getJSON<Library>("/api/library").then(setLib).catch(() => {});
  }, []);

  const edit = useCallback((...ops: Obj[]) => cmd("edit", { ops }), [cmd]);
  const rec = status?.rec_state ?? "idle";
  const locked = rec !== "idle";
  const editing = !!status?.edit;
  const data = file?.data ?? {};
  const objects = (data.objects ?? {}) as Record<string, Obj>;
  const movable = (task?.objects ?? []).filter((o) => !o.fixed).map((o) => o.name);
  const defs = schema?.task.$defs ?? {};

  return (
    <>
      <div className="card scene-ed">
        <div className="card-head"><h2>Scene editor</h2>
          <button className="btn ghost small" title="Rebuild the scene from the files on disk (after editing them by hand)"
            disabled={locked} onClick={() => cmd("reload_scene")}>⟳ Reload</button>
        </div>
        {tasks && (
          <label className="row-field"><span className="muted small">Task</span>
            <select value={tasks.tasks.find((t) => file?.path.startsWith(t.path))?.path ?? ""} disabled={locked}
              onChange={(e) => e.target.value && cmd("reload_scene", { task: e.target.value })}>
              {!tasks.tasks.some((t) => file?.path.startsWith(t.path)) && <option value="">{tasks.current}</option>}
              {tasks.tasks.map((t) => <option key={t.path} value={t.path}>{t.task}</option>)}
            </select>
          </label>
        )}
        {file && <div className="muted small mono">{file.path} · version {file.version.slice(0, 8)}</div>}
        {fileErr && <div className="alert warn small">{fileErr}</div>}
        {locked && <div className="alert info small">A take is in progress: the scene can't change until it is saved or discarded.</div>}
        <div className="edit-row">
          <button className={`btn small ${editing ? "ok" : ""}`} disabled={locked || !file}
            onClick={() => cmd("edit_mode", { enabled: !editing })}>{editing ? "✓ Done editing" : "✎ Edit mode"}</button>
          {editing && <>
            <button className="btn small ok" disabled={!status?.edit?.moved.length} onClick={() => cmd("edit_commit")}>
              Apply moves ({status?.edit?.moved.length ?? 0})</button>
            <button className="btn ghost small" disabled={!status?.edit?.moved.length}
              onClick={() => { cmd("edit_mode", { enabled: false }); cmd("edit_mode", { enabled: true }); }}>Discard</button>
          </>}
        </div>
        <p className="muted small hint">{editing
          ? <>Physics is paused. Click an object in the view, drag the gizmo to move it (<kbd>W</kbd>) or turn it (<kbd>E</kbd>), then <b>Apply moves</b>: its default pose and start region move with it.</>
          : <>Edit mode pauses physics and shows every object at its default pose, in the headset too.</>}</p>
        {editing && <CaptureGoal movable={movable} cmd={cmd} />}
      </div>

      {file && <ObjectsCard objects={objects} spawn={(data.spawn ?? {}) as Record<string, Obj>} regions={Object.keys((data.regions ?? {}) as Obj)}
        lib={lib} selected={selected} onSelect={onSelect} edit={edit} locked={locked} version={file.version} />}
      {file && schema && <MapCard title="Regions" what="region" hint="Named areas on the table, drawn as tape (cameras see it)."
        node={defs.Region} defs={defs} items={(data.regions ?? {}) as Record<string, Obj>} locked={locked} version={file.version}
        blank={{ center: [0.45, 0], half: [0.05, 0.05], tape: true }}
        onSet={(name, v) => edit({ op: "region_set", name, ...v })} onRemove={(name) => edit({ op: "region_remove", name })} />}
      {file && schema && <MapCard title="Cameras" what="camera" hint="Scene cameras added by this task (pos and look-at point in world metres)."
        node={defs.CameraSpec} defs={defs} items={(data.cameras ?? {}) as Record<string, Obj>} locked={locked} version={file.version}
        blank={{ pos: [0.9, 0, 1.4], lookat: [0.45, 0, 0.75], fovy: 60 }}
        onSet={(name, v) => edit({ op: "camera_set", name, ...v })} onRemove={(name) => edit({ op: "camera_remove", name })} />}
      {file && <StagesCard data={data} schema={schema} objects={Object.keys(objects)} edit={edit} locked={locked} version={file.version} />}
    </>
  );
}

function CaptureGoal({ movable, cmd }: { movable: string[]; cmd: Cmd }) {
  const [anchor, setAnchor] = useState("");
  const [tolMm, setTolMm] = useState(10);
  const [tolDeg, setTolDeg] = useState(10);
  return (
    <div className="capture">
      <div className="muted small">Arrange the objects as the finished task, then capture it: one stage per object (its pose relative to the anchor) and the goal picture. Earlier captured stages (<code>*_placed</code>) are replaced, hand-written ones kept, success reset to the default.</div>
      <div className="edit-row">
        <label><span className="muted small">anchor</span>
          <select value={anchor} onChange={(e) => setAnchor(e.target.value)}>
            <option value="">lowest object</option>
            {movable.map((n) => <option key={n} value={n}>{n}</option>)}
          </select>
        </label>
        <label><span className="muted small">± mm</span><NumberInput value={tolMm} step={1} onChange={setTolMm} /></label>
        <label><span className="muted small">± deg</span><NumberInput value={tolDeg} step={1} onChange={setTolDeg} /></label>
        <button className="btn small warn" disabled={movable.length < 2}
          onClick={() => cmd("capture_goal", { ...(anchor ? { anchor } : {}), tol_pos: tolMm / 1000, tol_rot_deg: tolDeg })}>◎ Capture goal</button>
      </div>
    </div>
  );
}

const OBJECT_FIELDS = ["label", "mass", "friction"] as const;

function ObjectsCard({ objects, spawn, regions, lib, selected, onSelect, edit, locked, version }: {
  objects: Record<string, Obj>; spawn: Record<string, Obj>; regions: string[]; lib: Library | null; selected: string | null;
  onSelect: (n: string | null) => void; edit: (...ops: Obj[]) => void; locked: boolean; version: string;
}) {
  const names = Object.keys(objects);
  const [newName, setNewName] = useState("");
  const [newAsset, setNewAsset] = useState("");
  const asset = newAsset || lib?.objects[0] || "";
  const name = newName.trim() || `${asset.split("/").pop() ?? "object"}_${names.length + 1}`;
  const nameOk = /^[A-Za-z_][A-Za-z0-9_]*$/.test(name) && !names.includes(name);
  return (
    <div className="card objects-ed">
      <div className="card-head"><h2>Objects</h2><span className="muted">{names.length}</span></div>
      <div className="obj-list">
        {names.map((n) => (
          <button key={n} className={`obj ${selected === n ? "sel" : ""}`} onClick={() => onSelect(selected === n ? null : n)}>
            <i style={{ background: objects[n].rgba ? hex(arr(objects[n].rgba)) : "#777" }} />
            <b>{n}</b><span className="muted small">{String(objects[n].asset ?? "")}</span>
          </button>
        ))}
      </div>
      {selected && objects[selected] && (
        <ObjectInspector key={`${selected}:${version}`} name={selected} entry={objects[selected]} spawn={spawn[selected] ?? null}
          regions={regions} lib={lib} edit={edit} locked={locked} onRemoved={() => onSelect(null)} />
      )}
      <div className="add-row">
        <select value={asset} onChange={(e) => setNewAsset(e.target.value)}>
          {lib?.objects.map((a) => <option key={a} value={a}>{a}</option>)}
        </select>
        <input type="text" placeholder={name} value={newName} onChange={(e) => setNewName(e.target.value)} className={nameOk ? "" : "bad"} />
        <button className="btn small" disabled={locked || !asset || !nameOk}
          onClick={() => { edit({ op: "object_add", name, asset, pos: [0.45, 0] }); setNewName(""); onSelect(name); }}>+ Add</button>
      </div>
    </div>
  );
}

function ObjectInspector({ name, entry, spawn, regions, lib, edit, locked, onRemoved }: {
  name: string; entry: Obj; spawn: Obj | null; regions: string[]; lib: Library | null; edit: (...ops: Obj[]) => void;
  locked: boolean; onRemoved: () => void;
}) {
  const [draft, setDraft] = useState<Obj>(entry);
  const spawnRegion = spawn?.region;
  const spawnCenter = spawnRegion && typeof spawnRegion === "object" && !Array.isArray(spawnRegion) ? arr((spawnRegion as Obj).center) : [];
  // No default pose in the file: the object starts in its start region, edit mode shows it at the region's centre.
  const pos0 = arr(entry.pos).length ? arr(entry.pos) : spawnCenter.length ? spawnCenter : [0.45, 0];
  const [pos, setPos] = useState<number[]>([pos0[0], pos0[1]]);
  const [yaw, setYaw] = useState(num(entry.yaw_deg));
  const region = spawn?.region;
  const inline = region && typeof region === "object" && !Array.isArray(region) ? (region as Obj) : null;
  const [sp, setSp] = useState<{ half: number[]; yaw: number[] }>({
    half: inline ? arr(inline.half) : [0.04, 0.04], yaw: spawn ? arr(spawn.yaw_deg).concat([0, 0]).slice(0, 2) : [-15, 15] });
  const rgba = arr(draft.rgba).length ? arr(draft.rgba) : [0.7, 0.7, 0.7, 1];

  const fieldOps = () => {
    const fields: Obj = {};
    for (const k of [...OBJECT_FIELDS, "rgba", "asset"] as const) {
      if (JSON.stringify(draft[k] ?? null) !== JSON.stringify(entry[k] ?? null)) fields[k] = draft[k] ?? null;
    }
    const ops: Obj[] = Object.keys(fields).length ? [{ op: "object_set", object: name, fields }] : [];
    if (pos[0] !== pos0[0] || pos[1] !== pos0[1] || yaw !== num(entry.yaw_deg)) ops.push({ op: "object_pose", object: name, pos, yaw_deg: yaw });
    return ops;
  };
  const ops = fieldOps();
  return (
    <div className="inspector">
      <div className="sform">
        <label><span className="muted small">asset</span>
          <select value={String(draft.asset ?? "")} onChange={(e) => setDraft({ ...draft, asset: e.target.value })}>
            {!lib?.objects.includes(String(draft.asset)) && <option>{String(draft.asset ?? "")}</option>}
            {lib?.objects.map((a) => <option key={a} value={a}>{a}</option>)}
          </select>
        </label>
        <label><span className="muted small">label</span>
          <input type="text" value={String(draft.label ?? "")} placeholder="(from the asset)"
            onChange={(e) => setDraft({ ...draft, label: e.target.value || null })} /></label>
        <label><span className="muted small">mass kg</span>
          <NumberInput value={typeof draft.mass === "number" ? draft.mass : null} placeholder="(asset)" step={0.01} onChange={(v) => setDraft({ ...draft, mass: v })} /></label>
        <label><span className="muted small">friction</span>
          <NumberInput value={typeof draft.friction === "number" ? draft.friction : null} placeholder="(asset)" step={0.05} onChange={(v) => setDraft({ ...draft, friction: v })} /></label>
        <label><span className="muted small">colour</span>
          <span className="vec"><input type="color" value={hex(rgba)} onChange={(e) => setDraft({ ...draft, rgba: fromHex(e.target.value, rgba[3]) })} />
            {draft.rgba != null && <button className="btn ghost small" title="Use the asset's colour" onClick={() => setDraft({ ...draft, rgba: null })}>×</button>}</span>
        </label>
        <label><span className="muted small">default x, y (m)</span>
          <span className="vec">{[0, 1].map((i) => <NumberInput key={i} value={pos[i]} onChange={(v) => setPos(pos.map((o, j) => (j === i ? v : o)))} />)}</span>
        </label>
        <label><span className="muted small">yaw deg</span><NumberInput value={yaw} step={5} onChange={setYaw} /></label>
      </div>
      <div className="edit-row">
        <button className="btn small ok" disabled={locked || !ops.length} onClick={() => edit(...ops)}>Apply</button>
        <button className="btn ghost small" disabled={!ops.length} onClick={() => { setDraft(entry); setPos([pos0[0], pos0[1]]); setYaw(num(entry.yaw_deg)); }}>Revert</button>
        <span className="grow" />
        <button className="btn ghost small danger" disabled={locked} onClick={() => { edit({ op: "object_remove", object: name }); onRemoved(); }}>🗑 Remove</button>
      </div>
      <div className="spawn-ed">
        <div className="muted small"><b>Start region</b>{" "}
          {!spawn ? "none: the object always starts at its default pose."
            : typeof region === "string" ? `uses the region "${region}"${regions.includes(region) ? "" : " (missing!)"}; saving makes it its own, centred on the default pose.`
            : "uniform in this rectangle around the default pose, with a random yaw."}</div>
        <div className="sform">
          <label><span className="muted small">half x, y (m)</span>
            <span className="vec">{[0, 1].map((i) => <NumberInput key={i} value={sp.half[i]} onChange={(v) => setSp({ ...sp, half: sp.half.map((o, j) => (j === i ? v : o)) })} />)}</span></label>
          <label><span className="muted small">yaw range deg</span>
            <span className="vec">{[0, 1].map((i) => <NumberInput key={i} value={sp.yaw[i]} step={5} onChange={(v) => setSp({ ...sp, yaw: sp.yaw.map((o, j) => (j === i ? v : o)) })} />)}</span></label>
        </div>
        <div className="edit-row">
          <button className="btn small" disabled={locked} onClick={() => edit({ op: "spawn_set", object: name, center: inline ? arr(inline.center) : pos, half: sp.half, yaw_deg: sp.yaw })}>
            {spawn ? "Save start region" : "+ Start region"}</button>
          {spawn && <button className="btn ghost small" disabled={locked} onClick={() => edit({ op: "spawn_remove", object: name })}>Fixed start</button>}
        </div>
      </div>
    </div>
  );
}

/** A name -> entry section of the task file (regions, cameras), each entry a schema form with Apply / Remove. */
function MapCard({ title, what, hint, node, defs, items, blank, onSet, onRemove, locked, version }: {
  title: string; what: string; hint: string; node: SchemaNode | undefined; defs: Record<string, SchemaNode>;
  items: Record<string, Obj>; blank: Obj; onSet: (name: string, v: Obj) => void; onRemove: (name: string) => void;
  locked: boolean; version: string;
}) {
  const [adding, setAdding] = useState<string | null>(null);
  const names = Object.keys(items);
  if (!node) return null;
  return (
    <div className="card map-ed">
      <div className="card-head"><h2>{title}</h2>
        <button className="btn ghost small" disabled={locked} onClick={() => setAdding(`${what}_${names.length + 1}`)}>+ Add</button>
      </div>
      <p className="muted small">{hint}</p>
      {!names.length && adding === null && <p className="muted small">None.</p>}
      {names.map((n) => <MapEntry key={`${n}:${version}`} name={n} value={items[n]} node={node} defs={defs} locked={locked}
        onSet={(v) => onSet(n, v)} onRemove={() => onRemove(n)} />)}
      {adding !== null && (
        <MapEntry key={`new:${version}`} name={adding} value={blank} node={node} defs={defs} locked={locked} isNew
          onRename={setAdding} onSet={(v) => { if (/^[A-Za-z_][A-Za-z0-9_]*$/.test(adding) && !names.includes(adding)) { onSet(adding, v); setAdding(null); } }}
          onRemove={() => setAdding(null)} />
      )}
    </div>
  );
}

function MapEntry({ name, value, node, defs, onSet, onRemove, onRename, locked, isNew = false }: {
  name: string; value: Obj; node: SchemaNode; defs: Record<string, SchemaNode>; onSet: (v: Obj) => void; onRemove: () => void;
  onRename?: (n: string) => void; locked: boolean; isNew?: boolean;
}) {
  const [draft, setDraft] = useState<Obj>(value);
  const dirty = isNew || JSON.stringify(draft) !== JSON.stringify(value);
  return (
    <div className="map-entry">
      <div className="edit-row">
        {isNew ? <input type="text" value={name} onChange={(e) => onRename?.(e.target.value)} /> : <b className="mono">{name}</b>}
        <span className="grow" />
        <button className="btn small ok" disabled={locked || !dirty} onClick={() => onSet(draft)}>{isNew ? "Add" : "Apply"}</button>
        <button className="btn ghost small danger" disabled={locked && !isNew} onClick={onRemove}>{isNew ? "Cancel" : "×"}</button>
      </div>
      <SchemaForm node={node} defs={defs} value={draft} onChange={setDraft} />
    </div>
  );
}

/** Stages (ordered predicates), success and instruction. Predicates are edited as JSON, with the catalogue at hand. */
function StagesCard({ data, schema, objects, edit, locked, version }: {
  data: Obj; schema: Schema | null; objects: string[]; edit: (...ops: Obj[]) => void; locked: boolean; version: string;
}) {
  const stages0 = useMemo(() => ((data.stages ?? []) as Obj[]).map((s) => ({
    name: String(s.name ?? ""), label: String(s.label ?? ""), object: String(s.object ?? ""), when: pretty(s.when ?? {}) })), [data]);
  const success0 = data.success == null ? "" : pretty(data.success);
  const [stages, setStages] = useState(stages0);
  const [success, setSuccess] = useState(success0);
  const [instruction, setInstruction] = useState(String(data.instruction ?? ""));
  const [pred, setPred] = useState("");
  useEffect(() => { setStages(stages0); setSuccess(success0); setInstruction(String(data.instruction ?? "")); }, [version]); // eslint-disable-line react-hooks/exhaustive-deps

  const parse = (t: string): Json | undefined => { try { return JSON.parse(t) as Json; } catch { return undefined; } };
  const bad = stages.map((s) => { const w = parse(s.when); return w === undefined || typeof w !== "object" || w === null || Array.isArray(w) || !s.name; });
  const succBad = success.trim() !== "" && parse(success) === undefined;
  const dirtyStages = JSON.stringify(stages) !== JSON.stringify(stages0);
  const set = (i: number, k: keyof (typeof stages)[number], v: string) => setStages(stages.map((s, j) => (j === i ? { ...s, [k]: v } : s)));
  const move = (i: number, d: number) => {
    const j = i + d;
    if (j < 0 || j >= stages.length) return;
    const s = stages.slice();
    [s[i], s[j]] = [s[j], s[i]];
    setStages(s);
  };
  const template = (p: string) => pretty({ [p]: schema?.predicates[p]?.args ?? {} });
  return (
    <div className="card stages-ed">
      <div className="card-head"><h2>Task logic</h2></div>
      <label className="row-field"><span className="muted small">Instruction</span>
        <span className="vec"><input type="text" value={instruction} onChange={(e) => setInstruction(e.target.value)} />
          <button className="btn small ok" disabled={locked || instruction === String(data.instruction ?? "")}
            onClick={() => edit({ op: "set", path: ["instruction"], value: instruction })}>Apply</button></span>
      </label>
      <h3>Stages <span className="muted small">in order: progress counts the stages that hold, from the first</span></h3>
      {stages.map((s, i) => (
        <div key={i} className={`stage-ed ${bad[i] ? "bad" : ""}`}>
          <div className="edit-row">
            <span className="muted">{i + 1}</span>
            <input type="text" value={s.name} placeholder="name" onChange={(e) => set(i, "name", e.target.value)} />
            <input type="text" value={s.label} placeholder="label (shown)" onChange={(e) => set(i, "label", e.target.value)} />
            <select value={s.object} onChange={(e) => set(i, "object", e.target.value)} title="The object this stage is about (goal panel, next-object hint)">
              <option value="">no object</option>
              {objects.map((o) => <option key={o} value={o}>{o}</option>)}
            </select>
            <button className="icon-btn" title="Up" onClick={() => move(i, -1)}>↑</button>
            <button className="icon-btn" title="Down" onClick={() => move(i, 1)}>↓</button>
            <button className="icon-btn" title="Remove" onClick={() => setStages(stages.filter((_, j) => j !== i))}>×</button>
          </div>
          <textarea rows={Math.min(8, s.when.split("\n").length)} spellCheck={false} value={s.when} onChange={(e) => set(i, "when", e.target.value)} />
        </div>
      ))}
      <div className="edit-row">
        <select value={pred} onChange={(e) => setPred(e.target.value)} title={schema?.predicates[pred]?.doc}>
          <option value="">predicate…</option>
          {Object.keys(schema?.predicates ?? {}).map((p) => <option key={p} value={p}>{p}</option>)}
        </select>
        <button className="btn ghost small" onClick={() => setStages([...stages, { name: `stage_${stages.length + 1}`, label: "", object: "", when: template(pred || "inside") }])}>+ Stage</button>
        <span className="grow" />
        <button className="btn ghost small" disabled={!dirtyStages} onClick={() => setStages(stages0)}>Revert</button>
        <button className="btn small ok" disabled={locked || !dirtyStages || bad.some(Boolean)}
          onClick={() => edit({ op: "stages_set", stages: stages.map((s) => ({ name: s.name, label: s.label || null, object: s.object || null, when: parse(s.when)! })) })}>Apply stages</button>
      </div>
      {pred && schema?.predicates[pred] && <p className="muted small"><b>{pred}</b>: {schema.predicates[pred].doc} <code>{JSON.stringify(schema.predicates[pred].args)}</code></p>}
      <h3>Success <span className="muted small">empty = all stages done, nothing held, everything at rest</span></h3>
      <textarea className={succBad ? "bad" : ""} rows={Math.min(8, Math.max(2, success.split("\n").length))} spellCheck={false} value={success}
        onChange={(e) => setSuccess(e.target.value)} />
      <div className="edit-row">
        <span className="grow" />
        <button className="btn small ok" disabled={locked || succBad || success === success0}
          onClick={() => edit({ op: "success_set", success: success.trim() ? parse(success)! : null })}>Apply success</button>
      </div>
    </div>
  );
}
