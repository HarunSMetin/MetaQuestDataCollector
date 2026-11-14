// Draws a task's goal (from TaskSpec.goal_view) as a small 3D-looking picture on a 2D canvas, and derives the
// per-stage states the panels show. Shared by the headset's goal panel and the dashboard. Nothing here knows any
// particular task: it draws whatever objects and placements the server describes.

import type { Status, TaskObject, TaskSpec } from "./protocol";

export type StageState = "done" | "next" | "todo";

/** Per stage: done (its predicate holds and all before it), the next one to do, or still to do. */
export function stageStates(task: TaskSpec, s: Pick<Status, "progress"> | null): StageState[] {
  const progress = s?.progress ?? 0;
  return task.stages.map((_, k) => (k < progress ? "done" : k === progress ? "next" : "todo"));
}

/** State of each object shown in the goal picture, from the stage that is about it. */
export function objectStates(task: TaskSpec, s: Pick<Status, "progress"> | null): Record<string, StageState> {
  const states = stageStates(task, s);
  const out: Record<string, StageState> = {};
  task.stages.forEach((st, k) => { if (st.object && !(st.object in out)) out[st.object] = states[k]; });
  return out;
}

type V3 = [number, number, number];
const EL = 0.6, AZ = -0.65; // view: from above (34°) and from the operator's right
const D: V3 = [Math.cos(EL) * Math.cos(AZ), Math.cos(EL) * Math.sin(AZ), -Math.sin(EL)]; // view direction
const R: V3 = norm([D[1], -D[0], 0]); // screen right = d x z
const U: V3 = cross(R, D); // screen up

function norm(v: V3): V3 { const n = Math.hypot(...v); return [v[0] / n, v[1] / n, v[2] / n]; }
function cross(a: V3, b: V3): V3 { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
const dot = (a: V3, b: V3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

function shade(rgba: number[], k: number, alpha: number) {
  const c = (i: number) => Math.round(255 * Math.min(1, rgba[i] * k));
  return `rgba(${c(0)},${c(1)},${c(2)},${alpha})`;
}

interface Placed { obj: TaskObject; pos: V3; yaw: number; state: StageState }

/** The objects of the goal view at their goal poses (the anchor at the origin). Empty if the task has none. */
export function goalItems(task: TaskSpec, s: Pick<Status, "progress"> | null): Placed[] {
  const gv = task.goal_view;
  if (!gv) return [];
  const byName = new Map(task.objects.map((o) => [o.name, o]));
  const states = objectStates(task, s);
  const items: Placed[] = [];
  const anchor = byName.get(gv.anchor);
  if (anchor) items.push({ obj: anchor, pos: [0, 0, 0], yaw: 0, state: states[anchor.name] ?? "done" });
  for (const e of gv.show) {
    const obj = byName.get(e.object);
    if (!obj) continue;
    const p = e.pose.pos ?? [0, 0, 0];
    items.push({ obj, pos: [p[0], p[1], p[2]], yaw: e.pose.yaw ?? 0, state: states[obj.name] ?? "todo" });
  }
  return items;
}

/** Draw the goal into the box (x, y, w, h) of a canvas context. Returns false if there is nothing to draw. */
export function drawGoal(g: CanvasRenderingContext2D, task: TaskSpec, s: Pick<Status, "progress"> | null,
  box: { x: number; y: number; w: number; h: number }, pulse = 1): boolean {
  const items = goalItems(task, s);
  if (!items.length) return false;
  const world = (it: Placed, lx: number, ly: number, lz: number): V3 => {
    const c = Math.cos(it.yaw), sn = Math.sin(it.yaw);
    return [it.pos[0] + c * lx - sn * ly, it.pos[1] + sn * lx + c * ly, it.pos[2] + lz];
  };
  const corners = (it: Placed, z: number): V3[] => {
    const [sx, sy] = [it.obj.size[0] / 2, it.obj.size[1] / 2];
    return [[-sx, -sy], [sx, -sy], [sx, sy], [-sx, sy]].map(([a, b]) => world(it, a, b, z));
  };
  const studH = (it: Placed) => (it.obj.studs ? it.obj.stud_h ?? 0 : 0);

  // Fit the structure into the box. If the task has taped regions, one is drawn hugging the base (the real one
  // is usually larger; fitting it would shrink the objects to nothing).
  const base = items[0], m = 0.35 * Math.max(base.obj.size[0], base.obj.size[1]);
  const [ax, ay] = [base.obj.size[0] / 2 + m, base.obj.size[1] / 2 + m];
  const floor: V3[] = [[-ax, -ay, 0], [ax, -ay, 0], [ax, ay, 0], [-ax, ay, 0]].map(([x, y]) => world(base, x, y, 0));
  const pts = [...floor, ...items.flatMap((it) => [...corners(it, 0), ...corners(it, it.obj.size[2] + studH(it))])];
  const sx = pts.map((p) => dot(p, R)), sy = pts.map((p) => -dot(p, U));
  const minX = Math.min(...sx), maxX = Math.max(...sx), minY = Math.min(...sy), maxY = Math.max(...sy);
  const scale = Math.min(box.w / (maxX - minX), box.h / (maxY - minY)) * 0.92;
  const ox = box.x + box.w / 2 - ((minX + maxX) / 2) * scale, oy = box.y + box.h / 2 - ((minY + maxY) / 2) * scale;
  const P = (p: V3): [number, number] => [ox + dot(p, R) * scale, oy - dot(p, U) * scale];
  const poly = (ps: V3[]) => { g.beginPath(); ps.forEach((p, i) => (i ? g.lineTo(...P(p)) : g.moveTo(...P(p)))); g.closePath(); };

  if (Object.keys(task.regions).length) {
    poly(floor);
    g.fillStyle = "rgba(235,235,225,0.06)";
    g.fill();
    g.setLineDash([10, 8]);
    g.strokeStyle = "rgba(235,235,225,0.55)";
    g.lineWidth = 3;
    g.stroke();
    g.setLineDash([]);
  }

  // Painter's order: lower objects first, then farther ones.
  const order = items.slice().sort((a, b) => a.pos[2] - b.pos[2] || dot(b.pos, D) - dot(a.pos, D));
  for (const it of order) {
    const st = it.state, rgba = it.obj.rgba;
    const alpha = st === "todo" ? 0.25 : 1;
    const h = it.obj.size[2];
    const bot = corners(it, 0), top = corners(it, h);
    // Visible side faces (outward normal facing the viewer).
    for (let i = 0; i < 4; i++) {
      const j = (i + 1) % 4;
      const mid = [0, 1, 2].map((a) => (bot[i][a] + bot[j][a]) / 2 - it.pos[a]) as V3;
      mid[2] = 0;
      if (dot(mid, D) >= 0) continue;
      poly([bot[i], bot[j], top[j], top[i]]);
      g.fillStyle = shade(rgba, i % 2 ? 0.72 : 0.58, alpha);
      g.fill();
    }
    poly(top);
    g.fillStyle = shade(rgba, 1.0, alpha);
    g.fill();
    if (it.obj.studs && it.obj.pitch && it.obj.stud_r) {  // studded bricks: studs, far to near
      const [nx, ny] = it.obj.studs, p = it.obj.pitch, r = it.obj.stud_r * scale, ry = r * Math.sin(EL), sh = studH(it);
      const studs: V3[] = [];
      for (let i = 0; i < nx; i++) for (let j = 0; j < ny; j++) studs.push(world(it, (i - (nx - 1) / 2) * p, (j - (ny - 1) / 2) * p, h));
      studs.sort((a, b) => dot(b, D) - dot(a, D));
      for (const c of studs) {
        const [bx, by] = P(c), [tx, ty] = P([c[0], c[1], c[2] + sh]);
        g.fillStyle = shade(rgba, 0.7, alpha);
        g.beginPath(); g.ellipse(bx, by, r, ry, 0, 0, Math.PI * 2); g.fill();
        g.fillRect(bx - r, ty, 2 * r, by - ty);
        g.fillStyle = shade(rgba, 1.12, alpha);
        g.beginPath(); g.ellipse(tx, ty, r, ry, 0, 0, Math.PI * 2); g.fill();
      }
    }
    // Outline: dashed for to-do objects, glowing for the next one.
    const outline = () => {
      g.beginPath();
      for (const q of [bot, top]) {
        q.forEach((v, i) => (i ? g.lineTo(...P(v)) : g.moveTo(...P(v))));
        g.closePath();
      }
      for (let i = 0; i < 4; i++) { g.moveTo(...P(bot[i])); g.lineTo(...P(top[i])); }
    };
    if (st === "todo") {
      g.setLineDash([6, 6]);
      g.strokeStyle = shade(rgba, 1.2, 0.6);
      g.lineWidth = 2;
      outline();
      g.stroke();
      g.setLineDash([]);
    } else if (st === "next") {
      g.save();
      g.shadowColor = "#ffffff";
      g.shadowBlur = 10 + 10 * pulse;
      g.strokeStyle = `rgba(255,255,255,${0.6 + 0.4 * pulse})`;
      g.lineWidth = 3;
      poly(top);
      g.stroke();
      g.restore();
    }
  }
  return true;
}
