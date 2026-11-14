// World-anchored panel that shows the task's goal: a picture of the goal view (if the task has one) and the ordered
// stage list with progress. Everything comes from /api/task and the status; no task is known here.

import { drawGoal, stageStates } from "../goal";
import type { Status, TaskSpec } from "../protocol";
import { glass, Panel, text, THEME, wrapText } from "./Panel";

const W = 0.32, H = 0.42;
const ROW = 44;

export class GoalPanel {
  readonly panel: Panel;
  private task: TaskSpec | null = null;
  private status: Status | null = null;

  constructor() {
    this.panel = new Panel(W, H, 1400, (g, w, h) => this.draw(g, w, h));
  }

  setTask(task: TaskSpec) {
    this.task = task;
    this.panel.invalidate();
  }

  update(status: Status | null) {
    const changed = status?.progress !== this.status?.progress || status?.next?.object !== this.status?.next?.object;
    this.status = status;
    if (changed) this.panel.invalidate();
  }

  private draw(g: CanvasRenderingContext2D, w: number, h: number) {
    const pad = 32, t = this.task, s = this.status;
    glass(g, 6, 6, w - 12, h - 12, 40, THEME.accent);
    text(g, "GOAL", pad, pad + 20, 34, { weight: 800, color: THEME.accent });
    if (!t) {
      text(g, "Loading…", w / 2, h / 2, 30, { align: "center", color: THEME.dim });
      return;
    }
    const states = stageStates(t, s);
    const done = states.filter((x) => x === "done").length, n = t.stages.length;
    text(g, `${done} / ${n}`, w - pad, pad + 20, 34, { weight: 800, align: "right", color: n && done === n ? THEME.ok : THEME.text });

    const rows = Math.min(n, 6);
    const listH = ROW * rows + 16;
    const box = { x: pad, y: pad + 50, w: w - 2 * pad, h: h - 2 * pad - 50 - listH };
    if (!drawGoal(g, t, s, box, 1)) wrapText(g, t.instruction, pad, box.y + 30, box.w, 40, 30, THEME.text, 8);

    // One row per stage, in order; the rows scroll so the next stage stays in view.
    const first = Math.max(0, Math.min(n - rows, done - Math.floor(rows / 2)));
    const byName = new Map(t.objects.map((o) => [o.name, o]));
    let y = h - pad - listH + 26;
    for (let k = first; k < first + rows; k++) {
      const st = states[k], stage = t.stages[k];
      const obj = stage.object ? byName.get(stage.object) : undefined;
      if (obj) {
        const [r, gg, bb] = obj.rgba.map((v) => Math.round(v * 255));
        g.fillStyle = `rgb(${r},${gg},${bb})`;
        g.globalAlpha = st === "todo" ? 0.4 : 1;
        g.fillRect(pad, y - 13, 26, 26);
        g.globalAlpha = 1;
      }
      // Colour and symbol together, so the state reads without colour vision.
      const color = st === "done" ? THEME.ok : st === "next" ? THEME.text : THEME.dim;
      g.font = `600 27px ${THEME.font}`;
      let label = `${k + 1}  ${stage.label}`;
      while (label.length > 8 && g.measureText(label).width > w - 2 * pad - 90) label = `${label.slice(0, -2)}…`;
      text(g, label, pad + 40, y, 27, { weight: st === "next" ? 800 : 600, color });
      text(g, st === "done" ? "✓" : st === "next" ? "◀" : "", w - pad, y, 26, { align: "right", weight: 800, color });
      y += ROW;
    }
  }

  /** Redraws only after a change (each redraw re-uploads the texture). */
  tick() {
    this.panel.redraw();
  }
}
