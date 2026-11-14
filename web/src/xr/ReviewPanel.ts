// Review card, shown at the end of every take: the verdict a dataset would give it (the same rules the audit and
// the exporters use) with its reasons, the success label, quality stars and tag chips from the task's tag set, then
// save or discard with one button. Labels go to the dataset's sidecar file, never into the episode.

import type { Status } from "../protocol";
import { button, glass, Panel, text, THEME, type Widget, wrapText } from "./Panel";

const W = 0.46, H = 0.36;
const PX = 1600; // canvas 736 x 576

export class ReviewPanel {
  readonly panel: Panel;
  private status: Status | null = null;
  private success: boolean | null = null;   // null = as the task's check said
  private quality = 0;
  private tags = new Set<string>();
  private tagKey = "";
  private takeKey = "";

  constructor(private command: (name: string, args?: Record<string, unknown>) => void) {
    this.panel = new Panel(W, H, PX, (g, w, h) => this.draw(g, w, h));
    this.panel.mesh.visible = false;
    this.rebuild([]);
  }

  private get ok() { return this.success ?? this.status?.success_seen ?? false; }

  private rebuild(tags: string[]) {
    const CW = this.panel.canvas.width, CH = this.panel.canvas.height, m = 28, gap = 12;
    const redraw = (f: () => void) => () => { f(); this.panel.invalidate(); };
    const half = (CW - 2 * m - gap) / 2;
    const widgets: Widget[] = [
      // Success and failure are separate, explicit choices: a failed take can be valuable data.
      button("ok", m, 232, half, 70, () => "✓  Success", redraw(() => (this.success = true)),
        { color: () => THEME.ok, active: () => this.ok }),
      button("fail", m + half + gap, 232, half, 70, () => "✗  Failure", redraw(() => (this.success = false)),
        { color: () => THEME.warn, active: () => !this.ok }),
      ...[1, 2, 3, 4, 5].map((n) => button(`star-${n}`, m + 150 + (n - 1) * 74, 318, 64, 60, () => (n <= this.quality ? "★" : "☆"),
        redraw(() => (this.quality = this.quality === n ? 0 : n)), { color: () => THEME.warn, active: () => n <= this.quality })),
      button("save", m, CH - m - 84, half, 84, () => "Save", () => this.command("save", {
        success: this.ok, tags: [...this.tags], quality: this.quality }), { icon: () => "💾", color: () => THEME.ok }),
      button("discard", m + half + gap, CH - m - 84, half, 84, () => "Discard", () => this.command("discard"),
        { icon: () => "🗑", color: () => THEME.rec }),
    ];
    const shown = tags.slice(0, 4);
    const tw = (CW - 2 * m - (shown.length - 1) * gap) / Math.max(shown.length, 1);
    shown.forEach((t, i) => widgets.push(button(`tag-${t}`, m + i * (tw + gap), 392, tw, 60, () => t,
      redraw(() => (this.tags.has(t) ? this.tags.delete(t) : this.tags.add(t))), { active: () => this.tags.has(t) })));
    this.panel.widgets = widgets;
  }

  update(s: Status | null) {
    this.status = s;
    const r = s?.rec_state === "review" ? s.review : null;
    const show = !!r;
    if (show !== this.panel.mesh.visible) {
      this.panel.mesh.visible = show;
      this.panel.setHover(null);
    }
    if (!r) { this.takeKey = ""; return; }
    const take = `${r.duration_s}|${s!.steps}`;
    if (take !== this.takeKey) {  // a new take: start from clean labels
      this.takeKey = take;
      this.success = null;
      this.quality = 0;
      this.tags.clear();
    }
    const key = r.tags.join("|");
    if (key !== this.tagKey) { this.tagKey = key; this.rebuild(r.tags); }
    this.panel.invalidate();
  }

  private draw(g: CanvasRenderingContext2D, w: number, _h: number) {
    const r = this.status?.review, m = 28;
    glass(g, 6, 6, w - 12, _h - 12, 40, r?.rejected ? THEME.warn : THEME.ok);
    if (!r) return;
    text(g, "REVIEW", m, 44, 32, { weight: 800, color: THEME.accent });
    text(g, `${r.duration_s.toFixed(1)} s  ·  stage ${r.progress}/${r.stages}`, w - m, 44, 26, { align: "right", color: THEME.dim, weight: 700 });
    // The verdict with its reason: discard here what a dataset would not use anyway.
    if (r.rejected) wrapText(g, `✗ A dataset would skip it: ${r.rejected}`, m, 92, w - 2 * m, 34, 26, THEME.warn, 2);
    else text(g, `✓ A dataset would use ${r.used_s.toFixed(1)} s of this take`, m, 100, 28, { color: THEME.ok, weight: 800 });
    const ev = r.events;
    const facts = [`idle ${r.idle_pct}%`, ...Object.entries(ev).map(([k, n]) => `${k.replace(/_/g, " ")} ${n}×`),
      `safety ${r.safety_events}×`].join("  ·  ");
    wrapText(g, facts, m, 168, w - 2 * m, 30, 22, THEME.dim, 2);
    text(g, "QUALITY", m, 348, 22, { color: THEME.dim, weight: 700 });
  }

  tick() {
    if (this.panel.mesh.visible) this.panel.redraw();
  }
}
