// Replay timeline in VR: watch a take at 1:1 scale from any viewpoint, scrub with the stick while pointing at the
// panel, step frame by frame, and take over from the shown frame ("branch": the new take starts at that state).

import type { AuditRows, EpisodeRow, Status } from "../protocol";
import { button, glass, Panel, roundRect, text, THEME, type Widget } from "./Panel";

const W = 0.6, H = 0.2;
const PX = 1500; // canvas 900 x 300
const BAR = { x: 30, y: 78, w: 840, h: 40 };

export class TimelinePanel {
  readonly panel: Panel;
  private status: Status | null = null;
  private episodes: EpisodeRow[] = [];
  private audit: AuditRows = {};
  private scrub = 0;

  constructor(private command: (name: string, args?: Record<string, unknown>) => void) {
    this.panel = new Panel(W, H, PX, (g, w, h) => this.draw(g, w, h));
    this.panel.mesh.visible = false;
    const r = () => this.status?.replay;
    const fps = () => r()?.fps ?? 50;
    const seek = (d: number) => () => this.command("replay_seek", { i: Math.max(0, (r()?.i ?? 0) + d) });
    const y = 150, bh = 84, bw = 84, gap = 10;
    let x = 30;
    const next = (w = bw) => { const at = x; x += w + gap; return at; };
    const widgets: Widget[] = [
      button("prev-ep", next(), y, bw, bh, () => "Prev", () => this.open(-1), { icon: () => "⏮" }),
      button("back1s", next(), y, bw, bh, () => "−1 s", () => seek(-fps())(), { icon: () => "⏪" }),
      button("step-", next(), y, bw, bh, () => "Step", () => this.command("replay_step", { delta: -1 }), { icon: () => "◀" }),
      button("play", next(100), y, 100, bh, () => (r()?.paused ? "Play" : "Pause"), () => this.command("replay_pause"),
        { icon: () => (r()?.paused ? "▶" : "⏸"), active: () => !r()?.paused }),
      button("step+", next(), y, bw, bh, () => "Step", () => this.command("replay_step", { delta: 1 }), { icon: () => "▶" }),
      button("fwd1s", next(), y, bw, bh, () => "+1 s", () => seek(fps())(), { icon: () => "⏩" }),
      button("next-ep", next(), y, bw, bh, () => "Next", () => this.open(1), { icon: () => "⏭" }),
      button("speed", next(), y, bw, bh, () => `×${r()?.speed ?? 1}`, () => {
        const speeds = [0.25, 0.5, 1, 2, 4], k = speeds.indexOf(r()?.speed ?? 1);
        this.command("replay_options", { speed: speeds[(k + 1) % speeds.length] });
      }, { icon: () => "⏱" }),
      button("loop", next(), y, bw, bh, () => "Loop", () => this.command("replay_options", { loop: !r()?.loop }),
        { icon: () => "🔁", active: () => !!r()?.loop }),
      // The cheapest recovery data: pause at a hard moment, take over, record from exactly there.
      button("branch", next(170), y, 170, bh, () => "Take over here", () => this.command("replay_branch"),
        { icon: () => "✋", color: () => THEME.rec, enabled: () => !r()?.foreign }),
      button("stop", x, y, 900 - 30 - x, bh, () => "Close", () => this.command("replay_stop"), { icon: () => "✕" }),
    ];
    this.panel.widgets = widgets;
  }

  /** Episodes of the recording folder (newest last) and their audits (phases for the bar). */
  async refresh() {
    try {
      const [eps, audit] = await Promise.all([
        fetch("/api/episodes", { cache: "no-store" }).then((x) => x.json() as Promise<EpisodeRow[]>),
        fetch("/api/audit", { cache: "no-store" }).then((x) => x.json() as Promise<AuditRows>),
      ]);
      this.episodes = eps;
      this.audit = audit;
      this.panel.invalidate();
    } catch { /* server restarting */ }
  }

  lastPath(): string | null { return this.episodes.at(-1)?.path ?? null; }

  private current(): number {
    const path = this.status?.replay?.path;
    return this.episodes.findIndex((e) => e.path === path);
  }

  /** Replay the previous / next episode of the folder. */
  private open(delta: number) {
    if (!this.episodes.length) return;
    const k = this.current();
    const j = Math.min(this.episodes.length - 1, Math.max(0, (k < 0 ? this.episodes.length - 1 : k) + delta));
    this.command("replay", { path: this.episodes[j].path, paused: true });
  }

  update(s: Status | null) {
    const was = !!this.status?.replay, now = !!s?.replay;
    this.status = s;
    if (now !== was) {
      this.panel.mesh.visible = now;
      this.panel.setHover(null);
      if (now) void this.refresh();
    }
    if (now) this.panel.invalidate();
  }

  /** While a hand points at the panel, its stick scrubs: left/right, faster the further it is pushed. */
  scrubBy(stickX: number, dt: number) {
    const r = this.status?.replay;
    if (!r || Math.abs(stickX) < 0.25) { this.scrub = 0; return; }
    this.scrub += stickX * Math.abs(stickX) * r.fps * 4 * dt;
    const steps = Math.trunc(this.scrub);
    if (steps) {
      this.scrub -= steps;
      this.command("replay_seek", { i: Math.min(r.n - 1, Math.max(0, r.i + steps)) });
    }
  }

  /** Seek to where the bar was clicked (local point on the panel), if it was on the bar. */
  clickBar(px: number, py: number): boolean {
    const r = this.status?.replay;
    if (!r || px < BAR.x || px > BAR.x + BAR.w || py < BAR.y - 12 || py > BAR.y + BAR.h + 12) return false;
    this.command("replay_seek", { i: Math.round(((px - BAR.x) / BAR.w) * (r.n - 1)) });
    return true;
  }

  private draw(g: CanvasRenderingContext2D, w: number, h: number) {
    glass(g, 6, 6, w - 12, h - 12, 36, THEME.accent);
    const r = this.status?.replay;
    if (!r) return;
    const file = r.path.split("/").at(-1) ?? "";
    const k = this.current();
    text(g, `REPLAY  ·  ${file}${k >= 0 ? `  (${k + 1}/${this.episodes.length})` : ""}${r.foreign ? "  ·  other scene" : ""}`,
      30, 42, 26, { weight: 800, color: THEME.accent });
    text(g, `${(r.i / r.fps).toFixed(2)} / ${(r.n / r.fps).toFixed(2)} s   step ${r.i}`, w - 30, 42, 26,
      { align: "right", weight: 700, color: THEME.text });
    const { x, y, w: bw, h: bh } = BAR;
    roundRect(g, x, y, bw, bh, 10);
    g.fillStyle = "rgba(0,0,0,0.45)";
    g.fill();
    const a = this.audit[file];
    const at = (step: number) => x + (bw * step) / Math.max(1, r.n - 1);
    if (a) {
      for (const [s0, s1] of a.spans) {         // what a dataset would use
        g.fillStyle = "rgba(52,211,153,0.22)";
        g.fillRect(at(s0), y, at(s1) - at(s0), bh);
      }
      for (const p of a.phases) {               // automatic phases: holds as bands, stages as ticks
        if (p.kind === "hold") {
          g.fillStyle = "rgba(124,156,255,0.5)";
          g.fillRect(at(p.start), y + bh - 10, Math.max(2, at(p.stop) - at(p.start)), 10);
        } else {
          g.fillStyle = THEME.ok;
          g.fillRect(at(p.start) - 2, y, 4, bh);
        }
      }
    }
    const ep = this.episodes.find((e) => e.path === r.path);
    for (const n of ep?.annotations ?? []) {    // the operator's own annotations
      g.fillStyle = "rgba(251,191,36,0.55)";
      g.fillRect(at(n.start), y, Math.max(2, at(n.stop) - at(n.start)), 8);
    }
    g.fillStyle = THEME.text;
    g.fillRect(at(r.i) - 3, y - 8, 6, bh + 16);
    g.strokeStyle = THEME.stroke;
    g.lineWidth = 2;
    roundRect(g, x, y, bw, bh, 10);
    g.stroke();
  }

  tick() {
    if (this.panel.mesh.visible) this.panel.redraw();
  }
}
