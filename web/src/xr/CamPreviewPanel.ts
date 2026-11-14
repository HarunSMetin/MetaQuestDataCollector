// Live previews of the recorded cameras, as a vertical column, a horizontal row or a grid (wrist menu → Layout;
// the chosen layout and cameras persist). Polls the server's JPEG previews (GET /api/preview/<cam>.jpg, ~8 Hz)
// only while visible. The panel resizes to fit its cells.

import { armColorHex } from "../MujocoScene";
import { glass, Panel, text, THEME } from "./Panel";

export interface PreviewCam { name: string; label: string; short: string; color: string }
export type PreviewLayout = "column" | "row" | "grid";

const PX = 1400;
const PAD = 18, GAP = 12, LABEL_H = 34;
const CELL_W = 342, CELL_H = (CELL_W * 3) / 4;  // px; the previews are 4:3
const POLL_MS = 125;
const LAYOUT_KEY = "xrc.preview.layout", CAMS_KEY = "xrc.preview.cams";
const WORLD_COLORS = [THEME.accent, THEME.warn, THEME.ok, "#c084fc"];

export class CamPreviewPanel {
  readonly panel: Panel;
  private images = new Map<string, ImageBitmap>();
  private visible = false;
  private inflight = false;
  private lastPoll = 0;
  private layout_: PreviewLayout = "column";
  private cams = new Set<string>();
  private wanted: string[] | null = null;  // the operator's stored choice (kept across scenes)
  /** The cameras of the current scene (set from /api/scene): the wrist menu lists these. */
  available: PreviewCam[] = [];

  constructor() {
    try {
      const l = localStorage.getItem(LAYOUT_KEY);
      if (l === "column" || l === "row" || l === "grid") this.layout_ = l;
      const c = JSON.parse(localStorage.getItem(CAMS_KEY) ?? "null");
      if (Array.isArray(c) && c.length) this.wanted = c.map(String);
    } catch { /* storage unavailable or corrupt: defaults */ }
    this.panel = new Panel(0.1, 0.1, PX, (g, w, h) => this.draw(g, w, h));
    this.fit();
    this.panel.mesh.visible = false;
  }

  setVisible(on: boolean) {
    this.visible = on;
    this.panel.mesh.visible = on;
    this.panel.invalidate();
  }

  /** The scene's cameras: wrist cameras take their arm's colour. Chosen cameras that still exist stay chosen. */
  setCameras(names: string[], wrist: Record<string, string>, arms: string[]) {
    let k = 0;
    this.available = names.map((name) => {
      const arm = Object.entries(wrist).find(([, cam]) => cam === name)?.[0];
      const pretty = name.replace(/^cam_/, "").replace(/_/g, " ");
      return { name, label: pretty.toUpperCase(), short: pretty,
        color: arm ? armColorHex(arms.indexOf(arm)) : WORLD_COLORS[k++ % WORLD_COLORS.length] };
    });
    const keep = (this.wanted ?? []).filter((n) => names.includes(n));
    this.cams = new Set(keep.length ? keep : names.slice(0, 4));
    for (const n of [...this.images.keys()]) if (!names.includes(n)) { this.images.get(n)?.close(); this.images.delete(n); }
    this.fit();
  }

  get shown() { return this.visible; }
  get layout() { return this.layout_; }

  setLayout(l: PreviewLayout) {
    this.layout_ = l;
    this.fit();
    this.persist();
  }

  camOn(name: string) { return this.cams.has(name); }

  /** Show/hide one camera; the last one stays on. */
  toggleCam(name: string) {
    if (this.cams.has(name)) { if (this.cams.size > 1) this.cams.delete(name); }
    else this.cams.add(name);
    this.fit();
    this.persist();
  }

  private persist() {
    try {
      localStorage.setItem(LAYOUT_KEY, this.layout_);
      this.wanted = [...this.cams];
      localStorage.setItem(CAMS_KEY, JSON.stringify(this.wanted));
    } catch { /* storage unavailable */ }
  }

  private shownCams() { return this.available.filter((c) => this.cams.has(c.name)); }

  private grid(): [number, number] {
    const n = Math.max(1, this.shownCams().length);
    if (this.layout_ === "column") return [1, n];
    if (this.layout_ === "row") return [n, 1];
    const cols = Math.ceil(Math.sqrt(n));
    return [cols, Math.ceil(n / cols)];
  }

  private fit() {
    const [cols, rows] = this.grid();
    const w = 2 * PAD + cols * CELL_W + (cols - 1) * GAP;
    const h = 2 * PAD + rows * (LABEL_H + CELL_H) + (rows - 1) * GAP;
    this.panel.resize(w / PX, h / PX);
    this.panel.invalidate();
  }

  private draw(g: CanvasRenderingContext2D, w: number, h: number) {
    glass(g, 4, 4, w - 8, h - 8, 30);
    const [cols] = this.grid();
    this.shownCams().forEach((c, i) => {
      const col = i % cols, row = Math.floor(i / cols);
      const x = PAD + col * (CELL_W + GAP), top = PAD + row * (LABEL_H + CELL_H + GAP), y = top + LABEL_H;
      text(g, c.label, x + 4, top + LABEL_H / 2, 24, { weight: 800, color: c.color });
      const img = this.images.get(c.name);
      if (img) g.drawImage(img, x, y, CELL_W, CELL_H);
      else {
        g.fillStyle = "rgba(0,0,0,0.4)";
        g.fillRect(x, y, CELL_W, CELL_H);
        text(g, "…", x + CELL_W / 2, y + CELL_H / 2, 40, { align: "center", color: THEME.dim });
      }
      g.strokeStyle = c.color;
      g.lineWidth = 3;
      g.strokeRect(x, y, CELL_W, CELL_H);
    });
  }

  private async poll() {
    this.inflight = true;
    try {
      const t = Date.now();
      await Promise.all(this.shownCams().map(async (c) => {
        const r = await fetch(`/api/preview/${c.name}.jpg?t=${t}`, { cache: "no-store" });
        if (!r.ok) return;  // 404 until the server has rendered a first frame (or the camera isn't in record.cameras)
        const img = await createImageBitmap(await r.blob());
        this.images.get(c.name)?.close();
        this.images.set(c.name, img);
      }));
      this.panel.invalidate();
    } catch { /* server restarting: try again next tick */ }
    this.inflight = false;
  }

  tick() {
    const now = performance.now();
    if (this.visible && !this.inflight && now - this.lastPoll > POLL_MS) {
      this.lastPoll = now;
      void this.poll();
    }
    if (this.visible) this.panel.redraw();
  }
}
