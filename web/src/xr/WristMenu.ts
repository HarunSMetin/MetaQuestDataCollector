// Wrist-attached control menu (left hand / left controller). Three pages: controls, layout (camera preview grid,
// which panels show and how they follow you), and tools (replay, scene editing, comfort, hand mapping).

import type { Status } from "../protocol";
import type { PreviewCam, PreviewLayout } from "./CamPreviewPanel";
import type { Follow } from "./Layout";
import { button, glass, Panel, text, THEME, type Widget, wrapText } from "./Panel";

const W = 0.24, H = 0.415;
const PX = 2400; // high density: viewed up close (canvas 576 x 996)

export type PanelId = "hud" | "goal" | "previews";
type Page = "main" | "layout" | "tools";

export interface MenuActions {
  command: (name: string, args?: Record<string, unknown>) => void;
  exit: () => void;
  toggleSound: () => void;
  soundOn: () => boolean;
  toggleHints: () => void;
  hintsOn: () => boolean;
  toggleCamViz: () => void;
  camVizOn: () => boolean;
  togglePanel: (id: PanelId) => void;
  panelOn: (id: PanelId) => boolean;
  previewLayout: () => PreviewLayout;
  setPreviewLayout: (l: PreviewLayout) => void;
  cams: () => PreviewCam[];
  toggleCam: (name: string) => void;
  camOn: (name: string) => boolean;
  toggleLock: () => void;
  locked: () => boolean;
  resetLayout: () => void;
  follow: (id: PanelId) => Follow;
  cycleFollow: (id: PanelId) => void;
  replayLast: () => void;
  ghostLast: () => void;
  toggleGuide: () => void;
  guideOn: () => boolean;
}

const FOLLOW_ICON: Record<Follow, string> = { world: "🌐", body: "🧍", wrist: "⌚" };

export class WristMenu {
  readonly panel: Panel;
  status: Status | null = null;
  visible = false;
  private page: Page = "main";
  private pages: Record<Page, Widget[]>;
  private camKey = "";

  constructor(private act: MenuActions) {
    this.panel = new Panel(W, H, PX, (g, w, h) => {
      glass(g, 4, 4, w - 8, h - 8, 48);
      const s = this.status;
      text(g, { main: "XRCOLLECT", layout: "LAYOUT", tools: "TOOLS" }[this.page], 40, 52, 30, { weight: 800, color: THEME.accent });
      const head = (label: string, y: number, right = "") => {
        text(g, label, 40, y, 22, { color: THEME.dim, weight: 700 });
        if (right) text(g, right, w - 40, y, 22, { align: "right", color: THEME.dim, weight: 700 });
      };
      if (this.page === "main") {
        head("INPUT MODE", 112);
        head("EPISODE", 272, s ? `${s.session.saved} saved${s.session.target ? ` of ${s.session.target}` : ""}` : "");
        head("VIEW", 558, s ? `scale ×${s.scale.toFixed(2)}` : "");
      } else if (this.page === "layout") {
        head("CAMERA PREVIEWS", 112);
        head("PANELS  ·  tap the lower row to change how one follows you", 448);
        wrapText(g, this.act.locked()
          ? "Panels are locked. Unlock them to move them."
          : "Move a panel: release the arm, point at the panel and hold Trigger (hands: pinch thumb + index). "
            + "While holding, the stick moves it nearer/farther (up/down) and resizes it (left/right).",
        40, 780, w - 80, 34, 24, THEME.dim, 6);
      } else {
        head("REPLAY", 112);
        head("SCENE", 272, s ? `task v${s.task_version.slice(0, 6)}` : "");
        head("HANDS  ·  CONTROL", 432, s ? Object.entries(s.mapping).map(([h, a]) => `${h.toUpperCase()}→${a}`).join("  ") : "");
        head("COMFORT", 688, s ? `table ${s.table_offset >= 0 ? "+" : ""}${Math.round(s.table_offset * 100)} cm` : "");
      }
    });
    this.pages = { main: this.buildMain(), layout: this.buildLayout(), tools: this.buildTools() };
    this.panel.widgets = this.pages.main;
    this.panel.mesh.visible = false;
  }

  private show(page: Page) {
    this.page = page;
    this.panel.widgets = this.pages[page];
    this.panel.setHover(null);
    this.panel.invalidate();
  }

  /** Page tabs, top right: the two pages you are not on. */
  private tabs(self: Page): Widget[] {
    const CW = this.panel.canvas.width;
    const labels: Record<Page, string> = { main: "← Main", layout: "⚙ Layout", tools: "🛠 Tools" };
    const others = (["main", "layout", "tools"] as Page[]).filter((p) => p !== self);
    return others.map((p, i) => button(`tab-${p}`, CW - 36 - (2 - i) * 142 + 6, 20, 136, 64, () => labels[p], () => this.show(p)));
  }

  private buildMain(): Widget[] {
    const s = () => this.status;
    const cmd = this.act.command;
    const CW = this.panel.canvas.width;
    const m = 36, gap = 14;
    const col2 = (CW - 2 * m - gap) / 2;
    const col3 = (CW - 2 * m - 2 * gap) / 3;
    const c3 = (i: number) => m + i * (col3 + gap);
    const rec = () => s()?.rec_state ?? "idle";
    const idle = () => rec() === "idle" && !s()?.replay;
    const canSave = () => rec() === "review" || rec() === "recording";

    return [
      ...this.tabs("main"),
      button("mode-ctrl", m, 134, col2, 110, () => "Controllers", () => cmd("set_mode", { mode: "controllers" }),
        { icon: () => "🎮", active: () => s()?.mode === "controllers" }),
      button("mode-hands", m + col2 + gap, 134, col2, 110, () => "Hands", () => cmd("set_mode", { mode: "hands" }),
        { icon: () => "✋", active: () => s()?.mode === "hands" }),

      // During a replay: continue it from the shown frame as a new take (the operator's correction).
      button("rec", m, 294, CW - 2 * m, 120,
        () => (rec() === "recording" ? "Stop" : rec() === "countdown" ? "Cancel" : s()?.replay ? "Take over here" : "Record"),
        () => cmd(s()?.replay && rec() === "idle" ? "replay_branch" : "record_toggle"),
        {
          icon: () => (rec() === "recording" ? "■" : s()?.replay && rec() === "idle" ? "✋" : "●"), color: () => THEME.rec,
          active: () => rec() === "recording" || rec() === "countdown",
          enabled: () => rec() !== "review" && !s()?.edit,
        }),
      button("save-ok", c3(0), 430, col3, 96, () => "Success", () => cmd("save", { success: true }),
        { icon: () => "✓", color: () => THEME.ok, enabled: canSave }),
      button("save-fail", c3(1), 430, col3, 96, () => "Failure", () => cmd("save", { success: false }),
        { icon: () => "✗", color: () => THEME.warn, enabled: canSave }),
      button("discard", c3(2), 430, col3, 96, () => "Discard", () => cmd("discard"),
        { icon: () => "🗑", color: () => THEME.rec, enabled: () => rec() !== "idle" }),

      button("recenter", c3(0), 580, col3, 96, () => "Recenter", () => cmd("recenter"), { icon: () => "⟲" }),
      button("scale-down", c3(1), 580, col3, 96, () => "Slower", () => cmd("set_scale", { scale: (s()?.scale ?? 1) - 0.25 }),
        { icon: () => "−" }),
      button("scale-up", c3(2), 580, col3, 96, () => "Faster", () => cmd("set_scale", { scale: (s()?.scale ?? 1) + 0.25 }),
        { icon: () => "+" }),

      button("reset", m, 690, col2, 80, () => "↺  New scene", () => cmd("reset"), { enabled: idle }),
      button("exit", m + col2 + gap, 690, col2, 80, () => "⏏  Exit XR", () => this.act.exit()),
      button("sound", m, 786, col2, 80, () => (this.act.soundOn() ? "🔊  Sound" : "🔇  Muted"),
        () => { this.act.toggleSound(); this.panel.invalidate(); }, { active: () => this.act.soundOn() }),
      button("hints", m + col2 + gap, 786, col2, 80, () => "🏷  Button hints",
        () => { this.act.toggleHints(); this.panel.invalidate(); }, { active: () => this.act.hintsOn() }),
      button("camviz", m, 882, col2, 80, () => "📷  Camera view",
        () => { this.act.toggleCamViz(); this.panel.invalidate(); }, { active: () => this.act.camVizOn() }),
      button("preview", m + col2 + gap, 882, col2, 80, () => "🖼  Previews",
        () => { this.act.togglePanel("previews"); this.panel.invalidate(); }, { active: () => this.act.panelOn("previews") }),
    ];
  }

  private buildLayout(): Widget[] {
    const a = this.act;
    const CW = this.panel.canvas.width;
    const m = 36, gap = 14;
    const col2 = (CW - 2 * m - gap) / 2;
    const col3 = (CW - 2 * m - 2 * gap) / 3;
    const c2 = (i: number) => m + i * (col2 + gap);
    const c3 = (i: number) => m + i * (col3 + gap);
    const redraw = (f: () => void) => () => { f(); this.panel.invalidate(); };
    const previewsOn = () => a.panelOn("previews");

    const layouts: [PreviewLayout, string, string][] = [["column", "Vertical", "↕"], ["row", "Horizontal", "↔"], ["grid", "Grid", "⊞"]];
    const panels: [PanelId, string][] = [["hud", "Status"], ["goal", "Goal"], ["previews", "Previews"]];
    return [
      ...this.tabs("layout"),
      ...layouts.map(([l, label, icon], i) => button(`layout-${l}`, c3(i), 134, col3, 96, () => label,
        redraw(() => { a.setPreviewLayout(l); if (!previewsOn()) a.togglePanel("previews"); }),
        { icon: () => icon, active: () => a.previewLayout() === l })),
      // The scene's cameras (up to six), whatever the task calls them.
      ...a.cams().slice(0, 6).map((c, i) => button(`cam-${c.name}`, c3(i % 3), 246 + Math.floor(i / 3) * 88, col3, 76,
        () => c.short, redraw(() => a.toggleCam(c.name)), { color: () => c.color, active: () => a.camOn(c.name) })),

      ...panels.map(([id, label], i) => button(`panel-${id}`, c3(i), 470, col3, 80, () => label,
        redraw(() => a.togglePanel(id)), { active: () => a.panelOn(id) })),
      ...panels.map(([id], i) => button(`follow-${id}`, c3(i), 562, col3, 72,
        () => `${FOLLOW_ICON[a.follow(id)]} ${a.follow(id)}`, redraw(() => a.cycleFollow(id)))),
      button("lock", c2(0), 656, col2, 80, () => (a.locked() ? "🔒  Locked" : "🔓  Movable"),
        redraw(() => a.toggleLock()), { active: () => a.locked(), color: () => THEME.warn }),
      button("reset-layout", c2(1), 656, col2, 80, () => "↺  Reset layout", redraw(() => a.resetLayout())),
    ];
  }

  private buildTools(): Widget[] {
    const a = this.act, s = () => this.status, cmd = a.command;
    const CW = this.panel.canvas.width;
    const m = 36, gap = 14;
    const col2 = (CW - 2 * m - gap) / 2;
    const col3 = (CW - 2 * m - 2 * gap) / 3;
    const c2 = (i: number) => m + i * (col2 + gap);
    const c3 = (i: number) => m + i * (col3 + gap);
    const idle = () => (s()?.rec_state ?? "idle") === "idle";
    const redraw = (f: () => void) => () => { f(); this.panel.invalidate(); };
    const controls: Status["teleop_control"][] = ["clutched", "absolute", "rate"];
    return [
      ...this.tabs("tools"),
      // Replay: your own last take, at 1:1 scale, from any viewpoint (the timeline panel appears).
      button("replay-last", c2(0), 134, col2, 110, () => (s()?.replay ? "Stop replay" : "Replay last take"),
        () => (s()?.replay ? cmd("replay_stop") : a.replayLast()), { icon: () => (s()?.replay ? "■" : "▶"), enabled: idle,
          active: () => !!s()?.replay }),
      button("ghost-last", c2(1), 134, col2, 110, () => (s()?.ghost ? "Ghost off" : "Ghost of last take"),
        () => (s()?.ghost ? cmd("ghost", {}) : a.ghostLast()), { icon: () => "👻", active: () => !!s()?.ghost }),

      button("edit", c2(0), 294, col2, 110, () => (s()?.edit ? "Leave edit mode" : "Edit scene"),
        () => cmd("edit_mode", { enabled: !s()?.edit }), { icon: () => "✎", active: () => !!s()?.edit, enabled: idle }),
      button("guide", c2(1), 294, col2, 110, () => "Goal guide", redraw(() => a.toggleGuide()),
        { icon: () => "◎", active: () => a.guideOn() }),

      button("swap", m, 454, CW - 2 * m, 80, () => "⇄  Swap hands (left-handed / other arm)", () => cmd("swap_hands"), { enabled: idle }),
      ...controls.map((c, i) => button(`control-${c}`, c3(i), 548, col3, 96, () => c[0].toUpperCase() + c.slice(1),
        () => cmd("set_control", { control: c }),
        { icon: () => ({ clutched: "✊", absolute: "⌖", rate: "➠" })[c], active: () => s()?.teleop_control === c })),

      // Seated use: lower the whole scene instead of raising your arms.
      button("table-down", c3(0), 710, col3, 96, () => "Lower", () => cmd("set_table_offset", { offset: (s()?.table_offset ?? 0) - 0.05 }),
        { icon: () => "▼" }),
      button("table-zero", c3(1), 710, col3, 96, () => "Standing", () => cmd("set_table_offset", { offset: 0 }),
        { icon: () => "◼", active: () => Math.abs(s()?.table_offset ?? 0) < 0.01 }),
      button("table-up", c3(2), 710, col3, 96, () => "Raise", () => cmd("set_table_offset", { offset: (s()?.table_offset ?? 0) + 0.05 }),
        { icon: () => "▲" }),
      button("seated", m, 820, CW - 2 * m, 80, () => "🪑  Seated preset (−35 cm)", () => cmd("set_table_offset", { offset: -0.35 }),
        { active: () => Math.abs((s()?.table_offset ?? 0) + 0.35) < 0.01 }),
    ];
  }

  update(status: Status | null) {
    this.status = status;
    // The camera buttons follow the scene: rebuild the layout page when the scene's cameras change.
    const key = this.act.cams().map((c) => c.name).join(",");
    if (key !== this.camKey) {
      this.camKey = key;
      this.pages.layout = this.buildLayout();
      if (this.page === "layout") this.panel.widgets = this.pages.layout;
    }
    this.panel.invalidate();
  }

  setVisible(v: boolean) {
    this.visible = v;
    this.panel.mesh.visible = v;
    if (!v) this.panel.setHover(null);
  }
}
