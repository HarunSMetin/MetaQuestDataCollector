// Scene editor in VR (wrist menu → Tools → Edit scene). Physics is paused; every change is an edit operation on
// the task file (xrcollect/editor.py), after which the server hot-reloads the scene.
//
//   move      point at an object, hold Trigger, carry it over the table; the stick turns it. "Apply" writes the
//             poses (an object's start region moves along with it).
//   region    two Trigger clicks on the table = two corners of a taped region.
//   camera    one Trigger click: the camera goes where the controller is, looking at the point you aim at.
//   capture   "Capture goal": the objects' current arrangement becomes the task's goal (relative poses as stages).

import * as THREE from "three";
import type { MujocoScene } from "../MujocoScene";
import type { Status, TaskSpec } from "../protocol";
import type { Side } from "./input";
import { button, glass, Panel, text, THEME, type Widget, wrapText } from "./Panel";

const W = 0.5, H = 0.3;
const PX = 1500; // canvas 750 x 450
const SEND_MS = 33;
type Tool = "move" | "region" | "camera";

interface Grab { side: Side; object: string; offset: THREE.Vector3; yaw: number; z: number; lift: number; last: number }

export class EditMode {
  readonly panel: Panel;
  private status: Status | null = null;
  private task: TaskSpec | null = null;
  private tool: Tool = "move";
  private grab: Grab | null = null;
  private selected: string | null = null;
  private corner: THREE.Vector3 | null = null;
  private assets: string[] = [];
  private assetIdx = 0;
  private regionIdx = 0;
  private camIdx = 0;
  private marker: THREE.Mesh;
  private tmp = new THREE.Vector3();

  constructor(private mj: MujocoScene, private command: (name: string, args?: Record<string, unknown>) => void,
    private pulse: (side: Side, intensity: number, ms: number) => void) {
    this.panel = new Panel(W, H, PX, (g, w, h) => this.draw(g, w, h));
    this.panel.mesh.visible = false;
    this.marker = new THREE.Mesh(new THREE.RingGeometry(0.008, 0.012, 24),
      new THREE.MeshBasicMaterial({ color: 0xfbbf24, transparent: true, depthTest: false }));
    this.marker.visible = false;
    this.marker.renderOrder = 12;
    mj.root.add(this.marker);
    this.build();
    fetch("/api/library").then((r) => r.json()).then((lib: { objects: string[] }) => {
      this.assets = lib.objects;
      this.panel.invalidate();
    }).catch(() => {});
  }

  get active() { return !!this.status?.edit; }

  private regionNames(): string[] { return [...Object.keys(this.task?.regions ?? {}), "new_region"]; }
  private cameraNames(): string[] {
    const world = (this.mj.desc?.cameras ?? []).filter((c) => c.body === 0).map((c) => c.name);
    return [...world, "cam_new"];
  }
  private movable(): string[] { return (this.task?.objects ?? []).filter((o) => !o.fixed).map((o) => o.name); }

  private edit(ops: Record<string, unknown>[]) { this.command("edit", { ops }); }

  private build() {
    const CW = this.panel.canvas.width, m = 24, gap = 10;
    const col3 = (CW - 2 * m - 2 * gap) / 3;
    const c3 = (i: number) => m + i * (col3 + gap);
    const redraw = (f: () => void) => () => { f(); this.panel.invalidate(); };
    const cycle = (n: number, i: number) => (n ? (i + 1) % n : 0);
    const short = (s: string) => (s.length > 16 ? `…${s.slice(-15)}` : s);
    const widgets: Widget[] = [
      button("apply", c3(0), 70, col3, 74, () => `Apply moves (${this.status?.edit?.moved.length ?? 0})`,
        () => this.command("edit_commit"), { color: () => THEME.ok, enabled: () => !!this.status?.edit?.moved.length }),
      button("capture", c3(1), 70, col3, 74, () => "◎ Capture goal", () => this.command("capture_goal"),
        { color: () => THEME.warn, enabled: () => this.movable().length > 1 }),
      button("done", c3(2), 70, col3, 74, () => "Done", () => this.command("edit_mode", { enabled: false })),

      button("region-name", c3(0), 162, col3, 66, () => short(this.regionNames()[this.regionIdx] ?? ""),
        redraw(() => (this.regionIdx = cycle(this.regionNames().length, this.regionIdx)))),
      button("region-draw", c3(1), 162, col3, 66, () => (this.tool === "region" ? (this.corner ? "2nd corner…" : "1st corner…") : "▭ Draw region"),
        redraw(() => { this.tool = this.tool === "region" ? "move" : "region"; this.corner = null; }), { active: () => this.tool === "region" }),
      button("region-del", c3(2), 162, col3, 66, () => "Remove region", () => {
        const name = this.regionNames()[this.regionIdx];
        if (name && name !== "new_region") this.edit([{ op: "region_remove", name }]);
      }, { enabled: () => this.regionNames()[this.regionIdx] !== "new_region" }),

      button("asset", c3(0), 240, col3, 66, () => short(this.assets[this.assetIdx] ?? "…"),
        redraw(() => (this.assetIdx = cycle(this.assets.length, this.assetIdx)))),
      button("add", c3(1), 240, col3, 66, () => "＋ Add object", () => {
        const asset = this.assets[this.assetIdx];
        if (!asset) return;
        const base = asset.split("/").at(-1)!.replace(/[^a-z0-9_]/gi, "_");
        const taken = new Set((this.task?.objects ?? []).map((o) => o.name));
        let k = 1;
        while (taken.has(`${base}_${k}`)) k++;
        this.edit([{ op: "object_add", name: `${base}_${k}`, asset, pos: [0.45, 0.0] }]);
      }, { enabled: () => this.assets.length > 0 }),
      button("remove", c3(2), 240, col3, 66, () => (this.selected ? `Remove ${short(this.selected)}` : "Remove object"), () => {
        if (this.selected) this.edit([{ op: "object_remove", object: this.selected }]);
        this.selected = null;
      }, { enabled: () => !!this.selected, color: () => THEME.rec }),

      button("cam-name", c3(0), 318, col3, 66, () => short(this.cameraNames()[this.camIdx] ?? ""),
        redraw(() => (this.camIdx = cycle(this.cameraNames().length, this.camIdx)))),
      button("cam-place", c3(1), 318, col3, 66, () => (this.tool === "camera" ? "Aim + Trigger…" : "📷 Place camera"),
        redraw(() => (this.tool = this.tool === "camera" ? "move" : "camera")), { active: () => this.tool === "camera" }),
      button("spawn", c3(2), 318, col3, 66, () => "Start area ±", () => this.resizeSpawn(), { enabled: () => !!this.selected }),
    ];
    this.panel.widgets = widgets;
  }

  /** Cycle the selected object's start region: none -> 2 cm -> 4 cm -> 6 cm half-size (yaw +-15 degrees). */
  private resizeSpawn() {
    const name = this.selected, t = this.task;
    if (!name || !t) return;
    const o = this.mj.bodyObject(name);
    const cur = t.spawn[name];
    const sizes = [0.02, 0.04, 0.06];
    const k = cur ? sizes.findIndex((v) => Math.abs(v - cur.half[0]) < 1e-3) : -1;
    if (k === sizes.length - 1) { this.edit([{ op: "spawn_remove", object: name }]); return; }
    const center = cur ? cur.center : o ? [o.position.x, o.position.y] : [0.45, 0];
    const h = sizes[k + 1];
    this.edit([{ op: "spawn_set", object: name, center, half: [h, h], yaw_deg: [-15, 15] }]);
  }

  update(s: Status | null, task: TaskSpec | null) {
    const was = this.active;
    this.status = s;
    this.task = task;
    if (this.active !== was) {
      this.panel.mesh.visible = this.active;
      this.panel.setHover(null);
      this.tool = "move";
      this.grab = this.corner = null;
      this.marker.visible = false;
    }
    if (this.active) this.panel.invalidate();
    this.showOverlays();
  }

  /** In edit mode the regions and the objects' start regions are outlined on the table. */
  showOverlays() {
    if (!this.active || !this.task) { this.mj.setOverlays(null); return; }
    const colors = Object.fromEntries(this.task.objects.map((o) => [o.name, o.rgba]));
    this.mj.setOverlays(this.task.regions, this.task.spawn, colors);
  }

  /** One hand's ray this frame (renderer world). Returns the point the ray ends at if the editor used it. */
  handleRay(side: Side, origin: THREE.Vector3, dir: THREE.Vector3, down: boolean, wasDown: boolean, stickX: number,
    dt: number, raycaster: THREE.Raycaster): THREE.Vector3 | null {
    if (!this.active) return null;
    const mj = this.mj;
    const table = mj.rayOnTable(origin, dir, this.tmp);
    const toWorld = (p: THREE.Vector3) => p.clone().applyMatrix4(mj.root.matrixWorld);

    if (this.grab && this.grab.side === side) {
      const g = this.grab;
      if (!down) { this.grab = null; return null; }
      if (!table) return null;
      if (Math.abs(stickX) > 0.2) g.yaw -= stickX * 2.0 * dt;
      const now = performance.now();
      if (now - g.last > SEND_MS) {
        g.last = now;
        this.command("edit_move", { object: g.object, pos: [table.x + g.offset.x, table.y + g.offset.y, g.z], yaw: g.yaw,
          free_z: g.lift > 0.002 });
      }
      return toWorld(table);
    }
    if (this.grab) return null;  // the other hand holds an object

    if (this.tool === "region") {
      if (!table) return null;
      this.marker.visible = true;
      this.marker.position.set(table.x, table.y, mj.tableZ + 0.002);
      if (down && !wasDown) {
        this.pulse(side, 0.5, 30);
        if (!this.corner) this.corner = table.clone();
        else {
          const a = this.corner, b = table;
          const names = this.regionNames();
          let name = names[this.regionIdx];
          if (name === "new_region") { let k = 1; while (names.includes(`region_${k}`)) k++; name = `region_${k}`; }
          this.edit([{ op: "region_set", name, center: [(a.x + b.x) / 2, (a.y + b.y) / 2],
            half: [Math.max(Math.abs(a.x - b.x) / 2, 0.01), Math.max(Math.abs(a.y - b.y) / 2, 0.01)] }]);
          this.corner = null;
          this.tool = "move";
          this.marker.visible = false;
          this.panel.invalidate();
        }
      }
      return toWorld(table);
    }

    if (this.tool === "camera") {
      if (down && !wasDown) {
        const inv = new THREE.Matrix4().copy(mj.root.matrixWorld).invert();
        const pos = origin.clone().applyMatrix4(inv);
        const look = table ? table.clone().setZ(mj.tableZ) : origin.clone().addScaledVector(dir, 0.6).applyMatrix4(inv);
        const names = this.cameraNames();
        let name = names[this.camIdx];
        if (name === "cam_new") { let k = 1; while (names.includes(`cam_${k}`)) k++; name = `cam_${k}`; }
        this.edit([{ op: "camera_set", name, pos: pos.toArray(), lookat: look.toArray(), fovy: 60 }]);
        this.pulse(side, 0.6, 40);
        this.tool = "move";
        this.panel.invalidate();
      }
      return table ? toWorld(table) : null;
    }

    // move: pick an object and carry it over the table
    raycaster.set(origin, dir);
    const hit = mj.pickObject(raycaster, this.movable());
    if (hit && down && !wasDown && table) {
      const o = mj.bodyObject(hit.name)!;
      const e = new THREE.Euler().setFromQuaternion(o.quaternion, "ZYX");
      const lift = o.position.z - mj.tableZ;  // an object placed on another keeps its height
      this.grab = { side, object: hit.name, offset: new THREE.Vector3(o.position.x - table.x, o.position.y - table.y, 0),
        yaw: e.z, z: o.position.z, lift, last: 0 };
      this.selected = hit.name;
      this.pulse(side, 0.5, 30);
      this.panel.invalidate();
    }
    mj.highlightBody(hit?.name ?? this.selected);
    return hit ? hit.point : null;
  }

  private draw(g: CanvasRenderingContext2D, w: number, h: number) {
    glass(g, 6, 6, w - 12, h - 12, 36, THEME.accent);
    text(g, "EDIT SCENE", 24, 40, 28, { weight: 800, color: THEME.accent });
    text(g, this.selected ? `selected: ${this.selected}` : "point at an object + Trigger to move it", w - 24, 40, 22,
      { align: "right", color: THEME.dim, weight: 600 });
    wrapText(g, "Stick turns a held object. Capture goal: arrange the objects as the finished task, then press it.",
      24, 410, w - 48, 26, 20, THEME.dim, 2);
  }

  tick() {
    if (this.panel.mesh.visible) this.panel.redraw();
  }
}
