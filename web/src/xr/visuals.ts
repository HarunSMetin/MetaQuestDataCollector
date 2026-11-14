// Holographic hand skeletons, Quest controller models with live button hints, and pointer rays.

import * as THREE from "three";
import { XRControllerModelFactory } from "three/examples/jsm/webxr/XRControllerModelFactory.js";
import {
  B_MENU, B_PRIMARY, B_SECONDARY, B_SQUEEZE, B_STICK_PRESS, B_TRIGGER, type ButtonHint, type HandSample, type RecState,
  SRC_CONTROLLER, SRC_HAND, type Status,
} from "../protocol";
import { Panel, roundRect, text, THEME } from "./Panel";

// Bone pairs (joint indices) for the skeleton lines.
const BONES: [number, number][] = [];
for (const base of [1, 5, 10, 15, 20]) {
  const len = base === 1 ? 4 : 5;
  BONES.push([0, base]);
  for (let k = 0; k < len - 1; k++) BONES.push([base + k, base + k + 1]);
}

// Real Touch Plus models (WebXR input profiles), vendored in web/public/xr-profiles. Their buttons, trigger, grip and
// stick animate from the live gamepad state. One factory so both hands share the asset cache.
const controllerModels = new XRControllerModelFactory().setPath("/xr-profiles");

type Hand = "l" | "r";

interface HintCtx { s: Status | null; engaged: boolean; hasArm: boolean; buttons: Float32Array }

/** A callout next to one controller component: what it does right now. */
interface Hint {
  comp: string; // WebXR input-profile component id
  key: string; // badge text
  slot: number; // button slot (lights the label while pressed)
  label: (c: HintCtx) => string;
  active?: (c: HintCtx) => boolean; // dimmed when false
}

const rec = (c: HintCtx): RecState => c.s?.rec_state ?? "idle";
// Where each button of the server's table sits on a Touch controller: [component id, badge, button slot].
const PARTS: Record<Hand, Record<ButtonHint["button"], [string, string, number]>> = {
  r: { primary: ["a-button", "A", B_PRIMARY], secondary: ["b-button", "B", B_SECONDARY],
       stick: ["xr-standard-thumbstick", "R3", B_STICK_PRESS], squeeze: ["xr-standard-squeeze", "GRIP", B_SQUEEZE],
       trigger: ["xr-standard-trigger", "TRIG", B_TRIGGER], menu: ["thumbrest", "MENU", B_MENU] },
  l: { primary: ["x-button", "X", B_PRIMARY], secondary: ["y-button", "Y", B_SECONDARY],
       stick: ["xr-standard-thumbstick", "L3", B_STICK_PRESS], squeeze: ["xr-standard-squeeze", "GRIP", B_SQUEEZE],
       trigger: ["xr-standard-trigger", "TRIG", B_TRIGGER], menu: ["menu", "MENU", B_MENU] },
};

/** The callouts of one controller, from the server's button table (GET /api/commands). The table is the one place
 *  that says what a button does, so a label can't drift from the behaviour. */
export function hintsFor(hand: Hand, table: ButtonHint[]): Hint[] {
  return table.filter((b) => b.hand === hand).map((b): Hint => {
    const [comp, key, slot] = PARTS[hand][b.button];
    if (b.client === "clutch") {
      return { comp, key, slot, label: (c) => (!c.hasArm ? "No arm" : c.engaged ? "Clutch ON" : b.label), active: (c) => c.hasArm };
    }
    if (b.client === "gripper") {
      return { comp, key, slot, active: (c) => c.engaged,
        label: (c) => (c.buttons[B_TRIGGER] > 0.05 && c.engaged ? `Gripper ${Math.round(c.buttons[B_TRIGGER] * 100)}%`
          : c.hasArm && !c.engaged ? "Select / grab" : b.label) };
    }
    return {
      comp, key, slot,
      label: (c) => {
        if (b.command === "record_toggle" && c.s?.replay && rec(c) === "idle") return "Record new";
        return b.labels?.[rec(c)] || b.label;
      },
      active: (c) => (b.labels ? b.labels[rec(c)] !== "" : true) && (!b.when || b.when.includes(rec(c))) && !(c.s?.edit && b.command),
    };
  });
}

const HINT_W = 0.07, HINT_H = 0.015, HINT_PX = 5000;
const COLUMN_X = 0.085; // label column, outboard of the controller (view space)
const HINT_GAP = 0.018;

interface MotionComponent { visualResponses: Record<string, { valueNode?: THREE.Object3D }> }

class HintLabel {
  readonly panel: Panel;
  private key = "";
  pressed = false;
  live = true;
  caption = "";

  constructor(readonly hint: Hint, private color: string) {
    this.panel = new Panel(HINT_W, HINT_H, HINT_PX, (g, w, h) => {
      roundRect(g, 3, 3, w - 6, h - 6, (h - 6) / 2);
      g.fillStyle = this.pressed ? this.color : "rgba(12,16,26,0.86)";
      g.fill();
      g.lineWidth = 3;
      g.strokeStyle = this.pressed ? "#ffffff" : this.color;
      g.stroke();
      const bw = Math.max(h * 1.05, this.hint.key.length * h * 0.32 + h * 0.4);
      roundRect(g, 10, 10, bw, h - 20, (h - 20) / 2);
      g.fillStyle = this.pressed ? "#0b1020" : this.color;
      g.fill();
      text(g, this.hint.key, 10 + bw / 2, h / 2 + 1, h * 0.36, { align: "center", weight: 800, color: this.pressed ? this.color : "#0b1020" });
      const x = 10 + bw + 14, avail = w - x - 18;
      g.font = `700 ${h * 0.42}px ${THEME.font}`;
      const size = h * 0.42 * Math.min(1, avail / g.measureText(this.caption).width);
      text(g, this.caption, x, h / 2 + 1, size, { weight: 700, color: this.pressed ? "#0b1020" : this.live ? THEME.text : THEME.dim });
    });
  }

  refresh(c: HintCtx, fade: boolean) {
    this.caption = this.hint.label(c);
    this.live = this.hint.active?.(c) ?? true;
    this.pressed = c.buttons[this.hint.slot] > 0.5;
    const key = `${this.caption}|${this.live}|${this.pressed}`;
    if (key !== this.key) { this.key = key; this.panel.invalidate(); }
    const opacity = this.pressed ? 1 : (this.live ? 1 : 0.55) * (fade ? 0.6 : 1);
    (this.panel.mesh.material as THREE.MeshBasicMaterial).opacity = opacity;
    this.panel.redraw();
  }
}

/** Status tag above a controller: arm, clutch, held object, recording, reach. */
class StatusChip {
  readonly panel: Panel;
  private key = "";
  private lines: [string, string, string] = ["", "", ""]; // title, detail, detail colour

  /** The name of the arm this hand drives ("" = none). */
  armName = "";

  constructor(private color: string) {
    this.panel = new Panel(0.075, 0.022, HINT_PX, (g, w, h) => {
      const [title, detail, dc] = this.lines;
      roundRect(g, 3, 3, w - 6, h - 6, 30);
      g.fillStyle = "rgba(12,16,26,0.86)";
      g.fill();
      g.lineWidth = 4;
      g.strokeStyle = dc === THEME.rec ? THEME.rec : this.color;
      g.stroke();
      text(g, this.armName, 28, h * 0.34, h * 0.24, { weight: 800, color: this.color });
      text(g, title, w - 28, h * 0.34, h * 0.24, { weight: 800, align: "right", color: title === "ENGAGED" ? THEME.ok : THEME.dim });
      text(g, detail, 28, h * 0.7, h * 0.22, { weight: 700, color: dc });
    });
  }

  /** next: the object this arm should take next (data-collection guide), or null. */
  refresh(s: Status | null, a: string | null, engaged: boolean, next: string | null = null) {
    let detail = "Press GRIP to take control", dc = THEME.dim;
    const name = !a ? "" : a.length <= 2 ? ({ l: "LEFT", r: "RIGHT" } as Record<string, string>)[a] ?? a.toUpperCase() : a.toUpperCase();
    if (name !== this.armName) { this.armName = name; this.key = ""; }
    if (!s) { detail = "Server offline"; dc = THEME.warn; }
    else if (s.mode === "hands") { detail = "Hands mode: put controllers down"; dc = THEME.warn; }
    else if (s.edit) { detail = "Edit: point + TRIGGER grabs"; dc = THEME.accent; }
    else if (s.replay) { detail = "Replay: use the timeline"; dc = THEME.accent; }
    else if (!a) { detail = "Buttons and menu only"; }
    else if (s.out_of_reach[a]) { detail = "⚠ Out of reach"; dc = THEME.rec; }
    else if (s.held[a]) { detail = `✊ ${s.held[a]!.replace(/_/g, " ")}`; dc = THEME.ok; }
    else if (next) { detail = `NEXT: ${next}`; dc = this.color; }
    else if (engaged) { detail = "TRIGGER closes gripper"; dc = THEME.text; }
    if (s?.rec_state === "recording") detail = `● REC  ${detail}`;
    const title = engaged ? "ENGAGED" : "IDLE";
    const key = `${title}|${detail}|${dc}`;
    if (key !== this.key) { this.key = key; this.lines = [title, detail, dc]; this.panel.invalidate(); }
    this.panel.redraw();
  }
}

export class HandViz {
  readonly group = new THREE.Group();
  /** Show the button callouts (toggle in the wrist menu). */
  showHints = true;
  private joints: THREE.InstancedMesh;
  private bones: THREE.LineSegments;
  private jointMat: THREE.MeshStandardMaterial;
  private boneMat: THREE.LineBasicMaterial;
  private controller = new THREE.Group(); // at the grip pose
  private model: THREE.Object3D & { motionController: unknown };
  private fallback: THREE.Mesh; // stylized stand-in until (or if never) the real model loads
  private modelScene: THREE.Object3D | null = null;
  private modelMats: THREE.MeshStandardMaterial[] = [];
  private source: XRInputSource | null = null;
  private hints: HintLabel[] = [];
  private parts: { h: HintLabel; local: THREE.Vector3 }[] = []; // labelled components, grip frame
  private ui = new THREE.Group(); // world-space callouts + chip, laid out facing the head
  private hintGroup = new THREE.Group();
  private leaders: THREE.LineSegments;
  private chip: StatusChip;
  private status: Status | null = null;
  private colorHex: string;
  private static readonly REACH = new THREE.Color(THEME.rec);
  private m = new THREE.Matrix4();
  private q = new THREE.Quaternion();
  private p = new THREE.Vector3();
  private s = new THREE.Vector3();

  /** The arm this hand drives (status.mapping), or null. */
  arm: string | null = null;

  constructor(readonly color: THREE.Color, readonly hand: Hand) {
    this.colorHex = `#${color.getHexString()}`;
    this.jointMat = new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.5, roughness: 0.4, transparent: true, opacity: 0.9 });
    this.joints = new THREE.InstancedMesh(new THREE.SphereGeometry(1, 12, 8), this.jointMat, 25);
    this.joints.frustumCulled = false;
    this.boneMat = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.7 });
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(BONES.length * 6), 3));
    this.bones = new THREE.LineSegments(geo, this.boneMat);
    this.bones.frustumCulled = false;

    this.model = controllerModels.createControllerModel(this.controller) as typeof this.model;
    this.fallback = new THREE.Mesh(
      new THREE.CapsuleGeometry(0.018, 0.07, 6, 16),
      new THREE.MeshStandardMaterial({ color: 0x1b2233, roughness: 0.35, metalness: 0.3 }),
    );
    this.fallback.rotation.x = Math.PI / 2 - 0.5;
    this.fallback.position.set(0, -0.01, 0.03);
    const lg = new THREE.BufferGeometry();
    lg.setAttribute("position", new THREE.BufferAttribute(new Float32Array(12 * 6), 3));
    this.leaders = new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.7, depthWrite: false }));
    this.leaders.frustumCulled = false;
    this.hintGroup.add(this.leaders);
    this.chip = new StatusChip(this.colorHex);
    this.controller.add(this.model, this.fallback);
    this.ui.add(this.hintGroup, this.chip.panel.mesh);

    this.group.add(this.joints, this.bones, this.controller, this.ui);
    this.group.visible = false; // until the first tracked sample
  }

  setStatus(s: Status | null) {
    this.status = s;
    this.arm = s?.mapping[this.hand] ?? null;
  }

  /** The button labels of this controller, from the server's table. */
  setHints(table: ButtonHint[]) {
    for (const h of this.hints) { this.hintGroup.remove(h.panel.mesh); h.panel.texture.dispose(); }
    this.hints = hintsFor(this.hand, table).slice(0, 12).map((h) => new HintLabel(h, this.colorHex));
    for (const h of this.hints) { h.panel.mesh.visible = false; this.hintGroup.add(h.panel.mesh); }
    this.parts = [];
    if (this.modelScene) this.onModelLoaded(this.modelScene);
  }

  /** Arm colour (changes when the hands are swapped). */
  setColor(color: THREE.Color) {
    this.color.copy(color);
    this.jointMat.color.copy(color);
    this.jointMat.emissive.copy(color);
    this.boneMat.color.copy(color);
    (this.leaders.material as THREE.LineBasicMaterial).color.copy(color);
  }

  /** The object this arm should take next (shown on the status chip), or null. */
  next: string | null = null;

  /** The model factory picks the left/right asset from the source's input profiles on "connected". */
  private setSource(src: XRInputSource | null) {
    if (src === this.source) return;
    if (this.source) this.controller.dispatchEvent({ type: "disconnected" } as never);
    this.source = src;
    if (src) this.controller.dispatchEvent({ type: "connected", data: src } as never);
  }

  /** Once the real model is in: own materials for the glow, and a callout next to each mapped component. */
  private onModelLoaded(scene: THREE.Object3D) {
    this.modelScene = scene;
    this.modelMats = [];
    scene.traverse((o) => {
      const mesh = o as THREE.Mesh;
      if (!mesh.isMesh) return;
      const mat = (mesh.material as THREE.MeshStandardMaterial).clone(); // the asset cache shares materials
      mesh.material = mat;
      if ("emissive" in mat) this.modelMats.push(mat);
    });

    this.controller.updateMatrixWorld(true);
    const comps = (this.model.motionController as { components?: Record<string, MotionComponent> } | null)?.components ?? {};
    const box = new THREE.Box3();
    this.parts = [];
    for (const h of this.hints) {
      const node = Object.values(comps[h.hint.comp]?.visualResponses ?? {}).find((r) => r.valueNode)?.valueNode;
      h.panel.mesh.visible = !!node;
      if (!node) continue;
      box.setFromObject(node);
      this.parts.push({ h, local: this.controller.worldToLocal(box.getCenter(new THREE.Vector3())) });
    }
    this.leaders.geometry.setDrawRange(0, this.parts.length * 2);
  }

  /**
   * Callouts in view space, so they read at any controller angle: a column outboard of the controller, parallel to
   * the view, ordered by each part's on-screen height, with a leader line to the part; the chip sits above.
   */
  private layoutUi(headQ: THREE.Quaternion) {
    const side = this.hand === "l" ? -1 : 1;
    const right = this.p.set(1, 0, 0).applyQuaternion(headQ);
    const up = this.s.set(0, 1, 0).applyQuaternion(headQ);
    const c = this.controller.position, cq = this.controller.quaternion;
    const world = this.parts.map((pt) => ({ ...pt, w: pt.local.clone().applyQuaternion(cq).add(c) }));
    world.sort((a, b) => b.w.dot(up) - a.w.dot(up));
    const pos = this.leaders.geometry.getAttribute("position") as THREE.BufferAttribute;
    world.forEach(({ h, w }, k) => {
      const off = ((world.length - 1) / 2 - k) * HINT_GAP;
      const m = h.panel.mesh;
      m.position.copy(c).addScaledVector(right, side * COLUMN_X).addScaledVector(up, off);
      m.quaternion.copy(headQ);
      pos.setXYZ(2 * k, w.x, w.y, w.z);
      pos.setXYZ(2 * k + 1, ...m.position.clone().addScaledVector(right, -side * HINT_W / 2).toArray());
    });
    pos.needsUpdate = true;
    this.chip.panel.mesh.position.copy(c).addScaledVector(up, 0.075);
    this.chip.panel.mesh.quaternion.copy(headQ);
  }

  update(h: HandSample, radii: Float32Array, engaged: boolean, src: XRInputSource | null, headQ: THREE.Quaternion) {
    const isHand = h.source === SRC_HAND, isCtrl = h.source === SRC_CONTROLLER;
    this.group.visible = isHand || isCtrl;
    this.joints.visible = this.bones.visible = isHand;
    this.controller.visible = this.ui.visible = isCtrl;
    this.jointMat.emissiveIntensity = engaged ? 1.4 : 0.35;
    this.boneMat.opacity = engaged ? 1.0 : 0.55;
    this.setSource(isCtrl ? src : null);
    if (isHand) {
      const j = h.joints;
      for (let i = 0; i < 25; i++) {
        const r = Math.max(radii[i] || 0.008, 0.004) * 0.85;
        this.p.set(j[i * 7], j[i * 7 + 1], j[i * 7 + 2]);
        this.s.setScalar(r);
        this.m.compose(this.p, this.q.identity(), this.s);
        this.joints.setMatrixAt(i, this.m);
      }
      this.joints.instanceMatrix.needsUpdate = true;
      const pos = this.bones.geometry.getAttribute("position") as THREE.BufferAttribute;
      BONES.forEach(([a, b], k) => {
        pos.setXYZ(2 * k, j[a * 7], j[a * 7 + 1], j[a * 7 + 2]);
        pos.setXYZ(2 * k + 1, j[b * 7], j[b * 7 + 1], j[b * 7 + 2]);
      });
      pos.needsUpdate = true;
    } else if (isCtrl) {
      const p = h.pose;
      this.controller.position.set(p[0], p[1], p[2]);
      this.controller.quaternion.set(p[3], p[4], p[5], p[6]);

      const scene = this.model.children[0] ?? null;
      if (scene !== this.modelScene) {
        if (scene) this.onModelLoaded(scene);
        else this.modelScene = null;
      }
      const loaded = !!this.modelScene;
      this.fallback.visible = !loaded;

      // The controller glows in its arm colour while it drives the arm and pulses red when the arm can't follow.
      const reach = !!(this.arm && this.status?.out_of_reach[this.arm]);
      const glow = reach ? 0.3 + 0.2 * Math.sin(performance.now() / 120) : engaged ? 0.22 : 0;
      for (const mat of loaded ? this.modelMats : [this.fallback.material as THREE.MeshStandardMaterial]) {
        mat.emissive.copy(reach ? HandViz.REACH : this.color);
        mat.emissiveIntensity = glow;
      }

      this.hintGroup.visible = this.showHints && loaded;
      if (this.hintGroup.visible) {
        // Callouts fade while operating so they don't clutter the view of the robot.
        const ctx: HintCtx = { s: this.status, engaged, hasArm: !!this.arm, buttons: h.buttons };
        for (const l of this.hints) {
          if (!l.panel.mesh.visible) continue;
          l.refresh(ctx, engaged);
        }
        (this.leaders.material as THREE.LineBasicMaterial).opacity = engaged ? 0.3 : 0.7;
      }
      this.chip.refresh(this.status, this.arm, engaged, this.next);
      this.layoutUi(headQ);
    }
  }
}

export class Pointer {
  readonly group = new THREE.Group();
  private line: THREE.Mesh;
  private dot: THREE.Mesh;

  constructor(color: THREE.Color) {
    const mat = new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.55, depthWrite: false });
    this.line = new THREE.Mesh(new THREE.CylinderGeometry(0.0012, 0.0025, 1, 6, 1, true).translate(0, 0.5, 0).rotateX(-Math.PI / 2), mat);
    this.dot = new THREE.Mesh(new THREE.RingGeometry(0.004, 0.007, 24), new THREE.MeshBasicMaterial({ color: 0xffffff, depthTest: false, transparent: true }));
    this.dot.renderOrder = 20;
    this.group.add(this.line, this.dot);
    this.group.visible = false;
  }

  /** Show a ray from origin along quat's -Z, ending at `hit` (or a short stub). */
  set(origin: THREE.Vector3, quat: THREE.Quaternion, hit: THREE.Vector3 | null, hitNormalQuat?: THREE.Quaternion) {
    this.group.visible = true;
    const len = hit ? origin.distanceTo(hit) : 0.25;
    this.line.position.copy(origin);
    this.line.quaternion.copy(quat);
    this.line.scale.set(1, 1, len);
    (this.line.material as THREE.MeshBasicMaterial).opacity = hit ? 0.8 : 0.25;
    this.dot.visible = !!hit;
    if (hit) {
      this.dot.position.copy(hit);
      if (hitNormalQuat) this.dot.quaternion.copy(hitNormalQuat);
    }
  }

  hide() { this.group.visible = false; }
}
