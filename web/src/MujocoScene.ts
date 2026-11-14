// Rebuilds the MuJoCo scene in three.js from /api/scene and animates it from the pose stream.
// Shared by the WebXR app and the desktop dashboard.

import * as THREE from "three";
import { toCreasedNormals } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import type { PoseFrame, Region } from "./protocol";

/** One colour per arm, by its position in the embodiment's arm list (also used for that arm's hand and panels). */
export const ARM_PALETTE = ["#22d3ee", "#f472b6", "#a3e635", "#fbbf24", "#c084fc", "#fb7185"];
export const armColorHex = (i: number) => ARM_PALETTE[((i % ARM_PALETTE.length) + ARM_PALETTE.length) % ARM_PALETTE.length];
export const armColor = (i: number) => new THREE.Color(armColorHex(i));
/** Geom group of operator-only visuals (grasp-point markers): drawn here, never by a recorded camera. */
export const OPERATOR_GROUP = 5;

interface GeomDesc {
  name: string;
  group: number;
  body: number;
  type: "plane" | "sphere" | "capsule" | "ellipsoid" | "cylinder" | "box" | "mesh";
  size: number[];
  pos: number[];
  quat: number[];
  rgba: number[];
  material: { specular: number; shininess: number; reflectance: number };
  mesh: number | null;
}
interface MeshDesc { name: string; vertOffset: number; nvert: number; faceOffset: number; nface: number }
/** A MuJoCo camera: looks along its -z with +y up (like three.js), vertical field of view in degrees. */
interface CameraDesc { name: string; body: number; pos: number[]; quat: number[]; fovy: number }
export interface SceneDesc {
  version: number;
  nbody: number;
  bodies: { id: number; name: string; parent: number }[];
  geoms: GeomDesc[];
  meshes: MeshDesc[];
  cameras?: CameraDesc[];
  arms: string[];                         // the embodiment's arms, in flag-bit order
  objects: Record<string, number>;        // task object -> body id
  wrist_cameras: Record<string, string>;  // arm -> camera name
}

interface BufferedFrame { t: number; bodies: Float32Array; flags: number }

/** Grasp-point marker (the server's `<arm>_tcp_marker` geom) plus a drop line to the table under it. */
interface TcpMarker { arm: string; body: THREE.Object3D; sphere: THREE.Mesh; line: THREE.Line; dot: THREE.Mesh }

/** A wrist camera's view: its frustum (on the hand) and the patch of table it sees (outline + faint fill). */
interface CamViz { body: THREE.Object3D; cam: THREE.Object3D; frustum: THREE.LineSegments; outline: THREE.LineLoop; fill: THREE.Mesh }

const INTERP_DELAY_MS = 30;
const CREASE_ANGLE = THREE.MathUtils.degToRad(35);
const OUT_OF_REACH = new THREE.Color("#ef4444");
const CAM_ASPECT = 4 / 3;      // the recorded images are 320 x 240
const FRUSTUM_LEN = 0.18;      // m drawn in front of the camera
const MAX_RAY = 1.5;           // m: a corner ray that doesn't reach the table within this hides the footprint

function gridTexture(): THREE.CanvasTexture {
  const c = document.createElement("canvas");
  c.width = c.height = 512;
  const g = c.getContext("2d")!;
  g.fillStyle = "#0d1117";
  g.fillRect(0, 0, 512, 512);
  g.strokeStyle = "rgba(120,160,255,0.10)";
  g.lineWidth = 2;
  for (let i = 0; i <= 512; i += 64) {
    g.beginPath(); g.moveTo(i, 0); g.lineTo(i, 512); g.stroke();
    g.beginPath(); g.moveTo(0, i); g.lineTo(512, i); g.stroke();
  }
  g.strokeStyle = "rgba(120,160,255,0.22)";
  g.lineWidth = 3;
  g.strokeRect(0, 0, 512, 512);
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 4;
  return t;
}

export class MujocoScene {
  readonly root = new THREE.Group();          // world -> xr placement applied here
  readonly environment = new THREE.Group();   // floor + backdrop (hidden in passthrough)
  desc: SceneDesc | null = null;
  engaged: Record<string, boolean> = {};
  recording = false;
  /** Bumped on every (re)build: who caches things per scene compares it. */
  generation = 0;
  private outOfReach: Record<string, boolean> = {};
  private ghost = new THREE.Group();        // a reference take's bodies, translucent (operator-only)
  private ghostBodies: (THREE.Object3D | null)[] = [];
  private ghostMat = new THREE.MeshBasicMaterial({ color: 0x9fd0ff, transparent: true, opacity: 0.22, depthWrite: false });
  private guide = new THREE.Group();        // translucent copy of the next object at its goal pose
  private guideFor: string | null = null;
  private guideMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.3, depthWrite: false });
  private overlays = new THREE.Group();     // region / start-region outlines on the table (editor)
  private loading: Promise<void> | null = null;
  private bodies: THREE.Object3D[] = [];
  private markers: TcpMarker[] = [];
  private camViz: CamViz[] = [];
  private camVizOn = false;
  private highlighted: { name: string; meshes: THREE.Mesh[]; originals: THREE.Material[]; mat: THREE.MeshStandardMaterial } | null = null;
  tableZ = 0;
  private frames: BufferedFrame[] = [];
  private clockOffset = Number.POSITIVE_INFINITY;
  private materials = new Map<string, THREE.Material>();
  private geometries: THREE.BufferGeometry[] = [];
  private tmpQa = new THREE.Quaternion();
  private tmpQb = new THREE.Quaternion();
  private tmpV = new THREE.Vector3();

  constructor(private opts: { shadows: boolean }) {
    this.root.matrixAutoUpdate = false;
    this.ghost.visible = this.guide.visible = false;
    this.root.add(this.ghost, this.guide, this.overlays);
    this.addLights();
  }

  private addLights() {
    const hemi = new THREE.HemisphereLight(0xdfe8ff, 0x20232b, 1.4);
    hemi.position.set(0, 0, 3);
    hemi.up.set(0, 0, 1);
    const key = new THREE.DirectionalLight(0xffffff, 2.2);
    key.position.set(0.8, 0.6, 3.0);
    key.target.position.set(0.4, 0, 0.75);
    if (this.opts.shadows) {
      key.castShadow = true;
      key.shadow.mapSize.set(1024, 1024);
      const cam = key.shadow.camera as THREE.OrthographicCamera;
      cam.left = -1.0; cam.right = 1.0; cam.top = 1.0; cam.bottom = -1.0; cam.near = 0.5; cam.far = 4;
      key.shadow.bias = -0.0005;
      key.shadow.normalBias = 0.01;
    }
    const rim = new THREE.DirectionalLight(0x8fb4ff, 0.8);
    rim.position.set(2.0, -1.5, 1.8);
    this.root.add(hemi, key, key.target, rim);
  }

  /** (Re)load the scene from the server: on every connect and on every `scene_changed` event. */
  load(): Promise<void> {
    const run = async () => {
      for (let attempt = 0; attempt < 3; attempt++) {
        const [desc, blob] = await Promise.all([
          fetch("/api/scene", { cache: "no-store" }).then((r) => r.json() as Promise<SceneDesc>),
          fetch("/api/scene.bin", { cache: "no-store" }).then((r) => r.arrayBuffer()),
        ]);
        const size = (desc as SceneDesc & { blobSize: number }).blobSize;
        if (blob.byteLength === size) { this.build(desc, blob); return; }
        // The scene changed between the two requests: fetch both again.
      }
    };
    this.loading = (this.loading ?? Promise.resolve()).then(run, run);
    return this.loading;
  }

  armIndex(arm: string) { return this.desc?.arms.indexOf(arm) ?? -1; }

  private material(g: GeomDesc): THREE.Material {
    const key = `${g.rgba.join(",")}|${g.material.shininess}|${g.material.specular}`;
    let m = this.materials.get(key);
    if (!m) {
      const [r, gg, b, a] = g.rgba;
      m = new THREE.MeshStandardMaterial({
        color: new THREE.Color().setRGB(r, gg, b, THREE.SRGBColorSpace),
        roughness: THREE.MathUtils.clamp(0.85 - g.material.shininess * 0.6 - g.material.specular * 0.2, 0.15, 0.95),
        metalness: 0.05,
        transparent: a < 1,
        opacity: a,
      });
      this.materials.set(key, m);
    }
    return m;
  }

  private build(desc: SceneDesc, blob: ArrayBuffer) {
    this.clear();
    this.desc = desc;
    this.generation++;
    this.engaged = Object.fromEntries(desc.arms.map((a) => [a, false]));
    const meshGeoms: THREE.BufferGeometry[] = desc.meshes.map((m) => {
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(blob, m.vertOffset, m.nvert * 3), 3));
      g.setIndex(new THREE.BufferAttribute(new Uint32Array(blob, m.faceOffset, m.nface * 3), 1));
      const creased = toCreasedNormals(g, CREASE_ANGLE);
      g.dispose();
      this.geometries.push(creased);
      return creased;
    });

    for (let b = 0; b < desc.nbody; b++) {
      const o = new THREE.Object3D();
      o.name = desc.bodies[b].name;
      o.matrixAutoUpdate = true;
      this.bodies.push(o);
      this.root.add(o);
    }

    for (const g of desc.geoms) {
      let geo: THREE.BufferGeometry;
      const s = g.size;
      let mat = this.material(g);
      switch (g.type) {
        case "box": geo = new THREE.BoxGeometry(2 * s[0], 2 * s[1], 2 * s[2]); break;
        case "sphere": geo = new THREE.SphereGeometry(s[0], 24, 16); break;
        case "ellipsoid": geo = new THREE.SphereGeometry(1, 24, 16).scale(s[0], s[1], s[2]); break;
        case "capsule": geo = new THREE.CapsuleGeometry(s[0], 2 * s[1], 8, 16).rotateX(Math.PI / 2); break;
        case "cylinder": geo = new THREE.CylinderGeometry(s[0], s[0], 2 * s[1], 24).rotateX(Math.PI / 2); break;
        case "plane": {
          const w = s[0] > 0 ? 2 * s[0] : 20, h = s[1] > 0 ? 2 * s[1] : 20;
          geo = new THREE.PlaneGeometry(w, h);
          if (g.name === "floor") {
            const tex = gridTexture();
            tex.repeat.set(w / 0.5, h / 0.5);
            mat = new THREE.MeshStandardMaterial({ map: tex, roughness: 0.9, metalness: 0 });
          }
          break;
        }
        case "mesh": geo = meshGeoms[g.mesh!]; break;
      }
      if (g.type !== "mesh") this.geometries.push(geo);
      const mesh = new THREE.Mesh(geo, mat);
      mesh.name = g.name;
      mesh.position.fromArray(g.pos);
      mesh.quaternion.fromArray(g.quat);
      const isFloor = g.name === "floor";
      mesh.castShadow = this.opts.shadows && !isFloor && g.type !== "plane";
      mesh.receiveShadow = this.opts.shadows;
      if (g.name === "table_top") this.tableZ = g.pos[2] + s[2];
      if (isFloor) this.environment.add(mesh);
      else this.bodies[g.body].add(mesh);
      mesh.userData.group = g.group;
      const arm = desc.arms.find((a) => g.name === `${a}_tcp_marker`);
      if (arm) this.addMarker(arm, mesh);
    }
    this.bodies[0].add(this.environment);
    const wrist = Object.entries(desc.wrist_cameras ?? {});
    for (const c of desc.cameras ?? []) {
      const arm = wrist.find(([, cam]) => cam === c.name)?.[0];
      if (arm) this.addCamViz(c, armColor(desc.arms.indexOf(arm)));
    }
    this.frames = [];
    const want = this.highlighted?.name ?? null;  // the old meshes are gone: highlight the same body again
    this.highlighted?.mat.dispose();
    this.highlighted = null;
    this.highlightBody(want);
  }

  private addCamViz(c: CameraDesc, color: THREE.Color) {
    const cam = new THREE.Object3D();
    cam.position.fromArray(c.pos);
    cam.quaternion.fromArray(c.quat);
    const hh = Math.tan(THREE.MathUtils.degToRad(c.fovy) / 2), hw = hh * CAM_ASPECT;
    cam.userData.corners = [[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]].map(([x, y]) => new THREE.Vector3(x, y, -1));
    const L = FRUSTUM_LEN, k = (x: number, y: number) => [x * L, y * L, -L];
    const [a, b, cc, d] = [k(-hw, -hh), k(hw, -hh), k(hw, hh), k(-hw, hh)];
    const pts = [[0, 0, 0], a, [0, 0, 0], b, [0, 0, 0], cc, [0, 0, 0], d, a, b, b, cc, cc, d, d, a].flat();
    const frustumGeo = new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
    const frustum = new THREE.LineSegments(frustumGeo, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.7 }));
    cam.add(frustum);
    const outlineGeo = new THREE.BufferGeometry().setAttribute("position", new THREE.BufferAttribute(new Float32Array(12), 3));
    const outline = new THREE.LineLoop(outlineGeo, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.9 }));
    const fillGeo = new THREE.BufferGeometry().setAttribute("position", new THREE.BufferAttribute(new Float32Array(12), 3));
    fillGeo.setIndex([0, 1, 2, 0, 2, 3]);
    const fill = new THREE.Mesh(fillGeo, new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.08, side: THREE.DoubleSide, depthWrite: false }));
    for (const o of [outline, fill]) { o.frustumCulled = false; this.root.add(o); }
    this.geometries.push(frustumGeo, outlineGeo, fillGeo);
    const body = this.bodies[c.body];
    body.add(cam);
    this.camViz.push({ body, cam, frustum, outline, fill });
    this.setCameraViz(this.camVizOn);
  }

  /** Show where the wrist cameras look: frustum on the hand and the patch of table in view. */
  setCameraViz(on: boolean) {
    this.camVizOn = on;
    for (const v of this.camViz) v.frustum.visible = v.outline.visible = v.fill.visible = on;
  }

  /** Make one body (e.g. the brick to take next) glow; null clears it. */
  highlightBody(name: string | null) {
    if (this.highlighted?.name === name) return;
    if (this.highlighted) {
      this.highlighted.meshes.forEach((m, i) => (m.material = this.highlighted!.originals[i]));
      this.highlighted.mat.dispose();
      this.highlighted = null;
    }
    const body = name ? this.bodies.find((b) => b.name === name) : undefined;
    const meshes = (body?.children.filter((o) => o instanceof THREE.Mesh) ?? []) as THREE.Mesh[];
    if (!name || !meshes.length) {
      if (name) this.highlighted = { name, meshes: [], originals: [], mat: new THREE.MeshStandardMaterial() };  // retried after a reload
      return;
    }
    const mat = (meshes[0].material as THREE.MeshStandardMaterial).clone();
    mat.emissive.copy(mat.color);
    this.highlighted = { name, meshes, originals: meshes.map((m) => m.material as THREE.Material), mat };
    for (const m of meshes) m.material = mat;
  }

  private updateCamViz() {
    const p = this.tmpV, q = this.tmpQa, d = new THREE.Vector3(), z = this.tableZ + 0.001;
    for (const v of this.camViz) {
      if (!this.camVizOn) continue;
      p.copy(v.cam.position).applyQuaternion(v.body.quaternion).add(v.body.position);  // bodies are children of root
      q.copy(v.body.quaternion).multiply(v.cam.quaternion);
      const hits: THREE.Vector3[] = [];
      for (const c of v.cam.userData.corners as THREE.Vector3[]) {
        d.copy(c).applyQuaternion(q);
        const t = d.z < -1e-6 ? (z - p.z) / d.z : Infinity;
        if (t * d.length() > MAX_RAY) break;
        hits.push(p.clone().addScaledVector(d, t));
      }
      const show = hits.length === 4;
      v.outline.visible = v.fill.visible = show;
      if (!show) continue;
      for (const o of [v.outline, v.fill]) {
        const pos = o.geometry.getAttribute("position") as THREE.BufferAttribute;
        hits.forEach((h, i) => pos.setXYZ(i, h.x, h.y, h.z));
        pos.needsUpdate = true;
      }
    }
  }

  private addMarker(arm: string, sphere: THREE.Mesh) {
    const color = armColor(this.armIndex(arm));
    // Drawn on top of the fingers so the grasp point stays visible between them.
    sphere.material = new THREE.MeshBasicMaterial({ color, depthTest: false, transparent: true });
    sphere.renderOrder = 10;
    sphere.castShadow = sphere.receiveShadow = false;
    const lineGeo = new THREE.BufferGeometry().setAttribute("position", new THREE.BufferAttribute(new Float32Array(6), 3));
    const line = new THREE.Line(lineGeo, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.6 }));
    line.frustumCulled = false;
    const dot = new THREE.Mesh(new THREE.RingGeometry(0.004, 0.007, 24), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.8 }));
    this.geometries.push(lineGeo, dot.geometry);
    this.root.add(line, dot);
    this.markers.push({ arm, body: sphere.parent!, sphere, line, dot });
  }

  /** Markers turn red while the IK can't get that gripper to its target. */
  setOutOfReach(outOfReach: Record<string, boolean>) {
    this.outOfReach = { ...outOfReach };
  }

  private clear() {
    for (const b of this.bodies) this.root.remove(b);
    for (const v of this.camViz) {
      this.root.remove(v.outline, v.fill);
      for (const o of [v.frustum, v.outline, v.fill]) (o.material as THREE.Material).dispose();
    }
    this.camViz = [];
    for (const mk of this.markers) {
      this.root.remove(mk.line, mk.dot);
      for (const o of [mk.sphere, mk.line, mk.dot]) (o.material as THREE.Material).dispose();
    }
    this.markers = [];
    this.ghost.clear();
    this.ghostBodies = [];
    this.ghost.visible = false;
    this.guide.clear();
    this.guideFor = null;
    for (const g of this.geometries) g.dispose();
    for (const m of this.materials.values()) m.dispose();
    this.bodies = [];
    this.geometries = [];
    this.materials.clear();
    this.environment.clear();
  }

  setPlacement(colMajor: number[]) {
    this.root.matrix.fromArray(colMajor);
    this.root.matrixWorldNeedsUpdate = true;
  }

  /** Buffer a frame; timestamps are mapped onto the local clock for jitter-free interpolation. */
  pushFrame(f: PoseFrame, recvMs: number) {
    if (!this.desc || f.nbody !== this.desc.nbody) return;
    this.clockOffset = Math.min(this.clockOffset + 0.05, recvMs - f.tServer);
    this.frames.push({ t: f.tServer + this.clockOffset, bodies: f.bodies.slice(), flags: f.flags });
    if (this.frames.length > 8) this.frames.shift();
  }

  update(nowMs: number) {
    const n = this.frames.length;
    if (!n) return;
    const t = nowMs - INTERP_DELAY_MS;
    let a = this.frames[n - 1], b = a, alpha = 0;
    for (let i = n - 1; i > 0; i--) {
      if (this.frames[i - 1].t <= t) {
        a = this.frames[i - 1];
        b = this.frames[i];
        alpha = THREE.MathUtils.clamp((t - a.t) / Math.max(b.t - a.t, 1e-3), 0, 1);
        break;
      }
    }
    if (t < this.frames[0].t) { a = b = this.frames[0]; alpha = 0; }

    for (let i = 1; i < this.bodies.length; i++) {
      const o = this.bodies[i], k = i * 7, A = a.bodies, B = b.bodies;
      o.position.set(A[k] + (B[k] - A[k]) * alpha, A[k + 1] + (B[k + 1] - A[k + 1]) * alpha, A[k + 2] + (B[k + 2] - A[k + 2]) * alpha);
      this.tmpQa.set(A[k + 3], A[k + 4], A[k + 5], A[k + 6]);
      this.tmpQb.set(B[k + 3], B[k + 4], B[k + 5], B[k + 6]);
      o.quaternion.slerpQuaternions(this.tmpQa, this.tmpQb, alpha);
    }

    const latest = this.frames[n - 1];
    this.desc?.arms.forEach((arm, i) => (this.engaged[arm] = !!(latest.flags & (1 << i))));
    this.recording = !!(latest.flags & 128);
    if (this.highlighted?.meshes.length) this.highlighted.mat.emissiveIntensity = 0.25 + 0.2 * Math.sin(nowMs / 220);
    this.updateCamViz();
    for (const mk of this.markers) {
      const color = this.outOfReach[mk.arm] ? OUT_OF_REACH : armColor(this.armIndex(mk.arm));
      const engaged = this.engaged[mk.arm];
      for (const o of [mk.sphere, mk.line, mk.dot]) (o.material as THREE.MeshBasicMaterial).color.copy(color);
      (mk.sphere.material as THREE.MeshBasicMaterial).opacity = engaged ? 1 : 0.6;
      // Grasp point in world coordinates (bodies are direct children of root).
      const p = this.tmpV.copy(mk.sphere.position).applyQuaternion(mk.body.quaternion).add(mk.body.position);
      const z = this.tableZ + 0.0005, above = p.z > z + 0.003;
      mk.line.visible = mk.dot.visible = above;
      if (!above) continue;
      const pos = mk.line.geometry.getAttribute("position") as THREE.BufferAttribute;
      pos.setXYZ(0, p.x, p.y, p.z);
      pos.setXYZ(1, p.x, p.y, z);
      pos.needsUpdate = true;
      mk.dot.position.set(p.x, p.y, z);
    }
  }

  // -- operator-only overlays ------------------------------------------------------------------------

  /** Translucent copies of a body's meshes (no markers, no invisible helpers). */
  private cloneBody(b: number, mat: THREE.Material): THREE.Object3D | null {
    const src = this.bodies[b];
    const meshes = src.children.filter((o) => o instanceof THREE.Mesh && o.userData.group !== OPERATOR_GROUP) as THREE.Mesh[];
    if (!meshes.length) return null;
    const o = new THREE.Object3D();
    for (const m of meshes) {
      const c = new THREE.Mesh(m.geometry, mat);
      c.position.copy(m.position);
      c.quaternion.copy(m.quaternion);
      c.renderOrder = 5;
      o.add(c);
    }
    return o;
  }

  /** Bodies of a reference take, drawn translucent over the live scene (never in a recorded camera). */
  setGhostFrame(f: PoseFrame) {
    if (!this.desc || f.nbody !== this.desc.nbody) return;
    if (!this.ghostBodies.length) {
      this.ghostBodies = this.bodies.map((_, b) => {
        const name = this.desc!.bodies[b].name;
        const o = b === 0 || name === "table" ? null : this.cloneBody(b, this.ghostMat);  // static bodies would only z-fight
        if (o) this.ghost.add(o);
        return o;
      });
    }
    this.ghostBodies.forEach((o, b) => {
      if (!o) return;
      const k = b * 7, B = f.bodies;
      o.position.set(B[k], B[k + 1], B[k + 2]);
      o.quaternion.set(B[k + 3], B[k + 4], B[k + 5], B[k + 6]);
    });
    this.ghost.visible = true;
  }

  hideGhost() { this.ghost.visible = false; }

  /** Ghost of an object at the pose where it belongs next (pos, quat wxyz in the world), or null to hide it. */
  setGuide(g: { object: string; pos: number[]; quat: number[]; dist_mm: number } | null) {
    if (!g || !this.desc || this.desc.objects[g.object] === undefined) { this.guide.visible = false; return; }
    if (this.guideFor !== `${this.generation}/${g.object}`) {
      this.guide.clear();
      const o = this.cloneBody(this.desc.objects[g.object], this.guideMat);
      if (o) this.guide.add(o);
      this.guideFor = `${this.generation}/${g.object}`;
    }
    this.guide.position.fromArray(g.pos);
    this.guide.quaternion.set(g.quat[1], g.quat[2], g.quat[3], g.quat[0]);
    // Green within a few millimetres, so "close enough to press" is visible without reading the number.
    this.guideMat.color.set(g.dist_mm < 4 ? 0x34d399 : g.dist_mm < 15 ? 0xfbbf24 : 0xffffff);
    this.guide.visible = true;
  }

  /** Where the guide is drawn, in the scene root's frame (for a distance readout next to it). */
  guidePosition(out: THREE.Vector3): THREE.Vector3 | null {
    return this.guide.visible ? out.copy(this.guide.position) : null;
  }

  /** Outlines on the table (operator-only): named regions solid, start regions dashed in the object's colour. */
  setOverlays(regions: Record<string, Region> | null, spawn: Record<string, Region> | null = null,
    colors: Record<string, number[]> = {}) {
    for (const o of this.overlays.children) {
      (o as THREE.Line).geometry.dispose();
      ((o as THREE.Line).material as THREE.Material).dispose();
    }
    this.overlays.clear();
    const z = this.tableZ + 0.0012;
    const add = (r: Region, color: THREE.Color, dashed: boolean, name: string) => {
      const [cx, cy] = r.center, [hx, hy] = r.half;
      const pts = [[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy], [-hx, -hy]].map(([x, y]) => new THREE.Vector3(cx + x, cy + y, z));
      const geo = new THREE.BufferGeometry().setFromPoints(pts);
      const mat = dashed ? new THREE.LineDashedMaterial({ color, dashSize: 0.012, gapSize: 0.008 }) : new THREE.LineBasicMaterial({ color });
      const line = new THREE.Line(geo, mat);
      if (dashed) line.computeLineDistances();
      line.name = name;
      line.userData.region = r;
      this.overlays.add(line);
    };
    for (const [name, r] of Object.entries(regions ?? {})) add(r, new THREE.Color(0xf8fafc), false, `region:${name}`);
    for (const [name, r] of Object.entries(spawn ?? {})) {
      const c = colors[name];
      add(r, c ? new THREE.Color().setRGB(c[0], c[1], c[2], THREE.SRGBColorSpace) : new THREE.Color(0x7c9cff), true, `spawn:${name}`);
    }
  }

  /** The three.js object of a body (a task object by name), posed in the scene root's frame. */
  bodyObject(name: string): THREE.Object3D | null {
    return this.bodies.find((b) => b.name === name) ?? null;
  }

  /** The task object under a ray (world space of the renderer), or null. */
  pickObject(raycaster: THREE.Raycaster, names: string[]): { name: string; point: THREE.Vector3 } | null {
    const targets = names.map((n) => this.bodyObject(n)).filter((o): o is THREE.Object3D => !!o);
    const hit = raycaster.intersectObjects(targets, true)[0];
    if (!hit) return null;
    let o: THREE.Object3D | null = hit.object;
    while (o && !targets.includes(o)) o = o.parent;
    return o ? { name: o.name, point: hit.point } : null;
  }

  /** Where a ray (in the renderer's world) meets the table plane, in MuJoCo world coordinates. */
  rayOnTable(origin: THREE.Vector3, dir: THREE.Vector3, out: THREE.Vector3): THREE.Vector3 | null {
    const inv = new THREE.Matrix4().copy(this.root.matrixWorld).invert();
    const o = origin.clone().applyMatrix4(inv), d = dir.clone().transformDirection(inv);
    if (Math.abs(d.z) < 1e-6) return null;
    const t = (this.tableZ - o.z) / d.z;
    return t > 0 && t < 5 ? out.copy(o).addScaledVector(d, t) : null;
  }
}
