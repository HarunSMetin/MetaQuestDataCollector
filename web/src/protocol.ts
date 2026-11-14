// Wire protocol, client side. Every constant, size and offset comes from protocol.gen.ts, which is generated from
// xrcollect/wire/schema.json (like the Python side): change the layout there and run `xrc codegen`.

import {
  BODY_FIELDS, HAND_FIELDS, HAND_FLOATS, HANDS_OFFSET, INPUT_BYTES, INPUT_HEADER_BYTES, INPUT_HEADER_OFFSETS,
  MSG_GHOST, MSG_INPUT, MSG_POSE, N_BUTTONS, N_HAND_JOINTS, POSE_HEADER_BYTES, POSE_HEADER_OFFSETS, SRC_NONE, VERSION,
} from "./protocol.gen";

export * from "./protocol.gen";

export interface HandSample {
  source: number;
  pose: Float32Array; // 7: pos xyz, quat xyzw
  buttons: Float32Array; // N_BUTTONS
  joints: Float32Array; // 25*7
  pinch: Float32Array; // 2
}

export function emptyHand(): HandSample {
  const pose = new Float32Array(7);
  pose[6] = 1;
  return { source: SRC_NONE, pose, buttons: new Float32Array(N_BUTTONS), joints: new Float32Array(N_HAND_JOINTS * 7), pinch: new Float32Array([0.1, 0.1]) };
}

export class InputPacker {
  private buf = new ArrayBuffer(INPUT_BYTES);
  private view = new DataView(this.buf);
  private floats = new Float32Array(this.buf, INPUT_HEADER_BYTES);
  private seq = 0;

  /** tEcho: t_server of the last pose frame received (the server measures the round trip from it). */
  pack(head: Float32Array, left: HandSample, right: HandSample, tEcho: number): ArrayBuffer {
    const v = this.view, H = INPUT_HEADER_OFFSETS;
    v.setUint8(H.type, MSG_INPUT);
    v.setUint8(H.src_left, left.source);
    v.setUint8(H.src_right, right.source);
    v.setUint8(H.version, VERSION);
    v.setUint32(H.seq, this.seq++ >>> 0, true);
    v.setFloat64(H.t_echo_ms, tEcho, true);
    const f = this.floats;
    f.set(head, BODY_FIELDS.head.offset);
    [left, right].forEach((h, i) => {
      const o = HANDS_OFFSET + i * HAND_FLOATS;
      f.set(h.pose, o + HAND_FIELDS.pose.offset);
      f.set(h.buttons, o + HAND_FIELDS.buttons.offset);
      f.set(h.joints, o + HAND_FIELDS.joints.offset);
      f.set(h.pinch, o + HAND_FIELDS.pinch.offset);
    });
    return this.buf;
  }
}

export interface PoseFrame {
  ghost: boolean; // a reference take's bodies (operator-only overlay), not the live scene
  flags: number;  // bit i: arm i engaged; FLAG_RECORDING
  nbody: number;
  seq: number;
  tServer: number;
  bodies: Float32Array; // nbody*7
}

export function parsePoseFrame(buf: ArrayBuffer): PoseFrame | null {
  const v = new DataView(buf), H = POSE_HEADER_OFFSETS;
  const type = v.getUint8(H.type);
  if (type !== MSG_POSE && type !== MSG_GHOST) return null;
  const nbody = v.getUint16(H.nbody, true);
  const f = new Float32Array(buf, POSE_HEADER_BYTES);
  return { ghost: type === MSG_GHOST, flags: v.getUint8(H.flags), nbody, seq: v.getUint32(H.seq, true),
    tServer: v.getFloat64(H.t_server_ms, true), bodies: f.subarray(0, nbody * 7) };
}

// ---- JSON messages ----------------------------------------------------------------------

export type RecState = "idle" | "countdown" | "recording" | "review";
type PerArm<T> = Record<string, T>;

export interface Review {
  duration_s: number;
  success: boolean;
  idle_pct: number;
  events: Record<string, number>; // contact kind -> times it happened
  safety_events: number;
  progress: number;
  stages: number;
  rejected: string | null; // why a dataset would skip it
  used_s: number; // seconds a dataset would use
  tags: string[]; // the task's tag set
}

export interface Status {
  type: "status";
  task: string;
  task_version: string;
  scene_version: number;
  instruction: string;
  arms: string[];
  mode: "controllers" | "hands";
  teleop_control: "clutched" | "absolute" | "rate";
  mapping: Record<string, string>;        // hand ("l" | "r") -> arm
  hand_of: PerArm<"l" | "r" | null>;      // arm -> the hand that drives it
  scale: number;
  rec_state: RecState;
  countdown: number;
  elapsed: number;
  steps: number;
  auto_stop: boolean;
  success: boolean;
  success_seen: boolean;
  progress: number;       // stages done, counted in order
  stages: boolean[];      // each stage's predicate
  held: PerArm<string | null>;
  next: { object: string; arm: string | null } | null;
  /** Where the next object belongs (world) and how far it is from there: operator-only ghost + readout. */
  guide: { object: string; pos: [number, number, number]; quat: [number, number, number, number]; dist_mm: number; xy_mm: number; z_mm: number } | null;
  operator: string;
  session: { name: string; saved: number; discarded: number; target: number; elapsed_s: number };
  engaged: PerArm<boolean>;
  tracked: PerArm<boolean>;
  mode_mismatch: boolean;
  sources: { l: number; r: number };
  grippers: PerArm<number>;
  ik_error_mm: PerArm<number>;
  out_of_reach: PerArm<boolean>;
  ee_pos: PerArm<[number, number, number]>;
  warnings: { arm_collision: boolean; table_collision: boolean; ik_saturated: boolean; blocked: boolean };
  input_hz: number;
  rtt_ms: number | null;
  loop_hz: number;
  connected: boolean;
  placement: number[]; // column-major 4x4, world -> xr
  table_offset: number;
  control: { mode: "teleop" | "policy" | "shared"; driver: "operator" | "policy"; policy: string | null; ref: string | null;
    loading: string | null; error: string | null; policy_s: number; safety_rad: number };
  review: Review | null;
  edit: { moved: string[]; file: string } | null;
  ghost: { path: string; i: number; n: number } | null;
  branch: { of: string; step: number; frozen: boolean } | null;
  replay: { path: string; i: number; n: number; fps: number; paused: boolean; speed: number; loop: boolean; foreign: boolean } | null;
}

export type ServerEvent =
  | { type: "toast"; text: string; kind: "info" | "success" | "warn" | "rec" }
  | { type: "haptic"; arm: string; hand: "l" | "r" | null; intensity: number; ms: number; repeat?: number; gap_ms?: number }
  | { type: "saved"; path: string }
  | { type: "sound"; name: string; arm: string | null; hand: "l" | "r" | null }
  | { type: "scene_changed"; version: number; task: string; task_version: string }
  | { type: "edit_error"; text: string }
  | { type: "ghost"; on: boolean; path?: string }
  | { type: "error"; text: string };

export interface Region { center: [number, number]; half: [number, number]; tape?: boolean }
export interface TaskObject { name: string; label: string; size: [number, number, number]; rgba: number[]; shape: string;
  studs: [number, number] | null; fixed: boolean; pitch?: number | null; stud_r?: number | null; stud_h?: number | null }
/** The mission as data (GET /api/task). The UI draws the goal, the stage list and the hints from this alone. */
export interface TaskSpec {
  task: string;
  version: string;
  instruction: string;
  objects: TaskObject[];
  regions: Record<string, Region>;
  spawn: Record<string, Region>;
  stages: { name: string; label: string; object: string | null }[];
  goal_view: { anchor: string; show: { object: string; pose: { pos?: number[]; yaw?: number } }[] } | null;
  tags: string[];
  arms: string[];
  table_height: number;
}

/** One row of the server's button table (GET /api/commands): what a controller button does. */
export interface ButtonHint {
  hand: "l" | "r";
  button: "trigger" | "squeeze" | "primary" | "secondary" | "stick" | "menu";
  command: string | null;
  label: string;
  labels?: Partial<Record<RecState, string>>; // label per recorder state
  when?: RecState[];                          // lit only in these states
  client?: "menu" | "clutch" | "gripper";     // handled without a server command
}

export async function getJSON<T>(url: string): Promise<T> {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}
export const fetchTask = () => getJSON<TaskSpec>("/api/task");
export const fetchButtons = () => getJSON<ButtonHint[]>("/api/commands");

export interface Annotation { start: number; stop: number; label: string; kind?: string }
export interface EpisodeRow {
  path: string;
  file: string;
  success: boolean; // the label: recorded_success unless changed by hand
  recorded_success: boolean;
  duration_s: number;
  num_steps: number;
  input_mode: string;
  operator: string;
  session: string | null;
  created_at: string;
  source: string;
  task: string;
  task_version: string;
  seed: number;
  branched_from: string | null;
  branch_step: number | null;
  generated_from: string | null;
  retimed_from: string | null;
  tags: string[];
  note: string;
  excluded: boolean;
  quality: number;
  annotations: Annotation[];
}
export interface DatasetRow { name: string; recording: boolean; count: number; title: string; description: string }
export type EpisodeEdit = Partial<{ tags: string[]; note: string; exclude: boolean; success: boolean | null; quality: number;
  annotations: Annotation[] }>;
export type AuditRows = Record<string, {
  contact_frac: Record<string, number>;
  contact_events: Record<string, number>;
  idle_frac: number;
  stage_times: (number | null)[];
  fps: number;
  num_steps: number;
  safety_events: number;
  phases: Annotation[];
  success: boolean;
  verdict: string | null; // why a dataset would leave it out; null = used
  spans: [number, number][];
  task_version: string | null;
}>;
