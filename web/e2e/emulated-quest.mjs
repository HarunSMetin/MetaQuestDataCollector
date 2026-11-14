// End-to-end test without a headset: an emulated Quest 3 (Meta's IWER runtime) in headless Chromium
// drives the real WebXR page -> websocket -> teleop -> IK -> MuJoCo, and checks the server's status.
//
//   cd web/e2e && npm install && npx playwright install chromium && npm run build
//   cp -r ../../tasks/stack_two_blocks /tmp/xrc_e2e_task         # the edit-mode checks rewrite the task file
//   uv run xrc serve --no-https --config <config with a scratch record.out_dir> --task /tmp/xrc_e2e_task   # in another shell
//   npm test                                                         # XRC_URL=http://host:port if not :8000
//
// NOTE: it records a few short episodes into the server's record.out_dir. The edit-mode checks run only when the
// server's task file is outside this repository (they are skipped otherwise, so tasks/ is never rewritten).
import { chromium } from "playwright";
import { mkdirSync } from "node:fs";
import { resolve } from "node:path";
const HERE = import.meta.dirname;
const OUT = `${HERE}/out`;
mkdirSync(OUT, { recursive: true });
const BASE = (process.env.XRC_URL ?? process.env.QC_URL) ?? "http://localhost:8000";
const status = async () => { const s = await (await fetch(`${BASE}/api/status`)).json(); lastHz = s.input_hz; return s; };
const cmd = (name, args = {}) => fetch(`${BASE}/api/command`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name, args }) });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let failures = 0;
let lastHz = 0;
const check = (ok, msg) => { console.log(`${ok ? "PASS" : "FAIL"}  ${msg}   [input ${lastHz} Hz]`); if (!ok) failures++; };

const browser = await chromium.launch({ args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] });
const page = await browser.newPage({ viewport: { width: 1280, height: 720 } });
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));
page.on("console", (m) => { if (m.type() === "error" && !m.text().includes("404")) errors.push(m.text()); });
await page.addInitScript({ path: `${HERE}/iwer.bundle.js` });
await cmd("set_mode", { mode: "controllers" });
await page.goto(`${BASE}/xr/`);
await page.waitForFunction(() => !document.getElementById("enter-vr").disabled, null, { timeout: 15000 });
await page.evaluate(() => {
  const d = window.__xr;
  d.position.set(0, 1.6, 0);
  d.controllers.left.position.set(-0.2, 1.1, -0.35);
  d.controllers.right.position.set(0.2, 1.1, -0.35);
});
// Software GL can't render the Quest-3 scene at real-time rates; hide it so the emulated frame loop
// (and therefore the input stream) runs near headset rate. This test checks the input pipeline.
await page.evaluate(() => { const q = window.__xrc; q.renderer.xr.setFramebufferScaleFactor(0.12); q.renderer.shadowMap.enabled = false; q.mj.root.visible = false; });
await page.click("#enter-vr");
// Wait for the input stream (slow under software GL on a busy machine), then let the filters settle.
let s = await status();
for (let t = 0; t < 15000 && !(s.sources.l === 1 && s.sources.r === 1 && s.input_hz > 20); t += 250) { await sleep(250); s = await status(); }
await sleep(500);
s = await status();
check(s.sources.l === 1 && s.sources.r === 1, `controllers seen by server (sources=${JSON.stringify(s.sources)})`);
check(s.tracked.l && s.tracked.r, "both arms tracked");
check(s.input_hz > 20, `input stream rate ${s.input_hz} Hz`);
check(s.rtt_ms !== null && s.rtt_ms < 100, `server-measured RTT ${s.rtt_ms} ms`);

// Grip press (and release) toggles the right arm on; move controller 10 cm right and 10 cm down.
const r0 = s.ee_pos.r;
const tapGrip = async () => {
  await page.evaluate(() => window.__xr.controllers.right.updateButtonValue("squeeze", 1));
  await sleep(150);
  await page.evaluate(() => window.__xr.controllers.right.updateButtonValue("squeeze", 0));
  await sleep(250);
};
await tapGrip();
s = await status();
check(s.engaged.r && !s.engaged.l, "grip press engages only the right arm, and it stays engaged");
for (let i = 1; i <= 20; i++) {
  await page.evaluate((k) => window.__xr.controllers.right.position.set(0.2 + 0.005 * k, 1.1 - 0.005 * k, -0.35), i);
  await sleep(40);
}
await sleep(1200);
s = await status();
const d = s.ee_pos.r.map((v, i) => v - r0[i]);
// XR +x (right) -> world -y ; XR -y (down) -> world -z
check(d[1] < -0.07 && d[2] < -0.07 && Math.abs(d[0]) < 0.03, `right EE followed controller: Δworld=${d.map((v) => v.toFixed(3))}`);
await page.evaluate(() => window.__xr.controllers.right.updateButtonValue("trigger", 1));
await sleep(600);
s = await status();
check(s.grippers[s.mapping.r] < 0.05, `trigger closes right gripper (cmd=${s.grippers[s.mapping.r]})`);
await page.evaluate(() => window.__xr.controllers.right.updateButtonValue("trigger", 0));
await tapGrip();
s = await status();
check(!s.engaged.r, "second grip press disengages");

// Record with A, stop with A, save with X.
const press = async (side, btn) => {
  await page.evaluate(([sd, b]) => window.__xr.controllers[sd].updateButtonValue(b, 1), [side, btn]);
  await sleep(150);
  await page.evaluate(([sd, b]) => window.__xr.controllers[sd].updateButtonValue(b, 0), [side, btn]);
  await sleep(150);
};
const before = (await (await fetch(`${BASE}/api/episodes`)).json()).length;
await press("right", "a-button");
await sleep(300);
check((await status()).rec_state === "countdown", "A starts countdown");
await sleep(3300);
check((await status()).rec_state === "recording", "countdown -> recording");
await sleep(1500);
await press("right", "a-button");
check((await status()).rec_state === "review", "A stops -> review");
await press("left", "x-button");
await sleep(1500);
const eps = await (await fetch(`${BASE}/api/episodes`)).json();
check(eps.length === before + 1, `X saves episode (${eps.length} total) -> ${eps.at(-1)?.file} ${eps.at(-1)?.duration_s}s mode=${eps.at(-1)?.input_mode}`);
await page.screenshot({ path: `${OUT}/xr_session_controllers.png` });

// Hand tracking mode.
await cmd("set_mode", { mode: "hands" });
await page.evaluate(() => { window.__xr.primaryInputMode = "hand"; });
await sleep(1200);
s = await status();
check(s.mode === "hands" && s.sources.l === 2 && s.sources.r === 2, `hands seen by server (sources=${JSON.stringify(s.sources)})`);
check(s.tracked.l && s.tracked.r && !s.mode_mismatch, "hands tracked, no mode mismatch");
await page.evaluate(() => window.__xr.hands.right.position.set(0.15, 1.0, -0.3));  // place hand before engaging
await sleep(500);
const rh0 = (await status()).ee_pos.r;
await page.evaluate(() => { window.__xr.hands.right.poseId = "point"; });  // middle, ring, little finger curled
await sleep(500);
await page.evaluate(() => { window.__xr.hands.right.poseId = "default"; });
await sleep(500);
s = await status();
check(s.engaged.r, "curling three fingers toggles right arm engaged");
await page.evaluate(() => window.__xr.hands.right.updatePinchValue(1));
await sleep(800);
s = await status();
check(s.grippers[s.mapping.r] < 0.3, `index pinch closes gripper (cmd=${s.grippers[s.mapping.r]})`);
for (let i = 1; i <= 20; i++) {
  await page.evaluate((k) => window.__xr.hands.right.position.set(0.15, 1.0, -0.3 - 0.005 * k), i);
  await sleep(40);
}
await sleep(1200);
s = await status();
const dh = s.ee_pos.r.map((v, i) => v - rh0[i]);
// hand moved 10 cm forward (XR -z) -> world +x
check(dh[0] > 0.07 && Math.abs(dh[1]) < 0.03 && Math.abs(dh[2]) < 0.03, `right EE follows tracked hand: Δworld=${dh.map((v) => v.toFixed(3))}`);
await page.screenshot({ path: `${OUT}/xr_session_hands.png` });

// ---- headset panels: pressed through each panel's own widgets (the same handlers a ray + Trigger runs) ----
const pressWidget = async (where, id) => {
  const r = await page.evaluate(([where, id]) => {
    const x = window.__xrc;
    const panel = where === "menu" ? x.menu.panel : x[where].panel;
    const all = where === "menu" ? Object.values(x.menu.pages).flat() : panel.widgets;
    const w = all.find((v) => v.id === id);
    return !w ? "missing" : panel.press(w) ? "ok" : "disabled";
  }, [where, id]);
  await sleep(300);  // panels ignore presses closer than 250 ms
  return r;
};
/** A panel's visibility, waited for: the page learns about state changes from the next status message (~10 Hz). */
const visible = async (where, want = true, ms = 1500) => {
  for (let t = 0; t < ms; t += 100) {
    const v = await page.evaluate((w) => window.__xrc[w].panel.mesh.visible, where);
    if (v === want) return v;
    await sleep(100);
  }
  return !want;
};
const waitFor = async (pred, ms = 4000) => { for (let t = 0; t < ms; t += 100) { const s = await status(); if (pred(s)) return s; await sleep(100); } return status(); };

// Review card: record a short take, label it on the card, save.
const nBefore = (await (await fetch(`${BASE}/api/episodes`)).json()).length;
await cmd("record_toggle");
s = await waitFor((x) => x.rec_state === "recording", 5000);
await sleep(1200);
await cmd("record_toggle");
s = await waitFor((x) => x.rec_state === "review");
await sleep(300);
check(s.rec_state === "review" && await visible("review"), "review card shows after a take");
check(s.review && typeof s.review.used_s === "number", `review summary from the server (rejected=${s.review?.rejected ?? "no"})`);
const pr = [await pressWidget("review", "star-4"), await pressWidget("review", "ok"), await pressWidget("review", "save")];
s = await waitFor((x) => x.rec_state === "idle");
const epsAfter = await (await fetch(`${BASE}/api/episodes`)).json();
const saved = epsAfter.at(-1);
check(pr.every((v) => v === "ok") && epsAfter.length === nBefore + 1, `review card Save writes the take (${pr.join(",")})`);
check(saved?.quality === 4 && saved?.success === true, `review labels go to curation (quality=${saved?.quality}, success=${saved?.success})`);
check(!(await visible("review", false)), "review card hides after saving");

// Replay timeline: wrist menu -> replay last take, step, play, close.
check(await pressWidget("menu", "replay-last") === "ok", "menu: replay last take");
s = await waitFor((x) => !!x.replay);
await sleep(300);
check(!!s.replay && s.replay.paused && await visible("timeline"), `timeline opens on a paused replay (${s.replay?.n ?? 0} steps)`);
const i0 = s.replay?.i ?? 0;
await pressWidget("timeline", "step+"); await pressWidget("timeline", "step+");
s = await status();
check(s.replay && s.replay.i === i0 + 2, `timeline steps frame by frame (i ${i0} -> ${s.replay?.i})`);
await pressWidget("timeline", "play");
s = await waitFor((x) => x.replay && !x.replay.paused, 1500);
check(s.replay && !s.replay.paused, "timeline play");
await pressWidget("timeline", "stop");
s = await waitFor((x) => !x.replay);
check(!s.replay && !(await visible("timeline", false)), "timeline close ends the replay");

// Edit mode (rewrites the task file: only against a scratch copy).
const tf = async () => (await fetch(`${BASE}/api/taskfile`)).json().catch(() => null);
const f0 = await tf();
const REPO = resolve(HERE, "../..");
if (!f0 || resolve(f0.path).startsWith(REPO + "/") || !f0.path.startsWith("/")) {
  console.log(`SKIP  edit mode checks: the server edits ${f0?.path ?? "no task file"} (serve a copy with --task, see the top of this file)`);
} else {
  check(await pressWidget("menu", "edit") === "ok", "menu: edit scene");
  s = await waitFor((x) => !!x.edit);
  await sleep(300);
  check(!!s.edit && await visible("edit"), "edit mode: physics paused, editor panel shown");
  const objs = Object.keys(f0.data.objects);
  // A grab-and-carry is the editor's edit_move stream; then the panel's Apply writes it.
  // A pose different from the file's (a rerun against the same copy must still change it).
  const target = Math.abs((f0.data.objects[objs[0]].pos ?? [])[0] - 0.5) < 1e-3 ? [0.48, 0.03] : [0.5, 0.05];
  await cmd("edit_move", { object: objs[0], pos: target, yaw: 0.3 });
  s = await waitFor((x) => x.edit?.moved.length === 1);
  check(s.edit?.moved[0] === objs[0], `object moved in edit mode (${s.edit?.moved.join(",")})`);
  await sleep(300);  // the panel learns about the move from the next status (~10 Hz)
  const applied = await pressWidget("edit", "apply");
  // The file is written first; the version changes once the runtime has rebuilt the scene (slow under load).
  let f1 = await tf();
  for (let t = 0; t < 10000 && f1.version === f0.version; t += 250) { await sleep(250); f1 = await tf(); }
  const p1 = f1.data.objects[objs[0]].pos ?? [];
  check(f1.version !== f0.version && Math.abs(p1[0] - target[0]) < 1e-3 && Math.abs(p1[1] - target[1]) < 1e-3, `Apply writes the pose into the task file (${applied}, pos=${JSON.stringify(p1)})`);
  if (objs.length > 1) {
    await cmd("edit_move", { object: objs[1], pos: [...target, 0.79], yaw: 0.3, free_z: true });
    await sleep(300);
    await pressWidget("edit", "capture");
    await sleep(1500);
    const f2 = await tf();
    check((f2.data.stages ?? []).some((x) => String(x.name).endsWith("_placed")) && f2.data.goal_view, `capture goal writes stages + goal view (${(f2.data.stages ?? []).map((x) => x.name).join(",")})`);
  }
  await pressWidget("edit", "done");
  s = await waitFor((x) => !x.edit);
  check(!s.edit && !(await visible("edit", false)), "Done leaves edit mode");
}

// Switch back to controllers while hands are tracked -> mismatch warning.
await cmd("set_mode", { mode: "controllers" });
s = await waitFor((x) => x.mode_mismatch, 5000);  // a single sample can land on a stale-input tick under load
check(s.mode_mismatch, "mode mismatch flagged when hands are used in controller mode");

check(errors.length === 0, `no page errors${errors.length ? ": " + errors.join(" | ") : ""}`);
await browser.close();
console.log(failures ? `\n${failures} check(s) failed` : "\nall checks passed");
process.exit(failures ? 1 : 0);
