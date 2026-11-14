// Dashboard end-to-end test: the Scene tab (click-select, gizmo drag, Apply, inspector, regions, capture goal) and
// the Data tab (health tiles, chart tooltips, export preview, an export job), in headless Chromium.
//
//   Same setup as emulated-quest.mjs (scratch task copy via --task, scratch record.out_dir), then
//   npm run test:dashboard                                            # XRC_URL=http://host:port if not :8000
//
// The Scene checks rewrite the task file and the export writes <out_dir>/../exports/e2e_*: both run only when the
// server's task file is outside this repository (a scratch setup), and are skipped otherwise.
import { chromium } from "playwright";
import { mkdirSync } from "node:fs";
import { resolve } from "node:path";
const HERE = import.meta.dirname;
const OUT = `${HERE}/out`;
mkdirSync(OUT, { recursive: true });
const BASE = process.env.XRC_URL ?? "http://localhost:8000";
const getJSON = async (p) => (await fetch(`${BASE}${p}`)).json();
const status = () => getJSON("/api/status");
const taskfile = () => getJSON("/api/taskfile").catch(() => null);
const cmd = (name, args = {}) => fetch(`${BASE}/api/command`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name, args }) });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let failures = 0;
const check = (ok, msg) => { console.log(`${ok ? "PASS" : "FAIL"}  ${msg}`); if (!ok) failures++; };
/** Poll `get` until `pred` holds (or time out); returns the last value. */
const until = async (get, pred, ms = 8000) => {
  let v = await get();
  for (let t = 0; t < ms && !pred(v); t += 200) { await sleep(200); v = await get(); }
  return v;
};

// Start from a clean state: an aborted run may have left edit mode or a replay on.
await cmd("edit_mode", { enabled: false });
await cmd("replay_stop");

const browser = await chromium.launch({ args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"] });
const page = await browser.newPage({ viewport: { width: 1500, height: 950 } });
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));
page.on("console", (m) => { if (m.type() === "error" && !m.text().includes("404")) errors.push(m.text()); });
page.setDefaultTimeout(10000);
await page.goto(`${BASE}/dashboard/`);
await page.waitForFunction(() => !!window.__xrcViewer?.mj.bodyObject, null, { timeout: 15000 });
await sleep(1500);  // the scene arrives after the websocket connects
const tab = (name) => page.locator(".tabs").getByRole("button", { name }).click();

/** Screen position of a world point given in the viewer's three.js frame. */
const toScreen = (pts) => page.evaluate((pts) => {
  const v = window.__xrcViewer, r = v.renderer.domElement.getBoundingClientRect();
  return pts.map(([x, y, z]) => {
    const q = v.proxy.position.clone().set(x, y, z).project(v.camera);
    return { x: r.left + (q.x + 1) / 2 * r.width, y: r.top + (1 - q.y) / 2 * r.height };
  });
}, pts);
const bodyPos = (name) => page.evaluate((n) => {
  const v = window.__xrcViewer, p = v.mj.bodyObject(n)?.getWorldPosition(v.proxy.position.clone());
  return p ? [p.x, p.y, p.z] : null;
}, name);

/** Wait until the Scene tab shows this task file version (forms reset when the file reloads). */
const showsVersion = (v) => page.locator(".mono", { hasText: `version ${v.slice(0, 8)}` }).waitFor().then(() => sleep(300));

// ---------------------------------------------------------------------------------------------------- Scene tab
let f0 = await taskfile();
const REPO = resolve(HERE, "../..");
const scratch = !!f0 && f0.path.startsWith("/") && !resolve(f0.path).startsWith(REPO + "/");
const HOME = [0.5, 0.05];  // the first object's default pose during this test: in view, away from the edges
if (!scratch) {
  console.log(`SKIP  scene and export checks: the server edits ${f0?.path ?? "no task file"} (serve a copy with --task)`);
} else {
  const first = Object.keys(f0.data.objects)[0];
  const atHome = (f) => JSON.stringify(f.data.objects[first].pos) === JSON.stringify(HOME);
  if (!atHome(f0)) {
    await cmd("edit", { ops: [{ op: "object_pose", object: first, pos: HOME, yaw_deg: 0 }] });
    const v0 = f0.version;  // the text changes on disk first, the version once the runtime has reloaded
    f0 = await until(taskfile, (f) => atHome(f) && f.version !== v0);
  }
  await tab("Scene");
  await showsVersion(f0.version);
  await page.getByRole("button", { name: /Edit mode/ }).click();
  let s = await until(status, (x) => !!x.edit);
  check(!!s.edit, "edit mode on from the Scene tab");
  // The page learns about edit mode from the next status message, then the objects jump to their default poses.
  await page.getByRole("button", { name: /Done editing/ }).waitFor();
  await sleep(1000);

  const objs = Object.keys(f0.data.objects);
  const [a, b] = objs;
  const pa = await bodyPos(a);
  const [c] = await toScreen([pa]);
  await page.mouse.click(c.x, c.y);
  const sel = await until(() => page.locator(".obj.sel b").textContent({ timeout: 500 }).catch(() => null), (v) => v === a, 3000);
  check(sel === a, `click in the 3D view selects ${a} (${sel})`);
  await page.screenshot({ path: `${OUT}/dash_scene_selected.png` });

  // Drag the gizmo's X arrow (three +x = MuJoCo -y): grab it part way along, pull outwards.
  const p0 = await bodyPos(a);
  const [o, e] = await toScreen([p0, [p0[0] + 0.08, p0[1], p0[2]]]);
  const g = { x: o.x + (e.x - o.x) * 0.6, y: o.y + (e.y - o.y) * 0.6 };
  await page.mouse.move(g.x, g.y);
  await sleep(200);
  await page.mouse.down();
  for (let i = 1; i <= 10; i++) { await page.mouse.move(g.x + (e.x - o.x) * i * 0.1, g.y + (e.y - o.y) * i * 0.1); await sleep(40); }
  await page.mouse.up();
  s = await until(status, (x) => x.edit?.moved.includes(a), 3000);
  check(!!s.edit?.moved.includes(a), `gizmo drag moves ${a} (moved=${s.edit?.moved.join(",")})`);

  const apply = page.getByRole("button", { name: /Apply moves/ });
  await until(() => apply.isEnabled(), (v) => v, 3000);
  if (await apply.isEnabled()) await apply.click();
  else await cmd("edit_commit");  // the drag failed (reported above): carry on with the rest of the checks
  const f1 = await until(taskfile, (f) => f.version !== f0.version);
  const y0 = f0.data.objects[a].pos?.[1], y1 = f1.data.objects[a].pos?.[1];
  check(f1.version !== f0.version && Array.isArray(f1.data.objects[a].pos), `Apply moves writes ${a}.pos (${JSON.stringify(f1.data.objects[a].pos)}, y ${y0} -> ${y1})`);

  // Inspector: a physical property through the schema form.
  await showsVersion(f1.version);
  const row = page.locator(".obj", { hasText: a });
  if (!(await page.locator(".inspector").count())) await row.click();
  const mass = +(f1.data.objects[a].mass ?? 0.05) === 0.07 ? "0.06" : "0.07";
  await page.locator(".inspector label", { hasText: "mass kg" }).locator("input").fill(mass);
  await page.locator(".inspector").getByRole("button", { name: "Apply" }).click();
  const f2 = await until(taskfile, (f) => String(f.data.objects[a].mass) === mass && f.version !== f1.version);
  check(String(f2.data.objects[a].mass) === mass, `inspector sets ${a}.mass = ${f2.data.objects[a].mass}`);

  // Regions: add one from the Region schema's defaults.
  await showsVersion(f2.version);
  const nReg = Object.keys(f2.data.regions ?? {}).length;
  const regions = page.locator(".map-ed", { hasText: "Regions" });
  await regions.getByRole("button", { name: "+ Add" }).click();
  await regions.getByRole("button", { name: "Add", exact: true }).click();
  const f3 = await until(taskfile, (f) => Object.keys(f.data.regions ?? {}).length === nReg + 1 && f.version !== f2.version);
  const added = Object.keys(f3.data.regions ?? {}).filter((k) => !(k in (f2.data.regions ?? {})));
  check(added.length === 1, `region added (${added.join(",")})`);
  if (added.length === 1) {  // leave the copy as it was, so reruns add the same name
    await cmd("edit", { ops: [{ op: "region_remove", name: added[0] }] });
    await until(taskfile, (f) => !(added[0] in (f.data.regions ?? {})) && f.version !== f3.version);
  }

  // Capture goal: stack b on a in edit mode, capture from the panel.
  if (b) {
    const v = (await taskfile()).version;
    await until(status, (x) => x.task_version === v && !!x.edit);
    const pos = (await taskfile()).data.objects[a].pos;
    await cmd("edit_move", { object: b, pos: [pos[0], pos[1], 0.79], yaw: 0, free_z: true });
    await until(status, (x) => x.edit?.moved.includes(b), 3000);
    await page.getByRole("button", { name: /Capture goal/ }).click();
    const f4 = await until(taskfile, (f) => (f.data.stages ?? []).some((x) => String(x.name).endsWith("_placed")) && !!f.data.goal_view);
    check((f4.data.stages ?? []).some((x) => String(x.name).endsWith("_placed")) && !!f4.data.goal_view,
      `Capture goal writes stages + goal_view (${(f4.data.stages ?? []).map((x) => x.name).join(",")})`);
    await page.locator(".col-side").evaluate((el) => el.scrollTo(0, 99999));
    await sleep(300);
    await page.screenshot({ path: `${OUT}/dash_scene_stages.png` });
  }
  await page.getByRole("button", { name: /Done editing/ }).click();
  s = await until(status, (x) => !x.edit);
  check(!s.edit, "Done editing leaves edit mode");
  // Put the dragged object (and its start region, which moves along) back, so reruns start from the same place.
  await cmd("edit", { ops: [{ op: "object_pose", object: a, pos: HOME, yaw_deg: 0 }] });
}

// ----------------------------------------------------------------------------------------------------- Data tab
await tab("Data");
const tiles = await until(() => page.locator(".tile").allInnerTexts(), (t) => t.length >= 4);
check(tiles.length >= 4 && /Takes/i.test(tiles.join(" ")), `health tiles (${tiles.map((t) => t.replace(/\s+/g, " ")).join(" | ")})`);
const n = await page.locator(".pt").count();
if (n) {
  const bb = await page.locator(".pt").first().boundingBox();
  await page.mouse.move(bb.x + bb.width / 2, bb.y + bb.height / 2);
  const tip = await until(() => page.locator(".viz-tip").innerText({ timeout: 500 }).catch(() => ""), (t) => !!t.trim(), 2000);
  check(!!tip.trim(), `coverage point tooltip ("${tip.replace(/\s+/g, " ").slice(0, 60)}")`);
} else {
  console.log("SKIP  chart tooltip: no recorded takes in this dataset");
}
const preview = await until(() => page.locator(".preview").innerText({ timeout: 500 }).catch(() => ""), (t) => /segment/.test(t), 5000);
check(/segment/.test(preview), `export preview (${preview.replace(/\s+/g, " ").slice(0, 80)})`);
await page.screenshot({ path: `${OUT}/dash_data.png` });

const exportBtn = page.locator(".export").getByRole("button", { name: /Export \d+ segment/ });
const mixed = page.locator(".export label.inline input");
if (scratch && await mixed.count()) await mixed.check();  // scratch data spans several task versions
if (scratch && await exportBtn.count() && await exportBtn.isEnabled()) {
  for (const box of await page.locator(".export .checks label input").all()) {
    const label = await box.evaluate((el) => el.closest("label")?.textContent ?? "");
    if (/^cam_/.test(label.trim()) && await box.isChecked()) await box.uncheck();  // no cameras: a fast export
  }
  const name = `e2e_${Date.now() % 100000}`;
  await page.locator(".export input[placeholder]").first().fill(name);
  await exportBtn.click();
  const job = await until(() => page.locator(".job").innerText({ timeout: 500 }).catch(() => ""), (t) => /done|failed|exports\//i.test(t), 90000);
  check(/exports\//.test(job) && !/failed/i.test(job), `export job finishes (${job.replace(/\s+/g, " ").slice(0, 90)})`);
  await page.locator(".col-side").evaluate((el) => el.scrollTo(0, 99999));
  await sleep(200);
  await page.screenshot({ path: `${OUT}/dash_export.png` });
} else if (scratch) {
  console.log("SKIP  export job: nothing selectable to export");
}

check(errors.length === 0, `no page errors${errors.length ? ": " + errors.join(" | ") : ""}`);
await browser.close();
console.log(failures ? `\n${failures} check(s) failed` : "\nall checks passed");
process.exit(failures ? 1 : 0);
