// README screenshots (docs/media/) from a running server, using the emulated Quest 3 for the in-headset view.
// Point the server at a scratch out_dir with a few takes in it (the Episodes and Data tabs show them), e.g.
//
//   uv run xrc validate-task <scratch>/task --n 8 --save <scratch>/data/episodes
//   uv run xrc serve --no-https --config <yaml with record.out_dir: <scratch>/data/episodes> --task <scratch>/task
//   XRC_URL=http://localhost:<port> npm run screenshots
//
// It records one short take (to show the review card) and discards it.
import { mkdirSync } from "node:fs";
import { chromium } from "playwright";
const OUT = `${import.meta.dirname}/../../docs/media`;
const BASE = (process.env.XRC_URL ?? process.env.QC_URL) ?? "http://localhost:8000";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const command = (name, args = {}) => fetch(`${BASE}/api/command`, {
  method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify({ name, args }),
});
const status = async () => (await fetch(`${BASE}/api/status`)).json();
mkdirSync(OUT, { recursive: true });

const browser = await chromium.launch({ args: ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"] });

// Desktop dashboard: one screenshot per tab.
const dash = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
await dash.goto(`${BASE}/dashboard/`);
await sleep(6000); // scene + camera previews
await dash.screenshot({ path: `${OUT}/dashboard.png` });
const tab = (name) => dash.locator(".tabs").getByRole("button", { name }).click();
for (const [name, file] of [["Episodes", "dash_episodes"], ["Data", "dash_data"]]) {
  await tab(name);
  await sleep(4000);
  await dash.screenshot({ path: `${OUT}/${file}.png` });
}
// Scene tab in edit mode with the first object selected (gizmo showing). Nothing is applied: the task file stays.
await tab("Scene");
await dash.getByRole("button", { name: /Edit mode/ }).click();
await dash.getByRole("button", { name: /Done editing/ }).waitFor();
await sleep(1500);
const first = await dash.evaluate(async () => Object.keys((await (await fetch("/api/taskfile")).json()).data.objects)[0]);
const at = await dash.evaluate((n) => {
  const v = window.__xrcViewer, r = v.renderer.domElement.getBoundingClientRect();
  const q = v.mj.bodyObject(n).getWorldPosition(v.proxy.position.clone()).project(v.camera);
  return { x: r.left + (q.x + 1) / 2 * r.width, y: r.top + (1 - q.y) / 2 * r.height };
}, first);
await dash.mouse.click(at.x, at.y);
await sleep(1500);
await dash.screenshot({ path: `${OUT}/dash_scene.png` });
await dash.getByRole("button", { name: /Discard/ }).click().catch(() => {});
await command("edit_mode", { enabled: false });
await tab("Record");

// Quest Browser landing page and the in-headset view.
const xr = await browser.newPage({ viewport: { width: 1280, height: 800 } });
await xr.addInitScript({ path: `${import.meta.dirname}/iwer.bundle.js` });
await xr.goto(`${BASE}/xr/`);
await xr.waitForFunction(() => !document.getElementById("enter-vr").disabled, null, { timeout: 15000 });
await sleep(3000);
await xr.screenshot({ path: `${OUT}/xr_landing.png` });

await xr.evaluate(() => {
  const d = window.__xr;
  d.position.set(0, 1.6, 0);
  d.quaternion.set(-0.2164, 0, 0, 0.9763); // look ~25° down at the table
  d.controllers.left.position.set(-0.2, 1.3, -0.4);
  d.controllers.right.position.set(0.2, 1.3, -0.4);
});
await xr.click("#enter-vr");
await sleep(6000); // auto-recenter runs 1.2 s after entering; controller models load
// Software GL sends input at only a few Hz, so hold each button long enough for the server to see it.
const press = async (side, button) => {
  await xr.evaluate(([s, b]) => window.__xr.controllers[s].updateButtonValue(b, 1), [side, button]);
  await sleep(1500);
  await xr.evaluate(([s, b]) => window.__xr.controllers[s].updateButtonValue(b, 0), [side, button]);
};
await press("right", "squeeze"); // take control of the right arm: glow, status chip, brighter grasp-point marker
await sleep(4000);
await xr.screenshot({ path: `${OUT}/xr_headset.png` });

await press("left", "y-button"); // wrist menu
await sleep(1500);
await xr.screenshot({ path: `${OUT}/xr_menu.png` });
await press("left", "y-button");
await press("right", "squeeze"); // release the arm

// A short take, then the review card.
await command("record_toggle");
for (let i = 0; i < 40 && (await status()).rec_state !== "recording"; i++) await sleep(250);
await sleep(3000);
await command("record_toggle");
for (let i = 0; i < 40 && (await status()).rec_state !== "review"; i++) await sleep(250);
await sleep(2000);
await xr.screenshot({ path: `${OUT}/xr_review.png` });
await command("discard");

await browser.close();
console.log(`screenshots written to ${OUT}`);
