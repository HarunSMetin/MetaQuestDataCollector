import { useEffect, useState } from "react";
import type { Status } from "../protocol";
import type { Cmd } from "./util";

/** Policy tab: load a PolicyRunner by reference (module:Class or path/to/file.py:Class), run it, take over, record. */

const REF_KEY = "xrc.policyRef";
const ARGS_KEY = "xrc.policyArgs";
const remembered = (k: string, d: string) => { try { return localStorage.getItem(k) ?? d; } catch { return d; } };
const remember = (k: string, v: string) => { try { localStorage.setItem(k, v); } catch { /* private window */ } };

/** The loaded policy and who drives: run / stop / shared, and a fresh scene to start from. */
export function PolicyControl({ status, cmd }: { status: Status | null; cmd: Cmd }) {
  const c = status?.control;
  const [ref, setRef] = useState(() => remembered(REF_KEY, ""));
  const [args, setArgs] = useState(() => remembered(ARGS_KEY, ""));
  const [seed, setSeed] = useState("");
  // Prefill with what the server was started with (--policy), if nothing was typed yet.
  useEffect(() => { if (!ref && c?.ref) setRef(c.ref); }, [c?.ref]); // eslint-disable-line react-hooks/exhaustive-deps
  let parsed: Record<string, unknown> | null = {};
  try { parsed = args.trim() ? JSON.parse(args) : {}; } catch { parsed = null; }
  const argsOk = parsed !== null && typeof parsed === "object" && !Array.isArray(parsed);
  const rec = status?.rec_state ?? "idle";
  const busy = rec === "recording" || rec === "countdown";
  const load = () => {
    remember(REF_KEY, ref.trim());
    remember(ARGS_KEY, args);
    cmd("load_policy", { ref: ref.trim(), args: parsed ?? {} });
  };
  return (
    <div className="card policy-ctl">
      <div className="card-head"><h2>Policy</h2>
        {c?.policy && <button className="btn ghost small" title="Free the GPU and drive by hand again" onClick={() => cmd("unload_policy")}>⏏ Unload</button>}
      </div>
      {c?.loading && <div className="alert info"><span className="spin" /> Loading {c.loading}…</div>}
      {c?.error && !c.loading && <div className="alert warn">Couldn't load: {c.error}</div>}
      {c?.policy ? (
        <>
          <div className="loaded">
            <b>{c.policy}</b>
            {c.ref && <span className="muted small">{c.ref}</span>}
          </div>
          <div className="seg three">
            {(["policy", "shared", "teleop"] as const).map((m) => (
              <button key={m} className={c.mode === m ? "active" : ""} onClick={() => cmd("set_control_mode", { mode: m })}
                title={{ policy: "The policy drives all arms", shared: "The policy drives until you press Grip (DAgger)", teleop: "You drive; the policy waits" }[m]}>
                {{ policy: "▶ Run", shared: "🤝 Shared", teleop: "■ Stop" }[m]}
              </button>
            ))}
          </div>
          <div className="policy-live">
            <span>driving: <b className={c.driver === "policy" ? "acc" : ""}>{c.driver === "policy" ? "policy" : "you"}</b></span>
            <span>policy time <b>{(c.policy_s ?? 0).toFixed(1)} s</b></span>
            <span title="How much the safety filter changed the last joint targets">safety Δ <b>{c.safety_rad.toFixed(3)}</b> rad</span>
          </div>
        </>
      ) : (
        !c?.loading && <p className="muted">No policy loaded. Enter a runner below and press <b>Load</b>.</p>
      )}
      <div className="load-row">
        <label>Runner
          <input type="text" placeholder="my_pkg.policies:Runner or path/to/file.py:Runner" value={ref} spellCheck={false}
            onChange={(e) => setRef(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter" && ref.trim() && argsOk) load(); }} />
        </label>
        <label title="Keyword arguments for the runner's constructor, as a JSON object">Args
          <input type="text" placeholder='{"checkpoint": "…"}' value={args} spellCheck={false} className={argsOk ? "" : "bad"}
            onChange={(e) => setArgs(e.target.value)} />
        </label>
        <button className="btn small ok" disabled={!ref.trim() || !argsOk || !!c?.loading} onClick={load}>
          {c?.policy && c.ref === ref.trim() ? "↻ Reload" : "Load"}
        </button>
      </div>
      <div className="scene-row">
        <button className="btn small" disabled={busy} onClick={() => cmd("reset")} title="New random layout">↺ New scene</button>
        <span className="muted small">or seed</span>
        <input type="number" min={0} value={seed} placeholder="–" onChange={(e) => setSeed(e.target.value)} />
        <button className="btn small" disabled={busy || seed === ""} onClick={() => cmd("reset", { seed: Math.max(0, Math.floor(+seed)) })}
          title="Reset to the layout of this seed, to compare runs on the same start">Load</button>
      </div>
      <p className="muted small hint">The scene and the policy restart together. Every action goes through the safety filter. Record below to keep a rollout (or your takeover) as an episode.</p>
    </div>
  );
}
