#!/usr/bin/env bash
# One-step setup from a fresh clone: uv + Python 3.12 + Python deps, Node (only to build the web client), the
# Franka model, the web build, then `xrc doctor`. Safe to re-run: every step skips what is already there.
#
#   ./scripts/install.sh                 # the basics
#   ./scripts/install.sh --lerobot       # + LeRobot exporter (brings torch, a few GB)
#   ./scripts/install.sh --mesh          # + `xrc import-mesh`
#   ./scripts/install.sh --e2e           # + headset-free end-to-end tests (Playwright + Chromium)
#   ./scripts/install.sh --all --yes     # everything, no questions
#
# Nothing needs sudo. uv is installed per user (~/.local/bin) with its official installer if missing; Node is
# downloaded into .tools/node inside the repo if the system one is missing or too old. System libraries for
# offscreen rendering (EGL) are only checked: the script prints the command if they are missing.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS="$ROOT/.tools"
NODE_LINE="22"  # LTS line downloaded when no usable Node is found

EXTRAS=() E2E=0 YES=0 DOCTOR=1
usage() { sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; }
for a in "$@"; do
  case "$a" in
    --lerobot) EXTRAS+=(lerobot) ;;
    --mesh) EXTRAS+=(mesh) ;;
    --e2e) E2E=1 ;;
    --all) EXTRAS=(lerobot mesh); E2E=1 ;;
    -y|--yes) YES=1 ;;
    --no-doctor) DOCTOR=0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $a" >&2; usage; exit 2 ;;
  esac
done

if [ -t 1 ]; then B=$'\e[1m' G=$'\e[32m' Y=$'\e[33m' R=$'\e[31m' N=$'\e[0m'; else B="" G="" Y="" R="" N=""; fi
step() { printf '\n%s==> %s%s\n' "$B" "$*" "$N"; }
ok()   { printf '  %s✓%s %s\n' "$G" "$N" "$*"; }
warn() { printf '  %s!%s %s\n' "$Y" "$N" "$*"; }
die()  { printf '%s✗ %s%s\n' "$R" "$*" "$N" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }
ask()  {  # ask "question" -> 0 for yes; --yes or no terminal = yes
  [ "$YES" = 1 ] || [ ! -t 0 ] && return 0
  read -r -p "  $1 [Y/n] " r; [[ -z "$r" || "$r" =~ ^[Yy] ]]
}
fetch() {  # fetch URL FILE
  if have curl; then curl -fsSL "$1" -o "$2"; elif have wget; then wget -q "$1" -O "$2"; else die "need curl or wget"; fi
}

cd "$ROOT"
export npm_config_update_notifier=false npm_config_fund=false npm_config_audit=false

# ---------------------------------------------------------------------------------------------------------------
step "System"
OS="$(uname -s)" ARCH="$(uname -m)"
case "$OS" in
  Linux) ok "Linux $ARCH" ;;
  Darwin) warn "macOS: not supported (offscreen rendering needs EGL); continuing anyway" ;;
  *) die "unsupported OS: $OS (xrcollect runs on Linux)" ;;
esac
for t in git tar; do have "$t" || die "$t is required (e.g. sudo apt install $t)"; done
have curl || have wget || die "curl or wget is required (e.g. sudo apt install curl)"
ok "git, tar, $(have curl && echo curl || echo wget)"

# ---------------------------------------------------------------------------------------------------------------
step "uv (Python package manager; it also installs Python 3.12)"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if have uv; then
  ok "uv $(uv --version | awk '{print $2}')"
else
  ask "uv not found. Install it for this user with the official installer (https://astral.sh/uv)?" \
    || die "uv is required: https://docs.astral.sh/uv/getting-started/installation/"
  tmp="$(mktemp)"; fetch https://astral.sh/uv/install.sh "$tmp"; sh "$tmp"; rm -f "$tmp"
  have uv || die "uv installed but not on PATH; open a new shell and re-run"
  ok "uv $(uv --version | awk '{print $2}') (open a new shell, or add ~/.local/bin to PATH, to use it later)"
fi

# ---------------------------------------------------------------------------------------------------------------
step "Node.js (only to build the web client)"
node_ok() {  # Vite needs Node 20.19+ or 22.12+
  have node || return 1
  local v maj min; v="$(node -v 2>/dev/null | tr -d v)"; maj="${v%%.*}"; min="$(echo "$v" | cut -d. -f2)"
  [ "$maj" -gt 22 ] || { [ "$maj" -eq 22 ] && [ "$min" -ge 12 ]; } || { [ "$maj" -eq 20 ] && [ "$min" -ge 19 ]; }
}
[ -x "$TOOLS/node/bin/node" ] && export PATH="$TOOLS/node/bin:$PATH"
if node_ok; then
  ok "node $(node -v) ($(command -v node))"
else
  have node && warn "node $(node -v) is too old (need 20.19+ or 22.12+)"
  case "$OS-$ARCH" in
    Linux-x86_64) plat=linux-x64 ;;  Linux-aarch64|Linux-arm64) plat=linux-arm64 ;;
    Darwin-arm64) plat=darwin-arm64 ;; Darwin-x86_64) plat=darwin-x64 ;;
    *) die "no Node download for $OS $ARCH: install Node 22 yourself (https://nodejs.org)" ;;
  esac
  have xz || [ "$OS" = Darwin ] || die "xz is required to unpack Node (e.g. sudo apt install xz-utils)"
  ask "Download Node $NODE_LINE LTS into .tools/node (inside this repo, nothing system-wide)?" \
    || die "Node 20.19+ or 22.12+ is required: https://nodejs.org"
  base="https://nodejs.org/dist/latest-v${NODE_LINE}.x"
  tmp="$(mktemp -d)"
  fetch "$base/SHASUMS256.txt" "$tmp/SHASUMS256.txt"
  file="$(grep -o "node-v[0-9.]*-$plat\.tar\.xz" "$tmp/SHASUMS256.txt" | head -1)"
  [ -n "$file" ] || die "could not find a Node build for $plat at $base"
  fetch "$base/$file" "$tmp/$file"
  (cd "$tmp" && grep " $file\$" SHASUMS256.txt | { have sha256sum && sha256sum -c - || shasum -a 256 -c -; }) >/dev/null \
    || die "Node download failed its checksum"
  rm -rf "$TOOLS/node"; mkdir -p "$TOOLS/node"
  tar -xJf "$tmp/$file" -C "$TOOLS/node" --strip-components=1
  rm -rf "$tmp"
  export PATH="$TOOLS/node/bin:$PATH"
  node_ok || die "downloaded Node does not run"
  ok "node $(node -v) in .tools/node"
fi

# ---------------------------------------------------------------------------------------------------------------
step "Robot model (Franka Panda from MuJoCo Menagerie)"
"$ROOT/scripts/fetch_assets.sh" | sed 's/^/  /'

# ---------------------------------------------------------------------------------------------------------------
step "Python environment (.venv)"
sync=(uv sync --quiet)
for e in ${EXTRAS[@]+"${EXTRAS[@]}"}; do sync+=(--extra "$e"); done
"${sync[@]}"
ok "${sync[*]}"

# ---------------------------------------------------------------------------------------------------------------
step "Web client (web/dist)"
(cd web && npm ci --no-audit --no-fund --loglevel=error && npm run build --silent) >/dev/null
ok "built web/dist"

if [ "$E2E" = 1 ]; then
  step "End-to-end test tools (web/e2e)"
  (cd web/e2e && npm ci --no-audit --no-fund --loglevel=error && npx playwright install chromium && npm run build --silent) >/dev/null
  ok "Playwright + Chromium + IWER bundle"
  warn "if Chromium fails to start, install its system libraries: cd web/e2e && sudo env PATH=\"\$PATH\" npx playwright install-deps chromium"
fi

# ---------------------------------------------------------------------------------------------------------------
if [ "$DOCTOR" = 1 ]; then
  step "Checking the environment (xrc doctor)"
  out="$(uv run xrc doctor 2>&1 | grep -v -i "eglerror\|^Exception ignored\|^Traceback\|^  File \|^    \|egl.*error" || true)"
  echo "$out"
  if echo "$out" | grep -q "✗ offscreen rendering"; then
    warn "offscreen rendering failed. On Debian/Ubuntu: sudo apt install libegl1 libgl1 libgles2"
    warn "no GPU? software rendering works too: sudo apt install libosmesa6, then run with MUJOCO_GL=osmesa"
  fi
fi

cat <<EOF

${G}${B}Done.${N} Next:
  uv run xrc serve                          # start the server (task from config.yaml)
  http://localhost:8000/dashboard/          # desktop dashboard
  uv run xrc setup-quest                    # USB headset: then open http://localhost:8000/xr/ in Quest Browser
  uv run xrc certs                          # Wi-Fi headset: then open the https address xrc serve prints
  uv run xrc validate-task tasks/stack_two_blocks   # no headset: watch the scripted expert solve a task
EOF
[ -d "$TOOLS/node" ] && printf '\nNode was installed into .tools/node. To rebuild the web client later:\n  PATH="$PWD/.tools/node/bin:$PATH" npm --prefix web run build\n'

exit 0
