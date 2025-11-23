#!/usr/bin/env bash
set -euo pipefail

# =======================
# PATHS & DEFAULTS
# =======================
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEIGHTS_DIR="$PROJECT_ROOT/weights"
DINO_DIR="$WEIGHTS_DIR/groundingdino"

ENV_NAME="${ENV_NAME:-lavis}"
PY="${PYTHON_BIN:-python}"
HF_TOKEN="${HF_TOKEN:-}"

DF_PATH="${DF_PATH:-$PROJECT_ROOT/images.csv}"
IMG_DIR="${IMG_DIR:-$PROJECT_ROOT}"
SAVE_DIR="${SAVE_DIR:-$PROJECT_ROOT/outputs}"
SAVE_FILENAME="${SAVE_FILENAME:-preds.jsonl}"

START_INDEX="${START_INDEX:-0}"
INTERVAL="${INTERVAL:-0}"         # 0 = all rows
SAVE_EVERY="${SAVE_EVERY:-50}"
STATE_PATH="${STATE_PATH:-$SAVE_DIR/state.json}"

mkdir -p "$WEIGHTS_DIR" "$DINO_DIR" "$SAVE_DIR"

echo "==> PROJECT_ROOT: $PROJECT_ROOT"
echo "==> DINO_DIR:     $DINO_DIR"

# =======================
# CONDA (best-effort)
# =======================
echo "==> Activating conda env '$ENV_NAME'"
if [ -f "/c/Users/$USERNAME/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "/c/Users/$USERNAME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
conda activate "$ENV_NAME" || echo "NOTE: proceeding without explicit conda activation; ensure '$PY' is correct."

# =======================
# TOOLS
# =======================
echo "==> Installing Hugging Face tools (safe to re-run)"
$PY -m pip -q install --upgrade huggingface_hub hf_transfer || true

# =======================
# HF LOGIN (optional; cached login is fine)
# =======================
if [[ -n "$HF_TOKEN" ]]; then
  echo "==> Logging into Hugging Face with HF_TOKEN (non-interactive)"
  "$PY" -c "from huggingface_hub import login; import os; login(os.environ['HF_TOKEN'])"
else
  echo "NOTE: HF_TOKEN not set; using cached HF login if available."
fi

# =======================
# WINDOWS PATH FOR PYTHON
# =======================
DINO_DIR_WIN="$DINO_DIR"
if command -v cygpath >/dev/null 2>&1; then
  DINO_DIR_WIN="$(cygpath -w "$DINO_DIR")"
fi
echo "==> Python will write to: $DINO_DIR_WIN"

# =======================
# DOWNLOAD: your myDino files
# =======================
echo "==> Downloading GroundingDINO (myDino) to $DINO_DIR"
"$PY" - <<PY
from huggingface_hub import hf_hub_download
import os, shutil

dst = r"${DINO_DIR_WIN}"
os.makedirs(dst, exist_ok=True)

files = [
  ("AlexieLinardatos/myDino", "GroundingDINO/weights/GroundingDINO_SwinB_cfg.py", "GroundingDINO_SwinB_cfg.py"),
  ("AlexieLinardatos/myDino", "GroundingDINO/weights/groundingdino_swinb_cogcoor.pth", "groundingdino_swinb_cogcoor.pth"),
]

for repo, remote, local in files:
    fp = hf_hub_download(repo_id=repo, filename=remote, local_dir=dst)
    out = os.path.join(dst, local)
    if fp != out:
        try:
            shutil.move(fp, out)
        except Exception:
            # if already there or cross-device weirdness, copy then remove
            shutil.copy2(fp, out)
            try:
                os.remove(fp)
            except Exception:
                pass
    print(f"Downloaded {remote} -> {out} (exists={os.path.exists(out)})")
PY

# =======================
# VERIFY
# =======================
echo "==> Verifying DINO files in: $DINO_DIR"
ls -lah "$DINO_DIR" || true

DINO_CFG="$DINO_DIR/GroundingDINO_SwinB_cfg.py"
DINO_CKPT="$DINO_DIR/groundingdino_swinb_cogcoor.pth"

if [[ ! -f "$DINO_CFG" ]]; then
  echo "Missing DINO cfg at: $DINO_CFG"
  echo "Contents:"
  ls -lah "$DINO_DIR" || true
  exit 1
fi
if [[ ! -f "$DINO_CKPT" ]]; then
  echo "Missing DINO checkpoint at: $DINO_CKPT"
  echo "Contents:"
  ls -lah "$DINO_DIR" || true
  exit 1
fi

echo "Using DINO cfg:   $DINO_CFG"
echo "Using DINO ckpt:  $DINO_CKPT"

# =======================
# RUN INFERENCE
# =======================
echo "==> Running infer.py"
"$PY" "$PROJECT_ROOT/infer.py" \
  --target_checkpoint "$DINO_CKPT" \
  --save_dir "$SAVE_DIR" \
  --save_filename "$SAVE_FILENAME" \
  --start_index "$START_INDEX" \
  --interval "$INTERVAL" \
  --df_path "$DF_PATH" \
  --image_dir "$IMG_DIR" \
  --state_path "$STATE_PATH" \
  --save_every "$SAVE_EVERY"

echo "==> Done."
