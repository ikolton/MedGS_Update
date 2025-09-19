#!/usr/bin/env bash
# SCRIPT LAST TESTED: 19.09.2025 on wsl2 Ubuntu 22.04 + conda and cuda 11.7
# MedGS setup & pipeline runner for WSL + conda
# Follows the README steps. CUDA auto-detection removed — set TORCH_CUDA_CHANNEL below.
# Run from the repo root (where train.py and requirements.txt live).

set -euo pipefail

# -------- Defaults (override via CLI flags) --------
ENV_NAME="medgs"          # name of the conda environment that will be created/used
PY_VER="3.8"              # python version for the conda environment

DO_INSTALL=1              # 1 = install dependencies, 0 = skip (useful if already installed)

DATASET_DIR="./data/prostate"            # path to your training dataset (folder with original/0000.png etc.)
MODEL_DIR="./output/prostate"              # path where training outputs (model checkpoints, renderings) will be saved

PIPELINE_MODE="seg"          # "img" (default) or "seg" — type of training/rendering pipeline
RENDER_INTERP="1"         # interpolation factor during rendering (1 = no interpolation, 2 = double frames, etc.)

TRAIN_USE_SEG=1          # 1 = train on segmentation pipeline (binary masks), 0 = normal images
TRAIN_RANDOM_BG=0         # 1 = randomize background, 0 = keep dataset background
TRAIN_POLY_DEGREE=""      # polynomial degree for folded gaussians (empty = default)
TRAIN_BATCH_SIZE=""       # training batch size (empty = default)

MESH_INPUT="./output"             # parent directory containing case/model subfolders with seg/render/*.png
MESH_OUTPUT="./output/mesh"            # directory where .ply meshes will be saved
MESH_THRESH="150"         # iso-level threshold for marching cubes
INTER="1"               # interpolation factor for mesh generation (1 = no interpolation, 2 = double frames, etc.)

# ---- PyTorch CUDA wheel channel (manual) ----
# Valid options (per PyTorch download site):
#   cu118  → for CUDA 11.8 (works fine with toolkit 11.7, too)
#   cu121  → for CUDA 12.1–12.3
#   cu124  → for CUDA 12.4+
TORCH_CUDA_CHANNEL="cu118"
CUDA_MIN_REQUIRED="11.7"

# -------- Helpers --------
die() { echo "Error: $*" >&2; exit 1; }
warn() { echo "Warning: $*" >&2; }

require_at_repo_root() {
  [[ -f "train.py" ]] || die "Run this script from the repo root (train.py not found)."
  [[ -f "requirements.txt" ]] || die "requirements.txt not found."
}

ensure_conda_shell() {
  # shellcheck disable=SC1091
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
  elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  else
    die "conda not found in PATH. Install Miniconda/Anaconda and retry."
  fi
}

create_env_if_missing() {
  if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Creating conda env '$ENV_NAME' (python=$PY_VER)..."
    conda create -y -n "$ENV_NAME" "python=$PY_VER"
  else
    echo "Conda env '$ENV_NAME' already exists."
  fi
}

activate_env() {
  conda activate "$ENV_NAME"
}

install_deps() {
  echo "Upgrading pip..."
  python -m pip install --upgrade pip

  # Validate channel
  case "$TORCH_CUDA_CHANNEL" in
    cu118|cu121|cu124)
      ;;
    cpu)
      warn "Using CPU-only wheels. Training/render speed will be limited."
      ;;
    *)
      die "Unknown TORCH_CUDA_CHANNEL: '$TORCH_CUDA_CHANNEL' (use cu118|cu121|cu124|cpu)"
      ;;
  esac

  local index_url
  if [[ "$TORCH_CUDA_CHANNEL" == "cpu" ]]; then
    index_url="https://download.pytorch.org/whl/cpu"
  else
    index_url="https://download.pytorch.org/whl/${TORCH_CUDA_CHANNEL}"
  fi

  echo "Installing PyTorch + torchvision from: $index_url  (min CUDA required: $CUDA_MIN_REQUIRED)"
  python -m pip install torch torchvision --index-url "$index_url"

  echo "Installing submodules..."
  python -m pip install submodules/diff-gaussian-rasterization
  python -m pip install submodules/simple-knn

  echo "Installing additional requirements..."
  python -m pip install -r requirements.txt

  echo "Dependency install complete."
}

# -------- Pipeline steps --------
run_training() {
  [[ -n "$DATASET_DIR" ]] || die "Training requested but --dataset not provided."
  [[ -n "$MODEL_DIR"   ]] || die "Training requested but --model-out not provided."
  mkdir -p "$MODEL_DIR"

  cmd=(python3 train.py -s "$DATASET_DIR" -m "$MODEL_DIR")
  if [[ $TRAIN_USE_SEG -eq 1 ]]; then
    cmd+=(--pipeline seg)
  fi
  if [[ $TRAIN_RANDOM_BG -eq 1 ]]; then
    cmd+=(--random_background)
  fi
  if [[ -n "$TRAIN_POLY_DEGREE" ]]; then
    cmd+=(--poly_degree "$TRAIN_POLY_DEGREE")
  fi
  if [[ -n "$TRAIN_BATCH_SIZE" ]]; then
    cmd+=(--batch_size "$TRAIN_BATCH_SIZE")
  fi

  echo ">>> Running training:"
  printf ' %q' "${cmd[@]}"; echo
  "${cmd[@]}"
}

run_render() {
  [[ -n "$MODEL_DIR" ]] || die "Rendering requested but --model-out/--model-dir not provided."

  local pipeline="img"
  if [[ -n "$PIPELINE_MODE" ]]; then
    pipeline="$PIPELINE_MODE"
  fi

  cmd=(python3 render.py --model_path "$MODEL_DIR" --interp "$RENDER_INTERP" --pipeline "$pipeline")

  echo ">>> Rendering frames:"
  printf ' %q' "${cmd[@]}"; echo
  "${cmd[@]}"

  echo "Rendered images saved to: $MODEL_DIR/render"
}

run_mesh() {
  [[ -n "$MESH_INPUT"  ]] || die "Mesh requested but --mesh-input not provided."
  [[ -n "$MESH_OUTPUT" ]] || die "Mesh requested but --mesh-output not provided."

  cmd=(python3 slices_to_ply.py --input "$MESH_INPUT" --output "$MESH_OUTPUT" --thresh "$MESH_THRESH" --inter "$INTER")

  echo ">>> Creating mesh (.ply) via marching cubes:"
  printf ' %q' "${cmd[@]}"; echo
  "${cmd[@]}"

  echo "Meshes saved to: $MESH_OUTPUT"
}

print_usage() {
  cat <<'USAGE'
MedGS runner (WSL + conda). Run from the repo root.

Set CUDA wheel channel at the top of the script:
  TORCH_CUDA_CHANNEL=cu118   # (default) works for CUDA >= 11.7
  # or cu121 / cu124 / cpu

Basic install only:
  ./run_medgs.sh

Install with custom env name / Python:
  ./run_medgs.sh --env-name medgs --python 3.8

End-to-end (install → train → render → mesh):
  ./run_medgs.sh \
    --dataset /path/to/data \
    --model-out /path/to/out_model \
    --render-interp 8 \
    --pipeline img \
    --mesh-input /path/with/case_subfolders \
    --mesh-output /path/to/meshes \
    --mesh-thresh 150

Segmentation training + randomized background + custom hyperparams:
  ./run_medgs.sh \
    --dataset /data/cardiac_pngs \
    --model-out /outputs/medgs_case1 \
    --seg \
    --random-background \
    --poly-degree 3 \
    --batch-size 4 \
    --render-interp 8 \
    --pipeline seg \
    --mesh-input /outputs \
    --mesh-output /outputs/ply \
    --mesh-thresh 150

Skip installation if already installed:
  ./run_medgs.sh --skip-install ...other flags...

Flags:
  --env-name NAME           Conda env name (default: medgs)
  --python X.Y              Python version for env (default: 3.8)
  --skip-install            Don’t (re)install dependencies

Training:
  --dataset PATH            Dataset directory (frames: original/{0000.png,...}, mirror/)
  --model-out PATH          Model output directory
  --seg                     Use segmentation pipeline (adds: --pipeline seg)
  --random-background       Randomize background during training
  --poly-degree N           Polynomial degree for folded Gaussians
  --batch-size N            Training batch size

Rendering:
  --render-interp N         Interpolation multiplier (default: 1)
  --pipeline img|seg        Render mode; default img

Mesh:
  --mesh-input PATH         Parent directory with case subfolders (each containing seg/render/*.png)
  --mesh-output PATH        Output directory for <case>.ply meshes
  --mesh-thresh N           Marching cubes iso-level (default: 150)
USAGE
}

# -------- Parse CLI --------
if [[ $# -eq 0 ]]; then
  print_usage
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name) ENV_NAME="${2:?}"; shift 2 ;;
    --python) PY_VER="${2:?}"; shift 2 ;;
    --skip-install) DO_INSTALL=0; shift ;;

    --dataset) DATASET_DIR="${2:?}"; shift 2 ;;
    --model-out|--model-dir) MODEL_DIR="${2:?}"; shift 2 ;;
    --seg) TRAIN_USE_SEG=1; shift ;;
    --random-background) TRAIN_RANDOM_BG=1; shift ;;
    --poly-degree) TRAIN_POLY_DEGREE="${2:?}"; shift 2 ;;
    --batch-size) TRAIN_BATCH_SIZE="${2:?}"; shift 2 ;;

    --render-interp) RENDER_INTERP="${2:?}"; shift 2 ;;
    --pipeline) PIPELINE_MODE="${2:?}"; shift 2 ;;

    --mesh-input) MESH_INPUT="${2:?}"; shift 2 ;;
    --mesh-output) MESH_OUTPUT="${2:?}"; shift 2 ;;
    --mesh-thresh) MESH_THRESH="${2:?}"; shift 2 ;;

    -h|--help) print_usage; exit 0 ;;
    *) die "Unknown option: $1" ;;
  esac
done

# -------- Execute --------
require_at_repo_root
ensure_conda_shell
create_env_if_missing
activate_env

if [[ $DO_INSTALL -eq 1 ]]; then
  install_deps
else
  echo "Skipping dependency install as requested."
fi

# If dataset/model provided, run training.
if [[ -n "$DATASET_DIR" && -n "$MODEL_DIR" ]]; then
  run_training
fi

# If model provided, run rendering.
if [[ -n "$MODEL_DIR" ]]; then
  run_render
fi

# If mesh inputs provided, run mesh.
if [[ -n "$MESH_INPUT" && -n "$MESH_OUTPUT" ]]; then
  run_mesh
fi

echo "All requested steps completed."
