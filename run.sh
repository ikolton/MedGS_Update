#!/usr/bin/env bash
# SCRIPT LAST TESTED: 19.09.2025 on wsl2 Ubuntu 22.04 + conda and cuda 11.7
# MedGS setup & pipeline runner for WSL + conda
# Follows the README steps. CUDA auto-detection removed — set TORCH_CUDA_CHANNEL below.
# Run from the repo root (where train.py and requirements.txt live).

set -euo pipefail

# -------- Defaults (override via CLI flags) --------
ENV_NAME="medgs"          # name of the conda environment that will be created/used
PY_VER="3.8"              # python version for the conda environment

DO_INSTALL=1              # 1 = install dependencies

DATASET_DIR="./data/prostate"            # path to your training dataset (folder with original/0000.png etc.)
MODEL_DIR="./output/prostate"              # path where training outputs (model checkpoints, renderings) will be saved

PIPELINE_MODE="seg"          # "img" (default) or "seg" — type of training/rendering pipeline
RENDER_INTERP="8"         # interpolation factor during rendering (1 = no interpolation, 2 = double frames, etc.)

TRAIN_USE_SEG=1          # 1 = train on segmentation pipeline (binary masks), 0 = normal images
TRAIN_RANDOM_BG=0         # 1 = randomize background, 0 = keep dataset background
TRAIN_POLY_DEGREE=""      # polynomial degree for folded gaussians (empty = default)
TRAIN_BATCH_SIZE=""       # training batch size (empty = default)

MESH_INPUT="./output"             # parent directory containing case/model subfolders with seg/render/*.png
MESH_OUTPUT="./output/mesh"            # directory where .ply meshes will be saved
MESH_THRESH="0"         # iso-level threshold for marching cubes
INTER="8"               # interpolation factor for mesh generation (1 = no interpolation, 2 = double frames, etc.)

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
