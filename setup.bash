#!/usr/bin/env bash

set -euo pipefail

# -------- settings you can tweak ----------
PYTHON_BIN="${PYTHON_BIN:-python3}"

export SCRATCH="$(pwd)"
export VENV_DIR="$SCRATCH/venvs/robolens"
export CACHE_DIR="$VENV_DIR/.cache"
export TMPDIR="$VENV_DIR/.tmp"

mkdir -p "$VENV_DIR"  "$CACHE_DIR/pip" "$CACHE_DIR/torch" "$CACHE_DIR/huggingface" "$TMPDIR"


python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"


python -m pip install --upgrade pip setuptools wheel


export PIP_CACHE_DIR="$CACHE_DIR/pip"
export XDG_CACHE_HOME="$CACHE_DIR"          
export TORCH_HOME="$CACHE_DIR/torch"        
export HF_HOME="$CACHE_DIR/huggingface"    
export CUDA_CACHE_PATH="$CACHE_DIR/nv"     
export TMPDIR                              
export PYTHONNOUSERSITE=1

TRANSFORMERS_VER="${TRANSFORMERS_VER:->=4.45,<5.0}"
ACCELERATE_VER="${ACCELERATE_VER:->=0.33}"
TORCH_VER_PIN="${TORCH_VER_PIN:-2.4.*}"       
TV_VER_PIN="${TV_VER_PIN:-0.19.*}"            


echo "==> Using python: ${PYTHON_BIN}"
echo "==> Venv dir:    ${VENV_DIR}"
echo "==> HF cache:     ${CACHE_DIR}"


echo "==> Detecting CUDA..."


CUDA_WHL_INDEX="https://download.pytorch.org/whl/cu121"
echo "==> Installing Torch (CUDA wheels from ${CUDA_WHL_INDEX})"
if command -v nvidia-smi >/dev/null 2>&1 || command -v nvcc >/dev/null 2>&1; then
  if command -v nvcc >/dev/null 2>&1 && nvcc --version | grep -q "release 12"; then
    CUDA_WHL_INDEX="https://download.pytorch.org/whl/cu121"
  elif command -v nvcc >/dev/null 2>&1 && nvcc --version | grep -q "release 11"; then
    CUDA_WHL_INDEX="https://download.pytorch.org/whl/cu118"
  else
   
    CUDA_WHL_INDEX="https://download.pytorch.org/whl/cu121"
  fi
  echo "==> Installing Torch (CUDA wheels from ${CUDA_WHL_INDEX})"
  pip install --index-url "${CUDA_WHL_INDEX}" \
    "torch==${TORCH_VER_PIN}" "torchvision==${TV_VER_PIN}" "torchaudio==2.4.*"
else
  echo "==> No CUDA found; installing CPU Torch"
  pip install "torch==${TORCH_VER_PIN}" "torchvision==${TV_VER_PIN}" "torchaudio==2.4.*"
fi


echo "==> Installing Python dependencies"
pip install \
  "transformers${TRANSFORMERS_VER}" \
  "accelerate${ACCELERATE_VER}" \
  "pillow>=10.0.0" \
  "safetensors>=0.4.2" \
  "sentencepiece>=0.2.0" \
  "protobuf<5" \
  "numpy>=1.24" \
  "regex>=2024.0" \
  "timm>=0.9.0" \
  "huggingface-hub>=0.24.0" \
  "hf-transfer>=0.1.6"


echo "==> Done."
