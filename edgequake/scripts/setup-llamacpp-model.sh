#!/usr/bin/env bash
# Setup llama.cpp model directory for VLM-OCR backend.
#
# Symlinks a GGUF model from the HuggingFace cache (or a custom path) into
# the directory mounted by the llamacpp Docker service.
#
# Usage:
#   ./scripts/setup-llamacpp-model.sh [path-to-model.gguf]
#
# Default model: GLM-OCR Q5_K_M from ~/.cache/huggingface

set -euo pipefail

TARGET_DIR="${LLAMACPP_MODEL_DIR:-$HOME/.cache/edgequake/llamacpp-models}"

# Default: GLM-OCR GGUF from HF cache
DEFAULT_MODEL="$HOME/.cache/huggingface/hub/models--mradermacher--GLM-OCR-GGUF/snapshots/3c1e642c0fa5df64831f0b04f3c674b57ce341af/GLM-OCR.Q5_K_M.gguf"

MODEL_PATH="${1:-$DEFAULT_MODEL}"

if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at: $MODEL_PATH"
    echo ""
    echo "To download GLM-OCR:"
    echo "  hf download mradermacher/GLM-OCR-GGUF GLM-OCR.Q5_K_M.gguf"
    echo ""
    echo "Or specify a custom model path:"
    echo "  $0 /path/to/your/model.gguf"
    exit 1
fi

MODEL_FILENAME="$(basename "$MODEL_PATH")"

mkdir -p "$TARGET_DIR"

# Create symlink (or update if already exists)
LINK_PATH="$TARGET_DIR/$MODEL_FILENAME"
if [ -L "$LINK_PATH" ] || [ -f "$LINK_PATH" ]; then
    echo "Updating: $LINK_PATH -> $MODEL_PATH"
    rm -f "$LINK_PATH"
fi

ln -s "$MODEL_PATH" "$LINK_PATH"
echo "Linked: $LINK_PATH -> $MODEL_PATH"

echo ""
echo "Model directory ready: $TARGET_DIR"
echo "  Model: $MODEL_FILENAME"
echo ""
echo "To use with docker-compose, set:"
echo "  LLAMACPP_MODEL=$MODEL_FILENAME"
echo "  LLAMACPP_MODEL_DIR=$TARGET_DIR"
