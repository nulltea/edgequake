#!/usr/bin/env bash
# Pre-download OAR-OCR ONNX models to the host cache directory mounted into
# the edgequake container (docker/docker-compose.yml). Running this before
# first use avoids a ~2.5 GB download on the first PDF upload with the OAR-OCR
# backend — and keeps the models persisted across container rebuilds.
#
# Override target with EDGEQUAKE_OAR_OCR_MODEL_DIR (default: ~/.edgequake/oar-ocr-models).

set -euo pipefail

DEST="${EDGEQUAKE_OAR_OCR_MODEL_DIR:-${HOME}/.edgequake/oar-ocr-models}"
BASE="https://github.com/GreatV/oar-ocr/releases/download/v0.3.0"

MODELS=(
    # Layout detection (PP-DocLayout_plus-L: 20 classes incl. algorithm/formula/table)
    "pp-doclayout_plus-l.onnx"
    # Region detection (PP-DocBlockLayout: multi-column block grouping for reading order)
    "pp-docblocklayout.onnx"
    # Text detection + recognition (server variants, higher accuracy than mobile)
    "pp-ocrv5_server_det.onnx"
    "pp-ocrv5_server_rec.onnx"
    "ppocrv5_dict.txt"
    # Formula recognition → LaTeX
    "pp-formulanet_plus-l.onnx"
    "unimernet_tokenizer.json"
    # Table recognition (wired/wireless auto-switch)
    "pp-lcnet_x1_0_table_cls.onnx"
    "slanext_wired.onnx"
    "slanet_plus.onnx"
    "rt-detr-l_wired_table_cell_det.onnx"
    "rt-detr-l_wireless_table_cell_det.onnx"
    "table_structure_dict_ch.txt"
)

mkdir -p "${DEST}"
echo "Downloading OAR-OCR models to: ${DEST}"

for name in "${MODELS[@]}"; do
    out="${DEST}/${name}"
    if [[ -f "${out}" ]]; then
        echo "  ✓ ${name} (already present)"
        continue
    fi
    echo "  ↓ ${name}"
    curl -fL --progress-bar -o "${out}.downloading" "${BASE}/${name}"
    mv "${out}.downloading" "${out}"
done

echo "Done. Models ready for the OAR-OCR backend."
