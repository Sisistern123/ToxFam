#!/usr/bin/env bash
# =========================================================
# SignalP6 preprocessing (runs inside conda env 'signalp6')
# =========================================================
# Usage:
#   ./run_signalp6.sh --extra-args "--organism euk"
# =========================================================

# Determine absolute path of this script and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

FASTA_DIR="$BASE_DIR/data/intermediate/fasta"
SP6_TOX_DIR="$BASE_DIR/data/intermediate/sp6/tox"
SP6_NONTox_DIR="$BASE_DIR/data/intermediate/sp6/nontox"

# Custom model directory
MODEL_DIR="$HOME/Desktop/Uni/signalp6/signalp-6-package/models"

mkdir -p "$SP6_TOX_DIR" "$SP6_NONTox_DIR"

echo "🔹 Activating conda environment: signalp6"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate signalp6

echo "🚀 Running SignalP6 on tox.fasta"
signalp6 --fastafile "$FASTA_DIR/tox.fasta" \
         --output_dir "$SP6_TOX_DIR" \
         --organism eukarya \
         --mode fast \
         --model_dir "$MODEL_DIR"

echo "🚀 Running SignalP6 on nontox.fasta"
signalp6 --fastafile "$FASTA_DIR/nontox.fasta" \
         --output_dir "$SP6_NONTox_DIR" \
         --organism eukarya \
         --mode fast \
         --model_dir "$MODEL_DIR"

conda deactivate
echo "✅ SignalP6 preprocessing completed successfully."

