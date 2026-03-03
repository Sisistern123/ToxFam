#!/usr/bin/env bash
# =========================================================
# SignalP6 preprocessing (runs via tools/signalp6 uv project)
# =========================================================
# Usage:
#   ./scripts/run_signalp6.sh [--extra-args "--organism eukarya"] [--bsize 10]
#
# Requires: SignalP6 set up in tools/signalp6/
#   See docs/signalp6_setup.md for installation instructions.
# =========================================================
set -euo pipefail

# Determine absolute paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

FASTA_DIR="$BASE_DIR/data/intermediate/fasta"
SP6_TOX_DIR="$BASE_DIR/data/intermediate/sp6/tox"
SP6_NONTOX_DIR="$BASE_DIR/data/intermediate/sp6/nontox"
SP6_PROJECT="$BASE_DIR/tools/signalp6"

# Parse arguments
EXTRA_ARGS="--organism eukarya"
BATCH_SIZE=10
while [[ $# -gt 0 ]]; do
    case $1 in
        --extra-args)
            EXTRA_ARGS="$2"
            shift 2
            ;;
        --bsize)
            BATCH_SIZE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Verify SignalP6 is set up
if [ ! -f "$SP6_PROJECT/pyproject.toml" ]; then
    echo "ERROR: SignalP6 project not found at $SP6_PROJECT" >&2
    echo "See docs/signalp6_setup.md for installation instructions." >&2
    exit 1
fi

if [ ! -d "$SP6_PROJECT/bin/signalp-6-package" ]; then
    echo "ERROR: signalp-6-package not found in $SP6_PROJECT/bin/" >&2
    echo "See docs/signalp6_setup.md for installation instructions." >&2
    exit 1
fi

# Resolve model directory (weights live in models/, not in signalp/model_weights/)
MODEL_DIR="$(cd "$SP6_PROJECT/bin/signalp-6-package/models" && pwd)"

# Clear VIRTUAL_ENV so uv doesn't warn about the parent project's venv
unset VIRTUAL_ENV

# Select best available prediction mode
# "slow-sequential" uses the full ensemble (6 models run one at a time) for best accuracy
# "fast" uses a distilled single model as fallback
if [ -d "$MODEL_DIR/sequential_models_signalp6" ]; then
    SP6_MODE="slow-sequential"
else
    SP6_MODE="fast"
fi

# Print mode for the Python caller to parse
echo "SP6_MODE=${SP6_MODE}"

mkdir -p "$SP6_TOX_DIR" "$SP6_NONTOX_DIR"

# Verify input FASTA files exist
if [ ! -f "$FASTA_DIR/tox.fasta" ]; then
    echo "ERROR: tox.fasta not found at $FASTA_DIR/tox.fasta" >&2
    echo "Run the preprocessing pipeline first to generate input FASTAs." >&2
    exit 1
fi

if [ ! -f "$FASTA_DIR/nontox.fasta" ]; then
    echo "ERROR: nontox.fasta not found at $FASTA_DIR/nontox.fasta" >&2
    echo "Run the preprocessing pipeline first to generate input FASTAs." >&2
    exit 1
fi

run_signalp6() {
    local label="$1"
    local fasta="$2"
    local outdir="$3"

    echo "SP6_START=${label}"
    uv run --quiet --project "$SP6_PROJECT" signalp6 \
        --fastafile "$fasta" \
        --output_dir "$outdir" \
        --model_dir "$MODEL_DIR" \
        $EXTRA_ARGS \
        --mode "$SP6_MODE" \
        --bsize "$BATCH_SIZE" \
        --format none >&2
    echo "SP6_DONE=${label}"
}

run_signalp6 "tox"    "$FASTA_DIR/tox.fasta"    "$SP6_TOX_DIR"
run_signalp6 "nontox" "$FASTA_DIR/nontox.fasta"  "$SP6_NONTOX_DIR"
