#!/usr/bin/env bash
# =========================================================
# Download raw UniProt data for ToxFam
# =========================================================
# Usage:
#   ./scripts/download_raw_data.sh
#
# Downloads toxin (KW-0800) and non-toxin protein TSVs from
# UniProt for reviewed metazoan proteins, pinned to entries
# created on or before 2026-03-03 for reproducibility.
# =========================================================
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BASE_DIR="$(dirname "$SCRIPT_DIR")"
RAW_DIR="$BASE_DIR/data/raw"

mkdir -p "$RAW_DIR"

# UniProt REST API base
API="https://rest.uniprot.org/uniprotkb/stream"

# Columns to export
FIELDS="accession,protein_families,organism_id,sequence,ft_signal"

# Date cutoff for reproducibility
DATE_CUTOFF="2026-03-03"

# Common filter: reviewed metazoan proteins, no fragments, date-pinned
BASE_QUERY="(taxonomy_id:33208) AND (reviewed:true) AND (fragment:false) AND (date_created:[* TO ${DATE_CUTOFF}])"

# Toxin query: KW-0800
TOX_QUERY="${BASE_QUERY} AND (keyword:KW-0800)"

# Non-toxin query: NOT KW-0800
NONTOX_QUERY="${BASE_QUERY} NOT (keyword:KW-0800)"

download() {
    local name="$1"
    local query="$2"
    local dest="$3"
    local basename
    basename=$(basename "$dest")

    if [ -f "$dest" ]; then
        echo "   ${basename}: already exists, skipping"
        return 0
    fi

    printf "   ${basename}: downloading ..."
    local http_code
    http_code=$(curl -s -w "%{http_code}" -o "$dest" \
        --retry 3 --retry-delay 5 \
        --get \
        --data-urlencode "query=${query}" \
        --data-urlencode "format=tsv" \
        --data-urlencode "fields=${FIELDS}" \
        "${API}")

    if [ "$http_code" -ne 200 ]; then
        echo " FAILED (HTTP $http_code)" >&2
        rm -f "$dest"
        return 1
    fi

    local lines
    lines=$(wc -l < "$dest" | tr -d ' ')
    echo " $((lines - 1)) proteins"
}

download "0800.tsv (toxins)"       "$TOX_QUERY"    "$RAW_DIR/0800.tsv"
download "nontox.tsv (non-toxins)" "$NONTOX_QUERY" "$RAW_DIR/nontox.tsv"
