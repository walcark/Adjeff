#!/usr/bin/env bash
# Generate PSF parameter LUTs for all 8 CAMS OPAC species, sequentially.
#
# Usage:
#   ./generate_psf_luts.sh --cache-dir /path/to/cache [--output-dir /path] [--device cuda]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."
PYTHON_SCRIPT="$SCRIPT_DIR/generate_psf_param_lut.py"

SPECIES=(
    ammonium
    blackcar
    dust
    nitrate
    organicm
    seasalt
    sulphate
    secondar
)

export SMARTG_DIR_AUXDATA=/home/kwalcarius/dev/third-party/smartg/auxdata/

for species in "${SPECIES[@]}"; do
    echo "=== Processing species: $species ==="
    python "$PYTHON_SCRIPT" --species "$species" "$@"
done

echo "=== All species done. ==="
