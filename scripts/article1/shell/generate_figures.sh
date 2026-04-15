#!/usr/bin/env bash
# Generate article figures 4, 5, and 7-17.
# Usage: ./generate_figures.sh figureX  (X = 4, 5, 7..17)

set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 figureX  (X in {4, 5, 7..17})"
    exit 1
fi

arg="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."

if [[ "$arg" =~ ^figure([0-9]+)$ ]]; then
    X="${BASH_REMATCH[1]}"
else
    echo "Argument must be of the form figureX"
    exit 1
fi

case "$X" in
    4)
        python "$SCRIPT_DIR/figure4.py"
        ;;
    5)
        python "$SCRIPT_DIR/figure5.py"
        ;;
    7)
        python "$SCRIPT_DIR/figure7_17.py" \
            --figure figure7 \
            --aot 0.1 0.3 0.5 0.7 \
            --rh 50.0 --h 0.0 --wl 560.0 --href 2.0 \
            --sza 40.0 --vza 8.0 --saa 0.0 --vaa 0.0 \
            --species blackcar --remove_rayleigh
        ;;
    8)
        python "$SCRIPT_DIR/figure7_17.py" \
            --figure figure8 \
            --aot 0.1 0.3 0.5 0.7 \
            --rh 50.0 --h 0.0 --wl 560.0 --href 2.0 \
            --sza 40.0 --vza 8.0 --saa 0.0 --vaa 0.0 \
            --species sulphate --remove_rayleigh
        ;;
    9)
        python "$SCRIPT_DIR/figure7_17.py" \
            --figure figure9 \
            --aot 0.1 0.3 0.5 0.7 \
            --rh 95.0 --h 0.0 --wl 560.0 --href 2.0 \
            --sza 40.0 --vza 8.0 --saa 0.0 --vaa 0.0 \
            --species sulphate --remove_rayleigh
        ;;
    10)
        python "$SCRIPT_DIR/figure7_17.py" \
            --figure figure10 \
            --aot 0.1 0.3 0.5 0.7 \
            --rh 50.0 --h 0.0 --wl 560.0 --href 2.0 \
            --sza 40.0 --vza 8.0 --saa 0.0 --vaa 0.0 \
            --species seasalt --remove_rayleigh
        ;;
    11)
        python "$SCRIPT_DIR/figure7_17.py" \
            --figure figure11 \
            --aot 0.4 \
            --rh 50.0 --h 0.0 --wl 560.0 865.0 2190.0 --href 2.0 \
            --sza 40.0 --vza 8.0 --saa 0.0 --vaa 0.0 \
            --species blackcar --remove_rayleigh
        ;;
    12)
        python "$SCRIPT_DIR/figure7_17.py" \
            --figure figure12 \
            --aot 0.4 \
            --rh 50.0 --h 0.0 --wl 560.0 865.0 2190.0 --href 2.0 \
            --sza 40.0 --vza 8.0 --saa 0.0 --vaa 0.0 \
            --species sulphate --remove_rayleigh
        ;;
    13)
        python "$SCRIPT_DIR/figure7_17.py" \
            --figure figure13 \
            --aot 0.4 \
            --rh 50.0 --h 0.0 --wl 560.0 865.0 2190.0 --href 2.0 \
            --sza 40.0 --vza 8.0 --saa 0.0 --vaa 0.0 \
            --species seasalt --remove_rayleigh
        ;;
    14)
        python "$SCRIPT_DIR/figure7_17.py" \
            --figure figure14 \
            --aot 0.4 \
            --rh 50.0 --h 0.0 1.5 3.0 --wl 560.0 --href 2.0 \
            --sza 40.0 --vza 8.0 --saa 0.0 --vaa 0.0 \
            --species sulphate
        ;;
    15)
        python "$SCRIPT_DIR/figure7_17.py" \
            --figure figure15 \
            --aot 0.05 \
            --rh 50.0 --h 0.0 1.5 3.0 --wl 560.0 --href 2.0 \
            --sza 40.0 --vza 8.0 --saa 0.0 --vaa 0.0 \
            --species sulphate
        ;;
    16)
        python "$SCRIPT_DIR/figure7_17.py" \
            --figure figure16 \
            --aot 0.4 \
            --rh 50.0 --h 0.0 --wl 560.0 --href 1.0 2.0 4.0 \
            --sza 40.0 --vza 8.0 --saa 0.0 --vaa 0.0 \
            --species sulphate
        ;;
    17)
        python "$SCRIPT_DIR/figure7_17.py" \
            --figure figure17 \
            --aot 0.4 \
            --rh 50.0 --h 0.0 --wl 560.0 --href 2.0 \
            --sza 40.0 --vza 0.0 8.0 16.0 --saa 0.0 --vaa 0.0 \
            --species sulphate
        ;;
    *)
        echo "Figure must be one of: 4, 5, 7..17"
        exit 1
        ;;
esac
