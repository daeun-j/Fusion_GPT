#!/usr/bin/env bash
# Profile fused LayerNorm+Linear kernels with NVIDIA Nsight Systems.
#
# Usage:
#   bash scripts/profile_nsys.sh [single-op|e2e]
#
# Output: results/profile_<timestamp>.nsys-rep
#
# View with: nsys-ui results/profile_*.nsys-rep
# or:        nsys stats results/profile_*.nsys-rep

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$PROJECT_DIR/results"
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)

mkdir -p "$RESULTS_DIR"

MODE="${1:-single-op}"
OUTPUT="$RESULTS_DIR/profile_${MODE}_${TIMESTAMP}"

echo "=== Nsight Systems Profiling ==="
echo "Mode:   $MODE"
echo "Output: ${OUTPUT}.nsys-rep"
echo ""

# Activate venv if available
if [ -f "$PROJECT_DIR/venv/bin/activate" ]; then
    source "$PROJECT_DIR/venv/bin/activate"
fi

# Set CUDA_HOME if cuda-12.8 is available
if [ -d "/usr/local/cuda-12.8" ]; then
    export CUDA_HOME=/usr/local/cuda-12.8
elif [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME=/usr/local/cuda
fi

case "$MODE" in
    single-op)
        nsys profile \
            --trace=cuda,nvtx \
            --output="$OUTPUT" \
            --force-overwrite=true \
            python -m src.benchmark --single-op
        ;;
    e2e)
        nsys profile \
            --trace=cuda,nvtx \
            --output="$OUTPUT" \
            --force-overwrite=true \
            python -m src.benchmark --e2e
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: $0 [single-op|e2e]"
        exit 1
        ;;
esac

echo ""
echo "=== Profiling complete ==="
echo "Report: ${OUTPUT}.nsys-rep"
echo ""
echo "View stats:  nsys stats ${OUTPUT}.nsys-rep"
echo "Open GUI:    nsys-ui ${OUTPUT}.nsys-rep"
