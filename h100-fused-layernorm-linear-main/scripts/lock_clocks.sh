#!/usr/bin/env bash
# Lock H100 GPU clocks for reproducible benchmarks.
# Requires sudo / root. Run before benchmarking, reset after.
#
# Usage:
#   sudo bash scripts/lock_clocks.sh lock    # Lock clocks to max
#   sudo bash scripts/lock_clocks.sh unlock  # Reset to default
set -euo pipefail

case "${1:-lock}" in
    lock)
        echo "Enabling persistence mode..."
        nvidia-smi -pm 1
        echo "Locking GPU graphics clock to 1980 MHz..."
        nvidia-smi -lgc 1980,1980
        echo "Clocks locked. Current state:"
        nvidia-smi --query-gpu=clocks.gr,clocks.max.gr,clocks.mem --format=csv
        ;;
    unlock|reset)
        echo "Resetting GPU clocks to default..."
        nvidia-smi -rgc
        echo "Clocks reset. Current state:"
        nvidia-smi --query-gpu=clocks.gr,clocks.max.gr,clocks.mem --format=csv
        ;;
    *)
        echo "Usage: $0 {lock|unlock}"
        exit 1
        ;;
esac
