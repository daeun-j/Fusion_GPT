#!/usr/bin/env bash
set -euo pipefail

echo "=== Installing system packages ==="
sudo apt-get update -qq
sudo apt-get install -y python3-pip python3-venv

echo "=== Creating virtual environment ==="
python3 -m venv /home/ubuntu/dev/fused-ln-linear/venv

echo "=== Installing PyTorch (cu128) + transformers ==="
source /home/ubuntu/dev/fused-ln-linear/venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install transformers accelerate

echo "=== Verifying CUDA ==="
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

echo "=== Done ==="
