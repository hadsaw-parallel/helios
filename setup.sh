#!/bin/bash
# HELIOS — AMD MI300X instance setup
# Run this inside the Docker container after a fresh instance spin-up:
#   docker exec -it rocm /bin/bash
#   bash setup.sh
#
# Full setup takes ~15-20 min (mostly Llama model download)

set -e
echo "=== HELIOS Setup ==="

# ── 1. Clone project repo ─────────────────────────────────────────────────────
cd /app
if [ ! -d "helios" ]; then
    git clone https://github.com/hadsaw-parallel/helios.git
fi
cd helios
git pull origin main
echo "[1/6] Repo ready"

# ── 2. Install Python dependencies ───────────────────────────────────────────
pip install --quiet langgraph langchain-core requests numpy scipy \
    astropy sunpy Pillow xarray netCDF4 pyyaml \
    folium plotly streamlit streamlit-folium pytest
echo "[2/6] Dependencies installed"

# ── 3. Clone Surya (NASA-IMPACT) ─────────────────────────────────────────────
if [ ! -d "Surya" ]; then
    git clone https://github.com/NASA-IMPACT/Surya
fi
cd Surya && pip install --quiet -r requirements.txt && cd ..
echo "[3/6] Surya cloned"

# ── 4. Clone SuryaBench (historical storm data) ───────────────────────────────
if [ ! -d "SuryaBench" ]; then
    git clone https://github.com/NASA-IMPACT/SuryaBench
fi
echo "[4/6] SuryaBench cloned"

# ── 5. Download Surya model weights from HuggingFace ────────────────────────
# Requires: huggingface-cli login (run once manually with your token)
mkdir -p weights
python3 -c "
from huggingface_hub import hf_hub_download
import os
for f in ['surya.366m.v1.pt', 'config.yaml', 'scalers.yaml']:
    if not os.path.exists(f'weights/{f}'):
        print(f'Downloading {f}...')
        hf_hub_download(repo_id='nasa-ibm-ai4science/Surya-1.0', filename=f, local_dir='./weights')
        print(f'  Done: {f}')
    else:
        print(f'  Already present: {f}')
"
echo "[5/6] Surya weights ready"

# ── 6. Start vLLM server with Llama 3.1 ─────────────────────────────────────
# First run downloads ~140GB — subsequent runs use HuggingFace cache
echo "[6/6] Starting vLLM (Llama 3.1 70B)..."
echo "      First run downloads ~140GB — this takes 10-15 min on first setup"
echo "      Watch progress: tail -f /app/helios/vllm.log"
nohup python3 -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --port 8000 \
    --dtype float16 \
    > /app/helios/vllm.log 2>&1 &
echo "      vLLM starting in background (PID $!)"
echo "      Test when ready: curl http://localhost:8000/v1/models"

echo ""
echo "=== Setup complete ==="
echo "Next: verify APIs with  python3 data/noaa_feed.py"
echo "      check vLLM with   curl http://localhost:8000/v1/models"
