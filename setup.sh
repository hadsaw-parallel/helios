#!/bin/bash
# HELIOS — Full instance setup (run INSIDE the Docker container)
#
# On a fresh AMD Developer Cloud instance:
#
#   STEP 1 (on host, run once):
#     bash <(curl -s https://raw.githubusercontent.com/hadsaw-parallel/helios/main/setup_host.sh)
#
#   STEP 2 (enter container):
#     docker exec -it rocm /bin/bash
#
#   STEP 3 (inside container):
#     cd /app && git clone https://github.com/hadsaw-parallel/helios.git && cd helios && bash setup.sh
#
#   Next time (instance already configured):
#     docker exec -it rocm /bin/bash
#     cd /app/helios && bash setup.sh
#
# Dashboard opens at: http://<YOUR_INSTANCE_IP>
# ─────────────────────────────────────────────────────────────────────────────

set -e
cd /app

echo "╔══════════════════════════════════════════════════╗"
echo "║         HELIOS Setup — AMD MI300X ROCm           ║"
echo "╚══════════════════════════════════════════════════╝"

# ── 1. Repo ───────────────────────────────────────────────────────────────────
if [ ! -d "helios" ]; then
    echo "[1/7] Cloning helios repo..."
    git clone https://github.com/hadsaw-parallel/helios.git
fi
cd helios
git pull origin main
echo "[1/7] Repo ready ✓"

# ── 2. Python dependencies ────────────────────────────────────────────────────
echo "[2/7] Installing dependencies..."
pip install --quiet --ignore-installed \
    langgraph langchain-core requests numpy scipy \
    astropy sunpy Pillow xarray netCDF4 pyyaml \
    folium plotly streamlit streamlit-folium pytest
echo "[2/7] Dependencies ready ✓"

# ── 3. Surya (NASA-IMPACT) ────────────────────────────────────────────────────
echo "[3/7] Setting up Surya..."
if [ ! -d "Surya" ]; then
    git clone https://github.com/NASA-IMPACT/Surya
fi
cd Surya
git pull origin main 2>/dev/null || true
pip install --quiet -r requirements.txt
cd ..
echo "[3/7] Surya ready ✓"

# ── 4. Surya weights ──────────────────────────────────────────────────────────
echo "[4/7] Checking Surya weights..."
if [ ! -f "Surya/data/Surya-1.0/surya.366m.v1.pt" ]; then
    echo "      Downloading weights (~500MB)..."
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='nasa-ibm-ai4science/Surya-1.0',
    local_dir='Surya/data/Surya-1.0',
    allow_patterns=['config.yaml','scalers.yaml','surya.366m.v1.pt'],
    token=None
)
print('Weights downloaded.')
"
else
    echo "      Weights already present, skipping download."
fi
echo "[4/7] Surya weights ready ✓"

# ── 5. SuryaBench ─────────────────────────────────────────────────────────────
echo "[5/7] Checking SuryaBench..."
if [ ! -d "SuryaBench" ]; then
    git clone https://github.com/NASA-IMPACT/SuryaBench
fi
echo "[5/7] SuryaBench ready ✓"

# ── 6. vLLM — Llama 3.1 8B ───────────────────────────────────────────────────
echo "[6/7] Starting vLLM (meta-llama/Meta-Llama-3-8B)..."
# Kill any existing vLLM process
pkill -f "vllm.entrypoints" 2>/dev/null || true
sleep 2

nohup python3 -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3-8B \
    --port 8000 \
    --dtype float16 \
    > /app/helios/vllm.log 2>&1 &

VLLM_PID=$!
echo "      vLLM PID: $VLLM_PID — waiting for startup..."

# Wait up to 3 minutes for vLLM to be ready
for i in $(seq 1 36); do
    sleep 5
    if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
        echo "      vLLM ready ✓"
        break
    fi
    echo "      Waiting... ($((i*5))s)"
done
echo "[6/7] vLLM ready ✓"

# ── 7. Streamlit dashboard ────────────────────────────────────────────────────
echo "[7/7] Starting Streamlit on port 30000..."
pkill -f streamlit 2>/dev/null || true
sleep 2

nohup streamlit run dashboard/app.py \
    --server.port 30000 \
    --server.address 0.0.0.0 \
    --server.headless true \
    > /app/helios/streamlit.log 2>&1 &

sleep 3
echo "[7/7] Streamlit ready ✓"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║             HELIOS is running!                   ║"
echo "║                                                  ║"
echo "║  Dashboard : http://<YOUR_IP>                    ║"
echo "║  vLLM logs : tail -f /app/helios/vllm.log        ║"
echo "║  UI logs   : tail -f /app/helios/streamlit.log   ║"
echo "╚══════════════════════════════════════════════════╝"
