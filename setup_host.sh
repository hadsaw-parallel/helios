#!/bin/bash
# HELIOS — Host-level setup (run on AMD host as root, NOT inside Docker container)
# Run this once after spinning up a fresh AMD Developer Cloud instance.
#
# Usage:
#   ssh root@<YOUR_IP>
#   bash <(curl -s https://raw.githubusercontent.com/hadsaw-parallel/helios/main/setup_host.sh)
# ─────────────────────────────────────────────────────────────────────────────

set -e

echo "[host] Configuring Caddy to proxy port 80 → Streamlit (container port 30000)..."

# Install Caddy if not present
if ! command -v caddy &> /dev/null; then
    apt-get install -y caddy
fi

# Write Caddyfile
printf ':80 {\n\treverse_proxy localhost:30000\n}\n' | tee /etc/caddy/Caddyfile

# Restart Caddy
systemctl restart caddy && systemctl enable caddy

echo "[host] Caddy configured ✓"
echo "[host] Dashboard will be accessible at http://$(curl -s ifconfig.me) once setup.sh runs inside container"
echo ""
echo "[host] Next step: enter the container and run setup.sh"
echo "  docker exec -it rocm /bin/bash"
echo "  cd /app && git clone https://github.com/hadsaw-parallel/helios.git && cd helios && bash setup.sh"
