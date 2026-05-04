"""
Agent 01 — Solar Vision
Runs Surya-1.0 inference on live SDO images to detect solar flares.

IMPORTANT — READ BEFORE RUNNING:
  The Surya model API (class name, forward pass signature, output attributes)
  must be verified from the real NASA-IMPACT/Surya repo BEFORE this file works:

    git clone https://github.com/NASA-IMPACT/Surya
    cat Surya/tests/test_surya.py          # find real class name + inference call
    ls  Surya/downstream_examples/         # find real output attribute names

  Replace the TODO markers below with the verified class/attribute names.
  This file will raise ImportError or AttributeError until that is done.
"""
import sys
import time
import json
import torch
import numpy as np
import yaml
import requests
import io
from datetime import datetime, timezone
from PIL import Image

# After cloning NASA-IMPACT/Surya, add it to path
sys.path.insert(0, "./Surya")

# TODO: Replace with real class name found in Surya/tests/test_surya.py
# Common patterns to look for: SuryaModel, SuryaForecaster, HelioModel
try:
    from surya.model import SuryaModel  # REPLACE if class name differs
    _SURYA_AVAILABLE = True
except ImportError:
    _SURYA_AVAILABLE = False
    print("[Agent01] WARNING: Surya not importable yet — run after cloning NASA-IMPACT/Surya")


FLARE_THRESHOLD = 0.6  # probability above which we emit an alert event
SDO_URL = "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_{wavelength}.jpg"


class SolarVisionAgent:

    def __init__(self, weights_dir: str = "weights", device: str = "cuda"):
        if not _SURYA_AVAILABLE:
            raise RuntimeError("Surya not installed. Clone NASA-IMPACT/Surya first.")
        self.device = device
        self.model = self._load_model(weights_dir)
        print(f"[Agent01] Surya loaded on {device}")
        print(f"[Agent01] VRAM used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    def _load_model(self, weights_dir: str):
        config_path = f"{weights_dir}/config.yaml"
        weights_path = f"{weights_dir}/surya.366m.v1.pt"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # TODO: Verify constructor signature from Surya/tests/test_surya.py
        model = SuryaModel(config)
        state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def fetch_sdo_image(self, wavelength: str = "0171") -> torch.Tensor:
        """Fetch latest SDO image and return as (1, 1, H, W) GPU tensor."""
        url = SDO_URL.format(wavelength=wavelength)
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("L")
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.tensor(arr).unsqueeze(0).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def run_inference(self, image_tensor: torch.Tensor) -> tuple[float, float]:
        """Run Surya forward pass. Returns (flare_probability, latency_ms)."""
        t0 = time.perf_counter()
        output = self.model(image_tensor)
        latency_ms = (time.perf_counter() - t0) * 1000

        # TODO: Replace with real attribute name from Surya output object
        # Look in downstream_examples/ for how the output is used
        flare_prob = float(output.flare_probability)  # REPLACE if attribute name differs
        return flare_prob, latency_ms

    def emit_event(self, flare_prob: float, latency_ms: float) -> dict:
        severity = "X-class" if flare_prob > 0.85 else \
                   "M-class" if flare_prob > 0.6 else \
                   "C-class" if flare_prob > 0.3 else "low"
        return {
            "agent": "agent_01_vision",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "flare_probability": round(flare_prob, 4),
            "flare_detected": flare_prob > FLARE_THRESHOLD,
            "severity": severity,
            "inference_ms": round(latency_ms, 2),
            "vram_gb": round(torch.cuda.memory_allocated() / 1e9, 3),
        }

    def run_loop(self, callback=None, interval: int = 12):
        """Poll SDO every `interval` seconds (matching SDO's 12-second cadence)."""
        print("[Agent01] Starting solar vision loop...")
        while True:
            try:
                tensor = self.fetch_sdo_image("0171")
                prob, latency = self.run_inference(tensor)
                event = self.emit_event(prob, latency)
                print(f"[Agent01] {event['timestamp']} | prob={prob:.3f} | {latency:.0f}ms")
                if callback:
                    callback(event)
            except Exception as e:
                print(f"[Agent01] Error: {e}")
            time.sleep(interval)


if __name__ == "__main__":
    agent = SolarVisionAgent()
    agent.run_loop(callback=lambda e: print(json.dumps(e, indent=2)))
