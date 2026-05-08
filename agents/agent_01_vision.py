"""
Agent 01 — Solar Vision
Runs NASA/IBM Surya-1.0 (HelioSpectFormer) inference on SDO solar image sequences.

Surya is a FORECASTING model — it predicts the next solar image from a 2-timestep input.
Flare signal = mean intensity in predicted AIA 131Å channel (sensitive to flare temperatures).

Two modes:
  bench  — SuryaBench .nc files (demo / historical storm replay)
  live   — GOES X-ray flux from NOAA (real-time; Surya needs .nc which isn't available at 12s)

Verified API from NASA-IMPACT/Surya tests/test_surya.py:
  class:  HelioSpectFormer  (surya.models.helio_spectformer)
  input:  {"ts": (B,13,2,H,W), "time_delta_input": (B,2)}
  output: (B,13,H,W) — predicted solar image one timestep ahead
"""
import sys
import os
import time
import json
import yaml
import requests
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime, timezone

sys.path.insert(0, "./Surya")

try:
    from surya.models.helio_spectformer import HelioSpectFormer
    from surya.utils.data import build_scalers
    _SURYA_AVAILABLE = True
except ImportError:
    _SURYA_AVAILABLE = False
    print("[Agent01] WARNING: Surya not importable — clone NASA-IMPACT/Surya and pip install -r requirements.txt")

# AIA 131Å is most sensitive to flare plasma (~10 MK)
SDO_CHANNELS = ["aia94","aia131","aia171","aia193","aia211","aia304","aia335","aia1600",
                 "hmi_m","hmi_bx","hmi_by","hmi_bz","hmi_v"]
AIA131_IDX = SDO_CHANNELS.index("aia131")
AIA171_IDX = SDO_CHANNELS.index("aia171")

# Threshold on normalised AIA 131 mean intensity (tune after first benchmark run)
FLARE_THRESHOLD = 0.65

GOES_URL = "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json"


class SolarVisionAgent:

    def __init__(self, weights_dir: str = "Surya/data/Surya-1.0", device: str = "cuda"):
        if not _SURYA_AVAILABLE:
            raise RuntimeError("Surya not installed. Clone NASA-IMPACT/Surya first.")
        self.device = device
        self.weights_dir = weights_dir
        self.model, self.config = self._load_model(weights_dir)
        print(f"[Agent01] HelioSpectFormer loaded on {device}")
        print(f"[Agent01] VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    def _load_model(self, weights_dir: str):
        config_path = os.path.join(weights_dir, "config.yaml")
        weights_path = os.path.join(weights_dir, "surya.366m.v1.pt")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        mc = config["model"]
        dc = config["data"]

        model = HelioSpectFormer(
            img_size=mc["img_size"],
            patch_size=mc["patch_size"],
            in_chans=len(dc["sdo_channels"]),
            embed_dim=mc["embed_dim"],
            time_embedding={
                "type": "linear",
                "time_dim": len(dc["time_delta_input_minutes"]),
            },
            depth=mc["depth"],
            n_spectral_blocks=mc["n_spectral_blocks"],
            num_heads=mc["num_heads"],
            mlp_ratio=mc["mlp_ratio"],
            drop_rate=mc["drop_rate"],
            dtype=torch.bfloat16,
            window_size=mc["window_size"],
            dp_rank=mc["dp_rank"],
            learned_flow=mc["learned_flow"],
            use_latitude_in_learned_flow=mc["learned_flow"],
            init_weights=False,
            checkpoint_layers=list(range(mc["depth"])),
            rpe=mc["rpe"],
            ensemble=mc["ensemble"],
            finetune=mc["finetune"],
        )

        weights = torch.load(weights_path, map_location=self.device, weights_only=True)
        model.load_state_dict(weights, strict=True)
        model.to(self.device)
        model.eval()
        return model, config

    @torch.no_grad()
    def run_inference_on_batch(self, ts: torch.Tensor, time_delta: torch.Tensor) -> tuple[float, float]:
        """
        Run HelioSpectFormer forward pass.

        ts:         (1, 13, 2, H, W) — 2 timesteps, 13 channels, bfloat16
        time_delta: (1, 2)           — time offsets in hours (e.g. [-1.0, 0.0])

        Returns (flare_prob, latency_ms)
        """
        ts = ts.to(self.device)
        time_delta = time_delta.to(self.device)

        t0 = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            forecast = self.model({"ts": ts, "time_delta_input": time_delta})
        latency_ms = (time.perf_counter() - t0) * 1000

        # forecast shape: (B, 13, H, W)
        # Use mean intensity of AIA 131 channel as flare proxy
        aia131_pred = forecast[0, AIA131_IDX].float().cpu().numpy()
        # Normalise: values are in model-space (signum-log scaled)
        # positive high values → high predicted coronal intensity → flare signal
        flare_signal = float(np.clip(aia131_pred.mean(), 0, 1))
        return flare_signal, latency_ms

    def run_inference_from_nc(self, nc_dir: str, timestep: int = 0) -> tuple[float, float]:
        """Load a SuryaBench .nc file and run inference. Used for demo/replay."""
        import xarray as xr, glob

        files = sorted(glob.glob(os.path.join(nc_dir, "*.nc")))
        assert files, f"No .nc files in {nc_dir}"

        ds = xr.open_dataset(files[0])
        var = list(ds.data_vars)[0]
        arr = ds[var].isel(time=slice(timestep, timestep + 2)).values.astype(np.float32)
        # arr shape: (2, channels, H, W) or similar — adjust to (1, C, 2, H, W)
        if arr.ndim == 4:
            arr = arr.transpose(1, 0, 2, 3)[np.newaxis]  # (1, C, 2, H, W)
        elif arr.ndim == 3:
            arr = arr[np.newaxis, np.newaxis, :, :, :]    # rough fallback

        ts = torch.tensor(arr, dtype=torch.bfloat16)
        time_delta = torch.tensor([[-1.0, 0.0]])  # -60 min and 0 min in hours
        return self.run_inference_on_batch(ts, time_delta)

    # ── Live mode: GOES X-ray flux ─────────────────────────────────────────────
    @staticmethod
    def fetch_goes_flare_signal() -> tuple[float, str]:
        """
        Real-time flare detection from GOES X-ray flux.
        GOES is what NOAA actually uses for operational flare monitoring.
        Returns (normalised_signal 0-1, flare_class string).
        """
        try:
            r = requests.get(GOES_URL, timeout=10)
            r.raise_for_status()
            data = r.json()
            flux = float(data[-1].get("flux", 0) or 0)
        except Exception:
            return 0.0, "unknown"

        # GOES flux → normalised signal and class
        if flux >= 1e-4:   return 1.0,  "X-class"
        elif flux >= 1e-5: return 0.85, "M-class"
        elif flux >= 1e-6: return 0.55, "C-class"
        elif flux >= 1e-7: return 0.25, "B-class"
        else:              return 0.05, "A-class"

    def emit_event(self, flare_prob: float, latency_ms: float,
                   source: str = "surya", flare_class: str = "") -> dict:
        severity = ("X-class" if flare_prob > 0.85 else
                    "M-class" if flare_prob > 0.6  else
                    "C-class" if flare_prob > 0.3  else "low")
        if flare_class:
            severity = flare_class
        return {
            "agent": "agent_01_vision",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "flare_probability": round(flare_prob, 4),
            "flare_detected": flare_prob > FLARE_THRESHOLD,
            "severity": severity,
            "source": source,
            "inference_ms": round(latency_ms, 2),
            "vram_gb": round(torch.cuda.memory_allocated() / 1e9, 3),
        }

    @staticmethod
    def fetch_kp_index() -> float:
        """Fetch current planetary Kp index from NOAA as a second opinion."""
        try:
            r = requests.get(
                "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json",
                timeout=8,
            )
            data = r.json()
            kp_val = data[-1][1] if len(data) > 1 else "0"
            return float(kp_val)
        except Exception:
            return 0.0

    def run_live_cycle(self) -> dict:
        """
        One live pipeline cycle. Decision loop:
        1. Always fetch GOES X-ray (primary sensor).
        2. If signal is borderline (C-class zone, 0.3–0.65), fetch Kp index
           as a second opinion and upgrade severity if Kp confirms elevated activity.
        This is an observation → decision → conditional re-observation loop.
        """
        t0 = time.perf_counter()
        signal, flare_class = self.fetch_goes_flare_signal()

        # Decision branch: borderline signal triggers a second observation
        if 0.3 <= signal <= 0.65:
            kp = self.fetch_kp_index()
            if kp >= 5:
                # Kp confirms elevated geomagnetic conditions — upgrade signal
                signal = max(signal, 0.70)
                flare_class = "M-class"
            elif kp <= 1:
                # Kp confirms quiet conditions — downgrade signal
                signal = min(signal, 0.25)
                flare_class = "B-class"

        latency_ms = (time.perf_counter() - t0) * 1000
        return self.emit_event(signal, latency_ms, source="goes_xray", flare_class=flare_class)

    def run_bench_cycle(self, nc_dir: str, timestep: int = 0) -> dict:
        """One pipeline cycle using SuryaBench .nc data (demo mode)."""
        signal, latency_ms = self.run_inference_from_nc(nc_dir, timestep)
        return self.emit_event(signal, latency_ms, source="surya_bench")


if __name__ == "__main__":
    agent = SolarVisionAgent()
    # Live cycle test
    event = agent.run_live_cycle()
    print(json.dumps(event, indent=2))
