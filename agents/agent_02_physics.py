"""
Agent 02 — CME Physics
Reads real-time DSCOVR magnetic + plasma data and estimates storm intensity.
Uses Burton empirical formula for Kp and the Drag-Based Model for context.

Detection mode: L1 realtime (DSCOVR at L1 — 15–60 min lead time before Earth impact).
Upstream CME speed is NOT estimated from flare probability — that has no physical basis.
"""
import requests
import numpy as np
from datetime import datetime, timezone


AU_KM = 1.496e8          # Sun–Earth distance in km
L1_KM = AU_KM - 1.5e6   # Sun–L1 (DSCOVR) distance in km


class CMEPhysicsAgent:

    def fetch_mag(self) -> dict:
        """DSCOVR magnetic field — Bz is key coupling parameter."""
        r = requests.get(
            "https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json",
            timeout=10
        )
        r.raise_for_status()
        data = r.json()
        return dict(zip(data[0], data[-1]))

    def fetch_plasma(self) -> dict:
        """DSCOVR plasma — measured solar wind speed, density, temperature."""
        r = requests.get(
            "https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json",
            timeout=10
        )
        r.raise_for_status()
        data = r.json()
        return dict(zip(data[0], data[-1]))

    def fetch_goes_xray(self) -> dict:
        r = requests.get(
            "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json",
            timeout=10
        )
        r.raise_for_status()
        data = r.json()
        return data[-1] if data else {}

    def estimate_kp(self, bz: float, sw_speed: float) -> float:
        """
        Estimate Kp from DSCOVR Bz and measured solar wind speed.
        Based on Burton et al. (1975) empirical electric field proxy.
        bz: southward component in nT (negative = southward = storm-inducing)
        sw_speed: solar wind bulk speed in km/s from DSCOVR plasma data
        """
        if bz >= 0:
            return 0.0
        E = (sw_speed * abs(bz)) / 1000.0  # mV/m
        if E < 0.5:
            return 1.0
        elif E < 1.0:
            return 2.0 + (E - 0.5) * 2.0
        elif E < 2.0:
            return 3.0 + (E - 1.0) * 2.0
        elif E < 5.0:
            return 5.0 + (E - 2.0)
        else:
            return min(9.0, 8.0 + (E - 5.0) * 0.2)

    def kp_to_storm_class(self, kp: float) -> str:
        if kp < 5: return "G0 (no storm)"
        elif kp < 6: return "G1 (minor)"
        elif kp < 7: return "G2 (moderate)"
        elif kp < 8: return "G3 (strong)"
        elif kp < 9: return "G4 (severe)"
        else: return "G5 (extreme)"

    def run(self, flare_event: dict = None) -> dict:
        mag = self.fetch_mag()
        plasma = self.fetch_plasma()
        goes = self.fetch_goes_xray()

        bz = self._safe_float(mag.get("bz_gsm"), 0.0)
        sw_speed = self._safe_float(plasma.get("speed"), 450.0)
        density = self._safe_float(plasma.get("density"), 5.0)

        kp = self.estimate_kp(bz, sw_speed)
        storm_class = self.kp_to_storm_class(kp)

        return {
            "agent": "agent_02_physics",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "bz_nT": round(bz, 2),
            "solar_wind_speed_kms": round(sw_speed, 1),
            "proton_density": round(density, 2),
            "kp_estimated": round(kp, 1),
            "storm_class": storm_class,
            "goes_flux": goes.get("flux"),
            "detection_mode": "L1_realtime",
        }

    @staticmethod
    def _safe_float(val, default: float) -> float:
        try:
            return float(val)
        except (TypeError, ValueError):
            return default


if __name__ == "__main__":
    import json
    agent = CMEPhysicsAgent()
    print(json.dumps(agent.run(), indent=2))
