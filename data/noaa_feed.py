"""
NOAA real-time space weather data feeds.
Polls DSCOVR magnetic + plasma, GOES X-ray, SWPC Kp index.
"""
import requests
import json
import time
from datetime import datetime

ENDPOINTS = {
    "solar_wind_mag":    "https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json",
    "solar_wind_plasma": "https://services.swpc.noaa.gov/products/solar-wind/plasma-7-day.json",
    "xray_flux":         "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json",
    "alerts":            "https://services.swpc.noaa.gov/products/alerts.json",
    "kp_index":          "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json",
}


def _latest(raw):
    """Most NOAA endpoints return list-of-lists; row[0] is headers, last row is newest."""
    if isinstance(raw, list) and len(raw) > 1:
        headers = raw[0]
        latest = raw[-1]
        return dict(zip(headers, latest))
    return raw


def fetch_mag() -> dict:
    """DSCOVR magnetic field — Bx, By, Bz (nT)."""
    r = requests.get(ENDPOINTS["solar_wind_mag"], timeout=10)
    r.raise_for_status()
    return _latest(r.json())


def fetch_plasma() -> dict:
    """DSCOVR plasma — solar wind speed (km/s), proton density, temperature."""
    r = requests.get(ENDPOINTS["solar_wind_plasma"], timeout=10)
    r.raise_for_status()
    return _latest(r.json())


def fetch_xray() -> dict:
    """GOES X-ray flux — flare class proxy."""
    r = requests.get(ENDPOINTS["xray_flux"], timeout=10)
    r.raise_for_status()
    data = r.json()
    return data[-1] if isinstance(data, list) else data


def fetch_kp() -> dict:
    """NOAA planetary Kp index."""
    r = requests.get(ENDPOINTS["kp_index"], timeout=10)
    r.raise_for_status()
    return _latest(r.json())


def fetch_all() -> dict:
    """Fetch all endpoints; return dict with per-source results and errors."""
    result = {}
    for name, url in ENDPOINTS.items():
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            raw = r.json()
            result[name] = _latest(raw)
            print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] {name}: OK")
        except Exception as e:
            result[name] = {"error": str(e)}
            print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] {name}: ERROR — {e}")
    return result


if __name__ == "__main__":
    while True:
        data = fetch_all()
        print(json.dumps(data, indent=2))
        time.sleep(60)
