"""
Historical space weather data fetcher for counterfactual storm replay.

All values are fetched live from public archives — nothing is hardcoded.

Sources:
  NASA DONKI  - GOES solar flare events (class, peak time)
                https://api.nasa.gov/DONKI/FLR
  GFZ Potsdam - Planetary Kp index (authoritative, 3-hour resolution)
                https://kp.gfz-potsdam.de/app/json/
  NASA OMNIWeb - Hourly solar wind Bz, speed, density at Earth's L1 point
                https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi
"""
import requests
from datetime import datetime, timedelta, timezone

NASA_DONKI = "https://api.nasa.gov/DONKI"
GFZ_KP_API = "https://kp.gfz-potsdam.de/app/json/"
OMNI_URL   = "https://omniweb.gsfc.nasa.gov/cgi/nx1.cgi"
NASA_KEY   = "DEMO_KEY"   # free tier — get a key at api.nasa.gov if rate-limited


# ── 1. GOES flares from NASA DONKI ────────────────────────────────────────────

def _class_to_signal(cls: str) -> tuple[float, str]:
    """Convert GOES flare class string (e.g. 'X3.98') to (prob 0-1, label)."""
    c = (cls or "").upper().strip()
    if c.startswith("X"):
        mag = float(c[1:]) if len(c) > 1 else 1.0
        return min(1.0, 0.85 + mag * 0.015), "X-class"
    if c.startswith("M"):
        mag = float(c[1:]) if len(c) > 1 else 1.0
        return min(0.84, 0.60 + mag * 0.024), "M-class"
    if c.startswith("C"):
        return 0.45, "C-class"
    return 0.10, "B-class"


def fetch_goes_flares(start_date: str, end_date: str) -> list[dict]:
    """
    Fetch GOES solar flare events from NASA DONKI for a date range.
    start_date / end_date: 'YYYY-MM-DD'
    Returns list sorted by peak time (ascending).
    """
    r = requests.get(
        f"{NASA_DONKI}/FLR",
        params={"startDate": start_date, "endDate": end_date, "api_key": NASA_KEY},
        timeout=15,
    )
    r.raise_for_status()
    events = r.json() or []
    out = []
    for ev in events:
        cls = ev.get("classType") or "C1.0"
        prob, severity = _class_to_signal(cls)
        out.append({
            "timestamp": ev.get("peakTime", ""),
            "flare_class": cls,
            "flare_probability": round(prob, 3),
            "severity": severity,
            "source": "NASA DONKI /FLR",
        })
    return sorted(out, key=lambda x: x["timestamp"])


def worst_flare(start_date: str, end_date: str) -> dict:
    """Return the highest-probability flare in the window, or quiet baseline."""
    flares = fetch_goes_flares(start_date, end_date)
    if not flares:
        return {"flare_probability": 0.05, "severity": "A-class",
                "flare_class": "A1.0", "source": "NASA DONKI /FLR (none found)"}
    return max(flares, key=lambda f: f["flare_probability"])


# ── 2. Kp index from GFZ Potsdam ──────────────────────────────────────────────

def fetch_kp(start_iso: str, end_iso: str) -> list[dict]:
    """
    Fetch 3-hourly Kp from GFZ Potsdam (the authoritative Kp source).
    start_iso / end_iso: ISO 8601, e.g. '2024-05-08T00:00:00Z'
    Returns list of {datetime, kp}.
    """
    r = requests.get(
        GFZ_KP_API,
        params={"start": start_iso, "end": end_iso, "index": "Kp"},
        timeout=15,
    )
    r.raise_for_status()
    data = r.json()
    times = data.get("datetime", [])
    vals  = data.get("Kp", [])
    return [{"datetime": t, "kp": float(k)} for t, k in zip(times, vals)]


def peak_kp(start_iso: str, end_iso: str) -> float:
    """Return the single highest Kp value in the window."""
    records = fetch_kp(start_iso, end_iso)
    if not records:
        return 0.0
    return max(r["kp"] for r in records)


# ── 3. Solar wind from NASA OMNIWeb ───────────────────────────────────────────

def fetch_omni_solar_wind(start_yyyymmdd: str, end_yyyymmdd: str) -> list[dict]:
    """
    Fetch hourly solar wind data from NASA OMNIWeb OMNI2 dataset.
    Returns list of {bz_gsm, speed_kms, density} dicts.
    Fill values (9999.9) are filtered out automatically.

    OMNIWeb OMNI2 variable IDs used:
      17 = BZ, GSM (nT)
      24 = Flow Speed (km/s)
      25 = Proton Density (N/cc)
    """
    params = {
        "activity": "retrieve",
        "res":       "hour",
        "spacecraft":"omni2",
        "start_date": start_yyyymmdd,
        "end_date":   end_yyyymmdd,
        "vars":       "17,24,25",
        "submit":     "Submit",
    }
    r = requests.get(OMNI_URL, params=params, timeout=30)
    r.raise_for_status()

    rows = []
    for line in r.text.splitlines():
        parts = line.split()
        # Data lines: YEAR DOY HR BZ SPEED DENSITY
        if len(parts) < 6 or not parts[0].isdigit():
            continue
        try:
            bz      = float(parts[3])
            speed   = float(parts[4])
            density = float(parts[5])
        except (ValueError, IndexError):
            continue
        # OMNIWeb fill values: 9999.9 for most variables
        if bz > 999 or speed > 9990 or density > 990:
            continue
        rows.append({"bz_gsm": bz, "speed_kms": speed, "density": density})
    return rows


def peak_solar_wind(start_yyyymmdd: str, end_yyyymmdd: str) -> dict:
    """
    Return peak solar wind conditions (most negative Bz, max speed)
    for the window. Falls back to quiet-Sun defaults if OMNIWeb fails.
    """
    try:
        rows = fetch_omni_solar_wind(start_yyyymmdd, end_yyyymmdd)
    except Exception:
        rows = []

    if not rows:
        return {"bz_gsm": 0.0, "speed_kms": 450.0, "density": 5.0,
                "source": "OMNIWeb unavailable — quiet-Sun default"}

    min_bz   = min(r["bz_gsm"]    for r in rows)
    max_spd  = max(r["speed_kms"] for r in rows)
    med_dens = sorted(r["density"] for r in rows)[len(rows) // 2]
    return {
        "bz_gsm":   round(min_bz,  1),
        "speed_kms": round(max_spd, 0),
        "density":  round(med_dens, 1),
        "source":   "NASA OMNIWeb OMNI2 hourly",
    }


# ── 4. Composite snapshot for pipeline input ──────────────────────────────────

def _kp_to_storm_class(kp: float) -> str:
    if kp < 5:  return "G0 (no storm)"
    if kp < 6:  return "G1 (minor)"
    if kp < 7:  return "G2 (moderate)"
    if kp < 8:  return "G3 (strong)"
    if kp < 9:  return "G4 (severe)"
    return "G5 (extreme)"


def build_pipeline_snapshot(target_iso: str, lookback_hours: int = 12) -> dict:
    """
    Build a pipeline-compatible data snapshot for a historical moment.
    All values fetched live from NASA DONKI, GFZ Potsdam, and NASA OMNIWeb.

    target_iso:      ISO 8601 timestamp, e.g. '2024-05-08T21:00:00Z'
    lookback_hours:  how far back to search for the active flare / Kp window

    Returns a dict with the same keys expected by pipeline/orchestrator.py:
      flare_event  — input for Agent 01
      physics_event — input for Agent 02
      should_alert  — whether the pipeline should escalate to Agents 03 + 04
    """
    dt = datetime.fromisoformat(target_iso.replace("Z", "+00:00"))
    window_start = dt - timedelta(hours=lookback_hours)

    start_date = window_start.strftime("%Y-%m-%d")
    end_date   = dt.strftime("%Y-%m-%d")
    start_iso  = window_start.strftime("%Y-%m-%dT%H:%M:%SZ")
    end_iso    = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    start_ymd  = window_start.strftime("%Y%m%d")
    end_ymd    = dt.strftime("%Y%m%d")

    flare = worst_flare(start_date, end_date)
    kp    = peak_kp(start_iso, end_iso)
    sw    = peak_solar_wind(start_ymd, end_ymd)

    flare_event = {
        "agent":             "agent_01_vision",
        "timestamp":          target_iso,
        "flare_probability":  flare["flare_probability"],
        "flare_detected":     flare["flare_probability"] > 0.6,
        "severity":           flare["severity"],
        "flare_class":        flare["flare_class"],
        "source":             flare.get("source", "NASA DONKI"),
        "inference_ms":       0,
        "vram_gb":            0,
        "data_fetched_from":  "NASA DONKI /FLR (api.nasa.gov)",
    }

    physics_event = {
        "agent":                 "agent_02_physics",
        "timestamp":              target_iso,
        "bz_nT":                  sw["bz_gsm"],
        "solar_wind_speed_kms":   sw["speed_kms"],
        "proton_density":         sw["density"],
        "kp_estimated":           kp,
        "storm_class":            _kp_to_storm_class(kp),
        "detection_mode":         "historical_archive",
        "data_fetched_from":      f"GFZ Potsdam Kp + {sw.get('source', 'OMNIWeb')}",
    }

    return {
        "flare_event":   flare_event,
        "physics_event": physics_event,
        "impact_event":  None,
        "alert_event":   None,
        "should_alert":  flare_event["flare_detected"],
        "meta": {
            "target_timestamp":    target_iso,
            "lookback_hours":      lookback_hours,
            "flare_source":        "NASA DONKI /FLR",
            "kp_source":           "GFZ Potsdam Kp API",
            "solar_wind_source":   sw.get("source", "NASA OMNIWeb"),
        },
    }


if __name__ == "__main__":
    import json
    # Test: fetch Gannon Storm peak (May 10, 2024)
    print("Fetching May 2024 Gannon Storm data from live archives...")
    snapshot = build_pipeline_snapshot("2024-05-10T17:00:00Z", lookback_hours=12)
    print(json.dumps({
        "flare_class":    snapshot["flare_event"]["flare_class"],
        "flare_prob":     snapshot["flare_event"]["flare_probability"],
        "bz_nT":          snapshot["physics_event"]["bz_nT"],
        "speed_kms":      snapshot["physics_event"]["solar_wind_speed_kms"],
        "kp":             snapshot["physics_event"]["kp_estimated"],
        "storm_class":    snapshot["physics_event"]["storm_class"],
        "sources":        snapshot["meta"],
    }, indent=2))
