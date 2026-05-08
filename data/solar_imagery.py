"""
Real solar imagery fetcher using NASA Helioviewer public API.
Returns actual SDO AIA and SOHO LASCO images for any historical timestamp.

API: https://api.helioviewer.org/v2/
"""
import io
import requests
from PIL import Image

HELIOVIEWER = "https://api.helioviewer.org/v2"

# SDO AIA wavelength → what it shows
WAVELENGTH_LABELS = {
    "131": "AIA 131Å — Flare plasma (10 MK)",
    "171": "AIA 171Å — Corona (1 MK)",
    "304": "AIA 304Å — Chromosphere",
    "193": "AIA 193Å — Coronal holes",
}


def fetch_sdo_image(timestamp_iso: str, wavelength: str = "131",
                    size: int = 512) -> tuple[Image.Image, str]:
    """
    Fetch real SDO AIA solar image from NASA Helioviewer for a given timestamp.

    timestamp_iso : e.g. '2024-05-08T21:00:00Z'
    wavelength    : '131' (flare, hot), '171' (corona, golden), '304' (red)
    Returns (PIL Image, source_label)
    """
    r = requests.get(
        f"{HELIOVIEWER}/takeScreenshot/",
        params={
            "date":       timestamp_iso,
            "imageScale": 2.4,
            "layers":     f"[SDO,AIA,AIA,{wavelength},1,100]",
            "x0": 0, "y0": 0,
            "width":      size,
            "height":     size,
            "display":    "true",
            "watermark":  "false",
        },
        timeout=30,
    )
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content))
    label = f"NASA SDO {WAVELENGTH_LABELS.get(wavelength, wavelength)} — {timestamp_iso[:10]}"
    return img, label


def fetch_lasco_c3_image(timestamp_iso: str, size: int = 512) -> tuple[Image.Image, str]:
    """
    Fetch SOHO LASCO C3 coronagraph image — shows CME leaving the Sun.
    Field of view: 3.7–30 solar radii.
    """
    r = requests.get(
        f"{HELIOVIEWER}/takeScreenshot/",
        params={
            "date":       timestamp_iso,
            "imageScale": 56.0,
            "layers":     "[SOHO,LASCO,C3,white-light,1,100]",
            "x0": 0, "y0": 0,
            "width":      size,
            "height":     size,
            "display":    "true",
            "watermark":  "false",
        },
        timeout=30,
    )
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content))
    label = f"SOHO LASCO C3 (CME coronagraph) — {timestamp_iso[:10]}"
    return img, label


def fetch_sdo_composite(timestamp_iso: str, size: int = 512) -> tuple[Image.Image, str]:
    """
    SDO AIA 131+171 composite — shows both hot flare plasma and coronal loops.
    """
    r = requests.get(
        f"{HELIOVIEWER}/takeScreenshot/",
        params={
            "date":       timestamp_iso,
            "imageScale": 2.4,
            "layers":     "[SDO,AIA,AIA,131,1,70],[SDO,AIA,AIA,171,1,50]",
            "x0": 0, "y0": 0,
            "width":      size,
            "height":     size,
            "display":    "true",
            "watermark":  "false",
        },
        timeout=30,
    )
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content))
    label = f"NASA SDO AIA 131+171 composite — {timestamp_iso[:10]}"
    return img, label


def get_storm_images(timestamp_iso: str) -> dict:
    """
    Fetch all relevant images for a storm phase timestamp.
    Returns dict with keys: sdo_131, sdo_171, lasco_c3
    Falls back gracefully if any image fails.
    """
    images = {}
    for key, fn in [
        ("sdo_131",  lambda: fetch_sdo_image(timestamp_iso, "131")),
        ("sdo_171",  lambda: fetch_sdo_image(timestamp_iso, "171")),
        ("lasco_c3", lambda: fetch_lasco_c3_image(timestamp_iso)),
    ]:
        try:
            img, label = fn()
            images[key] = {"image": img, "label": label}
        except Exception as e:
            images[key] = {"image": None, "label": f"Unavailable: {e}"}
    return images
