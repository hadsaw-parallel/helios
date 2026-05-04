"""
SDO (Solar Dynamics Observatory) image fetcher.
Downloads latest AIA images from NASA's public endpoint and converts to tensors.
"""
import os
import io
import requests
import numpy as np
from PIL import Image


SDO_BASE = "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_{wavelength}.jpg"

# Available AIA wavelength channels
WAVELENGTHS = ["0094", "0131", "0171", "0193", "0211", "0304", "0335", "1600"]


def fetch_latest_image(wavelength: str = "0171", save_dir: str = "./data/sdo_cache") -> str:
    """Download latest SDO AIA image for the given wavelength. Returns local path."""
    os.makedirs(save_dir, exist_ok=True)
    url = SDO_BASE.format(wavelength=wavelength)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    path = os.path.join(save_dir, f"aia_{wavelength}_latest.jpg")
    with open(path, "wb") as f:
        f.write(r.content)
    return path


def image_bytes_to_tensor(raw_bytes: bytes) -> "np.ndarray":
    """Convert raw image bytes to normalized float32 numpy array (H, W)."""
    img = Image.open(io.BytesIO(raw_bytes)).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def fetch_as_tensor(wavelength: str = "0171") -> "np.ndarray":
    """Fetch latest SDO image and return as normalized numpy array without saving."""
    url = SDO_BASE.format(wavelength=wavelength)
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return image_bytes_to_tensor(r.content)


def fetch_multichannel_tensor(wavelengths: list = None) -> "np.ndarray":
    """
    Fetch multiple wavelength channels and stack into (C, H, W) tensor.
    Surya expects multi-channel input — use this when feeding the model.
    """
    if wavelengths is None:
        wavelengths = ["0131", "0171", "0193"]  # flare + corona + EUV
    channels = [fetch_as_tensor(w) for w in wavelengths]
    return np.stack(channels, axis=0)  # (C, H, W)


if __name__ == "__main__":
    print("Testing SDO fetch...")
    arr = fetch_as_tensor("0171")
    print(f"Single channel shape: {arr.shape}, min: {arr.min():.3f}, max: {arr.max():.3f}")

    multi = fetch_multichannel_tensor()
    print(f"Multi-channel shape: {multi.shape}")
    print("SDO fetch OK.")
