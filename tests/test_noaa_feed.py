"""Verify all NOAA live endpoints return valid data."""
import pytest
from data.noaa_feed import fetch_mag, fetch_plasma, fetch_xray, fetch_kp


def test_mag_returns_bz():
    data = fetch_mag()
    assert "bz" in data, f"bz missing from mag data: {data}"
    assert data["bz"] not in (None, "null", ""), f"bz is null: {data}"


def test_plasma_returns_speed():
    data = fetch_plasma()
    assert "speed" in data, f"speed missing from plasma data: {data}"
    speed = float(data["speed"])
    assert 200 <= speed <= 2000, f"speed out of plausible range: {speed}"


def test_xray_returns_flux():
    data = fetch_xray()
    assert data, "xray data empty"


def test_kp_returns_index():
    data = fetch_kp()
    assert data, "kp data empty"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
