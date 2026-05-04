"""Test Agent 02 physics model with live DSCOVR data."""
import pytest
from agents.agent_02_physics import CMEPhysicsAgent


@pytest.fixture(scope="module")
def agent():
    return CMEPhysicsAgent()


def test_run_returns_required_fields(agent):
    result = agent.run()
    for key in ("bz_nT", "solar_wind_speed_kms", "kp_estimated", "storm_class", "detection_mode"):
        assert key in result, f"missing field: {key}"


def test_kp_in_valid_range(agent):
    result = agent.run()
    assert 0.0 <= result["kp_estimated"] <= 9.0, f"Kp out of range: {result['kp_estimated']}"


def test_storm_class_matches_kp(agent):
    result = agent.run()
    kp = result["kp_estimated"]
    cls = result["storm_class"]
    if kp < 5:
        assert "G0" in cls
    elif kp < 6:
        assert "G1" in cls


def test_kp_estimate_southward_bz():
    agent = CMEPhysicsAgent()
    # Strong southward Bz at high speed should give high Kp
    kp = agent.estimate_kp(bz=-30.0, sw_speed=700.0)
    assert kp >= 7.0, f"Expected high Kp for Bz=-30, speed=700: got {kp}"


def test_kp_estimate_northward_bz():
    agent = CMEPhysicsAgent()
    kp = agent.estimate_kp(bz=5.0, sw_speed=500.0)
    assert kp == 0.0, f"Northward Bz should give Kp=0, got {kp}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
