"""
March 17, 2015 St. Patrick's Day geomagnetic storm replay (G4 class).
Uses real SuryaBench .nc files — NOT hardcoded strings.

To run this test:
  1. Clone SuryaBench: git clone https://github.com/NASA-IMPACT/SuryaBench
  2. Download March 2015 data per SuryaBench README instructions
  3. Place .nc files under SuryaBench/data/
  4. pytest tests/test_storm_replay.py -v -s
"""
import os
import pytest
import numpy as np
import torch


SURYA_BENCH_PATH = os.environ.get("SURYA_BENCH_PATH", "./SuryaBench")
MARCH_2015_NC = os.path.join(SURYA_BENCH_PATH, "data", "march_2015")


def _surya_bench_available():
    return os.path.isdir(MARCH_2015_NC)


def load_bench_frame(nc_dir: str, timestep: int = 0) -> torch.Tensor:
    """
    Load one SDO frame from SuryaBench .nc files.
    NOTE: Adjust variable name after reading SuryaBench schema:
      python -c "import xarray as xr; ds = xr.open_dataset('<file.nc>'); print(ds)"
    """
    import xarray as xr
    import glob
    files = sorted(glob.glob(os.path.join(nc_dir, "*.nc")))
    assert files, f"No .nc files found in {nc_dir}"

    ds = xr.open_dataset(files[0])
    print(f"SuryaBench variables: {list(ds.data_vars)}")  # discover schema on first run

    # TODO: Replace variable name after reading schema above
    var_name = list(ds.data_vars)[0]
    arr = ds[var_name].isel(time=timestep).values.astype(np.float32)
    arr = arr / (arr.max() + 1e-8)
    return torch.tensor(arr).unsqueeze(0).unsqueeze(0).to("cuda")


@pytest.mark.skipif(not _surya_bench_available(), reason="SuryaBench data not present")
def test_march_2015_storm_replay():
    """Full pipeline replay on March 15, 2015 (2 days before G4 peak)."""
    from agents.agent_01_vision import SolarVisionAgent
    from agents.agent_02_physics import CMEPhysicsAgent
    from agents.agent_03_impact import ImpactMapperAgent
    from agents.agent_04_command import CommandAgent

    vision = SolarVisionAgent()
    physics = CMEPhysicsAgent()
    impact = ImpactMapperAgent()
    command = CommandAgent()

    # Load historical frame
    tensor = load_bench_frame(MARCH_2015_NC, timestep=0)

    # Run real Surya inference on historical data
    flare_prob, latency_ms = vision.run_inference(tensor)
    flare_event = vision.emit_event(flare_prob, latency_ms)
    print(f"\nMarch 15 flare probability: {flare_prob:.3f} (latency {latency_ms:.0f}ms)")

    # Agent 02 still uses live DSCOVR for demo simplicity
    # (historical DSCOVR data for March 2015 would need separate download)
    physics_event = physics.run(flare_event)
    impact_event = impact.run(physics_event)
    alert = command.synthesize(flare_event, physics_event, impact_event)

    assert flare_event is not None
    assert alert.get("severity") in ("WATCH", "WARNING", "ALERT", "ALL_CLEAR")
    assert alert.get("bulletin")
    print(f"Bulletin: {alert['bulletin']}")
    print(f"Severity: {alert['severity']}")
    print("Storm replay test PASSED.")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
