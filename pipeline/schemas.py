from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FlareEvent:
    timestamp: str
    flare_probability: float
    flare_detected: bool
    severity: str
    inference_ms: float
    vram_gb: float = 0.0
    agent: str = "agent_01_vision"


@dataclass
class PhysicsEvent:
    timestamp: str
    bz_nT: float
    solar_wind_speed_kms: float
    proton_density: float
    kp_estimated: float
    storm_class: str
    detection_mode: str
    agent: str = "agent_02_physics"


@dataclass
class ImpactEvent:
    timestamp: str
    kp: float
    storm_class: str
    affected_latitude_poleward_of: int
    impacts: dict
    risk_map_path: str
    agent: str = "agent_03_impact"


@dataclass
class AlertEvent:
    timestamp: str
    severity: str        # WATCH | WARNING | ALERT | ALL_CLEAR
    bulletin: str
    recommended_actions: list
    confidence: str      # HIGH | MEDIUM | LOW
    next_update_minutes: int
    model_used: str = ""
    agent: str = "agent_04_command"
