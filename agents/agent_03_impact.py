"""
Agent 03 — Infrastructure Impact Mapper
Translates Kp index into per-sector infrastructure risk and generates a Folium map.
Pure physics/lookup — no LLM, no Qwen-VL (removed from scope).
"""
import folium
from datetime import datetime, timezone


KP_LATITUDE = {9: 45, 8: 50, 7: 55, 6: 60, 5: 65, 4: 70, 3: 75, 2: 80, 1: 90}

KP_IMPACTS = {
    9: {
        "power_grids":    "Widespread outages likely above 45°. Major transformer damage risk.",
        "gps":            "Complete GPS outages. Navigation unreliable globally.",
        "aviation":       "All polar routes must reroute. Radiation exposure risk at any altitude.",
        "satellites":     "All LEO satellites at elevated risk. GEO anomalies expected.",
        "communications": "HF radio blackout global. Satellite comms degraded.",
    },
    7: {
        "power_grids":    "Possible outages above 55°. Reduce load, monitor GIC levels.",
        "gps":            "Errors up to 10m at high latitudes. Precision agriculture affected.",
        "aviation":       "Polar diversions recommended. HF radio outages increasing.",
        "satellites":     "LEO above 55° at risk. Atmospheric drag increase expected.",
        "communications": "HF disruption above 55°. Some satellite uplink degradation.",
    },
    5: {
        "power_grids":    "Minor fluctuations above 65°. Monitor transformer GIC levels.",
        "gps":            "Slight degradation at high latitudes. Sub-meter precision affected.",
        "aviation":       "Polar routes may need monitoring. Minor HF disruption.",
        "satellites":     "Minimal risk. Slight drag increase at very low orbits.",
        "communications": "Minor HF disruption at high latitudes.",
    },
    0: {
        "general": "No significant impacts expected.",
    },
}

KEY_INFRASTRUCTURE = [
    ("Power Grid — Scandinavia", 65.0, 17.0),
    ("Power Grid — Quebec Canada", 52.0, -72.0),
    ("Power Grid — Siberia", 62.0, 105.0),
    ("Satellite Ground — Svalbard", 78.2, 15.6),
    ("Satellite Ground — Alaska", 64.8, -147.7),
    ("Polar Aviation Route", 85.0, 0.0),
]


def _impacts_for_kp(kp: float) -> dict:
    for threshold in sorted(KP_IMPACTS.keys(), reverse=True):
        if int(kp) >= threshold:
            return KP_IMPACTS[threshold]
    return KP_IMPACTS[0]


def _severity_color(kp: float) -> str:
    if kp >= 7: return "red"
    if kp >= 5: return "orange"
    if kp >= 3: return "yellow"
    return "green"


class ImpactMapperAgent:

    def generate_risk_map(self, kp: float, output_path: str = "dashboard/risk_map.html") -> str:
        m = folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB dark_matter")
        color = _severity_color(kp)
        affected_lat = KP_LATITUDE.get(min(9, int(kp)), 90)

        if affected_lat < 90:
            for bounds, label in [
                ([[affected_lat, -180], [90, 180]], f"Risk zone — above {affected_lat}°N"),
                ([[-90, -180], [-affected_lat, 180]], f"Risk zone — below {affected_lat}°S"),
            ]:
                folium.Rectangle(
                    bounds=bounds, color=color, fill=True,
                    fill_color=color, fill_opacity=0.2, popup=label
                ).add_to(m)

        for name, lat, lon in KEY_INFRASTRUCTURE:
            folium.CircleMarker(
                [lat, lon], radius=8, color=color, fill=True,
                fill_opacity=0.8, popup=f"{name}: at risk (Kp={kp:.1f})"
            ).add_to(m)

        m.save(output_path)
        return output_path

    def run(self, physics_event: dict) -> dict:
        kp = float(physics_event.get("kp_estimated", 0))
        impacts = _impacts_for_kp(kp)
        affected_lat = KP_LATITUDE.get(min(9, int(kp)), 90)
        map_path = self.generate_risk_map(kp)

        return {
            "agent": "agent_03_impact",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "kp": round(kp, 1),
            "storm_class": physics_event.get("storm_class", "unknown"),
            "affected_latitude_poleward_of": affected_lat,
            "impacts": impacts,
            "risk_map_path": map_path,
        }


if __name__ == "__main__":
    import json
    agent = ImpactMapperAgent()
    result = agent.run({"kp_estimated": 7.2, "storm_class": "G3 (strong)"})
    print(json.dumps({k: v for k, v in result.items() if k != "impacts"}, indent=2))
    print("Impacts:", json.dumps(result["impacts"], indent=2))
