"""
Agent 04 — Command LLM
Synthesizes all agent outputs into a plain-language operator alert bulletin.
Uses Llama 3.1 70B (or 8B fallback) served by vLLM.
"""
import json
import requests
from datetime import datetime, timezone


SYSTEM_PROMPT = """You are HELIOS, an automated space weather intelligence system.
Your job is to synthesize solar observation data into clear, actionable operator alerts.

Respond ONLY with valid JSON in this exact format — no prose outside the JSON:
{
  "severity": "WATCH|WARNING|ALERT|ALL_CLEAR",
  "bulletin": "2-3 sentence plain language summary. Include timing, affected regions, and what operators should do.",
  "recommended_actions": ["action 1", "action 2", "action 3"],
  "confidence": "HIGH|MEDIUM|LOW",
  "next_update_minutes": 15
}

Rules:
- WATCH: elevated solar activity, no confirmed Earth-directed event
- WARNING: CME detected and possibly Earth-directed, arrival uncertain
- ALERT: storm imminent or in progress (Kp >= 5)
- ALL_CLEAR: conditions nominal
- Be specific about timing, latitude bands, and affected infrastructure.
- Do not speculate beyond what the data supports.
"""


class CommandAgent:

    def __init__(self, vllm_url: str = "http://localhost:8000", model: str = None):
        self.url = f"{vllm_url}/v1/chat/completions"
        # Model is set at runtime so we can swap 70B / 8B without code changes
        self.model = model or "meta-llama/Llama-3.1-70B-Instruct"

    def _build_context(self, flare: dict, physics: dict, impact: dict) -> str:
        return f"""SOLAR VISION (Agent 01 — Surya-1.0 on AMD MI300X):
- Flare probability: {flare.get('flare_probability', 0):.2%}
- Flare detected: {flare.get('flare_detected')}
- Severity class: {flare.get('severity')}
- Surya inference latency: {flare.get('inference_ms')}ms

CME PHYSICS (Agent 02 — DSCOVR L1 realtime):
- DSCOVR Bz: {physics.get('bz_nT')} nT
- Solar wind speed: {physics.get('solar_wind_speed_kms')} km/s
- Proton density: {physics.get('proton_density')} p/cc
- Estimated Kp: {physics.get('kp_estimated')}
- Storm class: {physics.get('storm_class')}
- Detection mode: {physics.get('detection_mode')}

INFRASTRUCTURE IMPACT (Agent 03):
- Affected latitude: poleward of {impact.get('affected_latitude_poleward_of')}°
- Power grids: {impact.get('impacts', {}).get('power_grids', 'N/A')}
- GPS: {impact.get('impacts', {}).get('gps', 'N/A')}
- Aviation: {impact.get('impacts', {}).get('aviation', 'N/A')}
- Satellites: {impact.get('impacts', {}).get('satellites', 'N/A')}
"""

    def synthesize(self, flare: dict, physics: dict, impact: dict) -> dict:
        context = self._build_context(flare, physics, impact)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Generate operator alert:\n{context}"},
            ],
            "max_tokens": 512,
            "temperature": 0.1,
        }
        r = requests.post(self.url, json=payload, timeout=60)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]

        # Strip markdown fences if model wraps JSON in ```
        content = content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        alert = json.loads(content)
        alert["timestamp"] = datetime.now(timezone.utc).isoformat()
        alert["agent"] = "agent_04_command"
        alert["model_used"] = self.model
        return alert


if __name__ == "__main__":
    # Smoke test with dummy data
    agent = CommandAgent()
    dummy_flare = {"flare_probability": 0.82, "flare_detected": True, "severity": "M-class", "inference_ms": 145}
    dummy_physics = {"bz_nT": -18.4, "solar_wind_speed_kms": 620, "proton_density": 12.3,
                     "kp_estimated": 7.2, "storm_class": "G3 (strong)", "detection_mode": "L1_realtime"}
    dummy_impact = {"affected_latitude_poleward_of": 55,
                    "impacts": {"power_grids": "Outages possible above 55°.",
                                "gps": "10m errors at high latitudes.",
                                "aviation": "Polar diversions recommended.",
                                "satellites": "LEO at risk above 55°."}}
    result = agent.synthesize(dummy_flare, dummy_physics, dummy_impact)
    print(json.dumps(result, indent=2))
