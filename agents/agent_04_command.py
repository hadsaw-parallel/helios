"""
Agent 04 — Command LLM (True ReAct Agent)

Implements the Reasoning + Acting (ReAct) loop:
  Thought → Action → Observation → Thought → ... → issue_alert

The LLM autonomously decides which tools to call and in what order.
It loops until it has enough information to issue a confident alert.
This makes it a true agent, not a pipeline stage.
"""
import json
import re
import requests
from datetime import datetime, timezone


# ── ReAct system prompt ───────────────────────────────────────────────────────
REACT_SYSTEM = """You are HELIOS Agent 04, an autonomous space weather intelligence agent.
You receive raw sensor readings and must reason step-by-step before issuing any alert.

Available tools (call each at most once per session):
  check_flare_severity(flare_prob, flare_class)         -> assesses solar flare risk
  check_storm_strength(kp, bz_nT, speed_kms)            -> assesses geomagnetic storm severity
  identify_at_risk_infrastructure(kp)                   -> lists sites inside the risk zone
  issue_alert(severity, bulletin, actions, confidence)  -> emits the final alert (ENDS the loop)

severity must be one of: WATCH | WARNING | ALERT | ALL_CLEAR

Output exactly one step per turn in this format (no other text):
Thought: <reason about what you know and which tool to call next>
Action: <tool_name>
Action Input: <JSON dict of arguments>
"""

# ── Tool implementations ──────────────────────────────────────────────────────
_KP_LATITUDE = {9: 45, 8: 50, 7: 55, 6: 60, 5: 65, 4: 70, 3: 75, 2: 80, 1: 90}

_INFRA_SITES = [
    ("Power Grid — Scandinavia", 65.0),
    ("Power Grid — Quebec Canada", 52.0),
    ("Power Grid — Siberia", 62.0),
    ("Satellite Ground — Svalbard", 78.2),
    ("Satellite Ground — Alaska", 64.8),
    ("Polar Aviation Route", 85.0),
]


def _check_flare_severity(flare_prob: float, flare_class: str) -> str:
    if flare_prob > 0.85 or flare_class == "X-class":
        return (
            "X-class flare detected. Extreme UV/X-ray radiation reaching Earth now. "
            "High CME probability. Immediate protective action required for vulnerable systems."
        )
    if flare_prob > 0.6 or flare_class == "M-class":
        return (
            "M-class flare detected. Elevated solar activity. "
            "CME possible — monitor for Earth-directed component over next 24 hours."
        )
    if flare_prob > 0.3 or flare_class == "C-class":
        return "C-class flare. Minor activity. No significant impact unless conditions escalate."
    return "Background solar activity (A/B class). Conditions nominal."


def _check_storm_strength(kp: float, bz_nT: float, speed_kms: float) -> str:
    if kp >= 8:
        return (
            f"G4-G5 SEVERE geomagnetic storm. Kp={kp:.1f}, Bz={bz_nT:.1f} nT southward, "
            f"solar wind {speed_kms:.0f} km/s. Major infrastructure threat poleward of 50°. "
            "Transformer damage risk, GPS blackout, all polar routes must divert."
        )
    if kp >= 7:
        return (
            f"G3 STRONG storm. Kp={kp:.1f}, Bz={bz_nT:.1f} nT southward. "
            "Power grid stress above 55°. Polar flight diversions strongly recommended. "
            "HF radio outages likely."
        )
    if kp >= 5:
        return (
            f"G1-G2 moderate storm. Kp={kp:.1f}. Minor power grid fluctuations above 65°. "
            "HF radio disruption at high latitudes. Satellite drag elevated."
        )
    if kp >= 3:
        return (
            f"Elevated activity. Kp={kp:.1f}. Below storm threshold. "
            "Monitor Bz — further southward turning could trigger storm onset."
        )
    return f"Quiet conditions. Kp={kp:.1f}, Bz={bz_nT:.1f} nT. No storm impacts expected."


def _identify_at_risk_infrastructure(kp: float) -> str:
    lat_threshold = _KP_LATITUDE.get(min(9, int(kp)), 90)
    at_risk = [name for name, lat in _INFRA_SITES if lat >= lat_threshold]
    return json.dumps({
        "risk_poleward_of_degrees": lat_threshold,
        "at_risk_sites": at_risk if at_risk else ["None — conditions too mild for infrastructure impacts"],
        "note": f"All sites above {lat_threshold}° latitude are in the auroral risk zone.",
    })


_TOOLS = {
    "check_flare_severity": _check_flare_severity,
    "check_storm_strength": _check_storm_strength,
    "identify_at_risk_infrastructure": _identify_at_risk_infrastructure,
}


# ── Agent class ───────────────────────────────────────────────────────────────
class CommandAgent:

    def __init__(self, vllm_url: str = "http://localhost:8000", model: str = None):
        self.url = f"{vllm_url}/v1/completions"
        self.model = model or "meta-llama/Meta-Llama-3-8B"

    def _call_llm(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": 350,
            "temperature": 0.1,
            # Stop before the LLM hallucinate its own Observation
            "stop": ["Observation:", "\n\n\n", "---END---"],
        }
        r = requests.post(self.url, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()["choices"][0]["text"].strip()

    def _parse_step(self, text: str) -> tuple[str, str, dict]:
        """Extract Thought, Action, Action Input from one LLM output."""
        thought_m = re.search(r"Thought:\s*(.+?)(?=\nAction:|\Z)", text, re.DOTALL)
        action_m = re.search(r"Action:\s*(\w+)", text)
        input_m = re.search(r"Action Input:\s*(\{.+?\})", text, re.DOTALL)

        thought = thought_m.group(1).strip() if thought_m else text.strip()
        action = action_m.group(1).strip() if action_m else ""
        try:
            action_input = json.loads(input_m.group(1)) if input_m else {}
        except (json.JSONDecodeError, AttributeError):
            # Try to salvage key-value pairs even if JSON is malformed
            action_input = {}
            if input_m:
                for m in re.finditer(r'"(\w+)":\s*"?([^",}]+)"?', input_m.group(1)):
                    k, v = m.group(1), m.group(2).strip()
                    try:
                        action_input[k] = float(v)
                    except ValueError:
                        action_input[k] = v
        return thought, action, action_input

    def synthesize(self, flare: dict, physics: dict, impact: dict) -> dict:
        """
        True ReAct loop. The LLM decides which tools to call and loops
        until it calls issue_alert — at which point the loop terminates.
        Max 6 steps to prevent runaway loops.
        """
        # Raw sensor data — not pre-interpreted; agent must reason about it
        context = (
            "Raw sensor readings:\n"
            f"  flare_probability : {flare.get('flare_probability', 0):.2%}\n"
            f"  flare_class       : {flare.get('severity', 'unknown')}\n"
            f"  DSCOVR Bz         : {physics.get('bz_nT', 0)} nT\n"
            f"  solar_wind_speed  : {physics.get('solar_wind_speed_kms', 0)} km/s\n"
            f"  proton_density    : {physics.get('proton_density', 0)} p/cc\n"
            f"  estimated_Kp      : {physics.get('kp_estimated', 0)}\n"
            f"  storm_class       : {physics.get('storm_class', 'unknown')}\n"
        )

        # Seed the conversation
        prompt = REACT_SYSTEM + "\n\n" + context + "\nBegin:\n"

        steps = []
        alert = None

        for _ in range(6):
            llm_output = self._call_llm(prompt)
            thought, action_name, action_input = self._parse_step(llm_output)

            if action_name == "issue_alert":
                alert = {
                    "severity": action_input.get("severity", "WATCH"),
                    "bulletin": action_input.get("bulletin", ""),
                    "recommended_actions": action_input.get("actions", []),
                    "confidence": action_input.get("confidence", "MEDIUM"),
                    "next_update_minutes": 15,
                }
                observation = "Alert issued. Reasoning loop complete."
                steps.append({
                    "thought": thought,
                    "action": action_name,
                    "input": action_input,
                    "observation": observation,
                })
                # Append final step to prompt (for completeness)
                prompt += (
                    f"Thought: {thought}\n"
                    f"Action: {action_name}\n"
                    f"Action Input: {json.dumps(action_input)}\n"
                    f"Observation: {observation}\n"
                )
                break

            tool_fn = _TOOLS.get(action_name)
            if tool_fn:
                try:
                    observation = tool_fn(**action_input)
                except Exception as e:
                    observation = f"Tool error: {e}"
            else:
                observation = (
                    f"Unknown tool '{action_name}'. "
                    f"Available: {list(_TOOLS.keys())} or issue_alert."
                )

            steps.append({
                "thought": thought,
                "action": action_name,
                "input": action_input,
                "observation": observation,
            })

            # Append this full step to the prompt — LLM sees its own reasoning history
            prompt += (
                f"Thought: {thought}\n"
                f"Action: {action_name}\n"
                f"Action Input: {json.dumps(action_input)}\n"
                f"Observation: {observation}\n"
            )

        # Fallback if LLM exhausted steps without calling issue_alert
        if not alert:
            kp = physics.get("kp_estimated", 0)
            if kp >= 7:
                sev, conf = "ALERT", "MEDIUM"
            elif kp >= 5:
                sev, conf = "WARNING", "MEDIUM"
            elif kp >= 3:
                sev, conf = "WATCH", "LOW"
            else:
                sev, conf = "ALL_CLEAR", "HIGH"
            alert = {
                "severity": sev,
                "bulletin": (
                    f"Automated fallback after {len(steps)} reasoning steps. "
                    f"Kp={kp:.1f}, storm class {physics.get('storm_class', 'unknown')}."
                ),
                "recommended_actions": ["Monitor SWPC bulletins", "Continue normal operations"],
                "confidence": conf,
                "next_update_minutes": 15,
            }

        alert["timestamp"] = datetime.now(timezone.utc).isoformat()
        alert["agent"] = "agent_04_command"
        alert["model_used"] = self.model
        alert["react_steps"] = len(steps)
        alert["reasoning_trace"] = steps
        return alert


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    agent = CommandAgent()
    dummy_flare = {
        "flare_probability": 0.82, "flare_detected": True,
        "severity": "M-class", "inference_ms": 145,
    }
    dummy_physics = {
        "bz_nT": -18.4, "solar_wind_speed_kms": 620, "proton_density": 12.3,
        "kp_estimated": 7.2, "storm_class": "G3 (strong)", "detection_mode": "L1_realtime",
    }
    dummy_impact = {
        "affected_latitude_poleward_of": 55,
        "impacts": {
            "power_grids": "Outages possible above 55°.",
            "gps": "10m errors at high latitudes.",
        },
    }
    result = agent.synthesize(dummy_flare, dummy_physics, dummy_impact)
    trace = result.pop("reasoning_trace", [])
    print(json.dumps(result, indent=2))
    print(f"\n--- Reasoning trace ({len(trace)} steps) ---")
    for i, step in enumerate(trace, 1):
        print(f"\nStep {i}:")
        print(f"  Thought    : {step['thought'][:120]}")
        print(f"  Action     : {step['action']}")
        print(f"  Observation: {str(step['observation'])[:120]}")
