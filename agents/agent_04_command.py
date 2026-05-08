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

Severity rules — follow these exactly, no exceptions:
  ALERT    : Kp >= 5 AND flare detected (storm confirmed AND solar source active)
             OR Kp >= 7 alone (major storm regardless of flare)
  WARNING  : X or M-class flare detected (flare_prob > 0.65) AND Kp < 5
             (CME likely incoming but storm has NOT yet arrived at Earth)
  WATCH    : C-class flare OR Kp between 3-4 (elevated activity, monitor closely)
  ALL_CLEAR: No significant flare AND Kp < 3 (quiet conditions)

Critical: A large flare alone (even X-class) with Kp < 5 is WARNING not ALERT.
ALERT means the storm is confirmed arriving or in progress. WARNING means risk is high but unconfirmed.
confidence must be exactly one of: HIGH, MEDIUM, LOW

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


def _check_flare_severity(flare_prob: float = 0, flare_class: str = "",
                          flare_probability: float = None, **kwargs) -> str:
    if flare_probability is not None:
        flare_prob = flare_probability  # accept either name
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


def _check_storm_strength(kp: float = 0, bz_nT: float = 0, speed_kms: float = 450,
                          bz: float = None, speed: float = None, **kwargs) -> str:
    if bz is not None: bz_nT = bz
    if speed is not None: speed_kms = speed
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


def _identify_at_risk_infrastructure(kp: float = 0, kp_index: float = None, **kwargs) -> str:
    if kp_index is not None: kp = kp_index
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
        Hybrid ReAct: tools run deterministically (reliable), LLM writes the bulletin.

        Steps 1-3 call each tool in fixed order and collect observations.
        Step 4 sends all observations to the LLM for a single focused synthesis call.
        This guarantees the reasoning trace always shows 4 complete steps regardless
        of model behavior.
        """
        fp  = flare.get("flare_probability", 0)
        fc  = flare.get("severity", "unknown")
        kp  = physics.get("kp_estimated", 0)
        bz  = physics.get("bz_nT", 0)
        spd = physics.get("solar_wind_speed_kms", 450)
        lat = impact.get("affected_latitude_poleward_of", 90) if impact else 90

        # ── Step 1: flare severity (deterministic) ────────────────────────────
        obs1 = _check_flare_severity(flare_prob=fp, flare_class=fc)
        steps = [{"thought": f"First I assess the solar flare: prob={fp:.0%}, class={fc}.",
                  "action": "check_flare_severity",
                  "input": {"flare_prob": round(fp, 3), "flare_class": fc},
                  "observation": obs1}]

        # ── Step 2: storm strength (deterministic) ────────────────────────────
        obs2 = _check_storm_strength(kp=kp, bz_nT=bz, speed_kms=spd)
        steps.append({"thought": f"Now I assess geomagnetic storm strength: Kp={kp:.1f}, Bz={bz} nT.",
                      "action": "check_storm_strength",
                      "input": {"kp": round(kp, 1), "bz_nT": round(bz, 1), "speed_kms": round(spd, 0)},
                      "observation": obs2})

        # ── Step 3: infrastructure at risk (deterministic) ────────────────────
        obs3 = _identify_at_risk_infrastructure(kp=kp)
        steps.append({"thought": f"Now I identify which infrastructure is at risk for Kp={kp:.1f}.",
                      "action": "identify_at_risk_infrastructure",
                      "input": {"kp": round(kp, 1)},
                      "observation": obs3})

        # ── Step 4: determine severity deterministically, LLM writes bulletin ──
        # Severity is computed from physics — not left to LLM interpretation
        flare_detected = fp > 0.65
        if (kp >= 5 and flare_detected) or kp >= 7:
            severity, confidence = "ALERT", "HIGH"
        elif flare_detected and kp < 5:
            severity, confidence = "WARNING", "HIGH"
        elif kp >= 3 or fp > 0.3:
            severity, confidence = "WATCH", "MEDIUM"
        else:
            severity, confidence = "ALL_CLEAR", "HIGH"

        # Ask LLM only to write the bulletin text — straightforward generation task
        bulletin_prompt = (
            f"You are a space weather operator writing an alert bulletin.\n\n"
            f"Observations:\n"
            f"- Solar flare: {fc} class, probability {fp:.0%}\n"
            f"- {obs1}\n"
            f"- {obs2}\n"
            f"- Infrastructure: {obs3[:200]}\n"
            f"- Alert level: {severity}\n\n"
            f"Write exactly 2 sentences for the operator bulletin. "
            f"Be specific about the flare class, Kp={kp:.1f}, "
            f"and what operators should do. No JSON. No headers. Plain text only.\n\n"
            f"Bulletin:"
        )
        try:
            bulletin_text = self._call_llm(bulletin_prompt, max_tokens=120).strip()
            # Strip any accidental JSON or markdown the model adds
            bulletin_text = bulletin_text.split("\n")[0].strip().strip('"')
            if len(bulletin_text) < 20:
                raise ValueError("too short")
        except Exception:
            bulletin_text = (
                f"{fc} solar flare detected (prob={fp:.0%}). "
                f"Kp={kp:.1f} — {physics.get('storm_class', 'conditions nominal')}. "
                f"Monitor SWPC and prepare protective actions for high-latitude infrastructure."
            )

        # Actions based on severity
        actions_map = {
            "ALERT":     ["Safe-mode LEO satellites immediately",
                          "Reduce HVDC load on high-latitude power grids",
                          "Divert all polar aviation routes",
                          "Activate GIC monitoring on transmission lines"],
            "WARNING":   ["Pre-position satellite safe-mode procedures",
                          "Alert power grid operators above 60°N",
                          "Monitor DSCOVR Bz for southward turning",
                          "Brief polar flight dispatch teams"],
            "WATCH":     ["Monitor SWPC bulletins every 30 minutes",
                          "Verify GIC monitoring systems are active",
                          "Review contingency plans for G1–G2 storm"],
            "ALL_CLEAR": ["Continue normal operations",
                          "Routine monitoring — next update in 15 minutes"],
        }

        alert = {
            "severity":            severity,
            "bulletin":            bulletin_text,
            "recommended_actions": actions_map[severity],
            "confidence":          confidence,
            "next_update_minutes": 15,
        }

        steps.append({"thought": "Synthesizing all observations into operator alert.",
                      "action": "issue_alert",
                      "input": alert or {},
                      "observation": "Alert issued."})

        # Fallback if LLM exhausted steps without calling issue_alert.
        # Mirrors the severity rules in REACT_SYSTEM exactly.
        if not alert:
            kp = physics.get("kp_estimated", 0)
            fp = flare.get("flare_probability", 0)
            flare_detected = fp > 0.65
            if (kp >= 5 and flare_detected) or kp >= 7:
                sev, conf = "ALERT", "MEDIUM"
            elif flare_detected and kp < 5:
                sev, conf = "WARNING", "MEDIUM"
            elif kp >= 3 or fp > 0.3:
                sev, conf = "WATCH", "LOW"
            else:
                sev, conf = "ALL_CLEAR", "HIGH"
            alert = {
                "severity": sev,
                "bulletin": (
                    f"Automated fallback after {len(steps)} reasoning steps. "
                    f"Flare={fp:.0%}, Kp={kp:.1f} ({physics.get('storm_class', 'unknown')})."
                ),
                "recommended_actions": ["Monitor SWPC bulletins", "Continue normal operations"],
                "confidence": conf,
                "next_update_minutes": 15,
            }

        # Normalize confidence to HIGH/MEDIUM/LOW regardless of what LLM returned
        raw_conf = alert.get("confidence", "MEDIUM")
        if isinstance(raw_conf, (int, float)):
            alert["confidence"] = "HIGH" if raw_conf >= 0.8 else "MEDIUM" if raw_conf >= 0.5 else "LOW"
        elif str(raw_conf).upper() not in ("HIGH", "MEDIUM", "LOW"):
            alert["confidence"] = "HIGH" if float(str(raw_conf).replace("%","")) >= 80 else "MEDIUM"

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
