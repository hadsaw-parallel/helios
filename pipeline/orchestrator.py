"""
HELIOS LangGraph pipeline orchestrator.
Agents 01 and 02 run in true parallel (both are I/O + GPU bound independently).
They sync at Agent 03, which feeds Agent 04.
"""
import concurrent.futures
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

from agents.agent_01_vision import SolarVisionAgent
from agents.agent_02_physics import CMEPhysicsAgent
from agents.agent_03_impact import ImpactMapperAgent
from agents.agent_04_command import CommandAgent


class HELIOSState(TypedDict):
    flare_event: Optional[dict]
    physics_event: Optional[dict]
    impact_event: Optional[dict]
    alert_event: Optional[dict]
    should_alert: bool


# Agents are singletons — models stay loaded between pipeline invocations
_vision = SolarVisionAgent()
_physics = CMEPhysicsAgent()
_impact = ImpactMapperAgent()
_command = CommandAgent()


def node_parallel(state: HELIOSState) -> HELIOSState:
    """Agents 01 + 02 run concurrently — SDO fetch+GPU and DSCOVR fetch are independent."""
    def vision_task():
        tensor = _vision.fetch_sdo_image()
        prob, latency = _vision.run_inference(tensor)
        return _vision.emit_event(prob, latency)

    def physics_task():
        return _physics.run()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        f1 = pool.submit(vision_task)
        f2 = pool.submit(physics_task)
        flare = f1.result()
        physics = f2.result()

    return {
        **state,
        "flare_event": flare,
        "physics_event": physics,
        "should_alert": flare["flare_detected"],
    }


def node_impact(state: HELIOSState) -> HELIOSState:
    event = _impact.run(state["physics_event"])
    return {**state, "impact_event": event}


def node_command(state: HELIOSState) -> HELIOSState:
    alert = _command.synthesize(
        state["flare_event"],
        state["physics_event"],
        state["impact_event"],
    )
    return {**state, "alert_event": alert}


def _should_escalate(state: HELIOSState) -> str:
    """Only run impact + command if a significant event was detected."""
    return "impact" if state["should_alert"] else END


graph = StateGraph(HELIOSState)
graph.add_node("parallel", node_parallel)
graph.add_node("impact", node_impact)
graph.add_node("command", node_command)

graph.set_entry_point("parallel")
graph.add_conditional_edges("parallel", _should_escalate)
graph.add_edge("impact", "command")
graph.add_edge("command", END)

helios_pipeline = graph.compile()


def run_once(initial_state: dict = None) -> dict:
    """Run one full pipeline cycle. Pass initial_state to inject historical data."""
    state = initial_state or {
        "flare_event": None,
        "physics_event": None,
        "impact_event": None,
        "alert_event": None,
        "should_alert": False,
    }
    return helios_pipeline.invoke(state)


if __name__ == "__main__":
    import json
    result = run_once()
    print(json.dumps(result, indent=2))
