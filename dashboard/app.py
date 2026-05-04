"""
HELIOS Operator Dashboard — Streamlit
Run: streamlit run dashboard/app.py --server.port 8501
"""
import sys
import io
import json
import time
import requests
import streamlit as st
import plotly.graph_objects as go
import folium
from folium import Map
from streamlit_folium import st_folium
from PIL import Image

sys.path.insert(0, ".")

st.set_page_config(
    page_title="HELIOS — Space Weather Intelligence",
    page_icon="☀",
    layout="wide",
)

st.title("☀ HELIOS — Real-Time Space Weather Intelligence")
st.caption("AMD Instinct MI300X · ROCm · NASA/IBM Surya-1.0 · Llama 3.1")

# ── Agent status bar ─────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("01 Solar Vision", "Surya-1.0", "AMD ROCm")
c2.metric("02 CME Physics", "DSCOVR L1", "Burton / DBM")
c3.metric("03 Impact Mapper", "Kp → Infra", "Folium")
c4.metric("04 Command LLM", "Llama 3.1", "vLLM")

st.divider()

left, right = st.columns(2)

# ── Left column: live solar image + Bz chart ─────────────────────────────────
with left:
    st.subheader("☀ Live SDO AIA 171Å")
    try:
        r = requests.get(
            "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_0171.jpg",
            timeout=15
        )
        img = Image.open(io.BytesIO(r.content))
        st.image(img, use_container_width=True)
    except Exception as e:
        st.error(f"SDO image unavailable: {e}")

    st.subheader("📡 DSCOVR Bz — Last 200 Minutes")
    try:
        r = requests.get(
            "https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json",
            timeout=10
        )
        data = r.json()
        headers = data[0]
        rows = data[1:]
        bz_col = headers.index("bz")
        bz_values = []
        for row in rows[-200:]:
            try:
                bz_values.append(float(row[bz_col]))
            except (TypeError, ValueError):
                pass

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=bz_values, mode="lines",
            line=dict(color="cyan", width=1.5), name="Bz (nT)"
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.4)
        fig.update_layout(
            height=220, margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.3)",
            font=dict(color="white"), yaxis_title="Bz (nT)"
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"DSCOVR data unavailable: {e}")

# ── Right column: risk map + alert panel ─────────────────────────────────────
with right:
    st.subheader("🗺 Infrastructure Risk Map")
    risk_map_placeholder = st.empty()
    with risk_map_placeholder:
        m = Map(location=[30, 0], zoom_start=2, tiles="CartoDB dark_matter")
        st_folium(m, height=320, use_container_width=True)

    st.subheader("🚨 HELIOS Alert Bulletin")
    alert_placeholder = st.empty()
    alert_placeholder.info("Click 'Run Pipeline' to generate an alert.")

# ── Control buttons ───────────────────────────────────────────────────────────
st.divider()
col_run, col_replay = st.columns(2)

with col_run:
    if st.button("▶ Run HELIOS Pipeline", type="primary", use_container_width=True):
        with st.spinner("Running 4-agent pipeline on AMD MI300X..."):
            try:
                from pipeline.orchestrator import run_once
                result = run_once()
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                result = None

        if result and result.get("alert_event"):
            alert = result["alert_event"]
            icons = {"ALERT": "🔴", "WARNING": "🟡", "WATCH": "🟠", "ALL_CLEAR": "🟢"}
            icon = icons.get(alert.get("severity", ""), "⚪")
            alert_placeholder.success(
                f"{icon} **{alert.get('severity')}** — {alert.get('bulletin')}\n\n"
                + "\n".join(f"- {a}" for a in alert.get("recommended_actions", []))
            )

            # Refresh risk map with actual Kp
            if result.get("physics_event"):
                from agents.agent_03_impact import _severity_color, KP_LATITUDE
                kp = result["physics_event"].get("kp_estimated", 0)
                color = _severity_color(kp)
                lat = KP_LATITUDE.get(min(9, int(kp)), 90)
                m2 = Map(location=[30, 0], zoom_start=2, tiles="CartoDB dark_matter")
                if lat < 90:
                    folium.Rectangle(
                        bounds=[[lat, -180], [90, 180]],
                        color=color, fill=True, fill_color=color, fill_opacity=0.25,
                        popup=f"Risk zone — above {lat}°N (Kp={kp:.1f})"
                    ).add_to(m2)
                with risk_map_placeholder:
                    st_folium(m2, height=320, use_container_width=True)

        elif result and not result.get("alert_event"):
            alert_placeholder.success("🟢 **ALL_CLEAR** — No significant events detected.")

with col_replay:
    if st.button("🔄 Replay March 2015 Storm (G4)", use_container_width=True):
        st.info("Running real Surya inference on SuryaBench March 2015 data...")
        with st.spinner("Loading historical .nc data and running pipeline..."):
            try:
                from tests.test_storm_replay import load_bench_frame, MARCH_2015_NC
                import os
                if not os.path.isdir(MARCH_2015_NC):
                    st.error(
                        "SuryaBench data not found. "
                        "Run: `git clone https://github.com/NASA-IMPACT/SuryaBench` "
                        "and download March 2015 data first."
                    )
                else:
                    from agents.agent_01_vision import SolarVisionAgent
                    from agents.agent_02_physics import CMEPhysicsAgent
                    from agents.agent_03_impact import ImpactMapperAgent
                    from agents.agent_04_command import CommandAgent

                    v = SolarVisionAgent()
                    tensor = load_bench_frame(MARCH_2015_NC, timestep=0)
                    prob, ms = v.run_inference(tensor)
                    flare = v.emit_event(prob, ms)

                    p = CMEPhysicsAgent()
                    phys = p.run(flare)
                    i = ImpactMapperAgent()
                    imp = i.run(phys)
                    c = CommandAgent()
                    alert = c.synthesize(flare, phys, imp)

                    icons = {"ALERT": "🔴", "WARNING": "🟡", "WATCH": "🟠", "ALL_CLEAR": "🟢"}
                    icon = icons.get(alert.get("severity", ""), "⚪")
                    alert_placeholder.warning(
                        f"[REPLAY — March 17, 2015] {icon} **{alert.get('severity')}** — "
                        f"{alert.get('bulletin')}"
                    )
            except Exception as e:
                st.error(f"Replay error: {e}")

# ── Auto-refresh ──────────────────────────────────────────────────────────────
if st.checkbox("Auto-refresh every 60 seconds"):
    time.sleep(60)
    st.rerun()
