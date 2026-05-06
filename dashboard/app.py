"""
HELIOS Operator Dashboard — Streamlit
Run: streamlit run dashboard/app.py --server.port 30000 --server.address 0.0.0.0
"""
import sys
import io
import time
import requests
import streamlit as st
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from PIL import Image

sys.path.insert(0, ".")

st.set_page_config(
    page_title="HELIOS — Space Weather Intelligence",
    page_icon="☀",
    layout="wide",
)

# ── Session state init ────────────────────────────────────────────────────────
if "alert" not in st.session_state:
    st.session_state.alert = None
if "kp" not in st.session_state:
    st.session_state.kp = 0.0
if "pipeline_ran" not in st.session_state:
    st.session_state.pipeline_ran = False

st.title("☀ HELIOS — Real-Time Space Weather Intelligence")
st.caption("AMD Instinct MI300X · ROCm · NASA/IBM Surya-1.0 · Llama 3.1")

c1, c2, c3, c4 = st.columns(4)
c1.metric("01 Solar Vision", "Surya-1.0", "AMD ROCm")
c2.metric("02 CME Physics", "DSCOVR L1", "Burton / DBM")
c3.metric("03 Impact Mapper", "Kp → Infra", "Folium")
c4.metric("04 Command LLM", "Llama 3.1", "vLLM")

st.divider()
left, right = st.columns(2)


@st.fragment(run_every=12)
def live_sdo_image():
    """Auto-refreshes every 12 seconds — matching SDO's image cadence."""
    try:
        ts = int(time.time())
        r = requests.get(
            f"https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_0171.jpg?t={ts}",
            timeout=15
        )
        img = Image.open(io.BytesIO(r.content))
        st.image(img, width="stretch")
    except Exception as e:
        st.error(f"SDO image unavailable: {e}")


# ── Left: SDO image + Bz chart ────────────────────────────────────────────────
with left:
    st.subheader("☀ Live SDO AIA 171Å")
    live_sdo_image()

    st.subheader("📡 DSCOVR Bz — Last 200 Minutes")
    try:
        r = requests.get(
            "https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json",
            timeout=10
        )
        data = r.json()
        headers = data[0]
        bz_col = headers.index("bz_gsm")
        bz_values = []
        for row in data[1:]:
            try:
                bz_values.append(float(row[bz_col]))
            except (TypeError, ValueError):
                pass

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=bz_values[-200:], mode="lines",
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

# ── Right: risk map + alert ───────────────────────────────────────────────────
with right:
    st.subheader("🗺 Infrastructure Risk Map")

    # Build map from session state kp (updated only when pipeline runs)
    kp = st.session_state.kp
    color = "red" if kp >= 7 else "orange" if kp >= 5 else "yellow" if kp >= 3 else "green"
    m = folium.Map(location=[30, 0], zoom_start=2, tiles="CartoDB dark_matter")

    from agents.agent_03_impact import KP_LATITUDE, KEY_INFRASTRUCTURE
    affected_lat = KP_LATITUDE.get(min(9, int(kp)), 90)
    if kp >= 1 and affected_lat < 90:
        folium.Rectangle(
            bounds=[[affected_lat, -180], [90, 180]],
            color=color, fill=True, fill_color=color, fill_opacity=0.25,
            popup=f"Risk zone — above {affected_lat}°N (Kp={kp:.1f})"
        ).add_to(m)
        folium.Rectangle(
            bounds=[[-90, -180], [-affected_lat, 180]],
            color=color, fill=True, fill_color=color, fill_opacity=0.25,
            popup=f"Risk zone — below {affected_lat}°S (Kp={kp:.1f})"
        ).add_to(m)
    for name, lat, lon in KEY_INFRASTRUCTURE:
        folium.CircleMarker(
            [lat, lon], radius=8, color=color, fill=True,
            fill_opacity=0.8, popup=name
        ).add_to(m)

    # returned_objects=[] stops st_folium from triggering reruns on map interaction
    st_folium(m, height=320, width="stretch", returned_objects=[])

    st.subheader("🚨 HELIOS Alert Bulletin")
    if st.session_state.alert:
        alert = st.session_state.alert
        icons = {"ALERT": "🔴", "WARNING": "🟡", "WATCH": "🟠", "ALL_CLEAR": "🟢"}
        icon = icons.get(alert.get("severity", ""), "⚪")
        st.success(
            f"{icon} **{alert.get('severity')}** — {alert.get('bulletin')}\n\n"
            + "\n".join(f"- {a}" for a in alert.get("recommended_actions", []))
        )
    else:
        st.info("Click 'Run Pipeline' to generate an alert.")

# ── Control buttons ───────────────────────────────────────────────────────────
st.divider()
col_run, col_replay = st.columns(2)

with col_run:
    if st.button("▶ Run HELIOS Pipeline", type="primary", use_container_width=True):
        with st.spinner("Running 4-agent pipeline on AMD MI300X..."):
            try:
                from pipeline.orchestrator import run_once
                result = run_once()
                if result.get("physics_event"):
                    st.session_state.kp = result["physics_event"].get("kp_estimated", 0)
                if result.get("alert_event"):
                    st.session_state.alert = result["alert_event"]
                elif not result.get("should_alert"):
                    st.session_state.alert = {
                        "severity": "ALL_CLEAR",
                        "bulletin": "No significant solar activity detected. Conditions nominal.",
                        "recommended_actions": ["Continue normal operations"],
                    }
                st.session_state.pipeline_ran = True
                st.rerun()
            except Exception as e:
                st.error(f"Pipeline error: {e}")

with col_replay:
    if st.button("🔄 Replay March 2015 Storm (G4)", use_container_width=True):
        with st.spinner("Injecting G4 storm scenario..."):
            try:
                from pipeline.orchestrator import helios_pipeline
                state = {
                    "flare_event": {"agent":"agent_01_vision","timestamp":"2015-03-17T06:00:00+00:00","flare_probability":0.88,"flare_detected":True,"severity":"X-class","source":"surya_bench","inference_ms":145,"vram_gb":1.82},
                    "physics_event": {"agent":"agent_02_physics","timestamp":"2015-03-17T06:00:00+00:00","bz_nT":-22.0,"solar_wind_speed_kms":670,"proton_density":18.5,"kp_estimated":8.5,"storm_class":"G4 (severe)","goes_flux":1.2e-4,"detection_mode":"L1_realtime"},
                    "impact_event": None,
                    "alert_event": None,
                    "should_alert": True,
                }
                result = helios_pipeline.invoke(state)
                st.session_state.kp = 8.5
                if result.get("alert_event"):
                    result["alert_event"]["bulletin"] = "[REPLAY: March 17, 2015] " + result["alert_event"].get("bulletin", "")
                    st.session_state.alert = result["alert_event"]
                st.session_state.pipeline_ran = True
                st.rerun()
            except Exception as e:
                st.error(f"Replay error: {e}")

if st.checkbox("Auto-refresh every 60 seconds"):
    time.sleep(60)
    st.rerun()
