"""
HELIOS Operator Dashboard — Streamlit
Run: streamlit run dashboard/app.py --server.port 30000 --server.address 0.0.0.0
"""
import sys
import io
import json
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
if "reasoning_trace" not in st.session_state:
    st.session_state.reasoning_trace = []

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
        meta_cols = st.columns(3)
        meta_cols[0].caption(f"Confidence: **{alert.get('confidence', '—')}**")
        meta_cols[1].caption(f"ReAct steps: **{alert.get('react_steps', '—')}**")
        meta_cols[2].caption(f"Model: **{alert.get('model_used', '—').split('/')[-1]}**")

        trace = st.session_state.reasoning_trace
        if trace:
            with st.expander(f"Agent 04 reasoning trace ({len(trace)} steps)", expanded=False):
                for i, step in enumerate(trace, 1):
                    st.markdown(f"**Step {i} — `{step['action']}`**")
                    st.markdown(f"> *{step['thought'][:200]}*")
                    st.code(json.dumps(step["input"], indent=2), language="json")
                    obs = step["observation"]
                    try:
                        obs_parsed = json.loads(obs) if isinstance(obs, str) else obs
                        st.json(obs_parsed)
                    except Exception:
                        st.caption(f"Observation: {str(obs)[:300]}")
                    st.divider()
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
                    alert = result["alert_event"]
                    st.session_state.alert = alert
                    st.session_state.reasoning_trace = alert.pop("reasoning_trace", [])
                elif not result.get("should_alert"):
                    st.session_state.alert = {
                        "severity": "ALL_CLEAR",
                        "bulletin": "No significant solar activity detected. Conditions nominal.",
                        "recommended_actions": ["Continue normal operations"],
                    }
                    st.session_state.reasoning_trace = []
                st.session_state.pipeline_ran = True
                st.rerun()
            except Exception as e:
                st.error(f"Pipeline error: {e}")

with col_replay:
    st.markdown("**🔄 Counterfactual Storm Replay**")
    st.caption("Data fetched live from NASA DONKI, GFZ Potsdam, and NASA OMNIWeb — not hardcoded.")

    # Named presets map to real historical timestamps
    REPLAY_PRESETS = {
        "── May 2024 Gannon G5 Storm ──────────────": None,
        "Gannon T-96h: First X-flare (May 6 18:00 UTC)":  "2024-05-06T18:00:00Z",
        "Gannon T-36h: X3.98 + CME launched (May 8 21:00 UTC)": "2024-05-08T21:00:00Z",
        "Gannon T-6h: CME approaching (May 10 10:00 UTC)": "2024-05-10T10:00:00Z",
        "Gannon T-0: G5 peak — Kp 9 (May 10 17:00 UTC)":  "2024-05-10T17:00:00Z",
        "── March 2015 St. Patrick's Day G4 ───────": None,
        "StPat T-48h: Pre-storm (Mar 15 06:00 UTC)":  "2015-03-15T06:00:00Z",
        "StPat T-24h: CME launched (Mar 16 06:00 UTC)": "2015-03-16T06:00:00Z",
        "StPat T-6h: Storm onset (Mar 17 04:00 UTC)": "2015-03-17T04:00:00Z",
        "StPat T-0: G4 peak (Mar 17 22:00 UTC)":      "2015-03-17T22:00:00Z",
    }

    valid_presets = {k: v for k, v in REPLAY_PRESETS.items() if v is not None}
    preset = st.selectbox("Historical moment:", list(REPLAY_PRESETS.keys()),
                          label_visibility="collapsed")
    timestamp = REPLAY_PRESETS.get(preset)

    if timestamp and st.button("▶ Fetch & Run Pipeline", use_container_width=True):
        with st.spinner(f"Fetching real archive data for {timestamp[:10]}…"):
            try:
                from data.historical_noaa import build_pipeline_snapshot
                from pipeline.orchestrator import helios_pipeline

                snapshot = build_pipeline_snapshot(timestamp, lookback_hours=12)
                meta = snapshot.pop("meta", {})

                result = helios_pipeline.invoke(snapshot)
                fetched_kp = snapshot["physics_event"].get("kp_estimated", 0)
                st.session_state.kp = fetched_kp

                if result.get("alert_event"):
                    alert = result["alert_event"]
                    flare_cls = snapshot["flare_event"].get("flare_class", "")
                    alert["bulletin"] = (
                        f"[ARCHIVE {timestamp[:10]} | {flare_cls} | Kp={fetched_kp:.1f}] "
                        + alert.get("bulletin", "")
                    )
                    st.session_state.reasoning_trace = alert.pop("reasoning_trace", [])
                    st.session_state.alert = alert
                elif not snapshot.get("should_alert"):
                    st.session_state.alert = {
                        "severity": "ALL_CLEAR",
                        "bulletin": (
                            f"[ARCHIVE {timestamp[:10]}] "
                            f"Kp={fetched_kp:.1f} — no storm threshold reached. "
                            f"Flare: {snapshot['flare_event'].get('flare_class', 'none')}."
                        ),
                        "recommended_actions": ["Continue normal operations"],
                    }
                    st.session_state.reasoning_trace = []

                # Show data provenance so judges can verify
                with st.expander("Data sources fetched from archives", expanded=True):
                    st.json({
                        "flare_class":   snapshot["flare_event"].get("flare_class"),
                        "flare_prob":    snapshot["flare_event"].get("flare_probability"),
                        "bz_nT":         snapshot["physics_event"].get("bz_nT"),
                        "speed_kms":     snapshot["physics_event"].get("solar_wind_speed_kms"),
                        "kp":            fetched_kp,
                        "storm_class":   snapshot["physics_event"].get("storm_class"),
                        "flare_source":  meta.get("flare_source"),
                        "kp_source":     meta.get("kp_source"),
                        "wind_source":   meta.get("solar_wind_source"),
                    })

                st.session_state.pipeline_ran = True
                st.rerun()
            except Exception as e:
                st.error(f"Replay error: {e}")

if st.checkbox("Auto-refresh every 60 seconds"):
    time.sleep(60)
    st.rerun()
