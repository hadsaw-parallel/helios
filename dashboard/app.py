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

    st.subheader("📡 Live Solar Wind & X-ray — Agent Inputs")
    try:
        # GOES X-ray flux — Agent 01 input
        r_goes = requests.get(
            "https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json",
            timeout=10
        )
        goes_data = r_goes.json()
        goes_flux = []
        for row in goes_data[-200:]:
            try:
                goes_flux.append(float(row.get("flux", 0) or 0))
            except (TypeError, ValueError):
                goes_flux.append(0.0)

        # DSCOVR Bz — Agent 02 input
        r_mag = requests.get(
            "https://services.swpc.noaa.gov/products/solar-wind/mag-7-day.json",
            timeout=10
        )
        mag_data = r_mag.json()
        bz_col = mag_data[0].index("bz_gsm")
        bz_values = []
        for row in mag_data[1:]:
            try:
                bz_values.append(float(row[bz_col]))
            except (TypeError, ValueError):
                pass

        fig = go.Figure()
        # GOES X-ray (Agent 01) — top trace
        fig.add_trace(go.Scatter(
            y=goes_flux[-200:], mode="lines", name="GOES X-ray (Agent 01)",
            line=dict(color="orange", width=1.5), yaxis="y1"
        ))
        # Bz (Agent 02) — bottom trace, secondary axis
        fig.add_trace(go.Scatter(
            y=bz_values[-200:], mode="lines", name="DSCOVR Bz (Agent 02)",
            line=dict(color="cyan", width=1.5), yaxis="y2"
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="cyan",
                      opacity=0.3, yref="y2")
        fig.update_layout(
            height=200, margin=dict(l=0, r=50, t=20, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.3)",
            font=dict(color="white", size=11),
            legend=dict(orientation="h", y=1.15, x=0),
            yaxis=dict(title="X-ray (W/m²)", titlefont=dict(color="orange"),
                       tickfont=dict(color="orange"), side="left"),
            yaxis2=dict(title="Bz (nT)", titlefont=dict(color="cyan"),
                        tickfont=dict(color="cyan"), overlaying="y", side="right"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Orange: GOES X-ray flux → flare detection (Agent 01) · Cyan: DSCOVR Bz → storm intensity (Agent 02)")
    except Exception as e:
        st.error(f"Live data unavailable: {e}")

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
        # Ensure recommended_actions is always a list (LLM sometimes returns a string)
        actions = alert.get("recommended_actions", [])
        if isinstance(actions, str):
            actions = [a.strip() for a in actions.split(".") if a.strip()]
        st.success(
            f"{icon} **{alert.get('severity')}** — {alert.get('bulletin')}\n\n"
            + "\n".join(f"- {a}" for a in actions)
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

                st.session_state["replay_snapshot"] = snapshot
                st.session_state["replay_meta"]     = meta
                st.session_state["replay_timestamp"] = timestamp
                st.session_state.pipeline_ran = True
                st.rerun()
            except Exception as e:
                st.error(f"Replay error: {e}")

# ── Storm replay imagery panel (full width, shown after replay runs) ───────────
if st.session_state.get("replay_snapshot"):
    snapshot  = st.session_state["replay_snapshot"]
    meta      = st.session_state["replay_meta"]
    ts        = st.session_state["replay_timestamp"]
    fe        = snapshot["flare_event"]
    pe        = snapshot["physics_event"]

    st.divider()
    st.subheader(f"Real Solar Imagery — {ts[:10]} (NASA Helioviewer Archive)")
    st.caption("Images fetched live from NASA Helioviewer public API — actual solar conditions at this timestamp")

    img_cols = st.columns(3)
    labels = [
        ("SDO AIA 131Å", "131", "Flare-sensitive channel. Hot plasma at 10 MK. Active Region AR3664 visible."),
        ("SDO AIA 171Å", "171", "Coronal loops at 1 MK. Shows large-scale structure disrupted by flare."),
        ("SOHO LASCO C3", "lasco", "Coronagraph — occludes Sun to reveal CME propagating outward."),
    ]
    for col, (title, wl, caption) in zip(img_cols, labels):
        with col:
            st.markdown(f"**{title}**")
            st.caption(caption)
            with st.spinner(f"Fetching {title}..."):
                try:
                    from data.solar_imagery import fetch_sdo_image, fetch_lasco_c3_image
                    if wl == "lasco":
                        img, label = fetch_lasco_c3_image(ts)
                    else:
                        img, label = fetch_sdo_image(ts, wl)
                    st.image(img, use_container_width=True)
                    st.caption(f"Source: {label}")
                except Exception as e:
                    st.error(f"Image unavailable: {e}")

    # Kp timeline chart for the storm
    st.markdown("**Kp Index Timeline — Storm Evolution**")
    try:
        from data.historical_noaa import fetch_kp
        # For Gannon: fetch full storm window May 6-11
        date_part = ts[:10]
        year = date_part[:4]
        storm_start = f"{year}-05-06T00:00:00Z" if "05" in date_part else f"{date_part[:7]}-01T00:00:00Z"
        storm_end   = f"{year}-05-12T00:00:00Z" if "05" in date_part else f"{date_part[:8]}T23:59:59Z"
        kp_records  = fetch_kp(storm_start, storm_end)

        if kp_records:
            import plotly.graph_objects as go
            fig = go.Figure()
            kp_times = [r["datetime"] for r in kp_records]
            kp_vals  = [r["kp"]      for r in kp_records]
            fig.add_trace(go.Bar(
                x=kp_times, y=kp_vals,
                marker_color=["red" if k >= 7 else "orange" if k >= 5
                              else "yellow" if k >= 3 else "green" for k in kp_vals],
                name="Kp index",
            ))
            fig.add_hline(y=5, line_dash="dash", line_color="orange",
                          annotation_text="G1 storm threshold (Kp=5)")
            fig.add_hline(y=7, line_dash="dash", line_color="red",
                          annotation_text="G3 threshold (Kp=7)")
            fig.add_vline(x=ts, line_dash="solid", line_color="white",
                          annotation_text="← This phase", line_width=2)
            fig.update_layout(
                height=220, margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0.3)",
                font=dict(color="white"), yaxis_title="Kp",
                xaxis_title="Date (UTC)", showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Source: GFZ Potsdam Planetary Kp Index (authoritative) — kp.gfz-potsdam.de")
    except Exception as e:
        st.caption(f"Kp chart unavailable: {e}")

    # Data provenance
    with st.expander("Data provenance — verify against sources", expanded=False):
        st.json({
            "flare_class":  fe.get("flare_class"),
            "flare_prob":   fe.get("flare_probability"),
            "bz_nT":        pe.get("bz_nT"),
            "speed_kms":    pe.get("solar_wind_speed_kms"),
            "kp":           pe.get("kp_estimated"),
            "storm_class":  pe.get("storm_class"),
            "flare_source": meta.get("flare_source"),
            "kp_source":    meta.get("kp_source"),
            "wind_source":  meta.get("solar_wind_source"),
        })

if st.checkbox("Auto-refresh every 60 seconds"):
    time.sleep(60)
    st.rerun()
