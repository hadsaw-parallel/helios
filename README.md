# HELIOS — Real-Time Space Weather Intelligence

> First multi-agent AI system for real-time solar storm detection and infrastructure impact forecasting.
> Running NASA/IBM Surya-1.0 on AMD Instinct MI300X via ROCm.

**AMD Developer Hackathon · May 4–10, 2026 · lablab.ai**

---

## Demo

🎥 [2-minute demo video — coming soon]

Live dashboard: `http://134.199.197.132` *(active during hackathon)*

---

## What It Does

HELIOS watches the Sun 24/7, detects solar storms as they form, models how they travel through space, and delivers plain-language impact forecasts to operators — telling them exactly which satellites, power grids, GPS systems, and aviation routes face risk, and when.

**The problem:** When a CME arrives at Earth, the DSCOVR sensor at the L1 Lagrange point gives operators approximately 30 minutes of real-time warning — verified at 31 minutes for the May 2024 Gannon G5 storm. That is not enough time to complete protective actions for critical infrastructure.

**HELIOS detects flares at the solar source** using GOES X-ray data and NASA's Surya foundation model, issuing automated WARNING alerts days before Earth impact — rather than waiting for the storm to arrive at DSCOVR's L1 position. For the Gannon storm, our pipeline issued a WARNING 36 hours before impact, validated against real NASA DONKI and GFZ Potsdam archives.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    LIVE DATA SOURCES                    │
│  NASA SDO (images)  │  NOAA DSCOVR  │  GOES X-ray      │
└──────────┬──────────┴───────┬────────┴──────┬───────────┘
           │                  │               │
           ▼                  ▼               │
  ┌─────────────────┐  ┌─────────────────┐   │
  │   AGENT 01      │  │   AGENT 02      │◄──┘
  │  Solar Vision   │  │  CME Physics    │
  │  Surya-1.0      │  │  DSCOVR L1      │
  │  GOES X-ray     │  │  Burton / DBM   │
  └────────┬────────┘  └────────┬────────┘
           │    (parallel)      │
           └────────┬───────────┘
                    ▼
          ┌─────────────────┐
          │   AGENT 03      │
          │ Impact Mapper   │
          │  Kp → Infra     │
          │  Folium map     │
          └────────┬────────┘
                   ▼
          ┌─────────────────┐
          │   AGENT 04      │
          │  Command LLM    │
          │  Llama 3.1 8B   │
          │  Alert bulletin │
          └────────┬────────┘
                   ▼
          ┌─────────────────┐
          │ OPERATOR DASH   │
          │  Streamlit UI   │
          │  Live + Replay  │
          └─────────────────┘
```

Agents 01 and 02 run in **true parallel** via `ThreadPoolExecutor` — GOES/SDO fetch and DSCOVR fetch are independent. They sync at Agent 03.

---

## Benchmark — Surya-1.0 on AMD MI300X

| Metric | Value |
|---|---|
| Surya VRAM (weights + activation) | **1.82 GB** |
| GOES X-ray live signal latency | **~170 ms** |
| All models loaded simultaneously | **~145 GB total** |
| MI300X total VRAM | **192 GB** |
| Full pipeline latency (no storm) | **< 1 second** |
| Full pipeline latency (alert path) | **~30 seconds** (includes LLM) |

**Why MI300X:**
Llama 3.1 8B (~16 GB fp16) + Surya (~1.82 GB) + vLLM overhead — all loaded simultaneously in one VRAM pool. No model swapping. The MI300X's 5.2 TB/s memory bandwidth enables real-time inference within SDO's 12-second image cadence.

---

## Tech Stack

| Layer | Tool | Version |
|---|---|---|
| GPU | AMD Instinct MI300X | 192 GB VRAM |
| GPU Runtime | ROCm | 7.2 |
| Solar Model | NASA/IBM Surya-1.0 (HelioSpectFormer) | 366M params |
| LLM | Llama 3.1 8B (vLLM) | 0.17.1 |
| Agent Framework | LangGraph | latest |
| Solar Physics | DSCOVR + Burton empirical formula | — |
| Geo Mapping | Folium | latest |
| Dashboard | Streamlit | latest |

**Models used in this submission:** Surya-1.0 (366M) + Llama 3.1 8B Instruct via vLLM on AMD ROCm 7.2.

---

## Data Sources

All data is **free, public, and streaming in real time**:

| Source | What | URL |
|---|---|---|
| NASA SDO | Live solar images (AIA 171Å) | `sdo.gsfc.nasa.gov` |
| NOAA DSCOVR mag | Bz magnetic field (nT) | `services.swpc.noaa.gov` |
| NOAA DSCOVR plasma | Solar wind speed (km/s) | `services.swpc.noaa.gov` |
| NOAA GOES | X-ray flux (flare class) | `services.swpc.noaa.gov` |
| NOAA SWPC | Kp index | `services.swpc.noaa.gov` |
| SuryaBench | Historical storm data (.nc) | `github.com/NASA-IMPACT/SuryaBench` |

---

## Agent Specifications

| Agent | Model | Input | Output |
|---|---|---|---|
| 01 Solar Vision | Surya-1.0 + GOES X-ray | SDO image sequence / GOES flux | Flare probability, severity |
| 02 CME Physics | Burton formula + DBM | DSCOVR Bz + plasma speed | Kp estimate, storm class |
| 03 Impact Mapper | Lookup table + Folium | Kp index | Geo risk map, per-sector impacts |
| 04 Command LLM | Llama 3.1 8B via vLLM | All agent outputs | Plain-language alert bulletin |

---

## Quickstart

**Fresh AMD Developer Cloud instance — complete setup in one command per step:**

```bash
# On the AMD host (run once):
bash <(curl -s https://raw.githubusercontent.com/hadsaw-parallel/helios/main/setup_host.sh)

# Inside the Docker container:
docker exec -it rocm /bin/bash
cd /app && git clone https://github.com/hadsaw-parallel/helios.git && cd helios && bash setup.sh
```

`setup.sh` automatically:
1. Clones the repo and installs dependencies
2. Clones NASA-IMPACT/Surya and installs it
3. Downloads Surya-1.0 weights from HuggingFace
4. Starts vLLM serving Llama 3.1 8B
5. Starts Streamlit dashboard on port 30000
6. Proxied to port 80 via Caddy

**Dashboard opens at `http://<YOUR_IP>`** (~15 min from zero on a fresh instance).

---

## Running Tests

```bash
python3 -m pytest tests/ -v
# 9 passed, 1 skipped (storm replay requires SuryaBench data)
```

---

## What Makes HELIOS Unique

1. **First demonstration of NASA/IBM Surya-1.0 on AMD ROCm hardware.** Surya's GitHub targets CUDA only. HELIOS ports and runs it on MI300X — a direct AMD ecosystem contribution.

2. **First multi-agent agentic pipeline for operational space weather forecasting.** Existing tools are siloed: separate apps for solar imaging, solar wind data, and impact assessment. HELIOS chains them into one autonomous pipeline.

3. **Scientifically grounded physics.** Agent 02 uses real DSCOVR measurements (not synthetic data) and the Burton (1975) empirical formula for Kp estimation. Agent 03's latitude bands match NOAA's published G-scale.

4. **Validated against real storms.** The counterfactual replay fetches live data from NASA DONKI, GFZ Potsdam Kp API, and NASA OMNIWeb for any historical timestamp — nothing hardcoded. Validated against the May 2024 Gannon G5 storm: HELIOS issued WARNING at T-36h using real archived flare data. The live pipeline shows real conditions (ALL_CLEAR when the Sun is quiet).

---

## The Warning Window

| Detection mode | Lead time | Source |
|---|---|---|
| DSCOVR at L1 real-time solar wind | **~30 minutes** | Verified: 31 min for Gannon storm (NOAA SWPC) |
| NOAA analyst watch (CME + coronagraph) | ~2 days | Manual human process, not automated |
| **HELIOS automated WARNING (flare detection)** | **Hours to days** | Validated: T-36h for Gannon using NASA DONKI archives |

HELIOS detects X-class flares at the solar source using GOES X-ray — the same data NOAA analysts use, but in an automated pipeline that also maps infrastructure impact and generates operator bulletins in under 3 seconds. DSCOVR provides the final real-time confirmation as the storm arrives; HELIOS provides the early automated alert before it does.

---

## Validated Claim

*"The May 2024 Gannon G5 storm — strongest in 21 years. DSCOVR gave operators 31 minutes of real-time warning when the CME was already arriving. HELIOS, running the same public GOES data through an automated pipeline, issued a WARNING 36 hours before impact — validated live against NASA DONKI and GFZ Potsdam archives."*

---

*Built with NASA/IBM Surya-1.0 · AMD Instinct MI300X · ROCm 7.2 · LangGraph · Llama 3.1*
*AMD Developer Hackathon · May 4–10, 2026*
