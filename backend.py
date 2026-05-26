"""
backend.py — Smart Factory Safety Platform
==========================================
Run:  python backend.py
      (pip install fastapi uvicorn jinja2 python-multipart)

Endpoints:
  GET  /               → simulation.html (3-D Three.js factory)
  GET  /dashboard      → dashboard.html  (AI Digital Twin + Supervisor)
  POST /update         → compute risk for one zone, return full result
  POST /update-all     → compute risk for all 4 zones at once
  GET  /status         → last-known risk state for every zone
  GET  /history/{zone} → last N risk readings (for trend chart)
"""

from __future__ import annotations

import asyncio
import math
import random
import time
from collections import deque
from typing import Dict, Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from risk_engine import RiskEngine, RiskResult

# ─────────────────────────────────────────────────────────────────
app = FastAPI(title="Smart Factory Safety Platform", version="2.0")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

engine = RiskEngine()

# Per-zone state store
ZONES = ["A", "B", "C", "D"]

_default_sensors: Dict[str, float | bool] = dict(
    temperature=25.0, humidity=55.0, gas=30.0, air_quality=45.0,
    radiation=0.1, vibration=0.5, machine_load=50.0, noise=62.0,
    pressure=101.0, motion=False, worker_density=3.0,
    fatigue=2.0, forklift=1.0, door_state=False, ventilation=85.0,
)

# Zone-specific baselines (Zone B is the "hot" manufacturing zone)
ZONE_BASELINES: Dict[str, dict] = {
    "A": dict(temperature=22, gas=25,   machine_load=30, noise=58,  worker_density=2),
    "B": dict(temperature=44, gas=120,  machine_load=82, noise=88,  worker_density=8),
    "C": dict(temperature=30, gas=55,   machine_load=65, noise=72,  worker_density=6),
    "D": dict(temperature=21, gas=15,   machine_load=18, noise=44,  worker_density=1),
}


def _make_state(zone: str) -> dict:
    s = dict(_default_sensors)
    s.update(ZONE_BASELINES.get(zone, {}))
    return s


zone_sensors: Dict[str, dict]    = {z: _make_state(z) for z in ZONES}
zone_prev_temp: Dict[str, float]  = {z: zone_sensors[z]["temperature"] for z in ZONES}
zone_last_result: Dict[str, dict] = {}
zone_history: Dict[str, deque]    = {z: deque(maxlen=300) for z in ZONES}  # 5 min @ 1 Hz
zone_manual: Dict[str, float]     = {z: 0.0 for z in ZONES}  # timestamp of last manual update


# ─────────────────────────────────────────────────────────────────
# PYDANTIC MODELS
# ─────────────────────────────────────────────────────────────────
class SensorPayload(BaseModel):
    zone: str = "A"
    temperature:    float = 25.0
    humidity:       float = 55.0
    gas:            float = 30.0
    air_quality:    float = 45.0
    radiation:      float = 0.1
    vibration:      float = 0.5
    machine_load:   float = 50.0
    noise:          float = 62.0
    pressure:       float = 101.0
    motion:         bool  = False
    worker_density: float = 3.0
    fatigue:        float = 2.0
    forklift:       float = 1.0
    door_state:     bool  = False
    ventilation:    float = 85.0


class AllZonesPayload(BaseModel):
    zones: Dict[str, dict]   # {zone_id: {sensor_key: value}}


# ─────────────────────────────────────────────────────────────────
# RESULT SERIALISER
# ─────────────────────────────────────────────────────────────────
def serialise(zone: str, r: RiskResult) -> dict:
    return {
        "zone":          zone,
        "risk_score":    r.risk_score,
        "risk_env":      r.risk_env,
        "amplification": r.amplification,
        "risk_state":    r.risk_state,
        "message":       r.message,
        "vent_on":       r.vent_on,
        "temp_trend":    r.temp_trend,
        "contributions": r.contributions,
        "timestamp":     time.time(),
    }


# ─────────────────────────────────────────────────────────────────
# BACKGROUND AUTO-SIMULATION
# Keeps all zones "alive" with realistic noise.
# Manual updates (from simulation.html) take priority for 10 s.
# ─────────────────────────────────────────────────────────────────
def _noise(val: float, std: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, val + random.gauss(0, std)))


_spike_state: Dict[str, Optional[dict]] = {z: None for z in ZONES}


def _auto_tick(zone: str):
    s = zone_sensors[zone]
    sp = _spike_state[zone]

    # Rare correlated spike: gas + motion (0.4 % per tick)
    if sp is None and random.random() < 0.004:
        _spike_state[zone] = {"ticks": 0, "duration": random.randint(12, 30)}
    if sp is not None:
        sp["ticks"] += 1
        phase = math.sin(math.pi * sp["ticks"] / sp["duration"])
        s["gas"]    = min(900, s["gas"] + phase * 120)
        s["motion"] = phase > 0.4
        if sp["ticks"] >= sp["duration"]:
            _spike_state[zone] = None
    else:
        s["temperature"]    = _noise(s["temperature"],    0.4, 10, 90)
        s["humidity"]       = _noise(s["humidity"],       0.6, 20, 95)
        s["gas"]            = _noise(s["gas"],            8,    0, 850)
        s["air_quality"]    = _noise(s["air_quality"],    4,   10, 450)
        s["radiation"]      = _noise(s["radiation"],      0.02, 0,  9)
        s["vibration"]      = _noise(s["vibration"],      0.15, 0,  9)
        s["machine_load"]   = _noise(s["machine_load"],   2,    0, 100)
        s["noise"]          = _noise(s["noise"],          1.5, 30, 115)
        s["pressure"]       = _noise(s["pressure"],       0.3, 98, 108)
        s["worker_density"] = _noise(s["worker_density"], 0.1,  0,  18)
        s["fatigue"]        = _noise(s["fatigue"],        0.1,  0,   9)
        s["forklift"]       = _noise(s["forklift"],       0.1,  0,   9)
        if random.random() < 0.01:
            s["motion"] = not s["motion"]
        if random.random() < 0.005:
            s["door_state"] = not s["door_state"]
        # Ventilation auto-responds to high gas
        if s["gas"] > 500:
            s["ventilation"] = min(100, s["ventilation"] + 3)


async def _simulation_loop():
    while True:
        await asyncio.sleep(1.0)
        for zone in ZONES:
            # Skip auto-sim if manually updated in last 10 s
            if time.time() - zone_manual.get(zone, 0) < 10:
                continue
            _auto_tick(zone)
            prev = zone_prev_temp[zone]
            r = engine.compute(zone_sensors[zone], prev)
            zone_prev_temp[zone] = zone_sensors[zone]["temperature"]
            result_dict = serialise(zone, r)
            zone_last_result[zone] = result_dict
            zone_history[zone].append({
                "t":     time.time(),
                "score": r.risk_score,
                "state": r.risk_state,
            })


@app.on_event("startup")
async def startup():
    # Pre-populate with initial values
    for z in ZONES:
        r = engine.compute(zone_sensors[z])
        zone_last_result[z] = serialise(z, r)
    asyncio.create_task(_simulation_loop())


# ─────────────────────────────────────────────────────────────────
# ROUTES — pages
# ─────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def page_simulation(request: Request):
    return templates.TemplateResponse("simulation.html", {"request": request})


@app.get("/dashboard", response_class=HTMLResponse)
async def page_dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


# ─────────────────────────────────────────────────────────────────
# ROUTES — API
# ─────────────────────────────────────────────────────────────────
@app.post("/update")
async def update_zone(data: SensorPayload):
    zone = data.zone.upper()
    sensors = data.dict(exclude={"zone"})
    zone_sensors[zone] = sensors
    zone_manual[zone]  = time.time()

    prev = zone_prev_temp[zone]
    r    = engine.compute(sensors, prev)
    zone_prev_temp[zone] = sensors["temperature"]

    result = serialise(zone, r)
    zone_last_result[zone] = result
    zone_history[zone].append({"t": time.time(), "score": r.risk_score, "state": r.risk_state})
    return result


@app.post("/update-all")
async def update_all_zones(data: AllZonesPayload):
    out = {}
    for zone, sensors in data.zones.items():
        zone = zone.upper()
        zone_sensors[zone] = sensors
        zone_manual[zone]  = time.time()
        prev = zone_prev_temp[zone]
        r    = engine.compute(sensors, prev)
        zone_prev_temp[zone] = sensors.get("temperature", 25)
        result = serialise(zone, r)
        zone_last_result[zone] = result
        zone_history[zone].append({"t": time.time(), "score": r.risk_score, "state": r.risk_state})
        out[zone] = result
    return out


@app.get("/status")
async def get_status():
    return {z: zone_last_result.get(z, {}) for z in ZONES}


@app.get("/history/{zone}")
async def get_history(zone: str, limit: int = 60):
    z = zone.upper()
    hist = list(zone_history.get(z, []))
    return {"zone": z, "data": hist[-limit:]}


# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
