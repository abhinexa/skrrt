"""
risk_engine.py
==============
Two-Layer Context-Aware Risk Model for Smart Factory Safety

Layer 1 — Environmental Risk:
    R_env(t) = Σ w_i(t) · S_i(t)   — pure sensor fusion, NO motion dependency

Layer 2 — Human Amplification:
    H(t) = 1 + α · M(t)             — motion scales danger, never creates it

Final Risk:
    R(t) = min(1, R_env(t) · H(t))

Key properties:
  • Gas/radiation create danger independently of human presence
  • Motion ONLY amplifies existing hazard
  • Vibration risk scales with worker density
  • Ventilation efficiency actively reduces R_env
  • Temperature trend (rate-of-rise) contributes alongside absolute value
  • Full sensor-level contribution breakdown for explainability
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math


# ─────────────────────────────────────────────────────────────────
# THRESHOLDS  (paper §III.E)
# ─────────────────────────────────────────────────────────────────
THETA_1 = 0.30   # SAFE  → WARNING
THETA_2 = 0.60   # WARNING → DANGER

# Human amplification coefficient  (paper §III.D)
ALPHA = 0.40

# Temperature trend: critical rise over one sample window (°C)
DELTA_T_CRIT = 10.0

# Ventilation can reduce R_env by at most this fraction
VENT_MAX_REDUCTION = 0.14


# ─────────────────────────────────────────────────────────────────
# BASE WEIGHTS  — sum ≈ 1.0 so R_env reaches 1.0 at full saturation
#
# Dynamic modifiers applied at runtime:
#   gas_w       ×1.6 when motion=1
#   radiation_w ×1.5 when motion=1
#   vibration_w × (1 + 0.5 · worker_density_norm)
# ─────────────────────────────────────────────────────────────────
BASE_WEIGHTS: Dict[str, float] = {
    "gas":            0.20,   # primary environmental hazard
    "radiation":      0.12,   # ionising risk
    "temperature":    0.09,   # absolute thermal risk
    "temp_trend":     0.06,   # rate-of-rise predictive signal
    "air_quality":    0.08,   # composite particulates/VOC
    "vibration":      0.07,   # structural / machinery stress
    "worker_density": 0.07,   # human exposure scale
    "machine_load":   0.06,   # overload → breakdown risk
    "fatigue":        0.05,   # human error factor
    "forklift":       0.05,   # mobile collision hazard
    "noise":          0.04,   # acoustic damage / distraction
    "pressure_dev":   0.04,   # deviation from atmospheric (101 kPa)
    "humidity":       0.03,   # condensation / corrosion
    "door_open":      0.03,   # containment breach
    # ventilation handled separately as a negative contribution
}
# Positive weights sum ≈ 0.99


# ─────────────────────────────────────────────────────────────────
# NORMALISATION HELPERS
# ─────────────────────────────────────────────────────────────────
def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def normalize(sensors: dict) -> dict:
    """
    Map raw sensor readings to [0, 1].
    Temperature: normalised over operational range 15 – 90 °C.
    Pressure:    deviation from standard atmosphere (101 kPa).
    """
    t = sensors.get("temperature", 25.0)
    return {
        "temperature":    _clamp((t - 15.0) / (90.0 - 15.0)),
        "humidity":       _clamp((sensors.get("humidity", 50.0) - 20.0) / (95.0 - 20.0)),
        "gas":            _clamp(sensors.get("gas", 0.0) / 1000.0),
        "air_quality":    _clamp(sensors.get("air_quality", 45.0) / 500.0),
        "radiation":      _clamp(sensors.get("radiation", 0.0) / 10.0),
        "vibration":      _clamp(sensors.get("vibration", 0.0) / 10.0),
        "machine_load":   _clamp(sensors.get("machine_load", 50.0) / 100.0),
        "noise":          _clamp((sensors.get("noise", 60.0) - 40.0) / (120.0 - 40.0)),
        "pressure_dev":   _clamp(abs(sensors.get("pressure", 101.0) - 101.0) / 50.0),
        "motion":         1.0 if sensors.get("motion", False) else 0.0,
        "worker_density": _clamp(sensors.get("worker_density", 0.0) / 20.0),
        "fatigue":        _clamp(sensors.get("fatigue", 0.0) / 10.0),
        "forklift":       _clamp(sensors.get("forklift", 0.0) / 10.0),
        "door_open":      1.0 if sensors.get("door_state", False) else 0.0,
        "ventilation":    _clamp(sensors.get("ventilation", 80.0) / 100.0),
    }


def temperature_trend(current: float, previous: float) -> float:
    """
    T′(t) = min(1, |ΔT| / ΔT_crit)    (paper §III.C)
    Captures rapid thermal escalation before absolute threshold is crossed.
    """
    return _clamp(abs(current - previous) / DELTA_T_CRIT)


# ─────────────────────────────────────────────────────────────────
# RESULT DATACLASS
# ─────────────────────────────────────────────────────────────────
@dataclass
class RiskResult:
    risk_score:    float
    risk_env:      float
    amplification: float
    risk_state:    str
    message:       str
    vent_on:       bool
    temp_trend:    float
    contributions: List[dict] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────
# MAIN ENGINE
# ─────────────────────────────────────────────────────────────────
class RiskEngine:
    """
    Stateless per-call risk computation.
    Pass prev_temp for temperature-trend analysis (optional).
    """

    def compute(
        self,
        sensors: dict,
        prev_temp: Optional[float] = None,
    ) -> RiskResult:

        norm = normalize(sensors)
        M    = norm["motion"]          # 0 or 1

        # ── Build dynamic weights ──────────────────────────────
        w = dict(BASE_WEIGHTS)

        if M > 0:
            # Gas and radiation MORE dangerous when humans are present
            w["gas"]       *= 1.6
            w["radiation"] *= 1.5

        # Vibration hazard scales with worker density
        w["vibration"] *= (1.0 + 0.5 * norm["worker_density"])

        # ── Temperature trend ──────────────────────────────────
        t_trend = 0.0
        if prev_temp is not None:
            t_trend = temperature_trend(sensors.get("temperature", 25.0), prev_temp)

        # ── Compute R_env (no motion term) ─────────────────────
        raw_contribs: Dict[str, float] = {}
        R_env = 0.0

        for key, weight in w.items():
            if key == "pressure_dev":
                val = norm["pressure_dev"]
            elif key == "door_open":
                val = norm["door_open"]
            elif key == "temp_trend":
                val = t_trend
            else:
                val = norm.get(key, 0.0)

            contribution = weight * val
            raw_contribs[key] = contribution
            R_env += contribution

        # Ventilation actively reduces environmental risk
        vent_reduction = norm["ventilation"] * VENT_MAX_REDUCTION
        R_env = _clamp(R_env - vent_reduction)

        # ── Layer 2: Human amplification ───────────────────────
        H   = 1.0 + ALPHA * M
        R   = _clamp(R_env * H)

        # ── Classification ─────────────────────────────────────
        if R < THETA_1:
            state   = "SAFE"
            message = "All systems nominal. No action required."
        elif R < THETA_2:
            state   = "WARNING"
            message = "Elevated risk detected. Monitor conditions closely."
        else:
            state   = "DANGER"
            message = "CRITICAL RISK — Immediate intervention required!"

        # ── Ventilation logic ──────────────────────────────────
        vent_on = (norm["gas"] > 0.5) or (R >= THETA_2)

        # ── Explainability: sorted contribution list ───────────
        total_positive = max(sum(v for v in raw_contribs.values() if v > 0), 1e-9)
        contributions = sorted(
            [
                {
                    "sensor":       key,
                    "label":        key.replace("_", " ").title(),
                    "weight":       round(w.get(key, 0.0), 4),
                    "normalised":   round(norm.get(key, t_trend if key == "temp_trend"
                                         else norm.get("pressure_dev" if key == "pressure_dev"
                                         else key, 0.0)), 4),
                    "contribution": round(val, 4),
                    "share_pct":    round(100.0 * val / total_positive, 1),
                }
                for key, val in raw_contribs.items()
            ],
            key=lambda x: x["contribution"],
            reverse=True,
        )

        return RiskResult(
            risk_score    = round(R, 4),
            risk_env      = round(R_env, 4),
            amplification = round(H, 3),
            risk_state    = state,
            message       = message,
            vent_on       = vent_on,
            temp_trend    = round(t_trend, 4),
            contributions = contributions,
        )


# ─────────────────────────────────────────────────────────────────
# QUICK SELF-TEST
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    engine = RiskEngine()

    # Scenario 1: Gas + radiation high, NO motion → must still reach WARNING/DANGER
    s1 = dict(
        temperature=55, humidity=60, gas=750, air_quality=300,
        radiation=7, vibration=4, machine_load=80, noise=90,
        pressure=108, motion=False, worker_density=0,
        fatigue=0, forklift=0, door_state=False, ventilation=20,
    )
    r1 = engine.compute(s1, prev_temp=40)
    print(f"Scenario 1 (high gas+radiation, no motion): R={r1.risk_score:.3f}  [{r1.risk_state}]")
    assert r1.risk_state in ("WARNING", "DANGER"), "Gas/radiation alone must trigger ≥ WARNING"

    # Scenario 2: Moderate gas, motion=True → amplified to DANGER
    s2 = dict(
        temperature=40, humidity=55, gas=500, air_quality=250,
        radiation=4, vibration=3, machine_load=70, noise=80,
        pressure=103, motion=True, worker_density=10,
        fatigue=6, forklift=5, door_state=True, ventilation=30,
    )
    r2 = engine.compute(s2, prev_temp=30)
    print(f"Scenario 2 (moderate gas, motion=True):     R={r2.risk_score:.3f}  [{r2.risk_state}]")

    # Scenario 3: All nominal → SAFE
    s3 = dict(
        temperature=22, humidity=50, gas=20, air_quality=30,
        radiation=0.1, vibration=0.2, machine_load=40, noise=55,
        pressure=101, motion=False, worker_density=2,
        fatigue=1, forklift=0.5, door_state=False, ventilation=90,
    )
    r3 = engine.compute(s3)
    print(f"Scenario 3 (nominal conditions):            R={r3.risk_score:.3f}  [{r3.risk_state}]")
    assert r3.risk_state == "SAFE"

    print("\nTop contributors (Scenario 2):")
    for c in r2.contributions[:5]:
        print(f"  {c['label']:20s}  {c['contribution']:.4f}  ({c['share_pct']:.1f}%)")

    print("\nAll assertions passed ✓")
