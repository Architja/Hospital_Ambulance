"""Ujala | AI-01 + AI-02: emergency rules + recommend(), all in one file (Jupyter friendly).

Only ONE data file is needed: ujala_emergency_rules_synthetic.csv
The demo at the bottom uses small built-in sample ambulances/hospitals (no other files needed).
Demo logic only - not clinical advice, not a diagnosis. Synthetic data only.

Field names are PLACEHOLDERS until agreed with Naga Siddeswar:
  ambulance: id, category, available
  hospital : id, category, freeBeds, pmjay, and either distanceKm or lat+lng
"""
import json
import math
from pathlib import Path

import pandas as pd

# ------------------------------------------------------------------ PATHS (edit here)
DATA_DIR = Path(__file__).parent if (Path(__file__).parent / "ujala_emergency_rules_synthetic.csv").exists() else Path(r"C:\MAD_internship\files")
RULES_CSV = DATA_DIR / "ujala_emergency_rules_synthetic.csv"
CONFIG_JSON = DATA_DIR / "config.json"          # optional; defaults are used if missing

# ------------------------------------------------------------------ CONFIG (scoring weights)
DEFAULT_CONFIG = {"w_distance": 0.5, "w_beds": 0.3, "w_pmjay": 0.2,
                  "top_n": 3, "exclude_full_hospitals": True}
CONFIG = {**DEFAULT_CONFIG, **(json.loads(CONFIG_JSON.read_text()) if CONFIG_JSON.exists() else {})}

REQUIRED_TYPES = {"cardiac", "accident_trauma", "stroke", "maternity",
                  "breathing", "burns", "general"}
VALID_AMBULANCE = {"BLS", "ALS", "ICU", "NEONATAL_OR_MATERNITY_SPECIALIST", "REQUIRES_TRIAGE"}
TRIAGE = "REQUIRES_TRIAGE"


# ------------------------------------------------------------------ RULES (AI-01)
def normalize(emergency_type):
    """'  Accident-Trauma ' -> 'accident_trauma'. Returns None for missing/blank."""
    if emergency_type is None or (isinstance(emergency_type, float) and pd.isna(emergency_type)):
        return None
    key = str(emergency_type).strip().lower().replace("-", "_").replace(" ", "_")
    return key or None


def _rule(row):
    return {"ambulance": row["ambulance_category"],
            "hospital": row["hospital_category"],
            "reason": row["demo_reason"]}


def load_rules(path=RULES_CSV):
    """Load + validate the rules CSV. Returns {"rules", "variants", "unknown", "incomplete"}."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Rules CSV not found: {path}\nCheck DATA_DIR at the top of this file.")
    df = pd.read_csv(path)
    if not df["is_synthetic"].all():
        raise ValueError("Rules file contains non-synthetic rows")
    bad = set(df["ambulance_category"].dropna()) - VALID_AMBULANCE
    if bad:
        raise ValueError(f"Invalid ambulance categories: {bad}")
    if df[["ambulance_category", "hospital_category", "demo_reason"]].isna().any().any():
        raise ValueError("Rules file has missing mapping fields")

    rules, variants, unknown, incomplete = {}, {}, None, None
    for row in df.to_dict("records"):
        key, status, rule = normalize(row["emergency_type"]), row["input_status"], _rule(row)
        if status == "unknown_type":
            unknown = rule
        elif status == "incomplete" or key is None:
            incomplete = rule
        elif status == "valid_variant":
            variants.setdefault(key, []).append(rule)
        elif key in rules:
            same = (rules[key]["ambulance"], rules[key]["hospital"]) == (rule["ambulance"], rule["hospital"])
            if not same:
                raise ValueError(f"Conflicting rules for '{key}'")
            # exact duplicate mapping -> ignore
        else:
            rules[key] = rule

    # A variant whose type has no main rule (e.g. critical_care_transport) is a rule of its own
    for key in [k for k in variants if k not in rules]:
        rules[key] = variants.pop(key)[0]

    missing = REQUIRED_TYPES - set(rules)
    if missing:
        raise ValueError(f"Missing required emergency types: {missing}")
    if unknown is None or incomplete is None:
        raise ValueError("Rules file needs an unknown-type row and an incomplete-input row")
    return {"rules": rules, "variants": variants, "unknown": unknown, "incomplete": incomplete}


def lookup(table, emergency_type, specialist=False):
    """Returns (rule, status) with status in matched | unknown | incomplete."""
    key = normalize(emergency_type)
    if key is None:
        return table["incomplete"], "incomplete"
    if key not in table["rules"]:
        return table["unknown"], "unknown"
    if specialist and table["variants"].get(key):
        return table["variants"][key][0], "matched"
    return table["rules"][key], "matched"


TABLE = load_rules()


# ------------------------------------------------------------------ RANKING (AI-02)
def haversine(lat1, lng1, lat2, lng2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin((p2 - p1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lng2 - lng1) / 2) ** 2)
    return 2 * 6371 * math.asin(math.sqrt(a))


def _distance(h, lat, lng):
    if h.get("distanceKm") is not None:
        return float(h["distanceKm"])
    if None not in (lat, lng, h.get("lat"), h.get("lng")):
        return haversine(lat, lng, h["lat"], h["lng"])
    return None  # unknown distance -> ranked behind known ones


def _rank(hospitals, lat, lng, cfg):
    rows = [{"id": h.get("id"), "name": h.get("name"),
             "distanceKm": _distance(h, lat, lng),
             "freeBeds": h.get("freeBeds"), "pmjay": bool(h.get("pmjay"))}
            for h in hospitals]
    max_d = max((r["distanceKm"] for r in rows if r["distanceKm"] is not None), default=0)
    max_b = max((r["freeBeds"] or 0 for r in rows), default=0)
    for r in rows:
        d = r["distanceKm"]
        d_score = 0.0 if d is None else (1 - d / max_d if max_d > 0 else 1.0)
        b_score = (r["freeBeds"] or 0) / max_b if max_b > 0 else 0.0
        r["score"] = round(cfg["w_distance"] * d_score
                           + cfg["w_beds"] * b_score
                           + cfg["w_pmjay"] * (1.0 if r["pmjay"] else 0.0), 4)
    rows.sort(key=lambda r: (-r["score"],
                             math.inf if r["distanceKm"] is None else r["distanceKm"],
                             str(r["id"])))
    return rows[:cfg["top_n"]]


def recommend(emergency_type, lat, lng, ambulances, hospitals, *, specialist=False, config=None):
    """Returns {ambulanceType, hospitalCategory, topHospitals, reason}.
    Never raises for empty / None / missing input; falls back to REQUIRES_TRIAGE."""
    cfg = {**CONFIG, **(config or {})}
    rule, status = lookup(TABLE, emergency_type, specialist=specialist)
    ambulances, hospitals = ambulances or [], hospitals or []
    notes = []

    # Ambulances: available + compatible (no ambulance is picked for triage fallbacks)
    if rule["ambulance"] == TRIAGE:
        notes.append("No ambulance level inferred; triage needed.")
    else:
        ok = [a for a in ambulances
              if a.get("available") is True and a.get("category") == rule["ambulance"]]
        notes.append(f"{len(ok)} compatible available ambulance(s)." if ok
                     else "No compatible available ambulance found.")

    # Hospitals: category match (all hospitals for fallbacks), then optionally drop full ones
    pool = hospitals if status != "matched" else [
        h for h in hospitals if h.get("category") == rule["hospital"]]
    if not pool:
        notes.append("No hospital found for this category.")
    elif cfg["exclude_full_hospitals"]:
        with_beds = [h for h in pool if h.get("freeBeds") is None or h["freeBeds"] > 0]
        if not with_beds:
            notes.append("Matching hospitals have no free beds.")
        pool = with_beds

    return {
        "ambulanceType": rule["ambulance"],
        "hospitalCategory": rule["hospital"],
        "topHospitals": _rank(pool, lat, lng, cfg),
        "reason": " ".join([rule["reason"], *notes]) + " (Demo logic, not clinical advice.)",
    }


# ------------------------------------------------------------------ DEMO (built-in fake data)
SAMPLE_AMBULANCES = [
    {"id": "A1", "category": "ALS", "available": True},
    {"id": "A2", "category": "ALS", "available": False},
    {"id": "A3", "category": "BLS", "available": True},
    {"id": "A4", "category": "ICU", "available": True},
    {"id": "A5", "category": "NEONATAL_OR_MATERNITY_SPECIALIST", "available": True},
]
SAMPLE_HOSPITALS = [
    {"id": "H1", "name": "Demo Heart Institute",   "category": "cardiac_capable_emergency_hospital", "distanceKm": 2.1, "freeBeds": 6, "pmjay": True},
    {"id": "H2", "name": "Demo Cardiac Clinic",    "category": "cardiac_capable_emergency_hospital", "distanceKm": 4.5, "freeBeds": 1, "pmjay": False},
    {"id": "H3", "name": "Demo Full Cardiac",      "category": "cardiac_capable_emergency_hospital", "distanceKm": 1.0, "freeBeds": 0, "pmjay": True},
    {"id": "H4", "name": "Demo Trauma Centre",     "category": "trauma_capable_emergency_hospital",  "distanceKm": 3.4, "freeBeds": 5, "pmjay": False},
    {"id": "H5", "name": "Demo General North",     "category": "general_emergency_hospital",         "distanceKm": 5.5, "freeBeds": 14, "pmjay": True},
    {"id": "H6", "name": "Demo General South",     "category": "general_emergency_hospital",         "distanceKm": 1.2, "freeBeds": 8, "pmjay": True},
]
PATIENT_LAT, PATIENT_LNG = 22.7196, 75.8577   # fake patient location

for etype in ["cardiac", "accident_trauma", "general", "banana", None]:
    res = recommend(etype, PATIENT_LAT, PATIENT_LNG, SAMPLE_AMBULANCES, SAMPLE_HOSPITALS)
    print(f"\n=== {etype!r} -> {res['ambulanceType']} | {res['hospitalCategory']}")
    print(res["reason"])
    for h in res["topHospitals"]:
        print("   ", h)
