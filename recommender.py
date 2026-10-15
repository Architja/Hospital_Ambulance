"""recommend(): rule lookup + ambulance filter + hospital ranking.

Demo logic only - output is a suggestion, never a diagnosis.

Field names are PLACEHOLDERS until agreed with Naga Siddeswar:
  ambulance: id, category, available
  hospital : id, category, freeBeds, pmjay, and either distanceKm or lat+lng
"""
import json
import math
from pathlib import Path

from rules import load_rules, lookup, TRIAGE

_DEFAULTS = {"w_distance": 0.5, "w_beds": 0.3, "w_pmjay": 0.2, "top_n": 3, "exclude_full_hospitals": True}
_CFG_FILE = Path(__file__).parent / "config.json"
CONFIG = {**_DEFAULTS, **(json.loads(_CFG_FILE.read_text()) if _CFG_FILE.exists() else {})}
TABLE = load_rules()


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

    Never raises for empty / None / missing input; falls back to REQUIRES_TRIAGE.
    specialist=True selects the maternity/newborn specialist variant rule.
    """
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
