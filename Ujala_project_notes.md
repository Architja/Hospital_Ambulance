# Ujala | AI/ML Home Page Phase 1: project notes

Everything from our conversation in one place: what the task is, what we built, the decisions made, and all the code and data files.

**Rules of the task:** synthetic/fake data only; rules-based demo logic (not clinical advice, not a diagnosis); agree JSON field names with Naga Siddeswar before finalising; feature branch + Pull Request, no pushes to main/dev; tell the project lead if blocked more than 2 hours.

---

## 1. The task in plain words

You are building the recommendation brain for an emergency healthcare app. Given an emergency type and the patient's location, the code suggests which ambulance type to send and which hospitals to go to.

| Ticket | Days | What |
|---|---|---|
| AI-01 | 1-2 | Emergency rules table: emergency type -> ambulance category + hospital category + plain-language reason; handle unknown/incomplete input |
| AI-02 | 3-4 | `recommend(emergency_type, lat, lng, ambulances, hospitals)`: filter, rank, config weights, safe empty/no-match handling, 10+ tests |
| AI-03 | 5-6 | FastAPI `POST /recommend` for the Node backend: validation, safe errors, sample requests/responses, startup instructions |

Output fields required by the contract: `ambulanceType`, `hospitalCategory`, `topHospitals`, `reason`.

## 2. What we did, in order

1. Explained the three tickets and a suggested working order.
2. Walked through how to build it (project layout, rules loader, `recommend()`, FastAPI).
3. Made a first fake dataset, then switched to **your two CSVs** (`ujala_emergency_rules_synthetic.csv` and `ujala_synthetic_recommendation_cases.csv`) and deleted mine. My old rules CSV duplicated yours and used different category names.
4. Rewrote the code around your dataset. Things I found in it and how they are handled:
   - exact duplicate `cardiac` row -> ignored;
   - two `maternity` rows with different mappings -> default is BLS / `maternity_emergency_hospital`; the `NEONATAL_OR_MATERNITY_SPECIALIST` row is a variant, used only with `specialist=True`;
   - `critical_care_transport` (ICU) is not one of the 7 required types -> kept as its own rule;
   - unknown and blank types -> `REQUIRES_TRIAGE` / `nearest_verified_emergency_service`;
   - the rules file is rejected if any row has `is_synthetic` false;
   - each case row has one ambulance and one hospital with a precomputed `distance_km`, so the code uses `distanceKm` if present and computes it from lat/lng otherwise.
5. Added a bigger synthetic candidate pool (11 ambulances, 17 hospitals, your category names) so ranking has real choices to make.
6. Built a demo notebook (`recommendation_demo.ipynb`), then a single-file version (`recommendation_full.py`) for pasting into Jupyter.
7. Fixed your Windows path error (the notebook was running from `C:\Users\10090`). Your rules CSV lives at `C:\MAD_internship\files\ujala_emergency_rules_synthetic.csv`, and the single-file code points there with built-in sample data for the demo.
8. Built the FastAPI wrapper (`api.py`) with validation, a consistent 422 error format, and 16 API tests. Total: 61 tests passing.

## 3. Design decisions (mine, worth checking with your lead)

- **Scoring:** distance and free beds are scaled 0-1 across candidates; score = 0.5 x closeness + 0.3 x free beds + 0.2 x PMJAY match (weights in `config.json`). Top 3 returned.
- **Full hospitals:** hospitals with 0 free beds are excluded (`exclude_full_hospitals`, configurable).
- **Fallback hospitals:** for unknown/blank types, all hospitals with beds are ranked, with no category filter.
- **Ambulance match:** exact category match only (no "ICU can serve ALS" upgrade).
- **Response:** exactly the four agreed fields; the ambulance result appears only in `reason`.
- **API:** unknown types are a safe 200 (`REQUIRES_TRIAGE`), not an error; invalid lat/lng, blank or missing `emergencyType`, negative beds are 422 with `{"error": "validation_error", "details": [...]}`. Candidate ambulances/hospitals are sent in the request body.

## 4. Still to settle with Naga Siddeswar / your lead

1. Final JSON field names. Placeholders now: ambulance `id, category, available`; hospital `id, name, category, freeBeds, pmjay, distanceKm` (or `lat`/`lng`).
2. Candidates sent in the request body, or fetched by your service?
3. Does the Node backend accept the 422 error format?
4. Do they want the matched ambulances in the response (a new field)?

## 5. How to run (Windows)

```
cd path\to\ml\recommendation
pip install -r requirements.txt
pip install pytest httpx
python -m pytest                       # expect 61 passed
python -m uvicorn api:app --reload     # then open http://127.0.0.1:8000/docs
```

The code finds the rules CSV in `data\` or next to `api.py`. Or set the `UJALA_RULES_CSV` environment variable.

Sample request for Swagger UI (`POST /recommend`):

```json
{
  "emergencyType": "cardiac",
  "patientLat": 22.7196,
  "patientLng": 75.8577,
  "ambulances": [{"id": "A1", "category": "ALS", "available": true}],
  "hospitals": [{"id": "H1", "name": "Demo Heart Institute", "category": "cardiac_capable_emergency_hospital", "distanceKm": 2.1, "freeBeds": 6, "pmjay": true}]
}
```

## 6. Final handover checklist

- [ ] Code pushed to a feature branch (e.g. `feature/ai-recommendation`)
- [ ] Tests run (61 passed)
- [ ] README / setup instructions written
- [ ] Errors, empty data and invalid input handled (done in code and tests)
- [ ] Sample requests/responses or Swagger screenshots saved
- [ ] Pull Request created and self-reviewed
- [ ] 5-minute demo prepared
- [ ] Limitations and blockers documented (section 3 and 4 above)

## 7. Folder layout

```
ml/recommendation/
  data/        rules CSV, cases CSV, ambulance + hospital pool JSON
  tests/       test_recommend.py, test_api.py
  samples/     sample_requests_responses.json
  api.py  rules.py  recommender.py  recommendation_full.py
  config.json  requirements.txt  conftest.py  make_pool_data.py
  recommendation_demo.ipynb   (notebook; included in the zip)
```

---

## 8. All files

### `config.json`

````json
{
  "w_distance": 0.5,
  "w_beds": 0.3,
  "w_pmjay": 0.2,
  "top_n": 3,
  "exclude_full_hospitals": true
}
````

### `requirements.txt`

````text
pandas
fastapi
uvicorn
````

### `rules.py`

````python
"""Loads and validates the synthetic emergency rules table.

Demo logic only - not clinical advice. Synthetic data only.
"""
import os
from pathlib import Path
import pandas as pd

_NAME = "ujala_emergency_rules_synthetic.csv"


def _find_rules_csv():
    """UJALA_RULES_CSV env var, else data/<csv>, else <csv> next to this file."""
    if os.environ.get("UJALA_RULES_CSV"):
        return Path(os.environ["UJALA_RULES_CSV"])
    here = Path(__file__).parent
    for p in (here / "data" / _NAME, here / _NAME):
        if p.exists():
            return p
    return here / "data" / _NAME


RULES_CSV = _find_rules_csv()

REQUIRED_TYPES = {"cardiac", "accident_trauma", "stroke", "maternity",
                  "breathing", "burns", "general"}
VALID_AMBULANCE = {"BLS", "ALS", "ICU", "NEONATAL_OR_MATERNITY_SPECIALIST", "REQUIRES_TRIAGE"}
TRIAGE = "REQUIRES_TRIAGE"


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
    """Returns {"rules": {type: rule}, "variants": {type: [rule]},
                "unknown": rule, "incomplete": rule}.

    Validation: synthetic-only, valid ambulance categories, all 7 required
    types present, no conflicting duplicates. Exact duplicate rows are ignored.
    """
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
````

### `recommender.py`

````python
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
````

### `api.py`

````python
"""Ujala | AI-03: FastAPI wrapper around recommend().

Start:  python -m uvicorn api:app --reload
Docs :  http://127.0.0.1:8000/docs   (Swagger UI)

Demo logic only - output is a suggestion, never a diagnosis. Synthetic data only.
Field names are PLACEHOLDERS until agreed with Naga Siddeswar. For now the candidate
ambulances/hospitals are sent in the request body (agree this with Naga).
"""
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from recommender import recommend

app = FastAPI(
    title="Ujala Recommendation API",
    version="0.1.0",
    description="Suggests an ambulance type, hospital category and top hospitals. "
                "Rule-based demo logic - not clinical advice, not a diagnosis.",
)


# ---------------------------------------------------------------- request models
class Ambulance(BaseModel):
    id: str
    category: str                       # BLS | ALS | ICU | NEONATAL_OR_MATERNITY_SPECIALIST
    available: bool


class Hospital(BaseModel):
    id: str
    name: Optional[str] = None
    category: str                       # e.g. cardiac_capable_emergency_hospital
    freeBeds: Optional[int] = Field(default=None, ge=0)
    pmjay: bool = False
    distanceKm: Optional[float] = Field(default=None, ge=0)   # if missing, lat/lng is used
    lat: Optional[float] = Field(default=None, ge=-90, le=90)
    lng: Optional[float] = Field(default=None, ge=-180, le=180)


class RecommendRequest(BaseModel):
    emergencyType: str
    patientLat: float = Field(ge=-90, le=90)
    patientLng: float = Field(ge=-180, le=180)
    ambulances: list[Ambulance] = []
    hospitals: list[Hospital] = []

    @field_validator("emergencyType")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("emergencyType must not be blank")
        return v.strip()


# ---------------------------------------------------------------- response models
class TopHospital(BaseModel):
    id: Optional[str] = None
    name: Optional[str] = None
    distanceKm: Optional[float] = None
    freeBeds: Optional[int] = None
    pmjay: bool = False
    score: float


class RecommendResponse(BaseModel):
    ambulanceType: str
    hospitalCategory: str
    topHospitals: list[TopHospital]
    reason: str


# ---------------------------------------------------------------- clear validation errors
@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    details = [{"field": ".".join(str(p) for p in e["loc"][1:]) or "body",
                "message": e["msg"].removeprefix("Value error, ")}
               for e in exc.errors()]
    return JSONResponse(status_code=422, content={"error": "validation_error", "details": details})


# ---------------------------------------------------------------- endpoints
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendResponse)
def post_recommend(req: RecommendRequest):
    """Unknown emergency types are NOT errors: they get the safe REQUIRES_TRIAGE fallback.
    No matching ambulance/hospital is also a normal 200 with an empty topHospitals list."""
    return recommend(
        req.emergencyType, req.patientLat, req.patientLng,
        [a.model_dump() for a in req.ambulances],
        [h.model_dump() for h in req.hospitals],
    )
````

### `recommendation_full.py`

````python
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
DATA_DIR = Path(r"C:\MAD_internship\files")
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
````

### `tests/test_recommend.py`

````python
import math
from pathlib import Path

import pandas as pd
import pytest

from recommender import recommend, haversine, TABLE
from rules import load_rules, lookup, TRIAGE

DATA = Path(__file__).parent.parent / "data"
CASES = pd.read_csv(DATA / "ujala_synthetic_recommendation_cases.csv")


def _val(x):
    return None if pd.isna(x) else x


def case_to_request(row):
    """One CSV row -> recommend() arguments (placeholder contract field names)."""
    amb = [] if _val(row.ambulance_id) is None else [
        {"id": row.ambulance_id, "category": row.ambulance_category,
         "available": bool(row.ambulance_available)}]
    hos = [] if _val(row.hospital_id) is None else [
        {"id": row.hospital_id, "category": row.hospital_category,
         "distanceKm": row.distance_km, "freeBeds": int(row.free_beds),
         "pmjay": bool(row.pmjay_match)}]
    return dict(emergency_type=_val(row.emergency_type), lat=_val(row.patient_lat),
                lng=_val(row.patient_lng), ambulances=amb, hospitals=hos)


def run_case(case_id, **kw):
    row = next(CASES[CASES.case_id == case_id].itertuples())
    return recommend(**case_to_request(row), **kw)


def ids(result):
    return [h["id"] for h in result["topHospitals"]]


# ---------- rules table ----------
def test_rules_load_and_cover_required_types():
    for t in ["cardiac", "accident_trauma", "stroke", "maternity", "breathing", "burns", "general"]:
        assert lookup(TABLE, t)[1] == "matched"


def test_duplicate_cardiac_row_is_ignored_and_maternity_default_is_bls():
    assert lookup(TABLE, "cardiac")[0]["ambulance"] == "ALS"
    assert lookup(TABLE, "maternity")[0]["ambulance"] == "BLS"          # not the specialist variant
    assert lookup(TABLE, "maternity", specialist=True)[0]["ambulance"] == "NEONATAL_OR_MATERNITY_SPECIALIST"


def test_extra_critical_care_type_is_promoted_to_its_own_rule():
    assert lookup(TABLE, "critical_care_transport")[0]["ambulance"] == "ICU"


def test_conflicting_duplicate_rules_are_rejected(tmp_path):
    df = pd.read_csv(DATA / "ujala_emergency_rules_synthetic.csv")
    bad = pd.concat([df, df.iloc[[0]].assign(ambulance_category="BLS", input_status="valid")])
    p = tmp_path / "bad.csv"; bad.to_csv(p, index=False)
    with pytest.raises(ValueError, match="Conflicting"):
        load_rules(p)


def test_non_synthetic_data_is_rejected(tmp_path):
    df = pd.read_csv(DATA / "ujala_emergency_rules_synthetic.csv")
    df.loc[0, "is_synthetic"] = False
    p = tmp_path / "real.csv"; df.to_csv(p, index=False)
    with pytest.raises(ValueError, match="non-synthetic"):
        load_rules(p)


@pytest.mark.parametrize("raw", ["Cardiac", " cardiac ", "CARDIAC"])
def test_emergency_type_is_normalized(raw):
    assert recommend(raw, None, None, [], [])["ambulanceType"] == "ALS"


# ---------- the 12 synthetic cases ----------
@pytest.mark.parametrize("case_id", [f"CASE00{i}" for i in range(1, 8)])
def test_normal_cases(case_id):
    row = next(CASES[CASES.case_id == case_id].itertuples())
    res = run_case(case_id)
    rule = lookup(TABLE, row.emergency_type)[0]
    assert res["ambulanceType"] == rule["ambulance"]
    assert res["hospitalCategory"] == rule["hospital"]
    assert ids(res) == [row.hospital_id]
    assert "1 compatible available ambulance" in res["reason"]


def test_case008_unavailable_ambulance_and_wrong_hospital():
    res = run_case("CASE008")
    assert res["ambulanceType"] == "ALS" and res["topHospitals"] == []
    assert "No compatible available ambulance" in res["reason"]


def test_case009_unknown_type_falls_back_to_triage():
    res = run_case("CASE009")
    assert res["ambulanceType"] == TRIAGE
    assert res["hospitalCategory"] == "nearest_verified_emergency_service"
    assert ids(res) == ["HOSP009"]


def test_case010_incomplete_input_is_safe():
    res = run_case("CASE010")
    assert res["ambulanceType"] == TRIAGE and res["topHospitals"] == []
    assert "missing" in res["reason"]


def test_case011_zero_distance_zero_beds_does_not_crash():
    res = run_case("CASE011")
    assert res["topHospitals"] == []                       # full hospital excluded
    assert ids(run_case("CASE011", config={"exclude_full_hospitals": False})) == ["HOSP011"]


def test_case012_specialist_variant():
    res = run_case("CASE012", specialist=True)
    assert res["ambulanceType"] == "NEONATAL_OR_MATERNITY_SPECIALIST"
    assert ids(res) == ["HOSP012"]
    assert run_case("CASE012")["topHospitals"] == []       # default maternity rule doesn't match it


# ---------- ranking ----------
H = lambda i, d, b, p: {"id": i, "category": "general_emergency_hospital",
                        "distanceKm": d, "freeBeds": b, "pmjay": p}


def test_closer_hospital_wins_when_else_equal():
    res = recommend("general", None, None, [], [H("far", 9, 5, False), H("near", 2, 5, False)])
    assert ids(res) == ["near", "far"]


def test_more_beds_wins_when_else_equal():
    res = recommend("general", None, None, [], [H("few", 3, 1, False), H("many", 3, 9, False)])
    assert ids(res)[0] == "many"


def test_pmjay_wins_when_else_equal():
    res = recommend("general", None, None, [], [H("no", 3, 5, False), H("yes", 3, 5, True)])
    assert ids(res)[0] == "yes"


def test_weights_come_from_config_and_change_the_order():
    hs = [H("near_empty", 1, 1, False), H("far_roomy", 10, 20, False)]
    assert ids(recommend("general", None, None, [], hs,
                         config={"w_distance": 1, "w_beds": 0, "w_pmjay": 0}))[0] == "near_empty"
    assert ids(recommend("general", None, None, [], hs,
                         config={"w_distance": 0, "w_beds": 1, "w_pmjay": 0}))[0] == "far_roomy"


def test_top_n_limit_and_deterministic_ties():
    hs = [H(f"H{i}", 3, 5, False) for i in range(6)]
    assert ids(recommend("general", None, None, [], hs)) == ["H0", "H1", "H2"]


def test_distance_computed_from_lat_lng_when_distance_missing():
    h = {"id": "X", "category": "general_emergency_hospital", "lat": 22.73, "lng": 75.86, "freeBeds": 3}
    res = recommend("general", 22.72, 75.86, [], [h])
    assert math.isclose(res["topHospitals"][0]["distanceKm"], haversine(22.72, 75.86, 22.73, 75.86))


# ---------- empty / bad input ----------
@pytest.mark.parametrize("amb,hos", [([], []), (None, None), ([], None), (None, [])])
def test_empty_candidate_lists(amb, hos):
    res = recommend("cardiac", 22.7, 75.8, amb, hos)
    assert res["topHospitals"] == [] and "No compatible available ambulance" in res["reason"]


@pytest.mark.parametrize("bad", [None, "", "   ", "not_a_real_type", float("nan")])
def test_bad_emergency_types_never_raise(bad):
    assert recommend(bad, None, None, [], [])["ambulanceType"] == TRIAGE


def test_response_has_exactly_the_agreed_fields():
    assert set(run_case("CASE001")) == {"ambulanceType", "hospitalCategory", "topHospitals", "reason"}


# ---------- bigger synthetic pool (ranking with real choice) ----------
import json
POOL_A = json.loads((DATA / "ambulances_pool_synthetic.json").read_text())
POOL_H = json.loads((DATA / "hospitals_pool_synthetic.json").read_text())
PATIENT = (22.7196, 75.8577)


@pytest.mark.parametrize("etype", ["cardiac", "accident_trauma", "stroke", "maternity",
                                   "breathing", "burns", "general"])
def test_pool_results_match_category_and_are_sorted(etype):
    res = recommend(etype, *PATIENT, POOL_A, POOL_H)
    assert res["topHospitals"], "every required type has a hospital with beds in the pool"
    cats = {h["category"] for h in POOL_H if h["id"] in ids(res)}
    assert cats == {res["hospitalCategory"]}
    scores = [h["score"] for h in res["topHospitals"]]
    assert scores == sorted(scores, reverse=True)
    assert all(h["freeBeds"] != 0 for h in res["topHospitals"])
    assert "1 compatible" in res["reason"] or "compatible available ambulance" in res["reason"]


def test_pool_full_hospital_never_recommended():
    res = recommend("cardiac", *PATIENT, POOL_A, POOL_H)
    assert "Demo Multispeciality Cardiac" not in [h["name"] for h in res["topHospitals"]]


def test_pool_unavailable_ambulance_not_counted():
    only_unavailable = [a for a in POOL_A if a["category"] == "ICU" and not a["available"]]
    res = recommend("critical_care_transport", *PATIENT, only_unavailable, POOL_H)
    assert "No compatible available ambulance" in res["reason"]
````

### `tests/test_api.py`

````python
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api import app

client = TestClient(app)
DATA = Path(__file__).parent.parent / "data"
AMB = json.loads((DATA / "ambulances_pool_synthetic.json").read_text())
HOS = json.loads((DATA / "hospitals_pool_synthetic.json").read_text())
PICK_A = lambda a: {k: a[k] for k in ("id", "category", "available")}
PICK_H = lambda h: {k: h[k] for k in ("id", "name", "category", "freeBeds", "pmjay", "lat", "lng")}


def body(**kw):
    base = {"emergencyType": "cardiac", "patientLat": 22.7196, "patientLng": 75.8577,
            "ambulances": [PICK_A(a) for a in AMB], "hospitals": [PICK_H(h) for h in HOS]}
    return {**base, **kw}


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_valid_request_returns_agreed_shape():
    r = client.post("/recommend", json=body())
    assert r.status_code == 200
    d = r.json()
    assert set(d) == {"ambulanceType", "hospitalCategory", "topHospitals", "reason"}
    assert d["ambulanceType"] == "ALS" and d["hospitalCategory"] == "cardiac_capable_emergency_hospital"
    assert 1 <= len(d["topHospitals"]) <= 3
    assert all(h["freeBeds"] != 0 for h in d["topHospitals"])


def test_unknown_type_is_safe_fallback_not_error():
    r = client.post("/recommend", json=body(emergencyType="banana"))
    assert r.status_code == 200 and r.json()["ambulanceType"] == "REQUIRES_TRIAGE"


def test_empty_candidate_lists_are_a_safe_no_match():
    r = client.post("/recommend", json=body(ambulances=[], hospitals=[]))
    d = r.json()
    assert r.status_code == 200 and d["topHospitals"] == []
    assert "No compatible available ambulance" in d["reason"]


def test_candidate_lists_are_optional():
    r = client.post("/recommend", json={"emergencyType": "stroke", "patientLat": 1, "patientLng": 1})
    assert r.status_code == 200 and r.json()["topHospitals"] == []


@pytest.mark.parametrize("patch,field", [
    ({"patientLat": 123}, "patientLat"),
    ({"patientLng": -999}, "patientLng"),
    ({"patientLat": "abc"}, "patientLat"),
    ({"emergencyType": "   "}, "emergencyType"),
    ({"emergencyType": None}, "emergencyType"),
    ({"emergencyType": 5}, "emergencyType"),
])
def test_invalid_input_gives_controlled_422(patch, field):
    r = client.post("/recommend", json=body(**patch))
    assert r.status_code == 422
    d = r.json()
    assert d["error"] == "validation_error"
    assert field in [e["field"] for e in d["details"]]


@pytest.mark.parametrize("missing", ["emergencyType", "patientLat", "patientLng"])
def test_missing_required_fields(missing):
    b = body(); b.pop(missing)
    r = client.post("/recommend", json=b)
    assert r.status_code == 422 and missing in [e["field"] for e in r.json()["details"]]


def test_bad_hospital_field_is_reported_with_its_path():
    r = client.post("/recommend", json=body(hospitals=[{"id": "H", "category": "x", "freeBeds": -3}]))
    assert r.status_code == 422
    assert "hospitals.0.freeBeds" in [e["field"] for e in r.json()["details"]]


def test_malformed_json_is_a_controlled_422():
    r = client.post("/recommend", content="{not json", headers={"content-type": "application/json"})
    assert r.status_code == 422 and r.json()["error"] == "validation_error"
````

### `conftest.py`

````python

````

### `make_pool_data.py`

````python
"""Generates FAKE candidate pools (data/ambulances_pool_synthetic.json, data/hospitals_pool_synthetic.json).
Uses the category names from ujala_emergency_rules_synthetic.csv. Field names are placeholders until agreed with Naga."""
import json, random
from pathlib import Path
random.seed(7)
LAT, LNG = 22.7196, 75.8577   # same patient point as the synthetic cases file
OUT = Path(__file__).parent / "data"

def near(s=0.05):
    return round(LAT + random.uniform(-s, s), 5), round(LNG + random.uniform(-s, s), 5)

CAT = {"cardiac": "cardiac_capable_emergency_hospital", "trauma": "trauma_capable_emergency_hospital",
       "stroke": "stroke_capable_emergency_hospital", "mat": "maternity_emergency_hospital",
       "resp": "emergency_hospital_with_respiratory_support", "burns": "burns_capable_emergency_hospital",
       "gen": "general_emergency_hospital", "neo": "maternity_and_neonatal_capable_hospital",
       "icu": "hospital_with_critical_care"}
# (key, name, freeBeds, pmjay)
H = [("cardiac","Demo Heart Institute",6,True),("cardiac","Demo Cardiac Clinic",1,False),("cardiac","Demo Multispeciality Cardiac",0,True),
     ("trauma","Demo Trauma Centre",4,True),("trauma","Demo Emergency Care",8,False),
     ("stroke","Demo Neuro Hospital",2,True),("stroke","Demo Stroke Unit",5,False),
     ("mat","Demo Mother Hospital",9,True),("mat","Demo Maternity Home",3,False),
     ("resp","Demo Chest Care",5,False),("resp","Demo Lung Institute",7,True),
     ("burns","Demo Burns Unit",3,True),
     ("gen","Demo General North",14,True),("gen","Demo General South",0,True),("gen","Demo Community Centre",11,False),
     ("neo","Demo Mother & Newborn Hospital",2,True),("icu","Demo Critical Care Hospital",3,True)]
A = [("ALS",True),("ALS",True),("ALS",False),("ALS",True),("BLS",True),("BLS",True),("BLS",False),("BLS",True),
     ("ICU",True),("ICU",False),("NEONATAL_OR_MATERNITY_SPECIALIST",True)]

hospitals = []
for n, (k, name, beds, pm) in enumerate(H, 1):
    lat, lng = near()
    hospitals.append({"id": f"HOSPP{n:03d}", "name": name, "category": CAT[k], "lat": lat, "lng": lng, "freeBeds": beds, "pmjay": pm})
ambulances = []
for n, (c, av) in enumerate(A, 1):
    lat, lng = near()
    ambulances.append({"id": f"AMBP{n:03d}", "category": c, "available": av, "lat": lat, "lng": lng})
(OUT / "hospitals_pool_synthetic.json").write_text(json.dumps(hospitals, indent=2))
(OUT / "ambulances_pool_synthetic.json").write_text(json.dumps(ambulances, indent=2))
print(len(ambulances), "ambulances,", len(hospitals), "hospitals")
````

### `data/ujala_emergency_rules_synthetic.csv`

````csv
emergency_type,ambulance_category,hospital_category,demo_reason,input_status,is_synthetic
cardiac,ALS,cardiac_capable_emergency_hospital,Demo mapping: cardiac symptoms may require advanced prehospital support and a hospital with cardiac emergency capability.,valid,True
accident_trauma,ALS,trauma_capable_emergency_hospital,Demo mapping: significant trauma may require advanced support and trauma-capable emergency services.,valid,True
stroke,ALS,stroke_capable_emergency_hospital,Demo mapping: suspected stroke requires urgent assessment at a facility able to evaluate stroke symptoms.,valid,True
maternity,BLS,maternity_emergency_hospital,Demo mapping only: actual ambulance level depends on the patient's condition and local protocols.,valid,True
breathing,ALS,emergency_hospital_with_respiratory_support,Demo mapping: breathing emergencies require urgent assessment; actual transport capability must be confirmed.,valid,True
burns,ALS,burns_capable_emergency_hospital,Demo mapping: serious burns may require advanced support and a facility able to assess and treat burns.,valid,True
general,BLS,general_emergency_hospital,Default demo mapping for a general emergency; triage and local protocols determine actual care.,valid,True
unknown,REQUIRES_TRIAGE,nearest_verified_emergency_service,"Safe fallback: emergency type is unknown, so do not infer a specific ambulance level or specialty.",unknown_type,True
,REQUIRES_TRIAGE,nearest_verified_emergency_service,Safe fallback: emergency type is missing; request clarification while directing the user to emergency services if urgent.,incomplete,True
cardiac,ALS,cardiac_capable_emergency_hospital,Synthetic duplicate scenario for duplicate-row validation testing.,valid_duplicate_test,True
maternity,NEONATAL_OR_MATERNITY_SPECIALIST,maternity_and_neonatal_capable_hospital,Demo mapping: use only when maternity/newborn context is specified; confirm real capabilities before ranking.,valid_variant,True
critical_care_transport,ICU,hospital_with_critical_care,Synthetic edge case for critical-care transport; not one of the seven required emergency types.,valid_variant,True
````

### `data/ujala_synthetic_recommendation_cases.csv`

````csv
case_id,emergency_type,patient_lat,patient_lng,ambulance_id,ambulance_category,ambulance_available,hospital_id,hospital_category,distance_km,free_beds,pmjay_match,scenario_label,is_synthetic
CASE001,cardiac,22.7196,75.8577,AMB001,ALS,True,HOSP001,cardiac_capable_emergency_hospital,2.1,3.0,True,normal,True
CASE002,accident_trauma,22.725,75.865,AMB002,ALS,True,HOSP002,trauma_capable_emergency_hospital,3.4,5.0,False,normal,True
CASE003,stroke,22.71,75.84,AMB003,ALS,True,HOSP003,stroke_capable_emergency_hospital,1.8,2.0,True,normal,True
CASE004,maternity,22.73,75.88,AMB004,BLS,True,HOSP004,maternity_emergency_hospital,4.0,4.0,True,normal,True
CASE005,breathing,22.715,75.85,AMB005,ALS,True,HOSP005,emergency_hospital_with_respiratory_support,2.6,6.0,False,normal,True
CASE006,burns,22.705,75.87,AMB006,ALS,True,HOSP006,burns_capable_emergency_hospital,5.2,3.0,True,normal,True
CASE007,general,22.74,75.83,AMB007,BLS,True,HOSP007,general_emergency_hospital,1.2,8.0,True,normal,True
CASE008,cardiac,22.7196,75.8577,AMB008,BLS,False,HOSP008,general_emergency_hospital,0.9,0.0,False,unavailable_ambulance_and_no_beds,True
CASE009,unknown,22.7196,75.8577,AMB009,BLS,True,HOSP009,general_emergency_hospital,1.5,2.0,False,unknown_emergency,True
CASE010,,,,,,,,,,,,incomplete_input,True
CASE011,breathing,22.7196,75.8577,AMB011,ALS,True,HOSP011,emergency_hospital_with_respiratory_support,0.0,0.0,True,zero_distance_zero_beds_edge,True
CASE012,maternity,22.7196,75.8577,AMB012,NEONATAL_OR_MATERNITY_SPECIALIST,True,HOSP012,maternity_and_neonatal_capable_hospital,3.0,2.0,True,specialist_variant,True
````

### `data/ambulances_pool_synthetic.json`

````json
[
  {
    "id": "AMBP001",
    "category": "ALS",
    "available": true,
    "lat": 22.67556,
    "lng": 75.8283
  },
  {
    "id": "AMBP002",
    "category": "ALS",
    "available": true,
    "lat": 22.73764,
    "lng": 75.85046
  },
  {
    "id": "AMBP003",
    "category": "ALS",
    "available": false,
    "lat": 22.70101,
    "lng": 75.86626
  },
  {
    "id": "AMBP004",
    "category": "ALS",
    "available": true,
    "lat": 22.71492,
    "lng": 75.83768
  },
  {
    "id": "AMBP005",
    "category": "BLS",
    "available": true,
    "lat": 22.74904,
    "lng": 75.8776
  },
  {
    "id": "AMBP006",
    "category": "BLS",
    "available": true,
    "lat": 22.69401,
    "lng": 75.86514
  },
  {
    "id": "AMBP007",
    "category": "BLS",
    "available": false,
    "lat": 22.72212,
    "lng": 75.89521
  },
  {
    "id": "AMBP008",
    "category": "BLS",
    "available": true,
    "lat": 22.74254,
    "lng": 75.83649
  },
  {
    "id": "AMBP009",
    "category": "ICU",
    "available": true,
    "lat": 22.76762,
    "lng": 75.81951
  },
  {
    "id": "AMBP010",
    "category": "ICU",
    "available": false,
    "lat": 22.71141,
    "lng": 75.88341
  },
  {
    "id": "AMBP011",
    "category": "NEONATAL_OR_MATERNITY_SPECIALIST",
    "available": true,
    "lat": 22.6848,
    "lng": 75.8566
  }
]
````

### `data/hospitals_pool_synthetic.json`

````json
[
  {
    "id": "HOSPP001",
    "name": "Demo Heart Institute",
    "category": "cardiac_capable_emergency_hospital",
    "lat": 22.70198,
    "lng": 75.82278,
    "freeBeds": 6,
    "pmjay": true
  },
  {
    "id": "HOSPP002",
    "name": "Demo Cardiac Clinic",
    "category": "cardiac_capable_emergency_hospital",
    "lat": 22.73469,
    "lng": 75.81494,
    "freeBeds": 1,
    "pmjay": false
  },
  {
    "id": "HOSPP003",
    "name": "Demo Multispeciality Cardiac",
    "category": "cardiac_capable_emergency_hospital",
    "lat": 22.72319,
    "lng": 75.84427,
    "freeBeds": 0,
    "pmjay": true
  },
  {
    "id": "HOSPP004",
    "name": "Demo Trauma Centre",
    "category": "trauma_capable_emergency_hospital",
    "lat": 22.6754,
    "lng": 75.85844,
    "freeBeds": 4,
    "pmjay": true
  },
  {
    "id": "HOSPP005",
    "name": "Demo Emergency Care",
    "category": "trauma_capable_emergency_hospital",
    "lat": 22.67335,
    "lng": 75.85106,
    "freeBeds": 8,
    "pmjay": false
  },
  {
    "id": "HOSPP006",
    "name": "Demo Neuro Hospital",
    "category": "stroke_capable_emergency_hospital",
    "lat": 22.67659,
    "lng": 75.81677,
    "freeBeds": 2,
    "pmjay": true
  },
  {
    "id": "HOSPP007",
    "name": "Demo Stroke Unit",
    "category": "stroke_capable_emergency_hospital",
    "lat": 22.71205,
    "lng": 75.89039,
    "freeBeds": 5,
    "pmjay": false
  },
  {
    "id": "HOSPP008",
    "name": "Demo Mother Hospital",
    "category": "maternity_emergency_hospital",
    "lat": 22.68198,
    "lng": 75.83002,
    "freeBeds": 9,
    "pmjay": true
  },
  {
    "id": "HOSPP009",
    "name": "Demo Maternity Home",
    "category": "maternity_emergency_hospital",
    "lat": 22.73234,
    "lng": 75.90247,
    "freeBeds": 3,
    "pmjay": false
  },
  {
    "id": "HOSPP010",
    "name": "Demo Chest Care",
    "category": "emergency_hospital_with_respiratory_support",
    "lat": 22.72731,
    "lng": 75.84737,
    "freeBeds": 5,
    "pmjay": false
  },
  {
    "id": "HOSPP011",
    "name": "Demo Lung Institute",
    "category": "emergency_hospital_with_respiratory_support",
    "lat": 22.76723,
    "lng": 75.81236,
    "freeBeds": 7,
    "pmjay": true
  },
  {
    "id": "HOSPP012",
    "name": "Demo Burns Unit",
    "category": "burns_capable_emergency_hospital",
    "lat": 22.75545,
    "lng": 75.83666,
    "freeBeds": 3,
    "pmjay": true
  },
  {
    "id": "HOSPP013",
    "name": "Demo General North",
    "category": "general_emergency_hospital",
    "lat": 22.68403,
    "lng": 75.81948,
    "freeBeds": 14,
    "pmjay": true
  },
  {
    "id": "HOSPP014",
    "name": "Demo General South",
    "category": "general_emergency_hospital",
    "lat": 22.70045,
    "lng": 75.88931,
    "freeBeds": 0,
    "pmjay": true
  },
  {
    "id": "HOSPP015",
    "name": "Demo Community Centre",
    "category": "general_emergency_hospital",
    "lat": 22.68767,
    "lng": 75.86586,
    "freeBeds": 11,
    "pmjay": false
  },
  {
    "id": "HOSPP016",
    "name": "Demo Mother & Newborn Hospital",
    "category": "maternity_and_neonatal_capable_hospital",
    "lat": 22.73349,
    "lng": 75.84494,
    "freeBeds": 2,
    "pmjay": true
  },
  {
    "id": "HOSPP017",
    "name": "Demo Critical Care Hospital",
    "category": "hospital_with_critical_care",
    "lat": 22.72437,
    "lng": 75.81398,
    "freeBeds": 3,
    "pmjay": true
  }
]
````

### `samples/sample_requests_responses.json`

````json
{
  "1_cardiac_normal": {
    "request": {
      "emergencyType": "cardiac",
      "patientLat": 22.7196,
      "patientLng": 75.8577,
      "ambulances": [
        {
          "id": "A1",
          "category": "ALS",
          "available": true
        },
        {
          "id": "A2",
          "category": "ALS",
          "available": false
        },
        {
          "id": "A3",
          "category": "BLS",
          "available": true
        },
        {
          "id": "A4",
          "category": "ICU",
          "available": true
        }
      ],
      "hospitals": [
        {
          "id": "H1",
          "name": "Demo Heart Institute",
          "category": "cardiac_capable_emergency_hospital",
          "distanceKm": 2.1,
          "freeBeds": 6,
          "pmjay": true
        },
        {
          "id": "H2",
          "name": "Demo Cardiac Clinic",
          "category": "cardiac_capable_emergency_hospital",
          "distanceKm": 4.5,
          "freeBeds": 1,
          "pmjay": false
        },
        {
          "id": "H3",
          "name": "Demo Full Cardiac",
          "category": "cardiac_capable_emergency_hospital",
          "distanceKm": 1.0,
          "freeBeds": 0,
          "pmjay": true
        },
        {
          "id": "H4",
          "name": "Demo General South",
          "category": "general_emergency_hospital",
          "distanceKm": 1.2,
          "freeBeds": 8,
          "pmjay": true
        }
      ]
    },
    "status": 200,
    "response": {
      "ambulanceType": "ALS",
      "hospitalCategory": "cardiac_capable_emergency_hospital",
      "topHospitals": [
        {
          "id": "H1",
          "name": "Demo Heart Institute",
          "distanceKm": 2.1,
          "freeBeds": 6,
          "pmjay": true,
          "score": 0.7667
        },
        {
          "id": "H2",
          "name": "Demo Cardiac Clinic",
          "distanceKm": 4.5,
          "freeBeds": 1,
          "pmjay": false,
          "score": 0.05
        }
      ],
      "reason": "Demo mapping: cardiac symptoms may require advanced prehospital support and a hospital with cardiac emergency capability. 1 compatible available ambulance(s). (Demo logic, not clinical advice.)"
    }
  },
  "2_unknown_type_safe_fallback": {
    "request": {
      "emergencyType": "banana",
      "patientLat": 22.7196,
      "patientLng": 75.8577,
      "ambulances": [
        {
          "id": "A1",
          "category": "ALS",
          "available": true
        },
        {
          "id": "A2",
          "category": "ALS",
          "available": false
        },
        {
          "id": "A3",
          "category": "BLS",
          "available": true
        },
        {
          "id": "A4",
          "category": "ICU",
          "available": true
        }
      ],
      "hospitals": [
        {
          "id": "H1",
          "name": "Demo Heart Institute",
          "category": "cardiac_capable_emergency_hospital",
          "distanceKm": 2.1,
          "freeBeds": 6,
          "pmjay": true
        },
        {
          "id": "H2",
          "name": "Demo Cardiac Clinic",
          "category": "cardiac_capable_emergency_hospital",
          "distanceKm": 4.5,
          "freeBeds": 1,
          "pmjay": false
        },
        {
          "id": "H3",
          "name": "Demo Full Cardiac",
          "category": "cardiac_capable_emergency_hospital",
          "distanceKm": 1.0,
          "freeBeds": 0,
          "pmjay": true
        },
        {
          "id": "H4",
          "name": "Demo General South",
          "category": "general_emergency_hospital",
          "distanceKm": 1.2,
          "freeBeds": 8,
          "pmjay": true
        }
      ]
    },
    "status": 200,
    "response": {
      "ambulanceType": "REQUIRES_TRIAGE",
      "hospitalCategory": "nearest_verified_emergency_service",
      "topHospitals": [
        {
          "id": "H4",
          "name": "Demo General South",
          "distanceKm": 1.2,
          "freeBeds": 8,
          "pmjay": true,
          "score": 0.8667
        },
        {
          "id": "H1",
          "name": "Demo Heart Institute",
          "distanceKm": 2.1,
          "freeBeds": 6,
          "pmjay": true,
          "score": 0.6917
        },
        {
          "id": "H2",
          "name": "Demo Cardiac Clinic",
          "distanceKm": 4.5,
          "freeBeds": 1,
          "pmjay": false,
          "score": 0.0375
        }
      ],
      "reason": "Safe fallback: emergency type is unknown, so do not infer a specific ambulance level or specialty. No ambulance level inferred; triage needed. (Demo logic, not clinical advice.)"
    }
  },
  "3_no_candidates_safe_no_match": {
    "request": {
      "emergencyType": "stroke",
      "patientLat": 22.7196,
      "patientLng": 75.8577,
      "ambulances": [],
      "hospitals": []
    },
    "status": 200,
    "response": {
      "ambulanceType": "ALS",
      "hospitalCategory": "stroke_capable_emergency_hospital",
      "topHospitals": [],
      "reason": "Demo mapping: suspected stroke requires urgent assessment at a facility able to evaluate stroke symptoms. No compatible available ambulance found. No hospital found for this category. (Demo logic, not clinical advice.)"
    }
  },
  "4_invalid_latitude_422": {
    "request": {
      "emergencyType": "cardiac",
      "patientLat": 500,
      "patientLng": 75.8577
    },
    "status": 422,
    "response": {
      "error": "validation_error",
      "details": [
        {
          "field": "patientLat",
          "message": "Input should be less than or equal to 90"
        }
      ]
    }
  },
  "5_missing_emergency_type_422": {
    "request": {
      "patientLat": 22.7196,
      "patientLng": 75.8577
    },
    "status": 422,
    "response": {
      "error": "validation_error",
      "details": [
        {
          "field": "emergencyType",
          "message": "Field required"
        }
      ]
    }
  }
}
````
