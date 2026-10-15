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
