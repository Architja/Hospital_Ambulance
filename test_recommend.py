import math
from pathlib import Path

import pandas as pd
import pytest

from recommender import recommend, haversine, TABLE
from rules import load_rules, lookup, TRIAGE

def _find_data_dir():
    here = Path(__file__).parent
    for d in (here, here / "data", here.parent / "data"):
        if (d / "ujala_synthetic_recommendation_cases.csv").exists():
            return d
    return here

DATA = _find_data_dir()
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
