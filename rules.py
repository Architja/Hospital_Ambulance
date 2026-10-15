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
