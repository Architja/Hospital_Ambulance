"""Generates FAKE candidate pools (data/ambulances_pool_synthetic.json, data/hospitals_pool_synthetic.json).
Uses the category names from ujala_emergency_rules_synthetic.csv. Field names are placeholders until agreed with Naga."""
import json, random
from pathlib import Path
random.seed(7)
LAT, LNG = 22.7196, 75.8577   # same patient point as the synthetic cases file
OUT = Path(__file__).parent / "data" if (Path(__file__).parent / "data").is_dir() else Path(__file__).parent

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
