# Ujala Hospital & Ambulance Recommendation System

Emergency healthcare recommendation engine that suggests compatible ambulance levels, hospital categories, and ranked emergency hospitals based on patient location, travel distance, bed availability, and PMJAY scheme coverage.

> **Note:** Demo logic only — suggestions are rule-based recommendations, not clinical advice or a medical diagnosis. Uses synthetic data only.

---

## 🚀 Quick Start (Running from Git)

### 1. Clone the Repository
```bash
git clone https://github.com/Architja/Hospital_Ambulance.git
cd Hospital_Ambulance
```

### 2. Set Up a Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🏃‍♂️ Running the Services

### Start the FastAPI Server
To launch the REST API server:
```bash
uvicorn api:app --reload
```
Once started:
* **Interactive Swagger UI**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **Alternative API Docs**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
* **Health Check**: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

### Test with a Sample Request
You can send a POST request to `/recommend`:
```bash
curl -X POST "http://127.0.0.1:8000/recommend" \
     -H "Content-Type: application/json" \
     -d "{\"emergencyType\": \"cardiac\", \"patientLat\": 22.7196, \"patientLng\": 75.8577}"
```

---

## 🧪 Running Tests

Run the full pytest suite (61 tests covering rules, rankings, and API endpoints):
```bash
pytest -v
```

---

## 📓 Running the Standalone Script / Notebook

* **Standalone script**:
  ```bash
  python recommendation_full.py
  ```
* **Jupyter Notebook**:
  ```bash
  jupyter notebook recommendation_demo.ipynb
  ```

---

## 📂 Project Structure

```text
├── api.py                                 # FastAPI application and endpoints
├── recommender.py                         # Hospital scoring and ambulance recommendation engine
├── rules.py                               # Emergency rules loader and validator
├── config.json                            # Scoring weights and top-n configuration
├── requirements.txt                       # Python dependencies
├── test_api.py                            # API validation and endpoint tests
├── test_recommend.py                      # Recommendation logic and rules tests
├── recommendation_full.py                 # Self-contained single-file demonstration
├── recommendation_demo.ipynb              # Interactive Jupyter Notebook demo
├── sample_requests_responses.json         # API request and response fixtures
├── ambulances_pool_synthetic.json         # Synthetic ambulance candidate pool
├── hospitals_pool_synthetic.json          # Synthetic hospital candidate pool
├── ujala_emergency_rules_synthetic.csv     # Emergency rules lookup table
├── ujala_synthetic_recommendation_cases.csv# Synthetic test cases
└── Ujala_project_notes.md                 # Detailed project architecture notes
```
