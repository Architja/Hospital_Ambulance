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
