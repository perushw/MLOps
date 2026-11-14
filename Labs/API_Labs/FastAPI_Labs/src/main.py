from contextlib import asynccontextmanager
from typing import Annotated, Literal
from fastapi import Body, FastAPI, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field
from predict import load_model, predict_data

ml_models = {}


# load the model once when the server starts instead of reading the file on every request
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        ml_models["penguin"] = load_model()
    except FileNotFoundError:
        print("No trained model found. Run train.py, then restart the API.")
    yield
    ml_models.clear()


app = FastAPI(
    title="Penguin Species API",
    description="Predicts the species of a Palmer Archipelago penguin (Adelie, Chinstrap or Gentoo) from its measurements.",
    lifespan=lifespan,
)


class PenguinData(BaseModel):
    # the first example is pre-filled in the request body under "try it out" on /docs
    model_config = ConfigDict(json_schema_extra={
        "examples": [{
            "island": "Torgersen",
            "sex": "male",
            "bill_length_mm": 39.1,
            "bill_depth_mm": 18.7,
            "flipper_length_mm": 181,
            "body_mass_g": 3750,
        }]
    })

    island: Literal["Biscoe", "Dream", "Torgersen"]
    sex: Literal["male", "female"]
    bill_length_mm: float = Field(gt=0, le=100, description="Bill (culmen) length in mm")
    bill_depth_mm: float = Field(gt=0, le=40, description="Bill (culmen) depth in mm")
    flipper_length_mm: float = Field(gt=0, le=300, description="Flipper length in mm")
    body_mass_g: float = Field(gt=0, le=10000, description="Body mass in grams")


class PenguinResponse(BaseModel):
    species: str
    confidence: float
    probabilities: dict[str, float]


class ModelInfo(BaseModel):
    algorithm: str
    classes: list[str]
    features: list[str]
    test_accuracy: float
    train_rows: int
    test_rows: int
    trained_at: str
    sklearn_version: str
    feature_importances: dict[str, float]


def get_model_bundle():
    if "penguin" not in ml_models:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Run train.py, then restart the API.",
        )
    return ml_models["penguin"]


@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy", "model_loaded": "penguin" in ml_models}


@app.get("/model-info", response_model=ModelInfo)
async def model_info():
    return get_model_bundle()["info"]


# plain def instead of async def: fastapi runs it in a thread pool, so the cpu-bound prediction does not block other requests
@app.post("/predict", response_model=PenguinResponse)
def predict_penguin(penguin: PenguinData):
    model = get_model_bundle()["model"]
    try:
        return predict_data(model, [penguin.model_dump()])[0]
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/predict/batch", response_model=list[PenguinResponse])
def predict_penguins(penguins: Annotated[list[PenguinData], Body(min_length=1, max_length=100)]):
    model = get_model_bundle()["model"]
    try:
        return predict_data(model, [p.model_dump() for p in penguins])
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
