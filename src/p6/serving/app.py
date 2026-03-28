import os
import mlflow
import pandas as pd
from mlflow import MlflowClient
from pydantic import BaseModel
from fastapi import FastAPI
from typing import Literal

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:../mlruns")

mlflow.set_tracking_uri(tracking_uri)

client = MlflowClient()
app = FastAPI()

loaded_model = mlflow.pyfunc.load_model("models:/pm-model@champion")

champion = client.get_model_version_by_alias(name= "pm-model", alias="champion")

class PredictRequest(BaseModel):
    # 原始感測器
    volt: float
    rotate: float
    pressure: float
    vibration: float
    age: int              # MLflow schema 是 long（int64），不能用 float

    # sensor rolling features
    volt_mean_3h: float
    volt_std_3h: float
    volt_mean_24h: float
    volt_std_24h: float
    rotate_mean_3h: float
    rotate_std_3h: float
    rotate_mean_24h: float
    rotate_std_24h: float
    pressure_mean_3h: float
    pressure_std_3h: float
    pressure_mean_24h: float
    pressure_std_24h: float
    vibration_mean_3h: float
    vibration_std_3h: float
    vibration_mean_24h: float
    vibration_std_24h: float

    # error rolling count
    errorID_error1: float
    errorID_error2: float
    errorID_error3: float
    errorID_error4: float
    errorID_error5: float

    # maintenance rolling count
    comp_comp1: float
    comp_comp2: float
    comp_comp3: float
    comp_comp4: float

    # categorical
    model: Literal["model1", "model2", "model3", "model4"]


class PredictResponse(BaseModel):
    prediction: int       # 0 = 正常, 1 = 會故障
    probability: float    # P(failure)

class ModelInfoResponse(BaseModel):
    model_name : str
    version : str
    run_id : str



@app.get("/health")
def health():
    return {"status" : "ok"}

@app.get("/model-info", response_model=ModelInfoResponse)
def modelinfo():
    return ModelInfoResponse(model_name=champion.name, version = str(champion.version), run_id = champion.run_id)
@app.post("/predict", response_model=PredictResponse)
def predict(request : PredictRequest):
    input_df = pd.DataFrame([request.model_dump()])
    pred = int(loaded_model.predict(input_df)[0])

    proba = loaded_model._model_impl.predict_proba(input_df)[0][1]

    return PredictResponse(prediction=pred, probability=round(float(proba), 4))

