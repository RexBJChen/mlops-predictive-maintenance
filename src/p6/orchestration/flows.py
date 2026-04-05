import os

import mlflow
import pandas as pd
from mlflow import MlflowClient
from prefect import flow, task

from p6.ingestion import load_raw_data
from p6.validation import validate
from p6.feature_pipeline import build_features
from p6.training import train
from p6.registry import promote_model
from p6.monitoring import check_drift

load_raw_data_task = task(load_raw_data, name="load-raw-data", retries=2, retry_delay_seconds=5)
validate_task = task(validate, name="validate")
build_features_task = task(build_features, name="build-features")
train_task = task(train, name="train")
promote_model_task = task(promote_model, name="promote-model")
check_drift_task = task(check_drift, name="check-drift")


@flow(name="retrain-pipeline", log_prints=True)
def retrain_flow(data_dir: str) -> None:
    raw = load_raw_data_task(data_dir)
    validated = validate_task(raw)
    features = build_features_task(validated)
    train_task(features)
    promote_model_task()
    print("Retrain pipeline completed")


@task(name="load-reference")
def load_reference_task(model_name: str = "pm-model") -> pd.DataFrame:
    """從 champion model 的 run artifact 讀回 reference.csv。"""
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:../mlruns"))
    client = MlflowClient()

    champion_mv = client.get_model_version_by_alias(name=model_name, alias="champion")
    run_id = champion_mv.run_id

    local_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="reference.csv")
    return pd.read_csv(local_path)


@flow(name="monitoring-pipeline", log_prints=True)
def monitoring_flow(data_dir: str, model_name: str = "pm-model", drift_threshold: int = 5) -> dict:
    """定期監控：讀 reference + 新資料 → drift check → 可能觸發 retrain。"""

    # 1. 從 champion run 讀回訓練時的特徵分佈
    reference = load_reference_task(model_name)

    # 2. 用新資料跑 M2→M3→M4 產出 current features
    raw = load_raw_data_task(data_dir)
    validated = validate_task(raw)
    feature_df = build_features_task(validated)

    # 3. 取出跟 reference 一樣的欄位（不含 datetime / machineID / label）
    columns = reference.columns.tolist()
    current = feature_df[columns]

    # 4. drift 偵測
    report = check_drift_task(reference, current, columns, drift_threshold)
    print(f"Drift report: {report}")

    # 5. 有漂移 → 觸發 retrain
    if report["drift_detected"]:
        print("Drift detected — triggering retrain")
        retrain_flow(data_dir)
    else:
        print("No significant drift — skipping retrain")

    return report
