from prefect import flow, task

from p6.ingestion import load_raw_data
from p6.validation import validate
from p6.feature_pipeline import build_features
from p6.training import train
from p6.registry import promote_model

load_raw_data_task = task(load_raw_data, name="load-raw-data", retries=2, retry_delay_seconds=5)
validate_task = task(validate, name="validate")
build_features_task = task(build_features, name="build-features")
train_task = task(train, name="train")
promote_model_task = task(promote_model, name="promote-model")


@flow(name="retrain-pipeline", log_prints=True)
def retrain_flow(data_dir: str) -> None:
    raw = load_raw_data_task(data_dir)
    validated = validate_task(raw)
    features = build_features_task(validated)
    train_task(features)
    promote_model_task()
    print("Retrain pipeline completed")
