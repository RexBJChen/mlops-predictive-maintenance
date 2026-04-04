"""One-time script: train a small model and register it in Docker MLflow."""

import mlflow
from mlflow import MlflowClient
from pathlib import Path
from p6.ingestion.ingest import load_raw_data
from p6.validation.validate import validate
from p6.feature_pipeline.features import build_features
from p6.training.train import NUMERIC_FEATURES, CATEGORICAL_FEATURES, DROP_COLS

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score

# Point to Docker MLflow (override hardcoded URI)
TRACKING_URI = "http://localhost:5000"
mlflow.set_tracking_uri(TRACKING_URI)

# Build features
raw = load_raw_data(Path("data/raw"))
validated = validate(raw)
features = build_features(validated)

# Sample to speed up training
features_sampled = features.sample(frac=0.1, random_state=42)
print(f"Training on {len(features_sampled)} rows (10% sample)")

# Train/test split
day_split = features_sampled["datetime"].max() - pd.Timedelta(days=30)
FEATURE_COLS = [c for c in features_sampled if c not in DROP_COLS]

train_data = features_sampled[features_sampled["datetime"] <= day_split]
test_data = features_sampled[features_sampled["datetime"] > day_split]

X_train, y_train = train_data[FEATURE_COLS], train_data["label"]
X_test, y_test = test_data[FEATURE_COLS], test_data["label"]

# Build pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(drop="first", sparse_output=False), CATEGORICAL_FEATURES),
    ],
    remainder="drop",
)
model = RandomForestClassifier(n_estimators=50, random_state=42)

# MLflow run
mlflow.set_experiment("predictive-maintenance")

with mlflow.start_run():
    pipe = Pipeline([("preprocessor", preprocessor), ("classifier", model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    mlflow.log_params({"n_estimators": 50, "random_state": 42})
    mlflow.log_metric("f1", f1)
    mlflow.log_metric("recall", recall)
    mlflow.sklearn.log_model(pipe, name="model", input_example=X_test.head(3))

    print(f"f1={f1:.4f}, recall={recall:.4f}")

# Promote
client = MlflowClient(TRACKING_URI)
experiment = client.get_experiment_by_name("predictive-maintenance")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.f1 DESC"],
)
best_run = runs[0]
model_id = best_run.outputs.model_outputs[0].model_id
mv = mlflow.register_model(f"models:/{model_id}", "pm-model")
client.set_registered_model_alias("pm-model", "champion", mv.version)

print(f"Model registered: pm-model v{mv.version} (champion)")
