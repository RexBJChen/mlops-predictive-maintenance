"""延伸專案一：用 XGBoost 訓練 challenger 模型，註冊到 Docker MLflow。"""

import os

# Step 1: 指向 Docker MLflow（必須在 import p6 模組之前設定）
MLFLOW_URL = "http://localhost:5000"
os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_URL

from pathlib import Path
from xgboost import XGBClassifier
from p6.ingestion.ingest import load_raw_data
from p6.validation.validate import validate
from p6.feature_pipeline.features import build_features
from p6.training.train import train
from p6.registry.promote import promote_model

# Step 2: 建立 feature matrix（M2 → M3 → M4）
raw = load_raw_data(Path("data/raw"))
validated = validate(raw)
features = build_features(validated)

# Step 3: 用 XGBoost 訓練（M5）
train(
    feature_df=features,
    model_class=XGBClassifier,
    model_params={"n_estimators": 200, "max_depth": 6, "random_state": 42},
)

# Step 4: 把最佳模型升 champion（M6）
mv = promote_model()
print(f"Champion: pm-model v{mv.version}")
