# mlflow設定、pipeline、前處理、模型

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, classification_report
import mlflow

NUMERIC_FEATURES = [
    "volt", "rotate", "pressure", "vibration", "age",
    "volt_mean_3h", "volt_std_3h", "volt_mean_24h", "volt_std_24h",
    "rotate_mean_3h", "rotate_std_3h", "rotate_mean_24h", "rotate_std_24h",
    "pressure_mean_3h", "pressure_std_3h", "pressure_mean_24h", "pressure_std_24h",
    "vibration_mean_3h", "vibration_std_3h", "vibration_mean_24h", "vibration_std_24h",
    "errorID_error1", "errorID_error2", "errorID_error3", "errorID_error4", "errorID_error5",
    "comp_comp1", "comp_comp2", "comp_comp3", "comp_comp4",
]

CATEGORICAL_FEATURES = ["model"]

DROP_COLS = ['datetime' , 'machineID', 'failuretime', 'label']

## mlflow 設定好

def train(
        feature_df: pd.DataFrame,
        MLflow_name : str ="predictive-maintenance",
        model_params : dict | None = None
        ) -> None :
    
    ## 準備資料集
    day_split = feature_df["datetime"].max() - pd.Timedelta(days=30)

    df = feature_df.copy()

    FEATURE_COLS = [c for c in df if c not in DROP_COLS] 

    train_data = df[df["datetime"] <= day_split]
    test_data = df[df["datetime"] > day_split]

    X_train = train_data[FEATURE_COLS]
    y_train = train_data["label"]

    X_test = test_data[FEATURE_COLS]
    y_test = test_data["label"]
    


    ## 準備 preprocess, model

    preprocesser = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), CATEGORICAL_FEATURES)
        ], remainder="drop"
    )

    if model_params is None:
      model_params = {"n_estimators": 100, "random_state": 42}

    model = RandomForestClassifier(**model_params)

    ## 啟動 mlflow 內部建 pipeline
    mlflow.set_tracking_uri("file:../mlruns")
    mlflow.set_experiment(experiment_name=MLflow_name)
    
    with mlflow.start_run():
        
        ## 建構pipe 
        pipe = Pipeline(steps=[
            ("preprocesser", preprocesser),
            ("classifier", model)
        ])

        ## 訓練
        pipe.fit(X_train, y_train)
        
        ## 預測
        y_pred = pipe.predict(X_test)
        
        ## 評估
        f1 = f1_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)


        mlflow.log_params(model_params)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("recall", recall)

        mlflow.sklearn.log_model(
            pipe,
            name="model",
            input_example= X_test.head(3)
        )
    
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print(f"Metrics: f1 is {f1}, recall is {recall}")
        print(f"\n{classification_report(y_test, y_pred)}")