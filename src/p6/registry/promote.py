import mlflow
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion  

def promote_model(
      experiment_name: str = "predictive-maintenance",
      model_name: str = "pm-model",
      metric: str = "f1",
  ) -> ModelVersion:
    
    mlflow.set_tracking_uri("file:../mlruns")
    client = MlflowClient()


    experiment = client.get_experiment_by_name(experiment_name)

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"]
    )
    
    bestrun = runs[0]

    model_id = bestrun.outputs.model_outputs[0].model_id
    model_uri = f"models:/{model_id}"

    mv = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
    )

    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=mv.version
    )

    return mv