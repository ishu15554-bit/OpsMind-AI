
import mlflow
from mlflow.tracking import MlflowClient


# ==========================================
# PROJECT CONFIGURATION
# ==========================================

BASE_DIR = __import__("pathlib").Path(__file__).resolve().parents[1]

TRACKING_DB = BASE_DIR / "mlops" / "mlflow.db"

mlflow.set_tracking_uri(
    f"sqlite:///{TRACKING_DB.as_posix()}"
)

EXPERIMENT_NAME = "OpsMind-AI-Model-Tracking"

client = MlflowClient()


# ==========================================
# MODELS TO REGISTER
# ==========================================

MODELS = {
    "Ticket Classifier": "OpsMind-Ticket-Classifier",
    "OpenStack Anomaly Detection": "OpsMind-OpenStack-Anomaly",
    "HDFS Anomaly Detection": "OpsMind-HDFS-Anomaly"
}


# ==========================================
# REGISTRATION FUNCTION
# ==========================================

def register_model(run_name, registered_name):

    experiment = client.get_experiment_by_name(
        EXPERIMENT_NAME
    )

    if experiment is None:
        print("Experiment not found.")
        return False

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["attributes.start_time DESC"],
        max_results=1
    )

    if not runs:
        print(f"No MLflow run found: {run_name}")
        return False

    run_id = runs[0].info.run_id
    model_uri = f"runs:/{run_id}/model"

    try:
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=registered_name
        )

        print(f"Registered: {registered_name}")
        print(f"Version: {registered_model.version}")

        return True

    except Exception as error:
        print(f"Registration failed for {registered_name}: {error}")
        return False


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    successful = 0
    failed = 0

    print("\n========== OPSMIND AI MODEL REGISTRY ==========\n")

    for run_name, registered_name in MODELS.items():

        if register_model(run_name, registered_name):
            successful += 1
        else:
            failed += 1

    print("\n========== REGISTRY SUMMARY ==========")
    print(f"Registered: {successful}")
    print(f"Failed: {failed}")
    print("\nModel registration completed.")
