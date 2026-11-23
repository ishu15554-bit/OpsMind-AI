
import mlflow
from pathlib import Path

# Project configuration
BASE_DIR = Path(__file__).resolve().parents[2]
TRACKING_DB = BASE_DIR / "mlops" / "mlflow.db"

mlflow.set_tracking_uri(
    f"sqlite:///{TRACKING_DB.as_posix()}"
)


# Registered model names
REGISTERED_MODELS = {
    "ticket": "OpsMind-Ticket-Classifier",
    "openstack": "OpsMind-OpenStack-Anomaly",
    "hdfs": "OpsMind-HDFS-Anomaly"
}


def load_registered_model(model_key, version="1"):
    """Load a specific registered MLflow model version."""

    if model_key not in REGISTERED_MODELS:
        raise ValueError(f"Unknown model key: {model_key}")

    model_name = REGISTERED_MODELS[model_key]

    model_uri = (
        f"models:/{model_name}/{version}"
    )

    model = mlflow.pyfunc.load_model(model_uri)

    print(f"Loaded: {model_name}, Version: {version}")

    return model


if __name__ == "__main__":

    for key in REGISTERED_MODELS:

        try:
            load_registered_model(key)

        except Exception as error:
            print(f"Failed to load {key}: {error}")
