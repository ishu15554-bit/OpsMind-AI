
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

BASE_DIR = Path(__file__).resolve().parents[1]
TRACKING_DB = BASE_DIR / "mlops" / "mlflow.db"

mlflow.set_tracking_uri(
    f"sqlite:///{TRACKING_DB.as_posix()}"
)

client = MlflowClient()

models = [
    "OpsMind-Ticket-Classifier",
    "OpsMind-OpenStack-Anomaly",
    "OpsMind-HDFS-Anomaly"
]

print("\n========== OPSMIND MODEL REGISTRY ==========\n")

for model_name in models:

    try:
        versions = client.search_model_versions(
            f"name='{model_name}'"
        )

        print(f"\nModel: {model_name}")

        if not versions:
            print("No versions found.")
            continue

        for version in versions:
            print(f"Version: {version.version}")
            print(f"Status: {version.status}")
            print(f"Run ID: {version.run_id}")
            print(f"Source: {version.source}")

    except Exception as error:
        print(f"Error: {error}")

print("\n========== VERIFICATION COMPLETE ==========")
