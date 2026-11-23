
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn


# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "data" / "processed"

TRACKING_DB = BASE_DIR / "mlops" / "mlflow.db"
TRACKING_DB.parent.mkdir(parents=True, exist_ok=True)


# ==========================================
# MLFLOW CONFIGURATION
# ==========================================

mlflow.set_tracking_uri(
    f"sqlite:///{TRACKING_DB.as_posix()}"
)

mlflow.set_experiment("OpsMind-AI-Model-Tracking")


# ==========================================
# MODEL TRACKING FUNCTION
# ==========================================

def track_model(model_file, model_name):
    """Track an existing model and its complete package."""

    model_path = MODEL_DIR / model_file

    if not model_path.exists():
        print(f"Model not found: {model_file}")
        return False

    try:
        print(f"\nProcessing: {model_name}")

        # Load existing trained model package
        package = joblib.load(model_path)

        # Extract estimator from saved package
        if isinstance(package, dict):

            model = package.get("model")

            if model is None:
                print(f"Estimator not found: {model_file}")
                return False

            model_type = package.get(
                "model_name",
                type(model).__name__
            )

        else:
            model = package
            model_type = type(model).__name__

        # Start MLflow run
        with mlflow.start_run(run_name=model_name):

            # Log model parameters
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("model_file", model_file)
            mlflow.log_param("model_type", model_type)

            # Project information
            mlflow.set_tag("project", "OpsMind AI")
            mlflow.set_tag("module", "Model Tracking")
            mlflow.set_tag(
                "tracking_type",
                "existing_model_logging"
            )

            # Log threshold if available
            if isinstance(package, dict):
                if "threshold" in package:
                    mlflow.log_param(
                        "decision_threshold",
                        float(package["threshold"])
                    )

            # Log sklearn estimator
            # Pickle is used for these trusted, locally trained models.
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                serialization_format="pickle"
            )

            # Log complete original package
            # Preserves vectorizer, threshold and other metadata
            mlflow.log_artifact(
                str(model_path),
                artifact_path="complete_model_package"
            )

            print(f"Successfully tracked: {model_name}")
            print(f"Model type: {model_type}")

        return True

    except Exception as error:
        print(f"Error tracking {model_name}: {error}")
        return False


# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":

    models = [
        (
            "best_ticket_classifier.pkl",
            "Ticket Classifier"
        ),
        (
            "best_log_anomaly_model.pkl",
            "OpenStack Anomaly Detection"
        ),
        (
            "best_hdfs_anomaly_model.pkl",
            "HDFS Anomaly Detection"
        )
    ]

    successful = 0
    failed = 0

    print("\n========== OPSMIND AI MODEL TRACKING ==========\n")

    for model_file, model_name in models:

        result = track_model(model_file, model_name)

        if result:
            successful += 1
        else:
            failed += 1

    print("\n========== TRACKING SUMMARY ==========")
    print(f"Successful models: {successful}")
    print(f"Failed models: {failed}")
    print("\nMLflow tracking completed.")
