# Import required library
import joblib
from pathlib import Path


# Get project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Define model directory
MODEL_DIR = PROJECT_ROOT / "data" / "processed"


# Generic model loading function
def load_model(model_name):

    model_path = MODEL_DIR / model_name

    # Check if model exists
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    # Load model
    model = joblib.load(model_path)

    print(f"Model loaded successfully: {model_name}")

    return model


# Load ticket classification model
def load_ticket_classifier():
    return load_model("best_ticket_classifier.pkl")


# Load HDFS anomaly detection model
def load_hdfs_model():
    return load_model("best_hdfs_anomaly_model.pkl")


# Load OpenStack anomaly detection model
def load_openstack_model():
    return load_model("openstack_validated_model.pkl")