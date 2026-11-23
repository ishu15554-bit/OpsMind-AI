
# Import text preprocessing functions
from src.preprocessing.text_preprocessor import clean_text
from src.feature_engineering.text_features import extract_text_features

# Import trained model loader functions
from src.ml.model_loader import (
    load_ticket_classifier,
    load_hdfs_model,
    load_openstack_model
)


# ==========================================
# 1. ITSM TICKET CLASSIFICATION
# ==========================================

def predict_ticket(ticket_text: str) -> str:

    # Validate input
    if not isinstance(ticket_text, str) or not ticket_text.strip():
        raise ValueError("Ticket text cannot be empty.")

    # Clean input text
    cleaned_text = clean_text(ticket_text)

    # Extract additional text features
    features = extract_text_features(cleaned_text)

    # Load trained classifier
    model = load_ticket_classifier()

    # Generate prediction
    prediction = model.predict([cleaned_text])[0]

    # Convert numeric class into readable output
    result = f"Category {int(prediction)}"

    # Print extracted features and prediction
    print("Extracted Features:", features)
    print("Ticket Classification:", result)

    return result


# ==========================================
# 2. HDFS LOG ANOMALY DETECTION
# ==========================================

def predict_hdfs_anomaly(log_sequence: str) -> str:

    # Validate input
    if not isinstance(log_sequence, str) or not log_sequence.strip():
        raise ValueError("HDFS log sequence cannot be empty.")

    # Load trained HDFS model package
    package = load_hdfs_model()

    model = package["model"]
    vectorizer = package["vectorizer"]

    # Convert event sequence into numerical features
    features = vectorizer.transform([log_sequence])

    # Generate prediction
    prediction = model.predict(features)[0]

    # Convert prediction into readable output
    result = "Anomaly" if prediction == 1 else "Normal"

    return result


# ==========================================
# 3. OPENSTACK LOG ANOMALY DETECTION
# ==========================================

def predict_openstack_anomaly(log_message: str) -> str:

    # Validate input
    if not isinstance(log_message, str) or not log_message.strip():
        raise ValueError("OpenStack log message cannot be empty.")

    # Clean input log message
    cleaned_log = clean_text(log_message)

    # Load trained OpenStack model package
    package = load_openstack_model()

    model = package["model"]
    vectorizer = package["vectorizer"]

    # Convert log message into numerical features
    features = vectorizer.transform([cleaned_log])

    # Generate prediction
    prediction = model.predict(features)[0]

    # Convert prediction into readable output
    result = "Abnormal" if prediction == 1 else "Normal"

    return result


# ==========================================
# 4. TEST ALL PREDICTION FUNCTIONS
# ==========================================

if __name__ == "__main__":

    print("\n========== OPSMIND AI ==========")

    # Test ticket classification
    ticket_result = predict_ticket(
        "Unable to reset my password"
    )
    print("\nTicket Prediction:", ticket_result)

    # Test HDFS anomaly detection
    hdfs_result = predict_hdfs_anomaly(
        "E5 E5 E5 E5"
    )
    print("HDFS Prediction:", hdfs_result)

    # Test OpenStack anomaly detection
    openstack_result = predict_openstack_anomaly(
        "ERROR: Service connection failed"
    )
    print("OpenStack Prediction:", openstack_result)

    print("\nPrediction Testing Completed!")
