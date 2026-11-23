# Import required libraries
import re


def extract_text_features(text):
    """
    Extract basic numerical features from incident text.
    """

    # Handle missing text
    if text is None:
        text = ""

    text = str(text)

    # Extract text features
    features = {
        "character_count": len(text),
        "word_count": len(text.split()),
        "uppercase_count": sum(1 for c in text if c.isupper()),
        "digit_count": sum(1 for c in text if c.isdigit()),
        "special_character_count": len(
            re.findall(r"[^a-zA-Z0-9\s]", text)
        ),
        "contains_error": int(
            "error" in text.lower()
        ),
        "contains_failure": int(
            "fail" in text.lower()
        )
    }

    return features


# Test feature extraction
if __name__ == "__main__":

    sample = "ERROR: Server connection failed 503!"

    result = extract_text_features(sample)

    print("Extracted Text Features:")

    for feature, value in result.items():
        print(f"{feature}: {value}")