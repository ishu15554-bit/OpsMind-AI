# Import required libraries
import re


def clean_text(text):
    """
    Clean and normalize incident or log text.
    """

    # Handle missing input
    if text is None:
        return ""

    # Convert text to lowercase
    text = str(text).lower()

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove leading and trailing spaces
    text = text.strip()

    return text


def preprocess_text_list(text_list):
    """
    Preprocess a list of text records.
    """

    return [clean_text(text) for text in text_list]


# Test preprocessing
if __name__ == "__main__":

    sample_text = "  ERROR: Server Connection FAILED   "

    print("Original Text:", sample_text)
    print("Cleaned Text:", clean_text(sample_text))