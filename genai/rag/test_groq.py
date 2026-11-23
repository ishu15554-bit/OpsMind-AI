
import os
from dotenv import load_dotenv
from groq import Groq

# Load API key from root .env file
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY not found. Check your root .env file."
    )

# Initialize Groq client
client = Groq(api_key=api_key)

try:
    # Fetch models available to this account
    models = client.models.list()

    print("Groq API connected successfully!")
    print("\nAvailable Llama models:")

    llama_models = [
        model.id for model in models.data
        if "llama" in model.id.lower()
    ]

    for model in llama_models:
        print("-", model)

except Exception as error:
    print("Groq API connection failed.")
    print("Error:", error)
