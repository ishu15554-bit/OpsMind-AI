
import requests

# FastAPI RAG endpoint
API_URL = "http://127.0.0.1:8000/rag/ask"

# Test questions and expected source documents
test_cases = [
    {
        "question": "How can I troubleshoot a password reset issue?",
        "expected": "password_reset.md"
    },
    {
        "question": "How can I fix network connectivity problems?",
        "expected": "network_connectivity.md"
    },
    {
        "question": "Why am I unable to send or receive emails?",
        "expected": "email_service.md"
    },
    {
        "question": "How can I troubleshoot a printer hardware issue?",
        "expected": None
    }
]

print("\n========== OPSMIND AI RAG EVALUATION ==========\n")

for i, test in enumerate(test_cases, start=1):
    print(f"\nTEST {i}: {test['question']}")

    try:
        response = requests.post(
            API_URL,
            json={"log_text": test["question"]},
            timeout=120
        )

        response.raise_for_status()
        result = response.json()

        answer = result.get("answer", "")
        sources = result.get("sources", [])

        print("Answer:", answer[:300])
        print("Sources:", sources)

        if test["expected"] is None:
            print("CHECK: Review whether unsupported question is safely rejected.")

        elif any(test["expected"] in str(source) for source in sources):
            print("RESULT: PASS - Expected source found.")

        else:
            print("RESULT: CHECK - Expected source not found.")

    except requests.RequestException as error:
        print("API ERROR:", error)

print("\n========== EVALUATION COMPLETE ==========")
