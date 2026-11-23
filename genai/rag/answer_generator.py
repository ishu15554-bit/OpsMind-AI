
import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

from genai.rag.retrieve import retrieve_documents


# Load environment variables from project root
BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")


# Initialize Groq client
api_key = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv(
    "GROQ_MODEL",
    "llama-3.3-70b-versatile"
)

if not api_key:
    raise ValueError("GROQ_API_KEY missing in .env file")

client = Groq(api_key=api_key)


def generate_answer(question):
    """
    Retrieve relevant documents and generate
    a grounded troubleshooting answer using Groq.
    """

    # Retrieve relevant knowledge-base chunks
    results = retrieve_documents(question, top_k=3)

    if not results:
        return {
            "answer": "No relevant troubleshooting information found.",
            "sources": []
        }

    # Prepare context from retrieved documents
    context_parts = []

    for result in results:
        context_parts.append(
            f"Source: {result['source']}\n"
            f"Content:\n{result['text']}"
        )

    context = "\n\n".join(context_parts)

    # Instructions for the LLM
    system_prompt = """
You are OpsMind AI, an enterprise IT troubleshooting assistant.

Rules:
1. Answer using only the supplied knowledge-base context.
2. Never invent troubleshooting steps or technical facts.
3. Give clear, practical, numbered steps.
4. If the context does not contain enough information,
   clearly say that the information is insufficient.
5. Never ask users to share passwords or secret credentials.
6. Never recommend destructive actions without approval.
7. Be concise and professional.
"""

    user_prompt = f"""
User Question:
{question}

Retrieved Knowledge Base Context:
{context}

Generate a troubleshooting answer based only on this context.
"""

    # Generate response using Groq
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
        max_tokens=700
    )

    answer = response.choices[0].message.content

    sources = list(set(
        result["source"] for result in results
    ))

    return {
        "answer": answer,
        "sources": sources,
        "model": MODEL_NAME
    }


if __name__ == "__main__":

    question = "How can I troubleshoot a password reset issue?"

    result = generate_answer(question)

    print("\nUSER QUESTION:")
    print(question)

    print("\nOPSMIND AI RESPONSE:")
    print(result["answer"])

    print("\nSOURCE DOCUMENTS:")

    for source in result["sources"]:
        print("-", source)

    print("\nMODEL:", result["model"])
