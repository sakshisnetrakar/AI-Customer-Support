
from pathlib import Path
import sys

from sentence_transformers import SentenceTransformer

# Allow imports from the src directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from vector_store import VectorStore
from llm import generate_answer


# 1. Configuration
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 3
SIMILARITY_THRESHOLD = 0.40

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_retriever():
    """Load the embedding model and saved vector store."""

    model = SentenceTransformer(MODEL_NAME)
    store = VectorStore()

    store.load(
        str(PROJECT_ROOT / "index" / "vectors.npy"),
        str(PROJECT_ROOT / "index" / "metadata.json")
    )

    print(f"Loaded {len(store.vectors)} vectors.")

    return model, store


def retrieve_context(question, model, store):
    """Retrieve relevant document chunks for a question."""

    question_embedding = model.encode(question)

    results = store.search(
        question_embedding,
        top_k=TOP_K
    )

    relevant_results = [
        result
        for result in results
        if result["score"] >= SIMILARITY_THRESHOLD
    ]

    if not relevant_results:
        return None, []

    # Combine retrieved text into one context
    context = "\n\n".join(
        result["metadata"]["text"]
        for result in relevant_results
    )

    return context, relevant_results


def answer_question(question, model, store):
    """Retrieve context and generate an answer using Gemini."""

    context, results = retrieve_context(
        question, model, store
    )

    print("\nRetrieval Results")
    print("-----------------")

    if not results:
        return (
            "I couldn't find sufficiently relevant "
            "information in the TechCare knowledge base."
        )

    for i, result in enumerate(results, start=1):
        print(f"\nResult {i}")
        print(f"Similarity: {result['score']:.3f}")
        print(f"Source: {result['metadata']['source']}")

    # Pass retrieved context to Gemini
    return generate_answer(question, context)


if __name__ == "__main__":
    model, store = load_retriever()

    while True:
        question = input(
            "\nAsk your TechCare question (or type 'exit'): "
        ).strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if not question:
            print("Please enter a question.")
            continue

        try:
            answer = answer_question(question, model, store)
            print("\nTechCare AI Answer:")
            print("-------------------")
            print(answer)

        except Exception as error:
            print(f"\nRequest failed: {error}")
