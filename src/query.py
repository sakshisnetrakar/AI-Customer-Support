
from sentence_transformers import SentenceTransformer

from vector_store import VectorStore


# 1. Configuration
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 3
SIMILARITY_THRESHOLD = 0.40


# 2. Load the embedding model
model = SentenceTransformer(MODEL_NAME)


# 3. Load the saved vector store
store = VectorStore()

store.load(
    "index/vectors.npy",
    "index/metadata.json"
)

print(f"Loaded {len(store.vectors)} vectors.")


# 4. Ask the user a question
question = input("\nAsk your question: ")


# 5. Convert the question into an embedding
question_embedding = model.encode(question)


# 6. Retrieve the top-K results
results = store.search(
    question_embedding,
    top_k=TOP_K
)


# 7. Filter results using the threshold
relevant_results = [
    result
    for result in results
    if result["score"] >= SIMILARITY_THRESHOLD
]


# 8. Display the results
print("\nRetrieval Results")
print("-----------------")

if not relevant_results:
    print(
        "I couldn't find sufficiently relevant "
        "information in the TechCare knowledge base."
    )

else:
    for i, result in enumerate(relevant_results, start=1):

        print(f"\nResult {i}")
        print(f"Similarity: {result['score']:.3f}")
        print(f"Source: {result['metadata']['source']}")
        print(f"Text: {result['metadata']['text']}")
