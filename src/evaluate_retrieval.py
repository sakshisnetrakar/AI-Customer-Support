
from sentence_transformers import SentenceTransformer

from vector_store import VectorStore


# Configuration
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 3


# Test dataset:
# Each question has an expected source document.
test_cases = [
    {
        "question": "How long does shipping take?",
        "expected_source": "shipping_policy.txt"
    },
    {
        "question": "How can I return an item?",
        "expected_source": "return_policy.txt"
    },
    {
        "question": "Does the warranty cover accidental damage?",
        "expected_source": "warranty.txt"
    },
    {
        "question": "What payment methods are supported?",
        "expected_source": "payment.txt"
    }
]


# Load embedding model
model = SentenceTransformer(MODEL_NAME)


# Load the existing vector store
store = VectorStore()

store.load(
    "index/vectors.npy",
    "index/metadata.json"
)


print(f"Loaded {len(store.vectors)} vectors.")


# Evaluation counters
hits = 0
total = len(test_cases)


# Evaluate each question
for case in test_cases:

    question = case["question"]
    expected_source = case["expected_source"]

    # Convert question into an embedding
    question_embedding = model.encode(question)

    # Retrieve the top-K results
    results = store.search(
        question_embedding,
        top_k=TOP_K
    )

    # Extract retrieved source filenames
    retrieved_sources = [
        result["metadata"]["source"]
        for result in results
    ]

    # Check whether the expected source was retrieved
    hit = expected_source in retrieved_sources

    if hit:
        hits += 1

    print("\nQuestion:", question)
    print("Expected:", expected_source)
    print("Retrieved:", retrieved_sources)
    print("Hit:", hit)


# Calculate Hit Rate@K
hit_rate = hits / total if total else 0

print("\n----------------------------")
print(f"Hit Rate@{TOP_K}: {hit_rate:.2%}")
print(f"Successful queries: {hits}/{total}")
print("----------------------------")
