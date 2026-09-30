from sentence_transformers import SentenceTransformer

from vector_store import VectorStore


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Create vector store
store = VectorStore()


# Load previously indexed data
store.load(
    "index/vectors.npy",
    "index/metadata.json"
)


print(
    f"Loaded {len(store.vectors)} vectors."
)


# Ask the user
question = input("\nAsk your question: ")


# Convert question into embedding
question_embedding = model.encode(question)


# Search vector store
results = store.search(
    question_embedding,
    top_k=3
)


# Display results
print("\nTop Results:")
print("-----------")


for i, result in enumerate(results):

    print(f"\nResult {i + 1}")

    print(
        f"Similarity: "
        f"{result['score']:.3f}"
    )

    print(
        f"Source: "
        f"{result['metadata']['source']}"
    )

    print(
        f"Text:\n"
        f"{result['metadata']['text']}"
    )