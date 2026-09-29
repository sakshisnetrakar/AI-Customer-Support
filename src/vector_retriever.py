from sentence_transformers import SentenceTransformer

from document_loader import load_documents
from chunker import chunk_text
from vector_store import VectorStore


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Load TechCare documents
documents = load_documents()


# Create vector store
store = VectorStore()


# Process every document
for filename, content in documents.items():

    chunks = chunk_text(
        content,
        chunk_size=100,
        overlap=20
    )

    for chunk in chunks:

        embedding = model.encode(chunk)

        metadata = {
            "source": filename,
            "text": chunk
        }

        store.add(
            embedding,
            metadata
        )


print(f"Loaded {len(store.vectors)} vectors.")


# Ask user
question = input("\nAsk your question: ")


# Convert question to embedding
question_embedding = model.encode(question)


# Retrieve top 3 chunks
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