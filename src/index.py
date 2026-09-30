from sentence_transformers import SentenceTransformer

from document_loader import load_documents
from chunker import chunk_text
from vector_store import VectorStore


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Load documents
documents = load_documents()


# Create vector store
store = VectorStore()


# Process documents
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


# Save vector store
store.save(
    "index/vectors.npy",
    "index/metadata.json"
)


print(
    f"Successfully indexed "
    f"{len(store.vectors)} chunks."
)