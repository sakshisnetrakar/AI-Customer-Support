from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from document_loader import load_documents
from chunker import chunk_text


# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Load TechCare documents
documents = load_documents()


# Create chunks
chunks = []

for filename, content in documents.items():

    document_chunks = chunk_text(
        content,
        chunk_size=100,
        overlap=20
    )

    for chunk in document_chunks:

        chunks.append({
            "source": filename,
            "text": chunk
        })


print(f"Total chunks created: {len(chunks)}")


# Create embeddings for every chunk
chunk_embeddings = []

for chunk in chunks:

    embedding = model.encode(chunk["text"])

    chunk_embeddings.append(embedding)


# Ask the user a question
question = input("\nAsk your question: ")


# Convert question into an embedding
question_embedding = model.encode(question)


# Calculate similarity with every chunk
similarities = []

for index, chunk_embedding in enumerate(chunk_embeddings):

    score = cosine_similarity(
        [question_embedding],
        [chunk_embedding]
    )[0][0]

    similarities.append({
        "index": index,
        "score": score
    })


# Sort by similarity score
similarities.sort(
    key=lambda item: item["score"],
    reverse=True
)


# Get the most relevant chunk
best_match = similarities[0]

best_chunk = chunks[best_match["index"]]


print("\nBest Match")
print("----------")

print(f"Source: {best_chunk['source']}")
print(f"Similarity: {best_match['score']:.3f}")

print("\nRelevant Information:")
print(best_chunk["text"])