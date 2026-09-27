from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from document_loader import load_documents


# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Load all TechCare documents
documents = load_documents()


# Create embeddings for all documents
document_embeddings = {}

for filename, content in documents.items():
    embedding = model.encode(content)
    document_embeddings[filename] = embedding


# Ask the user a question
question = input("\nAsk your question: ")


# Convert the question into an embedding
question_embedding = model.encode(question)


# Calculate similarity between the question and every document
scores = {}

for filename, document_embedding in document_embeddings.items():

    score = cosine_similarity(
        [question_embedding],
        [document_embedding]
    )[0][0]

    scores[filename] = score


# Find the document with the highest similarity
best_document = max(
    scores,
    key=scores.get
)

best_score = scores[best_document]


# Display similarity scores
print("\nSemantic Similarity Scores:")

for filename, score in scores.items():
    print(f"{filename}: {score:.3f}")


# Check whether the similarity is high enough
if best_score < 0.30:

    print(
        "\nI couldn't find relevant information "
        "in the TechCare knowledge base."
    )

else:

    print("\nMost relevant document:")
    print(best_document)

    print("\nSimilarity score:")
    print(f"{best_score:.3f}")

    print("\nRelevant information:")
    print(documents[best_document])