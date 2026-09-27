import math

from document_loader import load_documents
from text_processor import tokenize
from tfidf import calculate_idf, calculate_tfidf


def cosine_similarity(vector_a, vector_b):

    dot_product = 0

    magnitude_a = 0
    magnitude_b = 0

    # Calculate dot product
    for word, value_a in vector_a.items():
        value_b = vector_b.get(word, 0)

        dot_product += value_a * value_b

    # Calculate magnitude of vector A
    for value in vector_a.values():
        magnitude_a += value ** 2

    # Calculate magnitude of vector B
    for value in vector_b.values():
        magnitude_b += value ** 2

    magnitude_a = math.sqrt(magnitude_a)
    magnitude_b = math.sqrt(magnitude_b)

    # Avoid division by zero
    if magnitude_a == 0 or magnitude_b == 0:
        return 0

    return dot_product / (magnitude_a * magnitude_b)


# Load documents
documents = load_documents()

# Calculate IDF
idf = calculate_idf(documents)

# Create TF-IDF vectors for documents
document_vectors = {}

for filename, content in documents.items():

    words = tokenize(content)

    vector = calculate_tfidf(words, idf)

    document_vectors[filename] = vector


# Ask user a question
question = input("\nAsk your question: ")

question_words = tokenize(question)

question_vector = calculate_tfidf(question_words, idf)


# Compare question with every document
scores = {}

for filename, document_vector in document_vectors.items():

    score = cosine_similarity(
        question_vector,
        document_vector
    )

    scores[filename] = score


# Find most relevant document
best_document = max(
    scores,
    key=scores.get
)


# Display results
print("\nSimilarity Scores:")

for filename, score in scores.items():
    print(f"{filename}: {score:.3f}")


print("\nMost relevant document:")
print(best_document)