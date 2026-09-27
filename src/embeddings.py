from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


model = SentenceTransformer("all-MiniLM-L6-v2")


sentence_a = "How can I return my product?"

sentence_b = "What payment methods does TechCare accept?"


embedding_a = model.encode(sentence_a)

embedding_b = model.encode(sentence_b)


print("Sentence A embedding shape:", embedding_a.shape)

print("Sentence B embedding shape:", embedding_b.shape)


similarity = cosine_similarity(
    [embedding_a],
    [embedding_b]
)


print("Similarity:", similarity[0][0])