import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class VectorStore:

    def __init__(self):
        self.vectors = []
        self.metadata = []

    def add(self, vector, metadata):

        self.vectors.append(vector)
        self.metadata.append(metadata)

    def search(self, query_vector, top_k=3):

        # If there are no vectors
        if not self.vectors:
            return []

        # Convert stored vectors into NumPy array
        vectors = np.array(self.vectors)

        # Calculate cosine similarity
        similarities = cosine_similarity(
            [query_vector],
            vectors
        )[0]

        # Get indices sorted from highest similarity
        ranked_indices = np.argsort(
            similarities
        )[::-1]

        # Store search results
        results = []

        # Get top K results
        for index in ranked_indices[:top_k]:

            results.append({
                "score": float(similarities[index]),
                "metadata": self.metadata[index]
            })

        # IMPORTANT: return the results
        return results