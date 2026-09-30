import json
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

        if not self.vectors:
            return []

        vectors = np.array(self.vectors)

        similarities = cosine_similarity(
            [query_vector],
            vectors
        )[0]

        ranked_indices = np.argsort(
            similarities
        )[::-1]

        results = []

        for index in ranked_indices[:top_k]:

            results.append({
                "score": float(similarities[index]),
                "metadata": self.metadata[index]
            })

        return results

    def save(self, vector_file, metadata_file):

        vectors = np.array(self.vectors)

        np.save(vector_file, vectors)

        with open(
            metadata_file,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                self.metadata,
                file,
                indent=4
            )

    def load(self, vector_file, metadata_file):

        vectors = np.load(vector_file)

        with open(
            metadata_file,
            "r",
            encoding="utf-8"
        ) as file:

            metadata = json.load(file)

        self.vectors = list(vectors)
        self.metadata = metadata