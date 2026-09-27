import math

from document_loader import load_documents
from text_processor import tokenize


def calculate_tf(words):
    word_count = len(words)

    frequencies = {}

    for word in words:
        frequencies[word] = frequencies.get(word, 0) + 1

    tf = {}

    for word, count in frequencies.items():
        tf[word] = count / word_count

    return tf


def calculate_idf(documents):
    total_documents = len(documents)

    document_frequency = {}

    for content in documents.values():
        words = set(tokenize(content))

        for word in words:
            document_frequency[word] = (
                document_frequency.get(word, 0) + 1
            )

    idf = {}

    for word, count in document_frequency.items():
        idf[word] = math.log(total_documents / count)

    return idf


def calculate_tfidf(words, idf):
    tf = calculate_tf(words)

    tfidf = {}

    for word, tf_value in tf.items():
        tfidf[word] = tf_value * idf.get(word, 0)

    return tfidf


documents = load_documents()

idf = calculate_idf(documents)

for filename, content in documents.items():

    words = tokenize(content)

    tfidf = calculate_tfidf(words, idf)

    print(f"\n--- {filename} ---")

    for word, score in tfidf.items():
        print(f"{word}: {score:.3f}")