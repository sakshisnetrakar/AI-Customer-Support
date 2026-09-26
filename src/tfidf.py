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


documents = load_documents()

for filename, content in documents.items():
    words = tokenize(content)

    tf = calculate_tf(words)

    print(f"\n{filename}")
    print(tf)