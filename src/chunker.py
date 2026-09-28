def chunk_text(text, chunk_size=100, overlap=20):
    words = text.split()

    chunks = []

    start = 0

    while start < len(words):
        end = start + chunk_size

        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


if __name__ == "__main__":

    text = """
    TechCare products come with a 1-year warranty.
    The warranty covers manufacturing defects.
    The warranty does not cover damage caused by accidents,
    misuse, or unauthorized repairs.
    Customers should provide the original order ID
    when requesting warranty service.
    """

    chunks = chunk_text(
        text,
        chunk_size=10,
        overlap=3
    )

    for i, chunk in enumerate(chunks):

        print(f"\nChunk {i + 1}:")
        print(chunk)