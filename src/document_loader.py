import os

DATA_FOLDER = "data"

def load_documents():
    documents = {}

    for filename in os.listdir(DATA_FOLDER):
        if filename.endswith(".txt"):
            file_path = os.path.join(DATA_FOLDER, filename)

            with open(file_path, "r", encoding="utf-8") as file:
                documents[filename] = file.read()

    return documents


documents = load_documents()

for filename, content in documents.items():
    print(f"\n--- {filename} ---")
    print(content)