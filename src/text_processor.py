import string


def tokenize(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    words = text.split()
    return words

text = "How long can I return my product?"
tokens = tokenize(text)
print(tokens)