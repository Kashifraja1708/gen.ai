!pip install gensim
from gensim.models import KeyedVectors
import numpy as np

# Load vectors
print("Loading pre-trained word vectors...")
wv = KeyedVectors.load_word2vec_format("/kaggle/input/google-word2vec/GoogleNews-vectors-negative300.bin", binary=True)
print("word2vec loaded successfully!")

# Functions
def explore(w1, w2, w3):
    try:
        vec = wv[w1] - wv[w2] + wv[w3]
        res = [(w, s) for w, s in wv.similar_by_vector(vec, topn=10) if w not in {w1, w2, w3}]
        print(f"\nWord Relationship: {w1} - {w2} + {w3}")
        print("Most similar words to the result (excluding input words):")
        for w, s in res[:5]: print(f"{w}: {s:.4f}")
    except KeyError as e:
        print(f"Error: {e} not found in the vocabulary.")

def sim(w1, w2):
    try: print(f"\nSimilarity between '{w1}' and '{w2}': {wv.similarity(w1, w2):.4f}")
    except KeyError as e: print(f"Error: {e} not found in the vocabulary.")

def similar(w):
    try:
        print(f"\nMost similar words to '{w}':")
        for w2, s in wv.most_similar(w, topn=5): print(f"{w2}: {s:.4f}")
    except KeyError as e:
        print(f"Error: {e} not found in the vocabulary.")

# Examples
explore("paris", "france", "germany")
explore("apple", "fruit", "carrot")
sim("cat", "dog")
sim("computer", "keyboard")
sim("music", "art")
similar("happy")
similar("sad")
similar("technology")

