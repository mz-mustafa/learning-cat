import gensim
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity


# Load the Universal Sentence Encoder
use_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


def compute_similarity(phrase1, phrase2):
    """
    Compute the cosine similarity between embeddings of two phrases.

    Args:
    - phrase1 (str): The first phrase.
    - phrase2 (str): The second phrase.

    Returns:
    - float: The cosine similarity between the two phrase embeddings.
    """
    # Embed the phrases
    embeddings = use_model([phrase1, phrase2])

    # Compute cosine similarity
    similarity = cosine_similarity(embeddings[0].numpy().reshape(1, -1), embeddings[1].numpy().reshape(1, -1))

    return similarity[0][0]


w2v_model = gensim.models.KeyedVectors.load_word2vec_format('wiki-news-300d-1M.vec', binary=False)

w2v_model.save_word2vec_format('wiki-news-300d-1M_saved.bin', binary=True)





# Test for similarity
"""
phrase1 = 'a'
while phrase1 != '%':
    phrase1 = input("First phrase: ")
    phrase2 = input("Second phrase: ")
    similarity_score = compute_similarity(phrase1, phrase2)
    print(f"Cosine similarity between '{phrase1}' and '{phrase2}' is: {similarity_score}")
"""