import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from utils.batch_loader import BatchLoader

if __name__ == "__main__":

    embeddings_path = '../../data/preprocessings/word_embeddings.npy'

    if not os.path.exists(embeddings_path):
        raise FileNotFoundError("word embeddings file was't found")

    pca = PCA(n_components=2)
    word_embeddings = np.load(embeddings_path)
    word_embeddings_pca = pca.fit_transform(word_embeddings)

    batch_loader = BatchLoader()
    words = batch_loader.idx_to_word

    fig, ax = plt.subplots()
    fig.set_size_inches(150, 150)
    x = word_embeddings_pca[:, 0]
    y = word_embeddings_pca[:, 1]
    ax.scatter(x, y)

    for i, word in enumerate(words):
        ax.annotate(word, (x[i], y[i]))

    fig.savefig('word_embedding.png', dpi=100)