import numpy as np
from sentence_transformers import SentenceTransformer
from datasets import load_dataset

from lang_lens.explorer import web_explorer
from lang_lens.lens import Lens


class Store:
    def __init__(self, texts, model_name="all-MiniLM-L6-v2"):
        self.texts = texts

        # Load model only temporarily (not stored on self! → avoids pickle issues)
        model = SentenceTransformer(model_name)

        # Precompute embeddings
        self.vectors = model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )

    def get_vectors(self):
        return self.vectors

    def get_texts(self):
        return self.texts


def main():
    from lang_lens.axis_discovery import pca_discovery

    # Load IMDB dataset (small subset for speed)
    
    dataset = load_dataset("imdb", split="train")
    dataset = dataset.shuffle(seed=42)  # randomize order
    subset = dataset.select(range(1000))  # first 200 reviews
    texts = list(subset["text"])

    print (texts)

    # Build store
    store = Store(texts)

    # Build lens + explorer
    lens = Lens(store, pca_discovery.PCADiscovery(n_components=0.8))
    explorer = web_explorer.WebExplorer(lens)

    # Inspect the first review
    explorer.inspect(store.get_vectors()[0])
    explorer.launch()


if __name__ == "__main__":
    main()