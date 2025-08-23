import numpy as np

class Store:
    def __init__(self):
        pass

    def get_vectors(self):
        return np.random.randn(100, 5)
    
def main():
    from lang_lens.axis_discovery import pca_discovery

    X = Store()
    pca = pca_discovery.PCADiscovery(n_components=0.99)
    pca.discover(X)
    axes = pca.get_axes()

    vec = np.random.randn(1, 5)
    proj = pca.transform(vec)

    for axis in axes:
        print(f"Discovered {axis.label} with vector {axis.vec}, projection {axis.transform(vec)}")

    final = pca.pca.inverse_transform(proj)

    print (vec)
    print (proj)
    print (final)

if __name__ == "__main__":
    main()
