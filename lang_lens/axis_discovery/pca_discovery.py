from .axis import Axis
from ..text_store import TextStore
from .axis_discovery import AxisDiscovery
import numpy as np
from sklearn.decomposition import PCA

class PCADiscovery(AxisDiscovery):
    def __init__(
        self, 
        n_components: float = 0.95
    ):
        super().__init__()
        self.n_components = n_components
        self.pca = None
        self.axes = []

    def discover(self, text_store: TextStore):
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(text_store.get_vectors())
        self.build_axes()

    def transform(self, vec: np.array):
        if not self.pca:
            raise ValueError("PCA model has not been fitted. Call discover() first.")
        
        return self.pca.transform(vec.reshape(1, -1)).flatten()
    
    def build_axes(self):
        if not self.pca:
            raise ValueError("PCA model has not been fitted. Call discover() first.")
        
        basis_vectors = self.pca.components_

        for idx, basis_vector in enumerate(basis_vectors):
            axis_name = f"Axis {idx + 1}"
            transform_fn = lambda vec, bv=basis_vector: (
                np.multiply(vec - self.pca.mean_, bv).sum(axis=1)
            )
            
            self.axes.append(Axis(axis_name, basis_vector, transform_fn))

    def get_axes(self):
        return self.axes