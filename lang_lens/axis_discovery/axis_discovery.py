from abc import ABC, abstractmethod
import numpy as np
from ..text_store import TextStore

class AxisDiscovery(ABC):
    @abstractmethod
    def discover(self, text_store: TextStore):
        pass

    @abstractmethod
    def transform(self, vec: np.array) -> np.array:
        pass

    @abstractmethod
    def get_axes(self):
        pass