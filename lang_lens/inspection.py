from typing import List
import numpy as np
from .axis_discovery import Axis

class Inspection:
    def __init__(
        self,
        vector: np.array,
        axes: List[Axis],
        projection: np.array
    ):
        self.vector = vector
        self.axes = axes
        self.projection = projection