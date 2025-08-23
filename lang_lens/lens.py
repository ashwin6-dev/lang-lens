import numpy as np

from .inspection import Inspection
from .axis_discovery import AxisDiscovery
from .text_store import TextStore

class Lens:
    def __init__(
        self,
        text_store: TextStore,
        axis_discovery: AxisDiscovery
    ):
        self.text_store = text_store
        self.axis_discovery = axis_discovery

    def inspect(self, query_vec: np.array):
        projection = self.axis_discovery.transform(query_vec)
        inspection = Inspection(
            vector=query_vec,
            axes=self.axis_discovery.get_axes(),
            projection=projection
        )

        return inspection