from ..text_store import TextStore

class Axis:
    def __init__(
        self,
        label,
        vec,
        transform
    ):
        self.label = label
        self.vec = vec
        self.transform = transform

    def set_label(self, label):
        self.label = label