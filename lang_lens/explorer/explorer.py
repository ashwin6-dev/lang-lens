from abc import ABC, abstractmethod

class Explorer(ABC):
    @abstractmethod
    def inspect(self, vec):
        pass

    @abstractmethod
    def launch(self):
        pass