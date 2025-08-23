from abc import ABC, abstractmethod

class TextStore(ABC):
    @abstractmethod
    def search(self, query):
        pass

    @abstractmethod
    def get_vectors(self):
        pass

    @abstractmethod
    def get_texts(self):
        pass