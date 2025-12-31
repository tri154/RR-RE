from abc import abstractmethod, ABC


class DatasetHandler(ABC):

    @abstractmethod
    def get_features(self, tokenizer) -> dict:
        pass
