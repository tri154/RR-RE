from abc import abstractmethod, ABC
from torch.utils.data import DataLoader


class DatasetHandler(ABC):

    @abstractmethod
    def get_features(self, tokenizer) -> dict[str, list]:
        pass

    @abstractmethod
    def collate_fn(self, batch, training=False) -> tuple:
        pass

    def train_collate_fn(self, batch):
        return self.collate_fn(batch, training=True)

    def test_collate_fn(self, batch):
        return self.collate_fn(batch, training=False)


class BaseTrainer(ABC):

    @abstractmethod
    def train_step(self, batch):
        pass

    @abstractmethod
    def prepare_train_dataloader(self, train_features, collate_fn) -> DataLoader:
        pass
