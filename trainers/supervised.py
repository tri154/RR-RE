from common import BaseTrainer, logging

from torch.utils.data import DataLoader
import torch
import os.path as path
import math


class SupervisedTrainer(BaseTrainer):
    device: str
    result_path: str
    log_path: str

    epochs: int
    batch_size: int
    gradient_accumulation_steps: int
    model_save: str
    opt_kwargs: dict
    opt_name: str
    warmup_ratio: float

    def __init__(self, **trainer_kwargs):
        self.__dict__.update(trainer_kwargs)
        self.model_save= path.join(self.result_path, self.model_save)

    def prepare_train_dataloader(self, train_features, collate_fn):
        train_loader = DataLoader(
            train_features,
            collate_fn=collate_fn,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True
        )
        self.num_training_steps = math.ceil(len(train_loader) / self.gradient_accumulation_steps) * self.epochs
        return train_loader


    def train_step(self):

        pass

    def train_step(self, batch):
        pass
