from common import logging

from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from transformers.optimization import get_linear_schedule_with_warmup
import math
import torch
import os.path as path

optimizers = {"Adam": Adam, "AdamW": AdamW}

class Trainer:
    device: str
    result_path: str
    log_path: str

    epochs: int
    batch_size: int
    gradient_accumulation_steps: int
    model_save: str
    collate_fn: function
    opt_kwargs: dict
    opt_name: str
    warmup_ratio: float

    def __init__(self, **trainer_kwargs):
        self.__dict__.update(trainer_kwargs)
        self.model_save= path.join(self.result_path, self.model_save)
        print(self.model_save)

    def prepare_optimizer_scheduler(self, model, train_dataloader):
        # TODO: different learning rate for pretrain after implement model.
        opt = optimizers[self.opt_name](model.parameters(), **self.opt_kwargs)
        num_updates = math.ceil(len(train_dataloader) / self.gradient_accumulation_steps) * self.epochs
        num_warmups = int(self.warmup_ratio * num_updates)
        sched = get_linear_schedule_with_warmup(opt, num_warmups, num_updates)
        return opt, sched

    def train_one_epoch(self, train_dataloader, current_epoch):
        pass

    def train(self, model, train_features):
        train_dataloader = DataLoader(
            train_features,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True
        )

        self.opt, self.sched = self.prepare_optimizer_scheduler(model, train_dataloader)
        self.best_score_dev = -1
        for idx_epoch in range(self.epochs):
            logging(f'epoch {idx_epoch + 1}/{self.epochs} ' + '=' * 100, self.log_path, is_printed=True)

            epoch_loss = self.train_one_epoch(train_dataloader, idx_epoch)

            logging(f"epoch: {idx_epoch + 1}, loss={epoch_loss} .", self.log_path, is_printed=True)

        model.load_state_dict(torch.load(self.model_save, map_location=self.device))
        # continue
        logging(f"Test result: {t_score}", self.log_path, is_printed=True)
