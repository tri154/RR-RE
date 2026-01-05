from typing import Any
import torch
import math
from torch.nn.utils import clip_grad_norm_
from collections import defaultdict
from transformers.optimization import get_linear_schedule_with_warmup
from torch.optim import AdamW, Adam
from torch.utils.data import DataLoader

import logging
log = logging.getLogger(__name__)


OPTIMIZERS = {"Adam": Adam, "AdamW": AdamW}


class Trainer:
    device: str
    epochs: int
    batch_size: int
    model_save: str
    grad_accum_step: int
    pretrain_lr: float
    new_lr: float
    optimizer_cfg: Any
    warmup_ratio: float

    def __init__(self, trainer_cfg, model, tester, loss, *,train_features, train_collate_fn):
        for name in self.__class__.__annotations__: # only update defined annotations.
            setattr(self, name, trainer_cfg.get(name))

        self.model = model
        self.tester = tester
        self.train_features = train_features
        self.train_collate_fn = train_collate_fn
        self.loss = loss

    def prepare_optimizer_scheduler(self, train_loader):
        grouped_params = defaultdict(list)
        for name, param in self.model.named_parameters():
            if 'pretrain' in name:
                grouped_params['pretrained_lr'].append(param)
            else:
                grouped_params['new_lr'].append(param)

        grouped_lrs = [{'params': grouped_params[group], 'lr': lr} for group, lr in zip(['pretrained_lr', 'new_lr'], [self.pretrain_lr, self.new_lr])]
        opt_type = OPTIMIZERS[self.optimizer_cfg.name]
        opt = opt_type(grouped_lrs, **self.optimizer_cfg.kwargs)

        num_updates = math.ceil(len(train_loader) / self.grad_accum_step) * self.epochs
        num_warmups = int(num_updates * self.warmup_ratio)
        sched = get_linear_schedule_with_warmup(opt, num_warmups, num_updates)

        return opt, sched

    def train_one_epoch(self, current_epoch):
        device = self.device

        total_loss = 0.0
        tracking_loss = 0.0

        for idx_batch, batch_input in enumerate(self.train_loader):
            # CONTINUE: todo things, pin memory.
            breakpoint()
            # TODO: to device
            # batch_input = batch_input.to(device)

            self.model.train()
            batch_logits = self.model()
            batch_loss = self.loss_fn.compute_loss(batch_logits, batch_trg[:, 1:].contiguous())

            # DEBUG
            # print(batch_loss)
            # input("break")
            # DEBUG
            batch_loss.backward()

            is_updated = True
            is_evaluated = idx_batch == len(train_dataloader) - 1
            is_evaluated = is_evaluated or (self.cfg.eval_freq > 0 and idx_batch % self.cfg.eval_freq == 0 and idx_batch != 0)

            if is_updated:
                self.opt.step()
                self.opt.zero_grad()
                self.sched.step()

            if is_evaluated:
                d_score = self.tester.test(self.model, self.tokenizer, tag='dev', batch_size=self.cfg.test_batch_size)
                self.cfg.logging(f"batch id: {idx_batch}, Dev result : {d_score}", is_printed=True, print_time=True)
                if d_score > self.best_score_dev:
                    self.best_score_dev = d_score
                    torch.save(self.model.state_dict(), self.cfg.save_path)

            if idx_batch % self.cfg.print_freq == 0 and idx_batch != 0:
                self.cfg.logging(f"batch id: {idx_batch}, batch loss: {tracking_loss/ self.cfg.print_freq}", is_printed=True, print_time=True)
                tracking_loss = 0.0

            batch_loss_item = batch_loss.item()
            tracking_loss += batch_loss_item
            total_loss += batch_loss_item

        return total_loss / len(train_dataloader)



    def train(self):
        self.train_loader = DataLoader(
            self.train_features,
            batch_size=self.batch_size,
            collate_fn=self.train_collate_fn,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )

        self.opt, self.sched = self.prepare_optimizer_scheduler(self.train_loader)
        self.cur_epoch = 0

        self.best_f1_dev = 0
        for idx_epoch in range(self.epochs):
            log.info(f'epoch {idx_epoch + 1}/{self.epochs} ' + '=' * 100)

            epoch_loss = self.train_one_epoch(idx_epoch)
            breakpoint()

            log.info(f"epoch: {idx_epoch + 1}, loss={epoch_loss} .")
            self.cur_epoch += 1

        self.model.load_state_dict(torch.load(self.model_save, map_location=self.device))
        t_tp, t_fp, t_fn, self.precision_test, self.recall_test, self.f1_test = self.tester.test(self.model, dataset='test', run_both=run_both)
        log.info(f"Test result: TP={t_tp}, FP={t_fp}, FN={t_fn}, P={self.precision_test:.10f}, R={self.recall_test:.10f}, F1={self.f1_test:.10f}")

        return self.best_f1_dev
