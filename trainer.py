from typing import Any
import torch
import math
from torch.nn.utils import clip_grad_norm_
from collections import defaultdict
from transformers.optimization import get_linear_schedule_with_warmup
from torch.optim import AdamW, Adam
from torch.utils.data import DataLoader

from utils import move_to_cuda

import logging
log = logging.getLogger(__name__)


OPTIMIZERS = {"Adam": Adam, "AdamW": AdamW}


class Trainer:
    epochs: int
    batch_size: int
    grad_accum_step: int
    eval_freq: int
    print_freq: int
    max_grad_norm: float
    model_save: str

    pretrain_lr: float
    new_lr: float
    warmup_ratio: float
    optimizer_cfg: Any

    def __init__(self, trainer_cfg, model, tester, loss, *,train_features, train_collate_fn):
        for name in self.__class__.__annotations__: # only update defined annotations.
            setattr(self, name, trainer_cfg.get(name))

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.tester = tester
        self.train_features = train_features
        self.train_collate_fn = train_collate_fn
        self.loss_fn = loss

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

        for idx_batch, (batch_input, batch_label) in enumerate(self.train_loader):
            batch_input = move_to_cuda(**batch_input, device=device)
            # TODO: move batch_label to cuda if needed.

            self.model.train()
            batch_logits = self.model(**batch_input)
            continue
            batch_loss = self.loss_fn.compute_loss(batch_logits, batch_label)
            (batch_loss / self.grad_accum_step).backward()

            # DEBUG
            # print(batch_loss)
            # input("break")
            # DEBUG

            is_updated = True
            is_final_step = idx_batch == len(self.train_loader) - 1
            is_eval_step = self.eval_freq > 0 and idx_batch % self.eval_freq == 0 and idx_batch != 0
            is_evaluated = is_final_step or is_eval_step

            if is_updated:
                if self.max_grad_norm > 0:
                    clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.opt.step()
                self.opt.zero_grad()
                self.sched.step()

            if is_evaluated:
                d_score = self.tester.test(self.model, tag='dev')
                log.info(f"batch id: {idx_batch}, Dev result : {d_score}")
                if d_score > self.best_score_dev:
                    self.best_score_dev = d_score
                    torch.save(self.model.state_dict(), self.model_save)

            if idx_batch % self.print_freq == 0 and idx_batch != 0:
                log.info(f"batch id: {idx_batch}, batch loss: {tracking_loss/self.print_freq}")
                tracking_loss = 0.0

            batch_loss_item = batch_loss.item()
            tracking_loss += batch_loss_item
            total_loss += batch_loss_item

        return total_loss / len(self.train_loader)



    def train(self):
        self.train_loader = DataLoader(
            self.train_features,
            batch_size=self.batch_size,
            collate_fn=self.train_collate_fn,
            shuffle=True,
            drop_last=True,
            pin_memory= self.device == 'cuda',
        )

        self.opt, self.sched = self.prepare_optimizer_scheduler(self.train_loader)
        self.cur_epoch = 0

        self.best_f1_dev = 0
        for idx_epoch in range(self.epochs):
            log.info(f'epoch {idx_epoch + 1}/{self.epochs} ' + '=' * 100)

            epoch_loss = self.train_one_epoch(idx_epoch)

            log.info(f"epoch: {idx_epoch + 1}, loss={epoch_loss} .")
            self.cur_epoch += 1

        self.model.load_state_dict(torch.load(self.model_save, map_location=self.device))
        t_tp, t_fp, t_fn, self.precision_test, self.recall_test, self.f1_test = self.tester.test(self.model, dataset='test', run_both=run_both)
        log.info(f"Test result: TP={t_tp}, FP={t_fp}, FN={t_fn}, P={self.precision_test:.10f}, R={self.recall_test:.10f}, F1={self.f1_test:.10f}")

        return self.best_f1_dev
