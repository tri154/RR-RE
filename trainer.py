from typing import Any
import torch
import math
from torch.nn.utils import clip_grad_norm_
from collections import defaultdict
from transformers.optimization import get_linear_schedule_with_warmup
from torch.optim import AdamW, Adam
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils import move_to_cuda, dist_log, get_dist_info

log_info = lambda dump, **func: dump
OPTIMIZERS = {"Adam": Adam, "AdamW": AdamW}
RANK = 0
WORLD_SIZE = 1


class Trainer:
    use_bfloat16: bool
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

    def __init__(self, trainer_cfg, model, tester, *,train_features, train_collate_fn, wandb_run=None):
        global log_info; log_info = dist_log(__name__)
        global RANK, WORLD_SIZE; RANK, WORLD_SIZE= get_dist_info()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        for name in self.__class__.__annotations__: # only update defined annotations.
            setattr(self, name, trainer_cfg.get(name))

        self.model = model

        self.tester = tester
        self.train_features = train_features
        self.train_collate_fn = train_collate_fn
        self.wandb_run = wandb_run
        bf16_supported = torch.cuda.is_bf16_supported()
        if self.use_bfloat16 and not bf16_supported:
            log_info("GPUs doesn't support bfloat16, fall back to default (float32).")
        elif self.use_bfloat16 and bf16_supported:
            log_info("GPUs supported bfloat16, use bfloat16. ")
        else:
            log_info("Use float32.")
        self.use_bfloat16 = self.use_bfloat16 and torch.cuda.is_bf16_supported()

    def wandb_log(self, data, cur_step):
        if self.wandb_run is None:
            return
        self.wandb_run.log(data, step=cur_step)


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
        log_info(f"Total update steps: {num_updates}")
        log_info(f"Total warmup steps: {num_warmups}")
        sched = get_linear_schedule_with_warmup(opt, num_warmups, num_updates)

        return opt, sched

    def train_one_epoch(self, current_epoch):
        device = self.device

        total_loss = 0.0
        tracking_loss = 0.0

        for idx_batch, (batch_input, batch_label) in enumerate(self.train_loader):
            batch_input, batch_label = move_to_cuda(**batch_input, **batch_label, device=device)

            self.model.train()
            if self.use_bfloat16:
                with torch.amp.autocast(dtype=torch.bfloat16, device_type=device):
                    batch_loss = self.model(**batch_input, **batch_label)
            else:
                batch_loss = self.model(**batch_input, **batch_label)

            (batch_loss / self.grad_accum_step).backward()

            # DEBUG
            # print(batch_loss)
            # input("break")
            # DEBUG

            is_final_step = idx_batch == len(self.train_loader) - 1
            is_updated = idx_batch % self.grad_accum_step == 0 or is_final_step
            is_eval_step = self.eval_freq > 0 and idx_batch % self.eval_freq == 0 and idx_batch != 0
            is_evaluated = is_final_step or is_eval_step

            if is_updated:
                if self.max_grad_norm > 0:
                    clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.opt.step()
                self.opt.zero_grad()
                self.sched.step()
                self.cur_step += 1

            if is_evaluated:
                d_score, d_output = self.tester.test(self.model, tag='dev')
                log_info(f"batch id: {idx_batch}, Dev result : {d_output} .")
                self.wandb_log(d_output, self.cur_step)
                if RANK == 0:
                    if d_score > self.best_score_dev:
                        self.best_score_dev = d_score
                        self.best_output_dev = d_output
                        torch.save(self.model.state_dict(), self.model_save)

            if idx_batch % self.print_freq == 0 and idx_batch != 0:
                log_info(f"batch id: {idx_batch}, batch loss: {tracking_loss/self.print_freq}")
                self.wandb_log({"loss": tracking_loss/self.print_freq}, self.cur_step)
                tracking_loss = 0.0

            batch_loss_item = batch_loss.item()
            tracking_loss += batch_loss_item
            total_loss += batch_loss_item

        return total_loss / len(self.train_loader)



    def train(self):
        sampler = DistributedSampler(self.train_features) if WORLD_SIZE > 1 else None
        self.train_loader = DataLoader(
            self.train_features,
            batch_size=self.batch_size,
            collate_fn=self.train_collate_fn,
            shuffle=sampler is None,
            drop_last=True,
            pin_memory= self.device == 'cuda',
            sampler=sampler,
            num_workers=1
        )

        self.opt, self.sched = self.prepare_optimizer_scheduler(self.train_loader)
        self.cur_epoch = 0
        self.cur_step = 0

        self.best_score_dev= 0
        self.best_output_dev = 0
        for idx_epoch in range(self.epochs):
            log_info(f'epoch {idx_epoch + 1}/{self.epochs} ' + '=' * 100)

            if WORLD_SIZE > 1: self.train_loader.sampler.set_epoch(idx_epoch)
            epoch_loss = self.train_one_epoch(idx_epoch)

            log_info(f"epoch: {idx_epoch + 1}, loss={epoch_loss} .")
            self.cur_epoch += 1

        self.model.load_state_dict(
            torch.load(self.model_save, map_location=self.device, weights_only=True)
        )
        log_info(f"Best dev result: {self.best_output_dev}")
        self.score_test, test_output = self.tester.test(self.model, tag='test')
        log_info(f"Test result: {test_output} .")
        self.wandb_log(test_output, cur_step=self.cur_step)

        return self.best_score_dev
