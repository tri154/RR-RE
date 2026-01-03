from trainers.supervised import SupervisedTrainer
from common import (
    Config,
    Encoder,
    DatasetHandler,
    load_class,
    OPTIMIZERS,
)

from transformers.optimization import get_linear_schedule_with_warmup
import math
import torch

def load_handler(**dataset_cfg) -> DatasetHandler:
    name = dataset_cfg["name"]
    cls = load_class("datasets", name)
    return cls(**dataset_cfg)


def load_optimizer(model, **opt_cfg):
    # TODO: different learning rate for pretrain after implementing model.
    name = opt_cfg["name"]
    opt = OPTIMIZERS[name](model.parameters(), **opt_cfg["kwargs"])
    return opt


def load_model(model_cfg):
    pass


def load_scheduler(optimizer, num_training_steps, warmup_ratio):
    return get_linear_schedule_with_warmup(
        optimizer,
        int(num_training_steps * warmup_ratio),
        num_training_steps
    )


if __name__ == "__main__":
    cfg_path = "configs/config_redocred.yaml"
    cfg = Config(cfg_path)

    encoder = Encoder(**cfg.encoder, **cfg.shared)
    model = torch.nn.Linear(1, 2)

    handler = load_handler(**cfg.dataset, **cfg.shared)
    features = handler.get_features(encoder.tokenizer)

    trainer = SupervisedTrainer(**cfg.trainer, **cfg.shared)
    train_loader = trainer.prepare_train_dataloader(
        features["train"],
        handler.train_collate_fn
    )

    optimizer = load_optimizer(model, **cfg.optimizer)
    scheduler = load_scheduler(
        optimizer,
        trainer.num_training_steps,
        cfg.lr_scheduler["warmup_ratio"]
    )
    # continue

    print("done")
    breakpoint()
