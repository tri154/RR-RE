from trainers.supervised import SupervisedTrainer
from common import (
    Config,
    Encoder,
    DatasetHandler,
    load_class,
    OPTIMIZERS,
    SCHEDULERS,
)

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


def load_scheduler(train_metadata, **scheduler_cfg):
    # continue
    name = scheduler_cfg["name"]
    fn = SCHEDULERS[name]

    pass


if __name__ == "__main__":
    cfg_path = "configs/config_redocred.yaml"
    cfg = Config(cfg_path)

    encoder = Encoder(**cfg.encoder, **cfg.shared)
    # temp
    import torch
    model = torch.nn.Linear(1, 2)

    handler = load_handler(**cfg.dataset, **cfg.shared)
    features = handler.get_features(encoder.tokenizer)

    trainer = SupervisedTrainer(**cfg.trainer, **cfg.shared)
    train_loader = trainer.prepare_train_dataloader(
        features["train"],
        handler.train_collate_fn
    )

    optimizer = load_optimizer(model, **cfg.optimizer)
    # scheduler = load_scheduler(train_metadata, **cfg.lr_scheduler)
    print("done")
    breakpoint()
