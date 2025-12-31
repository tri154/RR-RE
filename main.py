from common import (
    Config,
    Encoder,
    load_class,
    DatasetHandler
    Trainer,
)


def load_handler(**dataset_cfg) -> DatasetHandler:
    name = dataset_cfg["name"]
    cls = load_class("datasets", name)
    return cls(**dataset_cfg)


def load_model(model_cfg):
    pass


if __name__ == "__main__":
    cfg_path = "configs/config_redocred.yaml"
    cfg = Config(cfg_path)

    encoder = Encoder(**cfg.encoder, **cfg.shared)

    handler = load_handler(**cfg.dataset, **cfg.shared)
    features = handler.get_features(encoder.tokenizer)
    train_features = features["train"]
    dev_features = features["dev"]
    test_features = features["test"]

    trainer = Trainer(**cfg.trainer, **cfg.shared, collate_fn=handler.get_features)

    breakpoint()
