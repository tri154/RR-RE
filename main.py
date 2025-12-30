from common import Config, Encoder, load_class

def load_handler(dataset_cfg):
    name = dataset_cfg["name"]
    kwargs = dataset_cfg["kwargs"]
    cls = load_class("datasets", name)
    return cls(**kwargs)


def load_model(model_cfg):
    pass


if __name__ == "__main__":
    cfg_path = "configs/config_docred.yaml"
    cfg = Config(cfg_path)
    dvc = cfg.device

    encoder = Encoder(**cfg.encoder)

    handler = load_handler(cfg.dataset)
    handler.get_features(encoder.tokenizer)

    breakpoint()
