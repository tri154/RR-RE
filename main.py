from common import Config, load_class

def load_handler(dataset_cfg):
    mapping = {
        "docred": "DocRED",
    }
    name = dataset_cfg["name"]
    kwargs = dataset_cfg["kwargs"]
    cls = load_class("datasets", mapping[name.lower()])
    return cls(**kwargs)


def load_model(model_cfg):
    pass


if __name__ == "__main__":
    cfg_path = "configs/config_docred.yaml"
    cfg = Config(cfg_path)
    handler = load_handler(cfg.dataset)

    breakpoint()
