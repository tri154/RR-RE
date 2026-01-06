import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf
from functools import partial

from datasets import ReDocRED
from models import Encoder, DocREModel
from trainer import Trainer
from tester import Tester
from loss import Loss
from utils import collate_fn

import logging
log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="configs", config_name="config_redocred")
def main(cfg: DictConfig):
    OmegaConf.resolve(cfg)
    log.info(OmegaConf.to_yaml(cfg))
    dataset = ReDocRED(cfg.dataset)
    encoder = Encoder(cfg.encoder)
    features = dataset.get_features(encoder.tokenizer)
    model = DocREModel(cfg.model, encoder)

    loss = Loss(cfg.loss)
    tester = Tester(
        cfg.tester,
        dev_features=features["dev"],
        test_features=features["test"],
        test_collate_fn=partial(collate_fn, training=False)
    )
    trainer = Trainer(
        cfg.trainer,
        model,
        tester,
        loss,
        train_features=features["train"],
        train_collate_fn=partial(collate_fn, training=True)
    )

    trainer.train()

if __name__ == "__main__":
    main()
