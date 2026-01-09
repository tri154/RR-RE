import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf

from datasets import ReDocRED
from utils import init_dist, load_synced_config
from models import Encoder

import logging
log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="configs", config_name="config_redocred")
def main(cfg: DictConfig):
    RANK, WORLD_SIZE = init_dist()
    cfg = load_synced_config(cfg, RANK, WORLD_SIZE)
    log.info(OmegaConf.to_yaml(cfg))

    dataset = ReDocRED(cfg.dataset)
    encoder = Encoder(cfg.encoder)
    # cache data
    features = dataset.get_features(encoder.tokenizer, load_cached=False)

if __name__ == "__main__":
    main()
