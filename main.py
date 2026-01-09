import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf
from functools import partial

from datasets import ReDocRED
from models import Encoder, DocREModel
from trainer import Trainer
from tester import Tester
from loss import Loss
from utils import collate_fn, seeding, init_wandb, init_dist, load_synced_config, dist_log

@hydra.main(version_base=None, config_path="configs", config_name="config_redocred")
def main(cfg: DictConfig):
    RANK, WORLD_SIZE = init_dist()
    log_info = dist_log(__name__)
    cfg = load_synced_config(cfg, RANK, WORLD_SIZE)
    run = init_wandb(cfg) if cfg.wandb.used else None
    seeding(cfg.seed, hard=False, rank=RANK)
    if RANK==0: log_info(OmegaConf.to_yaml(cfg))

    dataset = ReDocRED(cfg.dataset)

    loss = Loss(cfg.loss)
    encoder = Encoder(cfg.encoder)
    model = DocREModel(cfg.model, encoder, loss)

    features = dataset.get_features(encoder.tokenizer, load_cached=True)
    # DEBUG
    breakpoint()
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
        train_features=features["train"],
        train_collate_fn=partial(collate_fn, training=True),
        wandb_run=run
    )

    trainer.train()

    if run is not None:
        run.finish()

if __name__ == "__main__":
    main()
