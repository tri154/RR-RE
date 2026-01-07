import torch
import torch.nn.functional as F

class Loss:
    def __init__(self, loss_cfg):
        for name in self.__class__.__annotations__: # only update defined annotations.
            setattr(self, name, loss_cfg.get(name))
        self.compute_loss = self.at_loss
        self.predict = None

    def at_loss(self, logits, labels):
        n_rels, n_class = logits.shape
        dump_cols = torch.full([n_rels, 1], fill_value=float("-inf"))
        logits = torch.cat([logits, dump_cols], dim=1)

        selected_logits = torch.gather(
            logits,
            dim=1,
            index=labels
        )
        # CONTINUE here
        breakpoint()
