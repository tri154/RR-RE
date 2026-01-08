import torch
import torch.nn.functional as F

import logging
log = logging.getLogger(__name__)

from utils import benchmark, check_tensor, logsubexp

from functools import partial
# ct = partial(check_tensor, logger=log)

class Loss:
    def __init__(self, loss_cfg):
        for name in self.__class__.__annotations__: # only update defined annotations.
            setattr(self, name, loss_cfg.get(name))
        self.compute_loss = self.at_loss


    def at_loss(self, logits, labels_out):
        labels = labels_out["labels"]
        mask = labels_out["labels_mask"]
        n_rels, n_class = logits.shape

        selected_logits = torch.gather(
            logits,
            dim=1,
            index=labels
        )
        selected_logits.masked_fill_(mask, value=float("-inf"))
        na_col = logits[:, 0]

        loss1 = selected_logits - torch.logsumexp(
            torch.cat([selected_logits, na_col.unsqueeze(dim=-1)], dim=1),
            dim=1,
            keepdim=True
        )
        loss1.masked_fill_(mask, value=0.0)
        loss1 = torch.sum(loss1, dim=1)
        is_na = na_col == 0
        loss1 = loss1 * (~is_na)

        # CHANGED: normalize multi labels relation loss
        rev_n_labels = 1 / (~mask).sum(dim=1)
        loss1 = loss1 * rev_n_labels

        loss2 = na_col - logsubexp(
            torch.logsumexp(logits, dim=1),
            torch.logsumexp(selected_logits, dim=1).masked_fill(is_na, value=float("-inf"))
        )

        loss = - (loss1 + loss2).mean()
        return loss
