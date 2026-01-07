import torch
import torch.nn.functional as F
from utils import benchmark

class Loss:
    def __init__(self, loss_cfg):
        for name in self.__class__.__annotations__: # only update defined annotations.
            setattr(self, name, loss_cfg.get(name))
        self.compute_loss = self.at_loss
        self.predict = None

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
        logits_exp = selected_logits.exp().sum(dim=1)
        loss1 = selected_logits - torch.log(logits_exp + logits[:, 0].exp()).unsqueeze(-1)
        loss1.masked_fill_(mask, value=0.0)
        loss1 = torch.sum(loss1, dim=1)
        is_na = labels[:, 0] == 0
        loss1 = loss1 * (~is_na)

        # CHANGED: normalize multi labels relation loss
        rev_n_labels = 1 / (~mask).sum(dim=1)
        loss1 = loss1 * rev_n_labels

        logits_exp.masked_fill_(is_na, value=0.0)
        loss2 = logits[:, 0] - torch.log(logits.exp().sum(dim=1) - logits_exp)

        loss = loss1 + loss2
        loss = - loss.mean()
        return loss
