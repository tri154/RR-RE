import torch
import torch.nn.functional as F

import logging
log = logging.getLogger(__name__)

from utils import benchmark, check_tensor

from functools import partial
ct = partial(check_tensor, logger=log)

class Loss:
    def __init__(self, loss_cfg):
        for name in self.__class__.__annotations__: # only update defined annotations.
            setattr(self, name, loss_cfg.get(name))
        self.compute_loss = self.at_loss


    def at_loss(self, logits, labels_out):
        labels = labels_out["labels"]
        mask = labels_out["labels_mask"]
        n_rels, n_class = logits.shape
        dump_col = torch.full((logits.shape[0], 1), fill_value=float("-inf")).to(logits)
        logits = torch.cat([logits, dump_col], dim=-1)

        # TODO
        def opt1():
            selected_logits = torch.gather(
                logits,
                dim=1,
                index=labels
            )
            return selected_logits
        row_idx = torch.arange(logits.shape[0], device=logits.device).unsqueeze(-1)
        def opt2():
            selected = logits[row_idx, labels]
            return selected
        benchmark([opt1, opt2])

        na_col = logits[:, 0]

        loss1 = selected_logits - torch.logsumexp(
            torch.cat([selected_logits, na_col.unsqueeze(dim=-1)], dim=1),
            dim=1,
            keepdim=True
        )
        loss1.masked_fill_(mask, value=0.0)
        loss1 = torch.sum(loss1, dim=1)
        is_na = labels[:, 0] == 0
        loss1 = loss1 * (~is_na)

        # CHANGED: normalize multi labels relation loss
        rev_n_labels = 1 / (~mask).sum(dim=1)
        loss1 = loss1 * rev_n_labels


        # def opt1():
        #     row_idx = torch.arange(logits.shape[0], device=logits.device).unsqueeze(-1)
        #     selected = logits[row_idx, labels]
        #     breakpoint()
        # opt1()
        # loss2 = na_col -

        # ct(loss2, "loss2")

        # loss = - (loss1 + loss2).mean()
        # return loss
