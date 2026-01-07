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

        def option1():
            selected_logits.masked_fill_(mask, value=float("-inf"))
            loss1 = selected_logits - torch.log(selected_logits.exp().sum(dim=1) + logits[:, 0].exp()).unsqueeze(-1)
            loss1.masked_fill_(mask, value=0.0)
            loss1 = torch.sum(loss1, dim=1)
            loss1 = loss1 * (labels[:, 0] != 0)
            return loss1

        def option2():
            selected_logits.masked_fill_(mask, value=float("-inf"))
            exp_logits = selected_logits.exp()
            den = exp_logits.sum(dim=1) + logits[:, 0].exp()
            loss1 = exp_logits / den.unsqueeze(dim=-1)
            loss1.masked_fill_(mask, value=1.0)
            loss1 = torch.log(loss1).sum(dim=1)
            loss1 = loss1 * (labels[:, 0] != 0)
            return loss1

        def option3():
            # selected_logits.masked_fill_(mask, value=float("-inf"))
            sel_logits  = selected_logits - (mask * 1e10)
            loss1 = sel_logits  - torch.log(sel_logits.exp().sum(dim=1) + logits[:, 0].exp()).unsqueeze(-1)
            # loss1.masked_fill_(mask, value=0.0)
            loss1 = torch.sum(loss1 * ~mask, dim=1)
            loss1 = loss1 * (labels[:, 0] != 0)
            return loss1

        # o3 = option3()
        # o2 = option2()
        # o1 = option1()
        # breakpoint()
        benchmark([option3, option1, option2])
        breakpoint()
