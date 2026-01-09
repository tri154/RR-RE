import torch
import torch.nn.functional as F

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
        not_na = labels[:, 0] != 0
        loss1 = loss1 * not_na

        rev_n_labels = 1 / (~mask).sum(dim=1)
        loss1 = loss1 * rev_n_labels


        row_idx = (
            torch.arange(logits.shape[0], device=logits.device)
            .unsqueeze(-1)
            .expand(*labels.shape)
        )
        filter = (~mask) & not_na.unsqueeze(dim=-1)
        mask2 = torch.zeros_like(logits, dtype=torch.bool)
        mask2[row_idx[filter], labels[filter]] = True
        logits_masked = logits.masked_fill(mask2, float("-inf"))

        loss2 = na_col - logits_masked.logsumexp(dim=1)

        loss = - (loss1 + loss2).mean()
        return loss
