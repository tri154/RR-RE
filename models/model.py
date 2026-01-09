import torch.nn as nn
import torch
import torch.nn.functional as F

# import logging
# log = logging.getLogger(__name__)

from utils import cumsum_with_zero, check_tensor
# from functools import partial

# ct = partial(check_tensor, logger=log)

SMALL_NEGATIVE = -1e10

class DocREModel(nn.Module):
    emb_size: int
    block_size: int
    num_class: int

    def __init__(self, model_cfg, pretrain):
        super().__init__()
        for name in self.__class__.__annotations__: # only update defined annotations.
            setattr(self, name, model_cfg.get(name))
        pretrain.load_model()
        self.pretrain = pretrain
        self.hidden_size = self.pretrain.config.hidden_size

        self.head_extractor = nn.Linear(self.hidden_size * 2, self.emb_size)
        self.tail_extractor = nn.Linear(self.hidden_size * 2, self.emb_size)
        self.bilinear_nn = nn.Linear(self.emb_size * self.block_size, self.num_class)


    def __bilinear(self, hs, ts):
        hs = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        ts = ts.view(-1, self.emb_size // self.block_size, self.block_size)
        bl = (hs.unsqueeze(3) * ts.unsqueeze(2)).view(-1, self.emb_size * self.block_size)
        logits = self.bilinear_nn(bl)
        return logits


    def get_entity_embs_attns(self, seq_embs, attentions, entity_pos, n_entities):
        bs = seq_embs.shape[0]
        dids = (
            torch.arange(bs)
            .repeat_interleave(n_entities)
            .unsqueeze(dim=-1)
            .to(seq_embs.device)
        )
        entity_embs = seq_embs[dids, entity_pos].logsumexp(dim=-2)

        n_ments = (entity_pos != -1).sum(dim=1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        entity_attns = attentions[dids, :, entity_pos, :].sum(dim=1) / n_ments

        return entity_embs, entity_attns


    def offset_hts(self, hts, n_entities, n_rels):
        dvc = hts.device
        offsets = (
            cumsum_with_zero(n_entities, exclude_last=True)
            .repeat_interleave(n_rels)
            .to(dvc)
        )
        new_hts = hts + offsets.unsqueeze(-1)
        return new_hts


    def get_ht(self, entity_embs, hts):
        pairs = entity_embs[hts]
        hs, ts = pairs.unbind(dim=1)
        return hs, ts

    def get_rs(self, seq_embs, entity_attns, hts, n_rels):
        rs = entity_attns[hts]
        rs = rs[:, 0] * rs[:, 1]
        rs = rs.sum(dim=1)
        assert (rs.sum(dim=1) != 0).all()
        rs = rs / rs.sum(dim=1, keepdim=True) # n, mlen

        n_rels_cum = cumsum_with_zero(n_rels, exclude_last=False)
        res = list()
        for i in range(0, len(n_rels_cum) - 1):
            start = int(n_rels_cum[i])
            end = int(n_rels_cum[i + 1])

            seq_emb = seq_embs[i, :-1]
            rr = rs[start:end]
            res.append(torch.matmul(rr, seq_emb))
        res = torch.cat(res, dim=0)
        return res

    def forward(
        self,
        input_ids,
        input_mask,
        entity_pos,
        hts,
        n_entities,
        n_rels,
    ):
        # TODO: speed when output attentions with not, try another type of attention backends.
        # https://huggingface.co/docs/transformers/attention_interface
        # If attention is must, try implement another roberta encoder layer, freeze pretrained model.
        # https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
        # https://chatgpt.com/c/695d1003-361c-8322-938f-f2cf973e10ce

        seq_embs, attentions = self.pretrain(input_ids, input_mask, output_attentions=True)

        seq_embs = F.pad(seq_embs, (0, 0, 0, 1), value=SMALL_NEGATIVE)
        attentions = F.pad(attentions, ((0, 0, 0, 1)), value=0.0)

        entity_embs, entity_attns = self.get_entity_embs_attns(
            seq_embs,
            attentions,
            entity_pos,
            n_entities
        )

        hts = self.offset_hts(hts, n_entities, n_rels)
        hs, ts = self.get_ht(entity_embs, hts)
        rs = self.get_rs(seq_embs, entity_attns, hts, n_rels)

        hs = torch.cat([hs, rs], dim=1)
        ts = torch.cat([ts, rs], dim=1)
        hs = torch.tanh(self.head_extractor(hs))
        ts = torch.tanh(self.head_extractor(ts))
        logits = self.__bilinear(hs, ts)

        if self.training:
            return logits
        else:
            preds = logits > logits[:, 0, None]
            preds[:, 0] = (preds.sum(1) == 0)
            return preds
