import torch.nn as nn
import torch
import torch.nn.functional as F

from utils import cumsum_with_zero

SMALL_NEGATIVE = -1e10

class DocREModel(nn.Module):

    def __init__(self, model_cfg, pretrain):
        super().__init__()
        for name in self.__class__.__annotations__: # only update defined annotations.
            setattr(self, name, model_cfg.get(name))
        pretrain.load_model()
        self.pretrain  = pretrain

    def get_entity_embs(self, seq_embs, entity_pos, n_entities):
        bs = seq_embs.shape[0]
        dids = (
            torch.arange(bs)
            .repeat_interleave(n_entities)
            .unsqueeze(dim=-1)
            .to(seq_embs.device)
        )
        entity_embs = seq_embs[dids, entity_pos].logsumexp(dim=-2)
        return entity_embs

    def get_ht(self, entity_embs, hts, n_entities, n_rels):
        dvc = entity_embs.device
        offsets = (
            cumsum_with_zero(n_entities, exclude_last=True)
            .repeat_interleave(n_rels)
            .to(dvc)
        )
        hts = hts + offsets.unsqueeze(-1)
        pairs = entity_embs[hts]
        hs, ts = pairs.unbind(dim=1)
        return hs, ts

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
        bs = input_ids.shape[0]
        dvc = input_ids.device

        seq_embs, attentions = self.pretrain(input_ids, input_mask, output_attentions=True)

        seq_embs = F.pad(seq_embs, (0, 0, 0, 1), value=SMALL_NEGATIVE)
        entity_embs = self.get_entity_embs(seq_embs, entity_pos, n_entities)
        hs, ts = self.get_ht(entity_embs, hts, n_entities, n_rels)
        # =================
