import torch.nn as nn


class DocREModel(nn.Module):

    def __init__(self, model_cfg, pretrain):
        super().__init__()
        pretrain.load_model()
        self.pretrain  = pretrain

    def forward(
        self,
        input_ids,
        input_mask,
        entity_pos,
        hts,
        n_entities
    ):
        print(input_ids.shape)
        seq_embs, attentions = self.pretrain(input_ids, input_mask)
        print(f"done {seq_embs.shape}")
