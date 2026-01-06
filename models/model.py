import torch.nn as nn


class DocREModel(nn.Module):

    def __init__(self, model_cfg, pretrain):
        super().__init__()
        for name in self.__class__.__annotations__: # only update defined annotations.
            setattr(self, name, model_cfg.get(name))
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
        # TODO: speed when output attentions with not, try another type of attention backends.
        # https://huggingface.co/docs/transformers/attention_interface
        seq_embs, attentions = self.pretrain(input_ids, input_mask, output_attentions=True)
        breakpoint()
