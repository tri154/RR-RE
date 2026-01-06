from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch.nn as nn

class Encoder(nn.Module):
    result_path: str
    device: str
    log_path: str

    name: str
    transformer_type: str
    lazy: bool

    def __init__(self, encoder_cfg):
        super().__init__()

        for name in self.__class__.__annotations__: # only update defined annotations.
            setattr(self, name, encoder_cfg.get(name))

        self.config = AutoConfig.from_pretrained(self.name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        if not self.lazy:
            self.load_model()

        self.max_num_tokens = 512
        if self.transformer_type == "bert":
            self.start_tokens = [self.tokenizer.cls_token_id]
            self.end_tokens = [self.tokenizer.sep_token_id]
        elif self.transformer_type == "roberta":
            self.start_tokens = [self.tokenizer.cls_token_id]
            self.end_tokens = [self.tokenizer.sep_token_id, self.tokenizer.sep_token_id]
        else:
            raise NotImplementedError()

        self.start_token_len = len(self.start_tokens)
        self.end_token_len = len(self.end_tokens)


    def load_model(self):
        if not hasattr(self, "encoder"):
            self.encoder = AutoModel.from_pretrained(self.name, config=self.config)

    def forward(self, input_ids, input_mask):
        self.load_model()
        output = self.encoder(input_ids=input_ids, attention_mask=input_mask, output_attentions=True)
        seq_embs = output[0]
        attentions = output[-1][-1]
        return seq_embs, attentions
