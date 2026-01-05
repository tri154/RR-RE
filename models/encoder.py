from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch.nn as nn

class Encoder(nn.Module):
    result_path: str
    device: str
    log_path: str

    name: str
    lazy: bool

    def __init__(self, encoder_cfg):
        super().__init__()

        for name in self.__class__.__annotations__: # only update defined annotations.
            setattr(self, name, encoder_cfg.get(name))

        self.config = AutoConfig.from_pretrained(self.name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        if not self.lazy:
            self.load_model()


    def load_model(self):
        if not hasattr(self, "encoder"):
            self.encoder = AutoModel.from_pretrained(self.name, config=self.config)

    def forward(self):
        self.load_model()
        pass
