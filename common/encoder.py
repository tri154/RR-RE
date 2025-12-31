from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch.nn as nn

class Encoder(nn.Module):
    result_path: str
    device: str
    log_path: str

    name: str
    lazy: bool

    def __init__(self, **encoder_kwargs):
        super().__init__()
        self.__dict__.update(encoder_kwargs)
        self.config = AutoConfig.from_pretrained(self.name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.name)
        if not self.lazy:
            self.load_model()


    def load_model(self):
        self.model = AutoModel.from_pretrained(self.name, config=self.config).to(self.device)

    def forward(self):
        if not hasattr(self, "model"): self.load_model()
        pass
