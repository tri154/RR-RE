from transformers import AutoConfig, AutoModel, AutoTokenizer
import torch.nn as nn

class Encoder(nn.Module):

    def __init__(self, name, lazy):
        self.name = name
        self.config = AutoConfig.from_pretrained(name)
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        if not lazy:
            self.load_model()

    def load_model(self):
        self.model = AutoModel.from_pretrained(self.name, config=self.config)

    def forward(self):
        if not hasattr(self, "model"): self.load_model()
        pass
