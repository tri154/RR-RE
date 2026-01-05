import torch.nn as nn


class DocREModel(nn.Module):

    def __init__(self, model_cfg, pretrain):
        super().__init__()
        pretrain.load_model()
        self.pretrain  = pretrain

    def forward(self):
        pass
