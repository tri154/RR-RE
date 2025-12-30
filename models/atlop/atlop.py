import torch.nn as nn
import torch
from common import Encoder

class ATLOP(nn.Module):

    def __init__(self, model_cfg):
        super().__init__()
        self.encoder = Encoder(**model_cfg.encoder)
        breakpoint()
        # continue
