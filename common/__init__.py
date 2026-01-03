import importlib
import time
from torch.optim import Adam, AdamW

def load_class(module, class_name):
    module = importlib.import_module(module)
    cls = getattr(module, class_name)
    return cls

def logging(text, log_path, is_printed=False, print_time=False):
    if print_time:
        text = time.strftime("%Y %b %d %a, %H:%M:%S: ") + text
    if is_printed:
        print(text)
    with open(log_path, 'a') as file:
        print(text, file=file, flush=True)

OPTIMIZERS = {"Adam": Adam, "AdamW": AdamW}

from .interfaces import DatasetHandler, BaseTrainer
from .config import Config
from .encoder import Encoder
