from .interfaces import DatasetHandler
from .config import Config
from .encoder import Encoder

import importlib

def load_class(module, class_name):
    module = importlib.import_module(module)
    cls = getattr(module, class_name)
    return cls
