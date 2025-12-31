from common import logging
import argparse
import random
import torch
import numpy as np
import yaml
import os

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_config_path():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    args = parser.parse_args()
    return args.config

class Config:
    device: str
    result_path: str
    dataset: dict
    encoder: dict
    model: dict

    def __init__(self, path=None):
        path = path if path is not None else get_config_path()
        with open(path, "r") as f:
            file_config = yaml.safe_load(f)
        self.__dict__.update(file_config)

        # process config
        os.makedirs(self.result_path, exist_ok=True)
        self.log_path = os.path.join(self.result_path, 'log.txt')
        self.check_device(self.device)

        self.log_config(True)
        self.shared = {
            "device": self.device,
            "result_path": self.result_path,
            "log_path": self.log_path
        }

    def check_device(self, current_device):
        if current_device.lower() == 'cuda' and not torch.cuda.is_available():
            self.device = 'cpu'
            logging("cuda is not available, use cpu instead.", self.log_path, is_printed=True)

    def set_seed(self, seed):
        logging("=" * 50 + "\n\n\n", self.log_path)
        logging(f"Using seed {seed}.", self.log_path)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


    def log_config(self, is_printed=False):
        logging("Configuration Settings:", self.log_path,is_printed=is_printed)
        for key, value in sorted(self.__dict__.items()):
            # Avoid logging functions or modules
            if not key.startswith("__") and not callable(value):
                logging(f"{key}: {value}", self.log_path, is_printed=is_printed)
        logging("=" * 50, self.log_path, is_printed=is_printed)
