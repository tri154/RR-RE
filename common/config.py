import argparse
import random
import torch
import numpy as np
import time
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

    def __init__(self, path=None):
        path = path if path is not None else get_config_path()
        with open(path, "r") as f:
            file_config = yaml.safe_load(f)
        self.__dict__.update(file_config)

        # process config
        os.makedirs(self.result_path, exist_ok=True)
        self.save_path = os.path.join(self.result_path, 'best.pt')
        self.log_path = os.path.join(self.result_path, 'log.txt')
        self.check_device()


        self.log_config(True)

    def check_device(self):
        if self.device.lower() == 'cuda' and not torch.cuda.is_available():
            self.device = 'cpu'
            self.logging("cuda is not available, use cpu instead.", is_printed=True)

    def set_seed(self):
        self.logging("=" * 50 + "\n\n\n")
        self.logging(f"Using seed {self.seed}.")
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)

    def logging(self, text, is_printed=False, print_time=False):
        if print_time:
            text = time.strftime("%Y %b %d %a, %H:%M:%S: ") + text
        if is_printed:
            print(text)
        with open(self.log_path, 'a') as file:
            print(text, file=file, flush=True)

    def log_config(self, is_printed=False):
        self.logging("Configuration Settings:", is_printed=is_printed)
        for key, value in sorted(self.__dict__.items()):
            # Avoid logging functions or modules
            if not key.startswith("__") and not callable(value):
                self.logging(f"{key}: {value}", is_printed=is_printed)
        self.logging("=" * 50, is_printed=is_printed)
