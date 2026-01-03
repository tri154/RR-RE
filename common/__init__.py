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

# other schedulers follow this signature.
def linear_scheduler_warmup_lambdalr(current_step, *, num_warmup_steps, num_training_steps):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

OPTIMIZERS = {"Adam": Adam, "AdamW": AdamW}
SCHEDULERS = {"Linear": linear_scheduler_warmup_lambdalr}


from .interfaces import DatasetHandler, BaseTrainer
from .config import Config
from .encoder import Encoder
