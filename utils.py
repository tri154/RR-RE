import torch
from torch.nn.utils.rnn import pad_sequence
import time
import torch
from contextlib import contextmanager
import random
import numpy
import pickle as pkl
import json
import wandb
import omegaconf
import torch.distributed as dist
import os
from omegaconf import OmegaConf
from functools import partial

import logging

def move_to_cuda(
     input_ids,
     input_mask,
     entity_pos,
     hts,
     n_entities,
     n_rels,
     device,
     labels=None,
     labels_mask=None,
):
    if device == 'cpu':
        output = {
            "input_ids": input_ids,
            "input_mask": input_mask,
            "entity_pos": entity_pos,
            "hts": hts,

            "n_entities": n_entities,
            "n_rels": n_rels,
        }
        if labels is not None:
            label_out = {
                "labels": labels,
                "labels_mask": labels_mask,
            }
    else:
        output = {
            "input_ids": input_ids.cuda(non_blocking=True),
            "input_mask": input_mask.cuda(non_blocking=True),
            "entity_pos": entity_pos.cuda(non_blocking=True),
            "hts": hts.cuda(non_blocking=True),
            "n_entities": n_entities.cuda(non_blocking=True),
            "n_rels": n_rels.cuda(non_blocking=True),
        }
        if labels is not None:
            label_out = {
                "labels": labels.cuda(non_blocking=True),
                "labels_mask": labels_mask.cuda(non_blocking=True)
            }
    if labels is not None:
        return output, label_out
    else:
        return output


def collate_fn(batch, training):
    # CHANGED: use pad sequence for tensor.
    input_ids = [f["input_ids"] for f in batch]
    input_mask = [torch.ones(len(ts)) for ts in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    input_mask = pad_sequence(input_mask, batch_first=True, padding_value=0)

    # stacking
    n_entities = list()
    entity_pos = list()
    for f in batch:
        epos = f["entity_pos"]
        n_entities.append(len(epos))
        entity_pos.extend(epos)
    entity_pos = pad_sequence(entity_pos, padding_value=-1, batch_first=True)

    hts = [f["hts"] for f in batch]
    n_rels = [len(ts) for ts in hts]
    hts = torch.cat(hts, dim=0)


    labels = [f["labels"] for f in batch]
    labels = torch.cat(labels, dim=0)
    labels_mask = [f["labels_mask"] for f in batch]
    labels_mask = torch.cat(labels_mask, dim=0)

    input_ids = input_ids.long()
    input_mask = input_mask.bool()
    entity_pos = entity_pos.long()
    hts = hts.long()
    n_entities = torch.tensor(n_entities, dtype=torch.long)
    n_rels = torch.tensor(n_rels, dtype=torch.long)

    labels = labels.long()
    labels_mask = labels_mask.bool()

    output = {
        "input_ids": input_ids,
        "input_mask": input_mask,
        "entity_pos": entity_pos,
        "hts": hts,

        "n_entities": n_entities,
        "n_rels": n_rels,
    }
    # CHANGED: move labels out of batch.
    label_out = {
        "labels": labels,
        "labels_mask": labels_mask
    }
    return output, label_out



def cumsum_with_zero(x, exclude_last=False):
    """ Expect x as 1D tensor"""
    res = torch.cumsum(x, dim=0)
    res = torch.cat([torch.tensor([0], dtype=torch.long, device=res.device), res], dim=0)
    if exclude_last:
        res = res[:-1]
    return res


def old_collate_fn(batch, training):
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] for f in batch]
    entity_pos = [f["entity_pos"] for f in batch]
    hts = [f["hts"] for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    output = (input_ids, input_mask, labels, entity_pos, hts)
    return output


@contextmanager
def cuda_sync():
    """Ensure accurate timing on CUDA."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark(
    funcs,
    iters=100,
    warmup=10,
    device="cuda",
    measure_memory=True,
    reduce="mean",  # "mean" or "median"
):
    """
    funcs: list of callables, each callable is fn()
    iters: number of benchmark iterations
    warmup: warmup iterations (not measured)
    measure_memory: track max CUDA memory allocated
    reduce: mean or median time
    """

    assert reduce in ("mean", "median")

    results = []

    for fn in funcs:
        # Warmup
        for _ in range(warmup):
            fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        times = []
        mems = []

        for _ in range(iters):
            if measure_memory and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            with cuda_sync():
                start = time.perf_counter()
                fn()
                end = time.perf_counter()

            times.append(end - start)

            if measure_memory and torch.cuda.is_available():
                mems.append(torch.cuda.max_memory_allocated())

        times_t = torch.tensor(times)

        if reduce == "mean":
            time_stat = times_t.mean().item()
        else:
            time_stat = times_t.median().item()

        result = {
            "fn": fn.__name__,
            "time_sec": time_stat,
            "time_us": time_stat * 1e6,
        }

        if measure_memory and mems:
            result["max_mem_MB"] = max(mems) / (1024 ** 2)

        results.append(result)

    # Pretty print
    print("\nBenchmark results:")
    for r in results:
        mem = f", mem={r['max_mem_MB']:.2f} MB" if "max_mem_MB" in r else ""
        print(f"{r['fn']:<30}: {r['time_us']:.2f} µs{mem}")

    breakpoint()
    return results


def check_tensor(
    tensor: torch.Tensor,
    name: str,
    logger: logging.Logger,
    raise_on_error: bool = False,
) -> bool:
    """
    Check a tensor for NaN / Inf values and log using the caller's logger.

    Args:
        tensor: torch.Tensor to check
        name: human-readable tensor name
        logger: logger from calling file (logging.getLogger(__name__))
        raise_on_error: raise RuntimeError if invalid values found

    Returns:
        True if tensor is clean, False otherwise
    """

    if tensor is None:
        logger.error(f"{name} is None")
        if raise_on_error:
            raise RuntimeError(f"{name} is None")
        return False

    if not torch.is_tensor(tensor):
        logger.error(f"{name} is not a torch.Tensor (type={type(tensor)})")
        if raise_on_error:
            raise RuntimeError(f"{name} is not a torch.Tensor")
        return False

    if tensor.numel() == 0:
        logger.error(f"{name} is empty (numel=0)")
        if raise_on_error:
            raise RuntimeError(f"{name} is empty")
        return False

    # Fast checks (GPU-safe)
    has_nan = torch.isnan(tensor).any()
    has_inf = torch.isinf(tensor).any()

    if not (has_nan or has_inf):
        # logger.info(
        #     f"{name} OK | shape={tuple(tensor.shape)} "
        #     f"dtype={tensor.dtype} device={tensor.device}"
        # )
        return True

    # Only sync if needed
    has_nan = has_nan.item()
    has_inf = has_inf.item()

    msg = (
        f"{name} INVALID | "
        f"NaN={has_nan}, Inf={has_inf}, "
        f"shape={tuple(tensor.shape)}, "
        f"dtype={tensor.dtype}, "
        f"device={tensor.device}"
    )

    logger.warning(msg)

    if raise_on_error:
        raise RuntimeError(msg)

    return False


def seeding(seed, hard=False, rank=0):
    seed = seed + rank
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if hard:
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def load_json(p):
    with open(p, 'r') as file:
        data = json.load(file)
    return data

def load_cache(path):
    with open(path, 'rb') as file:
        loadded_data = pkl.load(file)
    return loadded_data

def save_cache(data, path):
    with open(path, 'wb') as file:
        pkl.dump(data, file)


def init_wandb(cfg):
    run = wandb.init(
        entity=cfg.wandb.entity,
        # Set the wandb project where this run will be logged.
        project=cfg.wandb.project,
        # Track hyperparameters and run metadata.
        config = omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        )
    )
    return run


def init_dist():
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(
            backend="nccl",
            init_method="env://"
        )
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

        return rank, world_size

    return 0, 1


def load_synced_config(cfg, rank, world_size):
    objects = [None]
    if rank == 0:
        OmegaConf.resolve(cfg)
        objects = [cfg]

    if world_size > 1:
        dist.broadcast_object_list(objects, src=0)

    return objects[0]

def get_dist_info():
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size

def log_info(msg, logger, rank, world_size, log_all=False):
    if log_all and world_size > 1:
        msg = f"[Rank {rank}, World Size {world_size}]: " + msg
        logger.info(msg)
        return
    if rank == 0:
        logger.info(msg)

def dist_log(log_name):
    logger = logging.getLogger(log_name)
    rank, world_size = get_dist_info()
    return partial(log_info, logger=logger, rank=rank, world_size=world_size)
