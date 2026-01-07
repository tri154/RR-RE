import torch
from torch.nn.utils.rnn import pad_sequence

def move_to_cuda(
     input_ids,
     input_mask,
     entity_pos,
     hts,
     n_entities,
     n_rels,
     labels,
     labels_mask,
     device
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

            "n_entities": n_entities.cpu(),
            "n_rels": n_rels.cpu(),
        }
        label_out = {
            "labels": labels.cuda(non_blocking=True),
            "labels_mask": labels_mask.cuda(non_blocking=True)
        }
    return output, label_out


def collate_fn(batch, training):
    # CHANGED: use pad sequence for tensor.
    input_ids = [f["input_ids"] for f in batch]
    input_mask = [torch.ones(len(ts)) for ts in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    input_mask = pad_sequence(input_mask, batch_first=True, padding_value=0)

    # NOTDONE: stack this one. Should run code to compare speed.
    # No need: high chance that one is faster.
    if True:
        n_entities = list()
        entity_pos = list()
        for f in batch:
            epos = f["entity_pos"]
            n_entities.append(len(epos))
            entity_pos.extend(epos)
        entity_pos = pad_sequence(entity_pos, padding_value=-1, batch_first=True)
    else:
        entity_pos = [f["entity_pos"] for f in batch]

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
    res = torch.cat([torch.tensor([0], dtype=torch.long), res], dim=0)
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
