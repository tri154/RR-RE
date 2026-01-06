import os.path as path
import json
import numpy as np
import pickle as pkl
from collections import defaultdict
import torch
import torch.functional as F
from torch.nn.utils.rnn import pad_sequence

import logging
log = logging.getLogger(__name__)

def _load_json(p):
    with open(p, 'r') as file:
        data = json.load(file)
    return data


def _load_cache(path):
    with open(path, 'rb') as file:
        loadded_data = pkl.load(file)
    return loadded_data


def _save_cache(data, path):
    with open(path, 'wb') as file:
        pkl.dump(data, file)


class ReDocRED:
    name: str
    sets: dict
    num_class: int
    max_seq_length: int
    cached_location: str
    rel2id: str

    def __init__(self, dataset_cfg):
        super().__init__()
        for name in self.__class__.__annotations__: # only update defined annotations.
            setattr(self, name, dataset_cfg.get(name))
        self.rel2id = _load_json(self.rel2id)


    def __read_docred(self, file_in, tokenizer, max_seq_length=1024, max_docs=None):
        len_freq = defaultdict(int)
        i_line = 0
        pos_samples = 0
        neg_samples = 0
        rel_nums = 0
        features = []
        # if file_in == "":
        #     return None
        with open(file_in, "r") as fh:
            data = json.load(fh)

        if max_docs is not None:
            data = data[:max_docs]

        re_fre = np.zeros(len(self.rel2id) - 1)
        for idx, sample in enumerate(data):
            sents = []
            sent_map = []

            entities = sample['vertexSet']
            entity_start, entity_end = [], []
            for entity in entities:
                for mention in entity:
                    sent_id = mention["sent_id"]
                    pos = mention["pos"]
                    entity_start.append((sent_id, pos[0],))
                    entity_end.append((sent_id, pos[1] - 1,))
            for i_s, sent in enumerate(sample['sents']):
                new_map = {}
                for i_t, token in enumerate(sent):
                    tokens_wordpiece = tokenizer.tokenize(token)

                    if (i_s, i_t) in entity_start:
                        tokens_wordpiece = ["*"] + tokens_wordpiece
                    if (i_s, i_t) in entity_end:
                        tokens_wordpiece = tokens_wordpiece + ["*"]

                    new_map[i_t] = len(sents)
                    sents.extend(tokens_wordpiece)
                new_map[i_t + 1] = len(sents)
                sent_map.append(new_map)

            train_triple = {}
            if "labels" in sample:
                for label in sample['labels']:
                    if 'evidence' not in label:
                        evidence = []
                    else:
                        evidence = label['evidence']
                    r = int(self.rel2id[label['r']])
                    re_fre[r - 1] += 1
                    if (label['h'], label['t']) not in train_triple:
                        train_triple[(label['h'], label['t'])] = [{'relation': r, 'evidence': evidence}]
                    else:
                        train_triple[(label['h'], label['t'])].append({'relation': r, 'evidence': evidence})

            entity_pos = []
            for e in entities:
                entity_pos.append([])
                for m in e:
                    start = sent_map[m["sent_id"]][m["pos"][0]]
                    end = sent_map[m["sent_id"]][m["pos"][1]]
                    # entity_pos[-1].append((start, end,))
                    # CHANGED: only take start of mention.
                    entity_pos[-1].append(start)

            # CHANGED: replace one-hot encoding to save memory.
            relations, hts = [], []
            for h, t in train_triple.keys():
                # relation = [0] * len(self.rel2id)
                relation = []
                for mention in train_triple[h, t]:
                    # relation[mention["relation"]] = 1
                    relation.append(mention["relation"])
                    evidence = mention["evidence"]
                    rel_nums += 1
                relations.append(torch.tensor(relation, dtype=torch.long))
                hts.append([h, t])
                pos_samples += 1

            for h in range(len(entities)):
                for t in range(len(entities)):
                    if h != t and [h, t] not in hts:
                        # relation = [1] + [0] * (len(self.rel2id) - 1)
                        relation = torch.tensor([0], dtype=torch.long)
                        relations.append(relation)
                        hts.append([h, t])
                        neg_samples += 1

            assert len(relations) == len(entities) * (len(entities) - 1)

            len_freq[len(sents)] += 1
            sents = sents[:max_seq_length - 2]
            input_ids = tokenizer.convert_tokens_to_ids(sents)
            input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
            # NOTE: adding sos token and eos token, making entity pos are off set of 1,
            # should check the behaviour of different tokenizers.

            # CHANGED: to tensor to save time at collate fn.
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            entity_pos = [torch.tensor(ments, dtype=torch.long) + 1 for ments in entity_pos]
            hts = torch.tensor(hts, dtype=torch.long)

            i_line += 1
            feature = {'input_ids': input_ids,
                        'entity_pos': entity_pos,
                        'labels': relations,
                        'hts': hts,
                        'title': sample['title']}
            features.append(feature)

        log.info("# of documents {}.".format(i_line))
        log.info("# of positive examples {}.".format(pos_samples))
        log.info("# of negative examples {}.".format(neg_samples))
        re_fre = 1. * re_fre / (pos_samples + neg_samples)
        # log("# rels per doc".format(1. * rel_nums / i_line))
        log.info(f"Max seq len: {max(list(len_freq.keys()))} .")
        return features, re_fre, len_freq

    def get_features(self, tokenizer):
        if path.exists(self.cached_location):
            log.info("use cached dataset.")
            return _load_cache(self.cached_location)

        res = dict()
        for k, file_path in self.sets.items():
            log.info(f"{k} stats: ")
            features, re_fre, len_freq  = self.__read_docred(file_path, tokenizer, self.max_seq_length)
            res[k] = features
        _save_cache(res, self.cached_location)
        log.info("dataset cached.")
        return res

    def collate_fn(self, batch, training):
        # CHANGED: use pad sequence for tensor.
        input_ids = [f["input_ids"] for f in batch]
        input_mask = [torch.ones(len(ts)) for ts in input_ids]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        input_mask = pad_sequence(input_mask, batch_first=True, padding_value=0)

        # TODO: stack this one. Should run code to compare speed.
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

        input_ids = input_ids.long()
        input_mask = input_mask.bool()
        entity_pos = entity_pos.long()
        hts = [f["hts"] for f in batch]
        labels = [f["labels"] for f in batch]

        output = {
            "input_ids": input_ids,
            "input_mask": input_mask,
            "entity_pos": entity_pos,
            "hts": hts,
            "n_entities": n_entities,
        }
        # CHANGED: move labels out of batch.
        return output, labels

    def old_collate_fn(self, batch, training):
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
