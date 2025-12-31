from common import DatasetHandler, logging

import os.path as path
import json
import numpy as np
import pickle as pkl


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



class ReDocRED(DatasetHandler):
    name: str
    sets: dict
    num_class: int
    max_seq_length: int
    cached_location: str
    result_path: str
    log_path: str
    device: str

    def __init__(self, **dataset_kwargs):
        super().__init__()
        self.__dict__.update(dataset_kwargs)
        self.sets = {k: path.join("data/redocred", self.sets[k]) for k in self.sets.keys()}
        self.rel2id = _load_json("data/redocred/rel2id.json")
        self.rel_info = _load_json("data/redocred/rel_info.json")
        self.cached_location = path.join(self.result_path, "cached.pkl")

    def read_docred(self, file_in, tokenizer, max_seq_length=1024, max_docs=None):
        from collections import defaultdict
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
                    entity_pos[-1].append((start, end,))


            relations, hts = [], []
            for h, t in train_triple.keys():
                relation = [0] * len(self.rel2id)
                for mention in train_triple[h, t]:
                    relation[mention["relation"]] = 1
                    evidence = mention["evidence"]
                    rel_nums += 1
                relations.append(relation)
                hts.append([h, t])
                pos_samples += 1

            for h in range(len(entities)):
                for t in range(len(entities)):
                    if h != t and [h, t] not in hts:
                        relation = [1] + [0] * (len(self.rel2id) - 1)
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

            entity_pos = [(np.array(ments) + 1).tolist() for ments in entity_pos]

            i_line += 1
            feature = {'input_ids': input_ids,
                        'entity_pos': entity_pos,
                        'labels': relations,
                        'hts': hts,
                        'title': sample['title']}
            features.append(feature)

        logging("# of documents {}.".format(i_line), self.log_path, is_printed=True)
        logging("# of positive examples {}.".format(pos_samples), self.log_path, is_printed=True)
        logging("# of negative examples {}.".format(neg_samples), self.log_path, is_printed=True)
        re_fre = 1. * re_fre / (pos_samples + neg_samples)
        # logging("# rels per doc".format(1. * rel_nums / i_line), self.log_path, is_printed=True)
        logging(f"Max seq len: {max(list(len_freq.keys()))} .", self.log_path, is_printed=True)
        return features, re_fre, len_freq

    def get_features(self, tokenizer):
        if path.exists(self.cached_location):
            logging("use cached dataset.", self.log_path, is_printed=True)
            return _load_cache(self.cached_location)

        res = dict()
        for k, file_path in self.sets.items():
            logging(f"{k} stats: ", self.log_path, is_printed=True)
            features, re_fre, len_freq  = self.read_docred(file_path, tokenizer, self.max_seq_length)
            res[k] = features
        _save_cache(res, self.cached_location)
        logging("dataset cached.", self.log_path, is_printed=True)
        return res
