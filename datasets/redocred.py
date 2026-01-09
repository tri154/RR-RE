import os
import os.path as path
import numpy as np
from collections import defaultdict
import tarfile
import torch

from utils import load_json, load_cache, save_cache, dist_log
log_info = lambda dump, **func: dump

class ReDocRED:
    name: str
    path: str
    sets: dict
    num_class: int
    max_seq_length: int
    cached_location: str
    rel2id: str

    def __init__(self, dataset_cfg):
        global log_info; log_info = dist_log(__name__)
        for name in self.__class__.__annotations__: # only update defined annotations.
            setattr(self, name, dataset_cfg.get(name))
        self.unzip_data()
        self.rel2id = load_json(self.rel2id)


    def __read_docred(self, file_in, tokenizer, max_seq_length=1024, max_docs=None):
        len_freq = defaultdict(int)
        i_line = 0
        pos_samples = 0
        neg_samples = 0
        rel_nums = 0
        features = []
        max_n_rels = -1
        # if file_in == "":
        #     return None
        data = load_json(file_in)

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
                relations.append(sorted(relation))
                assert relations[-1][0] != 0
                hts.append([h, t])
                pos_samples += 1

            for h in range(len(entities)):
                for t in range(len(entities)):
                    if h != t and [h, t] not in hts:
                        # relation = [1] + [0] * (len(self.rel2id) - 1)
                        relation = [0]
                        relations.append(relation)
                        hts.append([h, t])
                        neg_samples += 1

            assert len(relations) == len(entities) * (len(entities) - 1)
            max_n_rels = max(max([len(rel) for rel in relations]), max_n_rels)

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

        log_info("# of documents {}.".format(i_line))
        log_info("# of positive examples {}.".format(pos_samples))
        log_info("# of negative examples {}.".format(neg_samples))
        re_fre = 1. * re_fre / (pos_samples + neg_samples)
        # log("# rels per doc".format(1. * rel_nums / i_line))
        log_info(f"Max seq len: {max(list(len_freq.keys()))} .")
        log_info(f"max number of rels: {max_n_rels} .")

        # CHANGED: padding labels here. padding_value.
        for feature in features:
            relations = feature["labels"]
            temp = list()
            mask = list()
            for rel in relations:
                temp.append(rel + [0] * (max_n_rels - len(rel)))
                mask.append([False] * len(rel) + [True] * (max_n_rels - len(rel)))
            relations = torch.tensor(temp, dtype=torch.long)
            mask = torch.tensor(mask, dtype=torch.bool)
            feature["labels"] = relations
            feature["labels_mask"] = mask

        return features, re_fre, len_freq

    def unzip_data(self):
        if not os.path.exists(self.sets.train):
            with tarfile.open(path.join(self.path, "data.tar.gz"), "r:gz") as tar:
                tar.extractall(path=self.path)
            log_info("No json files found, extract data.")


    def get_features(self, tokenizer, load_cached):
        if load_cached:
            if path.exists(self.cached_location):
                log_info("use cached dataset.", log_all=True)
                return load_cache(self.cached_location)
            else:
                raise Exception("run 'python prepare_data.py' first.")

        res = dict()

        self.unzip_data()

        for k, file_path in self.sets.items():
            log_info(f"{k} stats: ")
            features, re_fre, len_freq  = self.__read_docred(file_path, tokenizer, self.max_seq_length)
            res[k] = features
        save_cache(res, self.cached_location)
        log_info("dataset cached.")
        return res
