from torch.utils.data import DataLoader
from typing import Any
import torch
import numpy as np
import os
import json
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from utils import move_to_cuda, load_json, get_dist_info, np_split

RANK = 0
WORLD_SIZE = 1


class Tester:
    batch_size: int
    rel2id: str
    data_dir: str
    sets: Any

    def __init__(self, tester_cfg, *, dev_features, test_features, test_collate_fn):
        global RANK, WORLD_SIZE; RANK, WORLD_SIZE = get_dist_info()
        for name in self.__class__.__annotations__: # only update defined annotations.
            setattr(self, name, tester_cfg.get(name))

        self.rel2id = load_json(self.rel2id)
        self.id2rel = {v:k for k,v in self.rel2id.items()}

        self.dev_features = dev_features
        self.test_features = test_features
        self.test_collate_fn = test_collate_fn
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.dev_loader = self.get_test_dataloader(dev_features)
        self.test_loader = self.get_test_dataloader(test_features)

    def get_test_dataloader(self, features):
        sampler = None
        if WORLD_SIZE > 1:
            sampler=DistributedSampler(
                features,
                shuffle=False,
                drop_last=False
            )
        return DataLoader(
            features,
            shuffle=False,
            drop_last=False,
            batch_size=self.batch_size,
            collate_fn=self.test_collate_fn,
            pin_memory=self.device == 'cuda',
            sampler=sampler
        )

    def to_official(self, preds, features):
        # NOTE:
        # - modified h_idx, t_idx to be suitable with feature as tensor.
        h_idx, t_idx, title = [], [], []
        for f in features:
            hts = f["hts"]
            h_idx.append(hts[:, 0])
            t_idx.append(hts[:, 1])
            # h_idx += [ht[0] for ht in hts]
            # t_idx += [ht[1] for ht in hts]
            title += [f["title"] for ht in hts]
        h_idx = torch.cat(h_idx, dim=0).tolist()
        t_idx = torch.cat(t_idx, dim=0).tolist()

        res = []
        for i in range(preds.shape[0]):
            pred = preds[i]
            pred = np.nonzero(pred)[0].tolist()
            for p in pred:
                if p != 0:
                    res.append(
                        {
                            'title': title[i],
                            'h_idx': h_idx[i],
                            't_idx': t_idx[i],
                            'r': self.id2rel[p],
                        }
                    )
        return res


    def gen_train_facts(self, data_file_name, truth_dir):
        fact_file_name = data_file_name[data_file_name.find("train_"):]
        fact_file_name = os.path.join(truth_dir, fact_file_name.replace(".json", ".fact"))

        if os.path.exists(fact_file_name):
            fact_in_train = set([])
            triples = json.load(open(fact_file_name))
            for x in triples:
                fact_in_train.add(tuple(x))
            return fact_in_train

        fact_in_train = set([])
        ori_data = json.load(open(data_file_name))
        for data in ori_data:
            vertexSet = data['vertexSet']
            for label in data['labels']:
                rel = label['r']
                for n1 in vertexSet[label['h']]:
                    for n2 in vertexSet[label['t']]:
                        fact_in_train.add((n1['name'], n2['name'], rel))

        json.dump(list(fact_in_train), open(fact_file_name, "w"))

        return fact_in_train



    def official_evaluate(self, tmp, path, tag):
        '''
            Adapted from the official evaluation code
        '''
        truth_dir = os.path.join(path, 'ref')

        if not os.path.exists(truth_dir):
            os.makedirs(truth_dir)

        fact_in_train_annotated = self.gen_train_facts(self.sets.train, truth_dir)
        fact_in_train_distant = self.gen_train_facts(os.path.join(path, "train_distant.json"), truth_dir)

        if tag == 'dev':
            truth = load_json(self.sets.dev)
        # elif tag == 'testtop10':
        #     truth = json.load(open(os.path.join(path, args.test_file_top10)))
        # elif tag == 'testbottom90':
        #     truth = json.load(open(os.path.join(path, args.test_file_bottom90)))
        else:
            truth = load_json(self.sets.test)

        std = {}
        tot_evidences = 0
        titleset = set([])

        title2vectexSet = {}
        for x in truth:
            title = x['title']
            titleset.add(title)

            vertexSet = x['vertexSet']
            title2vectexSet[title] = vertexSet

            for label in x['labels']:

                # if tag == 'testtop10':
                #     if label['r'] not in top10:
                #         continue
                # elif tag == 'testbottom90':
                #     if label['r'] in top10:
                #         continue

                r = label['r']
                h_idx = label['h']
                t_idx = label['t']
                std[(title, r, h_idx, t_idx)] = set([1])
                tot_evidences += len([1])

        tot_relations = len(std)
        tmp.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], x['r']))
        submission_answer = [tmp[0]]
        for i in range(1, len(tmp)):
            x = tmp[i]
            y = tmp[i - 1]
            if (x['title'], x['h_idx'], x['t_idx'], x['r']) != (y['title'], y['h_idx'], y['t_idx'], y['r']):
                submission_answer.append(tmp[i])

        correct_re = 0
        correct_evidence = 0
        pred_evi = 0

        correct_in_train_annotated = 0
        correct_in_train_distant = 0
        titleset2 = set([])
        for x in submission_answer:
            title = x['title']
            h_idx = x['h_idx']
            t_idx = x['t_idx']
            r = x['r']

            # if tag == 'testtop10':
            #     if r not in top10:
            #         continue
            # elif tag == 'testbottom90':
            #     if r in top10:
            #         continue

            titleset2.add(title)
            if title not in title2vectexSet:
                continue
            vertexSet = title2vectexSet[title]

            if 'evidence' in x:
                evi = set(x['evidence'])
            else:
                evi = set([])
            pred_evi += len(evi)

            if (title, r, h_idx, t_idx) in std:
                correct_re += 1
                stdevi = std[(title, r, h_idx, t_idx)]
                correct_evidence += len(stdevi & evi)
                in_train_annotated = in_train_distant = False
                for n1 in vertexSet[h_idx]:
                    for n2 in vertexSet[t_idx]:
                        if (n1['name'], n2['name'], r) in fact_in_train_annotated:
                            in_train_annotated = True
                        if (n1['name'], n2['name'], r) in fact_in_train_distant:
                            in_train_distant = True

                if in_train_annotated:
                    correct_in_train_annotated += 1
                if in_train_distant:
                    correct_in_train_distant += 1

        re_p = 1.0 * correct_re / len(submission_answer)
        re_r = 1.0 * correct_re / tot_relations
        if re_p + re_r == 0:
            re_f1 = 0
        else:
            re_f1 = 2.0 * re_p * re_r / (re_p + re_r)

        evi_p = 1.0 * correct_evidence / pred_evi if pred_evi > 0 else 0
        evi_r = 1.0 * correct_evidence / tot_evidences
        if evi_p + evi_r == 0:
            evi_f1 = 0
        else:
            evi_f1 = 2.0 * evi_p * evi_r / (evi_p + evi_r)

        re_p_ignore_train_annotated = 1.0 * (correct_re - correct_in_train_annotated) / (len(submission_answer) - correct_in_train_annotated + 1e-5)
        re_p_ignore_train = 1.0 * (correct_re - correct_in_train_distant) / (len(submission_answer) - correct_in_train_distant + 1e-5)

        if re_p_ignore_train_annotated + re_r == 0:
            re_f1_ignore_train_annotated = 0
        else:
            re_f1_ignore_train_annotated = 2.0 * re_p_ignore_train_annotated * re_r / (re_p_ignore_train_annotated + re_r)

        if re_p_ignore_train + re_r == 0:
            re_f1_ignore_train = 0
        else:
            re_f1_ignore_train = 2.0 * re_p_ignore_train * re_r / (re_p_ignore_train + re_r)

        return re_f1, evi_f1, re_f1_ignore_train_annotated, re_f1_ignore_train, re_p, re_r


    def test(self, model, *, tag):
        assert tag in ["dev", "test"]
        loader = self.dev_loader if tag == "dev" else self.test_loader
        features = self.dev_features if tag == "dev" else self.test_features

        model.eval()

        local_preds = []
        local_ids = []
        for batch_input, batch_ids in loader:
            batch_input = move_to_cuda(
                **batch_input,
                device=self.device
            )
            with torch.no_grad():
                batch_preds, n_rels_per_batch = model(**batch_input)
                batch_preds = batch_preds.cpu().numpy()
                n_rels_per_batch = n_rels_per_batch.cpu().numpy()

                batch_preds = np_split(batch_preds, n_rels_per_batch)
                batch_ids = np.array(batch_ids)
            local_preds.extend(batch_preds)
            local_ids.append(batch_ids)

        local_ids = np.concatenate(local_ids, axis=0).astype(int)
        if WORLD_SIZE > 1:
            obj = (local_ids, local_preds)
            if RANK == 0:
                all_objs = [None] * WORLD_SIZE
            else:
                all_objs = None
            dist.gather_object(
                obj=obj,
                object_gather_list=all_objs,
                dst=0
            )
        else:
            all_objs = [(local_ids, local_preds)]

        if RANK == 0:
            all_ids = []
            all_preds = []
            for l_ids, l_preds in all_objs:
                all_ids.append(l_ids)
                all_preds.extend(l_preds)

            all_ids = np.concatenate(all_ids, axis=0)
            assert len(all_ids) == len(all_preds)

            order = all_ids.argsort()
            all_preds = [all_preds[i] for i in order]

            all_preds = np.concatenate(all_preds, axis=0).astype(np.float32)
            ans = self.to_official(all_preds, features)
            if len(ans) > 0:
                best_f1, _, best_f1_ign, re_f1_ignore_train, re_p, re_r = \
                    self.official_evaluate(ans, self.data_dir, tag)
                output = {
                    tag + "_F1": best_f1 * 100,
                    tag + "_F1_ign": best_f1_ign * 100,
                    tag + "_precison": re_p * 100,
                    tag + "_recall": re_r * 100,
                }
            else:
                best_f1, best_f1_ign = -1, -1
                output = {
                    tag + "_F1": best_f1 * 100,
                    tag + "_F1_ign": best_f1_ign * 100,
                }
            return best_f1, output
