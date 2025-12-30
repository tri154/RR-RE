from common import DatasetHandler
import os.path as path
import json

def load_json(p):
    with open(p, 'r') as file:
        data = json.load(file)
    return data

class DocRED(DatasetHandler):
    def __init__(self, **dataset_kwargs):
        super().__init__()
        self.__dict__.update(dataset_kwargs)
        self.sets = {k: path.join("data/docred", self.sets[k]) for k in self.sets.keys()}

    def get_features(self, tokenizer, ):
        # need tokenizer
        pass
