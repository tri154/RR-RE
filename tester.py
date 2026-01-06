from torch.utils.data import DataLoader
import torch


def move_to_device(batch, device):
    if device == 'cpu':
        return batch
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.cuda(non_blocking=True)
    return batch

class Tester:
    batch_size: int

    def __init__(self, tester_cfg, *, dev_features, test_features, test_collate_fn):
        for name in self.__class__.__annotations__: # only update defined annotations.
            setattr(self, name, tester_cfg.get(name))

        self.test_feautres = test_features
        self.test_collate_fn = test_collate_fn
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.dev_loader = self.get_test_dataloader(dev_features)
        self.test_loader = self.get_test_dataloader(test_features)

    def get_test_dataloader(self, features):
        return DataLoader(
            features,
            shuffle=False,
            drop_last=False,
            batch_size=self.batch_size,
            collate_fn=self.test_collate_fn,
            pin_memory=self.device == 'cuda',
        )

    def test(self, model, *, tag):
        assert tag in ["dev", "test"]
        loader = self.dev_loader if tag == "dev" else self.test_loader
        model.eval()
        preds = []
        labels = []
        for batch_input, batch_label in loader:
            batch_input = move_to_device(batch_input, self.device)
            # batch_label = move_to_device(batch_label, self.device)
            # TODO: complete later.
            logits = model(**batch_input)

        # ...
        return 1
