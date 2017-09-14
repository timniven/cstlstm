"""For handling Natural Language Inference data."""
import json
import random

from torch.utils.data import dataset, dataloader

import glovar
from cstlstm import tree_batch
from ext import NLP

LABEL_MAP = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
    "hidden": -1}
NLI_DBS = ['snli', 'mnli']
NLI_COLLS = {
    'snli': ['train', 'dev', 'test'],
    'mnli': ['train',
             'dev_matched', 'dev_mismatched',
             'test_matched', 'test_mismatched']}


def get_data_loader(data_set, batch_size):
    return dataloader.DataLoader(
        data_set,
        batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=data_set.collate)


def get_text():
    premises = []
    hypotheses = []
    for db in NLI_DBS:
        for coll in NLI_COLLS[db]:
            for x in load_json(db, coll):
                premises.append(x['sentence1'])
                hypotheses.append(x['sentence2'])
    premises = ' '.join(premises)
    hypotheses = ' '.join(hypotheses)
    nli_text = ' '.join([premises, hypotheses])
    return nli_text


def load_json(db, coll):
    filename = '%s%s/%s_%s.jsonl' % (glovar.DATA_DIR, db, db, coll)
    data = []
    with open(filename, 'r') as file:
        for line in file.readlines():
            x = json.loads(line)
            if x['gold_label'] in LABEL_MAP.keys():
                data.append(x)
    return data


class NLIDataSet(dataset.Dataset):
    def __init__(self, data, vocab_dict):
        super(NLIDataSet, self).__init__()
        self.data = data
        self.vocab_dict = vocab_dict
        self.len = len(data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

    def collate(self, batch_data):
        # Create a forest from premises and hypotheses, in order
        premises = [NLP(x['sentence1'].rstrip()) for x in batch_data]
        hypotheses = [NLP(x['sentence2'].rstrip()) for x in batch_data]
        premises = [tree_batch.sent_to_tree(x) for x in premises]
        hypotheses = [tree_batch.sent_to_tree(x) for x in hypotheses]
        forest = tree_batch.Forest(premises + hypotheses)
        # Get the labels
        forest.labels = [LABEL_MAP[x['gold_label']] for x in batch_data]
        # Pre-lookup dictionary ixs - the encoder expects an attribute vocab_ix
        for node in forest.node_list:
            node.vocab_ix = self.vocab_dict[node.token]
        return forest


class NYUDataSet(NLIDataSet):
    def __init__(self, mnli_train, snli_train, vocab_dict, alpha=0.15):
        super(NYUDataSet, self).__init__([], vocab_dict)
        self.mnli_train = mnli_train
        self.snli_train = snli_train
        self.alpha = alpha
        self.n_snli = int(len(snli_train) * alpha)
        self.data = self.sample_dataset()
        self.len = len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        if index == self.len - 1:
            self.data = self.sample_dataset()
        return item

    def __len__(self):
        return self.len

    def sample_dataset(self):
        return self.mnli_train + random.sample(self.snli_train, self.n_snli)
