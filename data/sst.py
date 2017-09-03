"""For handling the Stanford Sentiment Treebank data."""
from nltk.tokenize import sexpr
from torch.utils.data import dataset, dataloader
import numpy as np
import spacy
from ext import tree_batch


DATA_DIR = 'D:\\dev\\data\\sst\\'
NLP = spacy.load('en')


def collate(batch_data):
    labels = np.array([s['label'] for s in batch_data])
    sents = [NLP(s['text'] for s in batch_data)]
    trees = [tree_batch.sent_to_tree(s) for s in sents]
    forest = tree_batch.Forest(trees)
    # need to create emb_mat and voc_dic
    # then I can define on a node in tree-Batch a vocab_ix
    # and do the dictionary lookup here
    # might save a smidgeon of time during training
    return forest, labels


def data():
    parsed = parsed_data()
    train = SSTDataset(parsed['train'])
    dev = SSTDataset(parsed['dev'])
    test = SSTDataset(parsed['test'])
    return train, dev, test


def get_data_loader(data_set, batch_size):
    return dataloader.DataLoader(
        data_set,
        batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate)


def parse(root_sexpr):
    label, sub_sexpr = root_sexpr[1:-1].split(None, 1)
    tokens = []
    stack = Stack()
    for sub_sexpr in reversed(sexpr.sexpr_tokenize(sub_sexpr)):
        stack.push(sub_sexpr)
    while not stack.empty:
        _, next_sexpr = stack.pop()[1:-1].split(None, 1)
        # Leaf: if the length of the next is 1 and the string isn't in brackets
        next_sexprs = sexpr.sexpr_tokenize(next_sexpr)
        if len(next_sexprs) == 1 and ('(' not in next_sexprs[0]
                                     and ')' not in next_sexprs):
            tokens.append(next_sexprs[0])
        # Otherwise, add them to the stack in reverse order
        else:
            for sub_sexpr in reversed(next_sexprs):
                stack.push(sub_sexpr)
    return label, ' '.join(tokens)


def parsed_data():
    data = raw_data()
    parsed = {}
    for key in data.keys():
        labels_texts = []
        for root_sexpr in data[key]:
            label, text = parse(root_sexpr)
            labels_texts.append({'label': label, 'text': text})
        parsed[key] = labels_texts
    return parsed


def raw_data():
    data = {}
    files = ['train.txt', 'dev.txt', 'test.txt']
    for file in files:
        with open(DATA_DIR + file) as f:
            lines = f.readlines()
            lines = [l.rstrip() for l in lines]
            data[file.split('.')[0]] = lines
    return data


class SSTDataset(dataset.Dataset):
    def __init__(self, data):
        super(SSTDataset, self).__init__()
        self.data = list(data)
        self.len = len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class Stack:
    def __init__(self):
        self.items = []

    @property
    def empty(self):
        return len(self.items) == 0

    def push(self, item):
        self.items.append(item)

    def pop(self):
        item = self.items[-1]
        del self.items[-1]
        return item
