"""For handling the Stanford Sentiment Treebank data."""
from nltk.tokenize import sexpr
from torch.utils.data import dataset, dataloader
import numpy as np
import spacy
from ext import tree_batch, pickling
import glovar


NLP = spacy.load('en')


class Batch:
    """Batch object for this data."""

    def __init__(self, forest, labels):
        self.forest = forest
        self.labels = labels


def data():
    vocab_dict = load_vocab_dict()
    parsed = parsed_data()
    train = SSTDataset(parsed['train'], vocab_dict)
    dev = SSTDataset(parsed['dev'], vocab_dict)
    test = SSTDataset(parsed['test'], vocab_dict)
    return train, dev, test


def get_data_loader(data_set, batch_size):
    return dataloader.DataLoader(
        data_set,
        batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=data_set.collate)


def load_vocab_dict():
    return pickling.load(glovar.PKL_DIR, 'vocab_dict.pkl')


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
        with open(glovar.DATA_DIR + file) as f:
            lines = f.readlines()
            lines = [l.rstrip() for l in lines]
            data[file.split('.')[0]] = lines
    return data


class SSTDataset(dataset.Dataset):
    """Dataset wrapper for the Stanford Sentiment Treebank."""

    def __init__(self, data, vocab_dict):
        super(SSTDataset, self).__init__()
        self.data = list(data)
        self.len = len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

    def collate(self, batch_data):
        """For collating a batch of trees.

        Args:
          batch_data: List of JSON objects.

        Returns:
          Batch object wrapping the labels and forest.
        """
        # Get the labels, and convert text to SpaCy docs.
        labels = np.array([s['label'] for s in batch_data])
        sents = [NLP(s['text'] for s in batch_data)]

        # Generate the forest.
        trees = [tree_batch.sent_to_tree(s) for s in sents]
        forest = tree_batch.Forest(trees)

        # Pre-emptively perform dictionary lookup to save time.
        for level in range(forest.max_level + 1):
            for node in forest.nodes[level]:
                node.vocab_ix = self.vocab_dict[node.token]

        # Wrap up a batch object and return.
        return Batch(forest, labels)


class Stack:
    # Internal utility class for parsing sexprs in the correct order.

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
