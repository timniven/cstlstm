"""For handling the Stanford Sentiment Treebank data."""
from nltk.tokenize import sexpr
from torch.utils.data import dataset, dataloader
import numpy as np
import spacy
from ext import tree_batch, pickling
import glovar
import os


NLP = spacy.load('en')
SST_DIR = os.path.join(glovar.DATA_DIR, 'sst/')


def annotate_data():
    raw_data = get_raw_data()
    parsed_data = get_parsed_data(raw_data)
    # combining text at nodes occurs within these functions
    sst_trees = get_sst_trees(raw_data)
    dep_trees = get_dep_trees(parsed_data)
    # use compare and annotate to annotate
    for dataset in dep_trees.keys():
        dep_set = dep_trees[dataset]
        sst_set = sst_trees[dataset]
        for i in range(len(dep_set)):
            compare_and_annotate(sst_set[i], dep_set[i])
            # report every so often to check integrity
            if i % 100 == 0:
                print('ORIGINAL')
                for node in sst_set[i].node_list:
                    print('%s\t%s\t%s' % (node.id, node.tag, node.text_at_node))
                print('DEP')
                for node in dep_set[i].node_list:
                    print('%s\t%s\t%s' % (
                        node.id, node.annotation, node.text_at_node))
    # save a pickle
    pickling.save(dep_trees, glovar.PKL_DIR, 'annotated_dep_trees.pkl')
    return dep_trees


def compare_and_annotate(sst_tree, dep_tree):
    for dep_node in dep_tree.node_list:
        # init an empty property as all nodes need one
        dep_node.annotation = None
        dep_doc = NLP(dep_node.text_at_node)
        # check for a match in the sst tree
        for sst_node in sst_tree.node_list:
            sst_doc = NLP(sst_node.text_at_node)
            if len(dep_doc) == len(sst_doc):
                match = True
                i = 0
                while match and i <= len(dep_doc) - 1:
                    match = dep_doc[i].text == sst_doc[i].text
                    i += 1
                if match:
                    dep_node.annotation = sst_node.tag


def get_data():
    vocab_dict = load_vocab_dict()
    data = pickling.load(glovar.PKL_DIR, 'annotated_dep_trees.pkl')
    train = SSTDataset(data['train'], vocab_dict)
    dev = SSTDataset(data['dev'], vocab_dict)
    test = SSTDataset(data['test'], vocab_dict)
    return train, dev, test


def get_data_loader(data_set, batch_size):
    return dataloader.DataLoader(
        data_set,
        batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=data_set.collate)


def get_dep_trees(parsed_data):
    dep_trees = {}
    for dataset in parsed_data.keys():
        trees = []
        for sample in parsed_data[dataset]:
            sent = NLP(sample['text'])
            tree = tree_batch.sent_to_tree(sent)
            tree_batch.combine_text_at_nodes(tree)
            trees.append(tree)
        dep_trees[dataset] = trees
    return dep_trees


def get_parsed_data(raw_data):
    parsed = {}
    for dataset in raw_data.keys():
        labels_texts = []
        for root_sexpr in raw_data[dataset]:
            label, text = parse(root_sexpr)
            labels_texts.append({'label': label, 'text': text})
        parsed[dataset] = labels_texts
    return parsed


def get_raw_data():
    data = {}
    files = ['train.txt', 'dev.txt', 'test.txt']
    for file in files:
        with open(SST_DIR + file, 'r') as f:
            lines = f.readlines()
            lines = [l.rstrip() for l in lines]
            data[file.split('.')[0]] = lines
    return data


def get_sst_trees(raw_data):
    sst_trees = {}
    for dataset in raw_data.keys():
        trees = []
        for sexpr in raw_data[dataset]:
            tree = tree_batch.sexpr_to_tree(sexpr)
            tree_batch.combine_text_at_nodes(tree)
            trees.append(tree)
        sst_trees[dataset] = trees
    return sst_trees


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

    @staticmethod
    def annotation_ixs(forest):
        ixs = {}
        for l in range(forest.max_level + 1):
            l_nodes = forest.nodes[l]
            l_ixs = [i for i in range(len(l_nodes)) if l_nodes[i].annotation]
            ixs[l] = l_ixs
        return ixs

    def collate(self, batch_data):
        """For collating a batch of trees.

        Args:
          batch_data: List of tree_batch.Tree.

        Returns:
          tree_batch.Forest.
        """
        forest = tree_batch.Forest(batch_data)
        forest.labels = []

        # Setting annotation_ixs here necessary downstream and for labels
        forest.annotation_ixs = self.annotation_ixs(forest)

        # Get labels and pre-emptively perform dictionary lookup.
        for l in range(forest.max_level + 1):
            forest.labels += [forest.nodes[l][i]
                              for i in forest.annotation_ixs[l]]
            for node in [n for n in forest.nodes[l] if n.token]:
                node.vocab_ix = self.vocab_dict[node.token]

        return forest


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
