# cstlstm
Child-Sum Tree-LSTM Implementation in PyTorch.

Baed on the paper "Improved Semantic Representations From Tree-Structure Long Short-Term Memory Networks" by Tai et al. 2015 (http://www.aclweb.org/anthology/P15-1150).

The standalone components are contained in the folder `cstlstm/`.

Dependencies:
```
python 3.5
PyTorch 0.2
spacy
numpy
```

## Implementation Notes

Tree-structured recursive neural networks are challenging because of computational efficiency issues, especially when compared to linear RNNs. Implementing Tai's model with recursion over nodes was found to be inefficient.

Therefore, the model is implemented here with a parallelization strategy: to process all nodes on each (depth) level of the forest in parallel, working upwards from the deepest leaves.  The LSTM cell (`cstlstm/cell.py`) is designed with this in mind: it is called a "BatchCell" for this reason.

This does indeed improve speed considerably, however it is still slow compared to a linear RNN.

WARNING: I have found that opting to tune the word embeddings during training is brutally inefficient. I hope this can be solved in the future and welcome any advice (as always).

## Encoder

The file `cstlm/encoder.py` is a reusable encoder, a PyTorch `nn.module`. It will return a hidden state for every node in the forest. These are returned as a dictionary structure, mirroring the input structure (see "Input" section below).

Usage example of the encoder:

```
import cstlstm
encoder = cstlstm.ChildSumTreeLSTMEncoder(
    embed_size, hidden_size, embeddings, p_keep_input, p_keep_rnn)
encodings = encoder(forest)
```

## Input

The file `cstlstm/tree_batch.py` deals with converting SpaCy sents, or dependency trees given as sexpressions, to Tree and Forest objects that contain all the information (critically, the upward wiring information) needed by the encoder.

For example, with NLI data we are given Stanford Parser dependency parses as strings, which can be turned into a Tree as such:

```
from cstlstm import tree_batch
s = '(ROOT (S (NP ... )))'
tree = tree_batch.sexpr_to_tree(s)
```

Alternatively (and my personal preference), we can pass a sentence as a string to SpaCy to do our dependency parsing for us, and use `tree_batch.py` to turn it into a tree:

```
from cstlstm import tree_batch
import spacy
nlp = spacy.load('en')
s = nlp('If he had a mind, there was something on it.')
tree = tree_batch.sent_to_tree(s)
```

This gives us individual trees, but for mini-batching (or even for a single NLI training sample) we need to input more than one sentence. We need a Forest. `tree_batch.py` takes care of this, defining the `Forest` object with a constructor that takes a list of `Tree`s. 

```
from cstlstm import tree_batch
import spacy
nlp = spacy.load('en')
sents = [nlp('First sentence.'), nlp('Second sentence.')]
trees = [tree_batch.sent_to_tree(s) for s in sents]
forest = tree_batch.Forest(trees)
```

Inputting the trees in the order you want, the forest keeps the nodes ordered along each level - e.g. the roots are in order on the first level, so `Forest.nodes[0]` gives a list of root nodes that can be indexed by this order. This ordering is mirrored in the encoding output - so `encodings[0]` gives a list of hidden states in the same order. The same applies to the nodes on all levels.

## Models

I have implemented this model for Natural Language Inference data (`models/inference.py`), my own research interest. That model trains successfully.

The Stanford Sentiment Treebank model (`models/sentiment.py`) is incomplete.

## Getting Started

TODO: explain glovar.py, necessary folders, data downloads, pre-processing.
