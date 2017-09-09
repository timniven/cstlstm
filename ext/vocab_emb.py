"""For creating vocab dictionaries and word embedding matrices."""
import numpy as np
import spacy
import collections


PADDING = "<PAD>"
UNKNOWN = "<UNK>"
LBR = '('
RBR = ')'


def create_embeddings(vocab, emb_size, embedding_file_path):
    """Create embeddings for the vocabulary.

    Creates an embedding matrix given the pre-trained word vectors, and any OOV
    tokens are initialized to random vectors.

    Args:
      vocab: Dictionary for the vocab with {token: id}.
      emb_size: Integer, the size of the word embeddings.
      embedding_file_path: String, file path to the pre-trained embeddings to
        use.

    Returns:
      embeddings, oov: 2D numpy.ndarray of shape vocab_size x emb_size,
        Dictionary of OOV vocab items.
    """
    print('Creating word embeddings from %s...' % embedding_file_path)
    vocab_size = max(vocab.values()) + 1
    print('vocab_size = %s' % vocab_size)
    oov = dict(vocab)
    embeddings = np.random.normal(size=(vocab_size, emb_size))\
        .astype('float32', copy=False)
    with open(embedding_file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            s = line.split()
            if len(s) > 301:  # a hack I have seemed to require for GloVe 840B
                s = [s[0]] + s[-300:]
                assert len(s) == 301
            if s[0] in vocab.keys():
                if s[0] in oov.keys():  # seems we get some duplicate vectors.
                    oov.pop(s[0])
                try:
                    embeddings[vocab[s[0]], :] = np.asarray(s[1:])
                except Exception as e:
                    print('i: %s' % i)
                    print('s[0]: %s' % s[0])
                    print('vocab_[s[0]]: %s' % vocab[s[0]])
                    print('len(vocab): %s' % len(vocab))
                    print('vocab_min_val: %s' % min(vocab.values()))
                    print('vocab_max_val: %s' % max(vocab.values()))
                    raise e
    print('Success.')
    print('OOV count = %s' % len(oov))
    print(oov)
    return embeddings, oov


def create_vocab_dict(text):
    """Create vocab dictionary.

    Args:
      text: String. Join all the text in the corpus on a space. It will be
        tokenized by SpaCy.

    Returns:
      Dictionary {token: id}, collections.Counter() with token counts.
    """
    nlp = spacy.load('en')
    doc = nlp(text)
    counter = collections.Counter()
    counter.update([t.text for t in doc])
    tokens = set([t for t in counter] + [UNKNOWN, LBR, RBR])
    # Make sure 0 is padding.
    vocab_dict = dict(zip(tokens, range(1, len(tokens) + 1)))
    assert PADDING not in vocab_dict.keys()
    assert 0 not in vocab_dict.values()
    vocab_dict[PADDING] = 0
    return vocab_dict, counter
