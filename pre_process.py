"""For pre-processing the data."""
from ext import vocab_emb, pickling
from data import sst, nli
import glovar
import os


if not os.path.exists(glovar.PKL_DIR):
    os.makedirs(glovar.PKL_DIR)
if not os.path.exists(glovar.CKPT_DIR):
    os.makedirs(glovar.CKPT_DIR)


# Create the vocab dictionary
print('Creating vocab dict...')
sst_text = sst.get_text()
nli_text = nli.get_text()
all_text = ' '.join([sst_text, nli_text])
vocab_dict, _ = vocab_emb.create_vocab_dict(all_text)
pickling.save(vocab_dict, glovar.PKL_DIR, 'vocab_dict.pkl')
print('Success.')


# Create GloVe embeddings
print('Creating GloVe embeddings...')
embedding_mat = vocab_emb.create_embeddings(vocab_dict, 300, glovar.GLOVE_DIR)
pickling.save(embedding_mat, glovar.PKL_DIR, 'glove_embeddings.pkl')
print('Success.')
