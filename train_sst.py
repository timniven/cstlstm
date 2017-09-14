"""Train Child-Sum Tree-LSTM model on the Stanford Sentiment Treebank."""
import glovar
from data import sst
from ext import parameters, pickling, training, histories
from models import sentiment

# Parse configuration settings from command line
params, arg_config = parameters.parse_arguments()


# Get or create History
history = histories.get(
    glovar.PKL_DIR, params.name, params.override, arg_config)


# Report config to be used
config = history.config
print(config)


print('Load embedding matrix...')
embedding_matrix = pickling.load(glovar.PKL_DIR, 'glove_embeddings.pkl')[0]


print('Loading data...')
train_data, dev_data, _ = sst.get_data()
train_loader = sst.get_data_loader(train_data, config.batch_size)
dev_loader = sst.get_data_loader(dev_data, config.batch_size)


print('Loading model...')
model = sentiment.SentimentModel(params.name, config, embedding_matrix)


print('Loading trainer...')
trainer = training.PyTorchTrainer(
    model, history, train_loader, dev_loader, glovar.CKPT_DIR)


print('Training...')
trainer.train()
