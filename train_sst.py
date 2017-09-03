"""Train Child-Sum Tree-LSTM model on the Stanford Sentiment Treebank."""
from data import sst
from cstlstm import sentiment
from ext import parameters, pickling, training, histories, models
import glovar


# Parse configuration settings from command line and get History object
params, arg_config = parameters.parse_arguments()
print('Getting or creating History...')
history_exists = histories.DBI.history.train.exists(_id=params.name)
print('History exists: %s' % history_exists)
if params.override and history_exists:
    print('Override selected - deleting history "%s"' % params.name)
    histories.DBI.history.train.delete(_id=params.name)
history = histories.History(params.name, models.Config(**arg_config))
config = history.config

# Report config to be used
print('Config as follows:')
for key in sorted(list(config.keys())):
    print('\t%s \t%s%s' % (key, '\t' if len(key) < 15 else '', config[key]))

print('Load embedding matrix...')
embedding_matrix = pickling.load(glovar.PKL_DIR, 'glove_embeddings.pkl')[0]

print('Loading data...')
train_data, dev_data, _ = sst.data()
train_loader = sst.get_data_loader(train_data, config.batch_size)
dev_loader = sst.get_data_loader(dev_data, config.batch_size)

print('Loading model...')
model = sentiment.SentimentModel(config, embedding_matrix)

print('Loading trainer...')
trainer = training.PyTorchTrainer(
    model, history, train_loader, dev_loader, glovar.CKPT_DIR)

print('Training...')
trainer.train()
