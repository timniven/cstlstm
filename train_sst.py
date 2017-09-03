"""Train Child-Sum Tree-LSTM model on the Stanford Sentiment Treebank."""
from data import sst
from cstlstm import sentiment
from ext import parameters, pickling, training
import glovar


# Parse configuration settings from command line
params, arg_config = parameters.parse_arguments()
print('Getting or creating History...')
history_exists = dbi.history.train.exists(_id=params.name)
print('History exists: %s' % history_exists)
if params.override and history_exists:
    print('Override selected - deleting history "%s"' % params.name)
    dbi.history.train.delete(_id=params.name)
history = histories.History(params.name, base.Config(**arg_config))
config = history.config
config.learning_rate = arg_config['learning_rate']
print('Config as follows:')
for key in sorted(list(config.keys())):
    print('\t%s \t%s%s' % (key, '\t' if len(key) < 15 else '', config[key]))


print('Load embedding matrix...')
embedding_matrix = pickling.load(glovar.PKL_DIR, 'glove_embeddings.pkl')


print('Loading data...')
train_data, dev_data = sst.data()
train_loader = sst.get_data_loader(train_data, config.batch_size)
dev_loader = sst.get_data_loader(dev_data, config.batch_size)


print('Loading model...')
model = sentiment.SentimentModel(config, embedding_matrix)


print('Loading trainer...')
trainer = training.PyTorchTrainer(
    model, history, train_loader, dev_loader, glovar.CKPT_DIR)


print('Training...')
trainer.train()
