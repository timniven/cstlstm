"""Train Child-Sum Tree-LSTM model on the Stanford Sentiment Treebank."""
from data import sst
from cstlstm import sentiment


# Parse configuration settings from command line
params, arg_config = configuration.parse_arguments()
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


# Load embedding matrix
print('Load embedding matrix...')
embedding_matrix = None


# Load data
print('Loading data...')
train_data, dev_data = sst.data()
train_loader = sst.get_data_loader(train_data, config.batch_size)
dev_loader = sst.get_data_loader(dev_data, config.batch_size)


# Load model
print('Loading model...')
model = sentiment.SentimentModel(config, embedding_matrix)


# Load trainer
print('Loading trainer...')
trainer = None


# Train
print('Training...')
trainer.train()
