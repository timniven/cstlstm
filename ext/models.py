"""Base classes for models."""
import torch
from torch import nn


FRAMEWORKS = ['tf', 'torch']
_DEFAULT_CONFIG = {
    'batch_size': 32,
    'embed_size': 300,
    'hidden_size': 300,
    'projection_size': 200,
    'learning_rate': 1e-3,
    'grad_clip_norm': 0.0,
    '_lambda': 0.0,
    'p_keep_input': 0.9,
    'p_keep_rnn': 0.9,
    'p_keep_fc': 0.9,
    'tune_embeddings': True
}


class Config:
    """Wrapper of config variables."""

    def __init__(self, default=_DEFAULT_CONFIG, **kwargs):
        """Create a new Config.

        Args:
          default: Dictionary of default values. These can be passed in, or else
            the _DEFAULT_CONFIG from this file will be used.
        """
        self.default = default
        self.kwargs = kwargs
        self.batch_size = self._value('batch_size', kwargs)
        self.embed_size = self._value('embed_size', kwargs)
        self.hidden_size = self._value('hidden_size', kwargs)
        self.projection_size = self._value('projection_size', kwargs)
        self.learning_rate = self._value('learning_rate', kwargs)
        self.grad_clip_norm = self._value('grad_clip_norm', kwargs)
        self._lambda = self._value('_lambda', kwargs)
        self.p_keep_input = self._value('p_keep_input', kwargs)
        self.p_keep_rnn = self._value('p_keep_rnn', kwargs)
        self.p_keep_fc = self._value('p_keep_fc', kwargs)
        self.tune_embeddings = self._value('tune_embeddings', kwargs)
        for key in [k for k in kwargs.keys()
                    if k not in self.default.keys()]:
            setattr(self, key, kwargs[key])

    def __delitem__(self, key):
        pass

    def __getitem__(self, key):
        return self.__getattribute__(key)

    def __repr__(self):
        x = 'Config as follows:\n'
        for key in sorted(self.keys()):
            x += '\t%s \t%s%s\n' % \
                 (key, '\t' if len(key) < 15 else '', self[key])
        return x

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    def dropout_keys(self):
        return [k for k in self.__dict__.keys() if k.startswith('p_keep_')]

    def keys(self):
        return [key for key in self.__dict__.keys()
                if key not in ['default', 'kwargs']]

    def to_json(self):
        return dict(self.__dict__)

    def _value(self, key, kwargs):
        if key in kwargs.keys():
            return kwargs[key]
        else:
            return self.default[key]


class Model:
    """Base class for a model of any kind."""

    def __init__(self, framework, config):
        """Create a new Model.

        Args:
          framework: String, the framework of the model, e.g. 'pytorch'.
          config: Config object, a configuration settings wrapper.
        """
        self.framework = framework
        self.config = config
        for key in config.keys():
            setattr(self, key, config[key])

    def accuracy(self, *args):
        raise NotImplementedError

    def forward(self, *args):
        """Forward step of the network.

        Returns:
          predictions, loss, accuracy.
        """
        raise NotImplementedError

    def logits(self, *args):
        raise NotImplementedError

    def loss(self, *args):
        raise NotImplementedError

    def optimize(self, *args):
        raise NotImplementedError

    def predictions(self, *args):
        raise NotImplementedError


class PyTorchModel(Model, nn.Module):
    """Base for a PyTorch model."""

    def __init__(self, name, config, embedding_matrix):
        Model.__init__(self, 'pytorch', config)
        nn.Module.__init__(self)

        self.name = name

        # Define embedding.
        self.embedding = nn.Embedding(
            embedding_matrix.shape[0], embedding_matrix.shape[1], sparse=False)
        embedding_tensor = torch.from_numpy(embedding_matrix)
        self.embedding.weight = nn.Parameter(
            embedding_tensor,
            requires_grad=self.tune_embeddings)
        self.embedding.cuda()

        # Define loss
        self.criterion = torch.nn.CrossEntropyLoss().cuda()

    @staticmethod
    def accuracy(correct_predictions, batch_size):
        # batch_size may vary - i.e. the last batch of the data set.
        correct = correct_predictions.cpu().sum().data.numpy()
        return correct / float(batch_size)

    def _biases(self):
        return [p for n, p in self.named_parameters() if n in ['bias']]

    @staticmethod
    def correct_predictions(predictions, labels):
        return predictions.eq(labels)

    def forward(self, forest):
        # Need to return predictions, loss, accuracy
        raise NotImplementedError

    def logits(self, forest):
        raise NotImplementedError

    def loss(self, logits, labels):
        loss = self.criterion(logits, labels)
        return loss

    def optimize(self, loss):
        loss.backward()
        self.optimizer.step()

    @staticmethod
    def predictions(logits):
        return logits.max(1)[1]

    def _weights(self):
        return [p for n, p in self.named_parameters() if n in ['weight']]

    def zero_grad(self):
        self.optimizer.zero_grad()
