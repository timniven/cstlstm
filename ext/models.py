"""Base classes for models."""


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
