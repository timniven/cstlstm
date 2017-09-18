"""Various configuration settings and mappings."""
import argparse
from ext import models


class Params:
    def __init__(self, name, override):
        self.name = name
        self.override = override


def parse_arguments():
    base_config = models.Config()
    parser = argparse.ArgumentParser()
    parser.add_argument('name',
                        type=str,
                        help='The name for the training run (unique).')
    parser.add_argument('--override',
                        action='store_true',
                        help='Set true to overwrite an old history.')
    parser.add_argument('--tune_embeddings',
                        action='store_true',
                        help='Set true to tune embeddings.')
    parser.add_argument('--train_subset',
                        type=int,
                        help='Size of subset to select from training data.',
                        default=None)
    parser.add_argument('--train_subset',
                        type=int,
                        help='Size of subset to select from tuning data.',
                        default=None)
    arg_config = {}
    for key in [k for k in base_config.keys() if k != 'tune_embeddings']:
        parser.add_argument(
            '--%s' % key,
            help='Set config.%s' % key,
            type=type(base_config[key]))
        arg_config[key] = base_config[key]
    args = parser.parse_args()
    params = Params(
        args.name,
        args.override)
    for key in base_config.keys():
        passed_value = getattr(args, key)
        if passed_value is not None:
            arg_config[key] = passed_value
    return params, arg_config
