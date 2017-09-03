"""Various configuration settings and mappings."""
import argparse


class Params:
    def __init__(self, model_type, name, override):
        self.model_type = model_type
        self.name = name
        self.override = override


def parse_arguments(model_types, base_config):
    parser = argparse.ArgumentParser()
    parser.add_argument('model_type',
                        choices=[m for m in model_types],
                        help='The type of model, e.g. parikh.')
    parser.add_argument('name',
                        type=str,
                        help='The name for the training run (unique).')
    parser.add_argument('--override',
                        action='store_true',
                        help='Set true to overwrite an old history.')
    arg_config = {}
    for key in base_config.keys():
        parser.add_argument(
            '--%s' % key,
            help='Set config.%s' % key,
            type=type(base_config[key]))
        arg_config[key] = base_config[key]
    args = parser.parse_args()
    params = Params(
        args.model_type,
        args.name,
        args.override)
    for key in base_config.keys():
        passed_value = getattr(args, key)
        if passed_value is not None:
            arg_config[key] = passed_value
    return params, arg_config
