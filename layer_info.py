import argparse
import sys

import pandas as pd
from tabulate import tabulate

if './' not in sys.path:
    sys.path.append('./')

from src.models import get_model


def _parse_args(args):
    parser = argparse.ArgumentParser(description='Simple settings.')
    parser.add_argument('classifier', type=str)
    return parser.parse_args(args)


def load_model(model_name):
    n_classes, n_channels = 100, 3
    model = get_model(model_name, n_classes, n_channels)
    return model


def main(args):
    # Load config
    conf = _parse_args(args)

    # Structure skeleton model
    model = load_model(conf.classifier).eval()

    # Create table with indexing layers available for transformation
    df = pd.DataFrame({'layer': model.layer_info})

    # Show table
    print(tabulate(df, headers='keys', tablefmt="fancy_grid"))


if __name__ == '__main__':
    main(sys.argv[1:])
