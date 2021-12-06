import argparse
import sys

if './' not in sys.path:
    sys.path.append('./')

from src.trainer import ModelStepHandler
from src.utils.eval_values import eval_net
from src.models import get_info_from_path


def _parse_args(args):
    parser = argparse.ArgumentParser(description='Simple settings.')
    parser.add_argument('m', help='Path to model')
    return parser.parse_args(args)


def run(model_path, verbose=False):
    # Initialize variables
    model_trainer = ModelStepHandler.for_eval(model_path)
    dataset_name = get_info_from_path(model_path)[1]

    # Calculate loss, accuracy and hits
    mean_loss, mean_acc, hits = eval_net(model_trainer,
                                         dataset_name,
                                         verbose=verbose)

    # Print if requested
    if verbose:
        print('Loss: {:.3f} | Accuracy: {:2.2f}%'.format(
            mean_loss, mean_acc * 100.))

    return mean_loss, mean_acc, hits


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = _parse_args(args)
    run(args.m, verbose=True)


if __name__ == '__main__':
    main()
