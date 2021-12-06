import argparse
import pickle
import sys

from src.trainer import FrankStepHandler
from src.utils.eval_values import eval_net


def _parse_args(args):
    parser = argparse.ArgumentParser(description='Simple settings.')
    parser.add_argument('pickle', help='Path to pickle')
    parser.add_argument('mode', help='Which matrix to use', choices=['before', 'after', 'ps_inv'])
    return parser.parse_args(args)


def eval_frank(data_dict, mode, verbose=False):
    # Init variables
    data_name = data_dict['params']['dataset']

    # Load frankenstein
    model_trainer = FrankStepHandler.from_data_dict(data_dict, mode)

    # Calculate loss and accuracy
    mean_loss, mean_acc, hits = eval_net(model_trainer, data_name, verbose=verbose)

    # Print if requested
    if verbose:
        print('Loss: {:.3f} | Accuracy: {:2.2f}%'.format(mean_loss, mean_acc * 100.))

    return mean_loss, mean_acc, hits


def eval_frank_from_pickle(pickle_path, mode, verbose=False):
    data_dict = pickle.load(open(pickle_path, "br"))
    return eval_frank(data_dict, mode, verbose)


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = _parse_args(args)
    eval_frank_from_pickle(args.pickle, args.mode, verbose=True)


if __name__ == '__main__':
    main()
