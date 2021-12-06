import argparse
import pickle
import sys

from src.utils import config
from src.comparators import ActivationComparator
from src.models import FrankeinsteinNet


def _parse_args(args):
    parser = argparse.ArgumentParser(description='Simple settings.')
    parser.add_argument('pickle', help='Path to pickle')
    parser.add_argument('mode', help='Which stitching matrix to use', choices=['before', 'after', 'ps_inv'])
    parser.add_argument('methods', help='Name of evaluation method', nargs="+", choices=['cka', 'l2', 'ps_inv', 'corr'])
    parser.add_argument('-l', '--layer', help='Layer to compare at. Default is one at pickle.', type=str, default=None)
    parser.add_argument('-b', '--batch-size', help='Batch size', type=int, default=256)
    parser.add_argument('-g', '--group-at', help='Group data by this value', type=float, default=float('inf'))
    parser.add_argument('-t', '--dataset-type', help='Type of data: val or train', type=str, default='val')
    return parser.parse_args(args)


def run_comparison_by_conf(conf, verbose=False):
    data_dict = pickle.load(open(conf.pickle, "br"))
    frank_model = FrankeinsteinNet.from_data_dict(data_dict, conf.mode)
    frank_model.to(config.device)
    comparator = ActivationComparator.from_frank_model(frank_model,
                                                       m1='frank',
                                                       m2='end')
    if conf.layer is not None:
        comparator.layers = [conf.layer, conf.layer]

    results = comparator(data_dict['params']['dataset'], conf.methods,
                         conf.batch_size, conf.group_at, conf.dataset_type)

    if verbose:
        print(results)

    return results


def run_comparison_by_list(args=None, verbose=True):
    if args is None:
        args = sys.argv[1:]
    args = _parse_args(args)
    return run_comparison_by_conf(args, verbose=verbose)


if __name__ == '__main__':
    run_comparison_by_list()
