import argparse
import sys

from src.comparators import ActivationComparator
from src.models import load_from_path


def _parse_args(args):
    parser = argparse.ArgumentParser(description='Simple settings.')
    parser.add_argument('front_model', help='Path to first model', type=str)
    parser.add_argument('end_model', help='Path to second model', type=str)
    parser.add_argument('front_layer', help='Last layer of first model', type=str)
    parser.add_argument('end_layer', help='Last layer of second model', type=str)
    parser.add_argument('dataset', help='Name of dataset',
                        choices=['fashion', 'mnist', 'cifar10', 'cifar100', 'celeba'])
    parser.add_argument('methods', help='Name of evaluation method', nargs="+", choices=['cka', 'l2', 'ps_inv', 'corr'])
    parser.add_argument('-b', '--batch-size', help='Batch size', type=int, default=256)
    parser.add_argument('-g', '--group_at', help='Group data by this value', type=float, default=float('inf'))
    parser.add_argument('-t', '--dataset-type', help='Type of data: val or train', type=str, default='val')
    return parser.parse_args(args)


def run_comparison_by_conf(conf, verbose=False):
    front_model = load_from_path(conf.front_model)
    end_model = load_from_path(conf.end_model)

    comparator = ActivationComparator(front_model, end_model, conf.front_layer,
                                      conf.end_layer)

    # Comparison of two models on provided dataset
    measures = comparator(
        conf.dataset,
        conf.methods,
        batch_size=conf.batch_size,
        group_at=conf.group_at,
        dataset_type=conf.dataset_type,
    )

    # Eval results
    if verbose:
        print(measures)
    return measures


def run_comparison_by_list(args=None, verbose=True):
    if args is None:
        args = sys.argv[1:]
    args = _parse_args(args)
    return run_comparison_by_conf(args, verbose=verbose)


if __name__ == '__main__':
    run_comparison_by_list()
