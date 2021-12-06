import argparse
import logging
import os
import pickle
import sys
from datetime import datetime

import git

from src.comparators import ActivationComparator
from src.dataset import get_datasets
from src.trainer import Trainer, ModelStepHandler, FrankStepHandler
from src.utils.eval_values import eval_net, frank_m2_similarity, relative_eval_net


def _parse_args(args):
    parser = argparse.ArgumentParser(description='Simple settings.')
    parser.add_argument('front_model', help='Path first model', type=str)
    parser.add_argument('end_model', help='Path to second model', type=str)
    parser.add_argument('front_layer', help='Last layer of first model', type=str)
    parser.add_argument('end_layer', help='Last layer of second model', type=str)
    parser.add_argument('-o', '--out-dir', help='Directory to save results to', default='results/')
    parser.add_argument('--run-name', help='Name of the current run', type=str, default=None)
    parser.add_argument('--optimizer', help='Optimizer', choices=['adam', 'sgd'], default='adam')
    parser.add_argument('--seed', help='Seed of stitching.', type=int, default=0)
    parser.add_argument('-e', '--epochs', help='Number of epochs to train', type=int, default=30)
    parser.add_argument('-b', '--batch-size', help='Batch size', type=int, default=128)
    parser.add_argument('-lr', '--lr', help='Learning rate', type=float, default=1e-3)
    parser.add_argument('-wd', '--weight-decay', help='Weight decay', type=float, default=0)
    parser.add_argument('--debug', help='Debug True/False', action='store_true')
    parser.add_argument('--l1', help='l1 regularization value', type=float, default=0.)
    parser.add_argument('--cka-reg', help='Weight of CKA regularization', type=float, default=0.)
    parser.add_argument('-r', '--low-rank', help='Rank', type=int, default=None)
    parser.add_argument('--temperature', help='Temperature of soft labels', type=float, default=1.)
    parser.add_argument('-d',
                        '--dataset',
                        help='Name of dataset',
                        default='cifar10',
                        choices=['fashion', 'mnist', 'cifar10', 'cifar100', 'celeba'])
    parser.add_argument('-s',
                        '--save-frequency',
                        help='Save frequency of stitching matrix in iterations. ' \
                             'Note that this number is multiplied by the number of epochs.',
                        type=int,
                        default=10)

    parser.add_argument('-f',
                        '--flatten',
                        help='Flatten around transformation. This changes the stitching ' \
                             'from "channel to channel" to "pixel to pixel"',
                        action='store_true')
    parser.add_argument('-i',
                        '--init',
                        help='Initial matrix',
                        choices=['random', 'perm', 'eye', 'ps_inv', 'ones-zeros'],
                        default='random')
    parser.add_argument('-m',
                        '--mask',
                        help='Mask applied on transformation. These methods ' \
                             'allow you to make a pairwise stitching between the channels.',
                        choices=['identity', 'semi-match', 'abs-semi-match', 'random-permutation'],
                        default='identity')
    parser.add_argument('--target-type',
                        help='Type of label to use for training',
                        choices=['hard', 'soft_1', 'soft_2', 'soft_12', 'soft_1_plus_2'],
                        default='hard')
    return parser.parse_args(args)


def get_hits_overlap(m1_hits, m2_hits):
    total_overlap = (m1_hits == m2_hits).sum()
    m1_right_m2_right = (m1_hits[m2_hits == m1_hits]).sum()
    m1_right_m2_wrong = (m1_hits[m2_hits != m1_hits]).sum()
    m1_wrong_m2_right = (m2_hits[m2_hits != m1_hits]).sum()
    m1_wrong_m2_wrong = total_overlap - m1_right_m2_right
    return {'rr': m1_right_m2_right, 'rw': m1_right_m2_wrong, 'wr': m1_wrong_m2_right, 'ww': m1_wrong_m2_wrong}


def create_data(conf, trainer, trans_m_before_train):
    # The dictionary in which we accumulate the results
    out_data = {}

    # Save trans learning curve
    out_data['trans_fit'] = trainer.model_trainer.stats

    # Save running parameters (arguments)
    out_data['params'] = vars(conf)

    # Losses and accuracies
    frank_model = trainer.model_trainer.model
    multilabel = conf.dataset == 'celeba'
    m1_runner = ModelStepHandler(frank_model.front_model, multilabel=multilabel)
    m2_runner = ModelStepHandler(frank_model.end_model, multilabel=multilabel)
    m1_loss, m1_acc, m1_hits = eval_net(m1_runner, conf.dataset)
    m2_loss, m2_acc, m2_hits = eval_net(m2_runner, conf.dataset)
    trans_loss, trans_acc, frank_hits = eval_net(trainer.model_trainer, conf.dataset)
    m1_m2_rel_results = relative_eval_net(frank_model.front_model,
                                          frank_model.end_model,
                                          out_data['params']['dataset'],
                                          verbose=False)

    out_data['model_results'] = {
        'front': {
            'loss': m1_loss,
            'acc': m1_acc
        },
        'end': {
            'loss': m2_loss,
            'acc': m2_acc
        },
        'trans': {
            'loss': trans_loss,
            'acc': trans_acc
        }
    }

    # Extract weight and bias
    out_data['trans_m'] = {}
    out_data['trans_m']['before'] = trans_m_before_train
    out_data['trans_m']['after'] = _extract_trans_w_and_b(frank_model)
    out_data['trans_m']['diff'] = _extract_trans_diff(out_data['trans_m'])
    if hasattr(frank_model.transform, "mask") and frank_model.transform.mask is not None:
        out_data['trans_m']['mask'] = {'w': frank_model.transform.mask}

    # Save transformation matrices changing in time
    out_data['trans_history'] = trainer.model_trainer.transform_history

    if conf.low_rank is None:
        # Comparisons
        comparator = ActivationComparator.from_frank_model(frank_model)
        comparator_frank = ActivationComparator.from_frank_model(frank_model, 'frank', 'end')
        group_at = 2500 if conf.dataset == 'celeba' else float('inf')
        batch_size = 50 if conf.dataset == 'celeba' else 2500
        stop_at = 5 if conf.dataset == 'celeba' else float('inf')
        dataset_type = 'val'
        frank_sim = frank_m2_similarity(frank_model, out_data['params']['dataset'], verbose=False)

        print('\n','='*80)
        print('M1 --> M2\n')
        measures = comparator(conf.dataset, ['cka', 'ps_inv', 'l2'], batch_size, group_at, stop_at, dataset_type)
        print_measures = measures.copy()
        print_measures.pop('ps_inv')
        print_measures.update(m1_m2_rel_results)
        print('\n', {k:round(v,3) for (k,v) in print_measures.items() if k!="same_class_out"})

        print('='*80)
        print('M1 + Dr.Frankenstein Stitch --> M2\n')
        measures_frank = comparator_frank(conf.dataset, ['cka', 'l2'], batch_size, group_at, stop_at, dataset_type)
        print_measures = measures_frank.copy()
        print_measures.update(frank_sim)
        print('\n', {k:round(v,3) for (k,v) in print_measures.items() if k!="same_class_out"})

        # Cka
        out_data['cka'] = measures['cka']
        out_data['cka_frank'] = measures_frank['cka']

        # l2
        out_data['l2'] = measures['l2']
        out_data['l2_frank'] = measures_frank['l2']

        # Pseudo inverse
        w = measures['ps_inv']['w'][..., None, None]
        b = measures['ps_inv']['b']
        out_data['trans_m']['ps_inv'] = {'w': w, 'b': b}

        # Difference to plot
        diff = out_data['trans_m']['ps_inv']['w'] - out_data['trans_m']['after']['w']
        out_data['trans_m']['ps_frank'] = {'w': diff}

        # psinv loss
        ps_inv_model_trainer = FrankStepHandler.from_data_dict(out_data, 'ps_inv')
        ps_inv_loss, ps_inv_acc, ps_inv_hits = eval_net(ps_inv_model_trainer, conf.dataset)

        ps_inv_model = ps_inv_model_trainer.model
        comparator_ps_inv = ActivationComparator.from_frank_model(ps_inv_model, 'frank', 'end')
        ps_inv_sim = frank_m2_similarity(ps_inv_model, out_data['params']['dataset'], verbose=False)

        print('='*80)
        print('M1 + Least Squares Stitch --> M2\n')
        measures_ps_inv = comparator_ps_inv(conf.dataset, ['cka', 'l2'], batch_size, group_at, stop_at, dataset_type)
        print_measures = measures_ps_inv.copy()
        print_measures.update(ps_inv_sim)
        print('\n', {k:round(v,3) for (k,v) in print_measures.items() if k!="same_class_out"})

        # Cka
        out_data['cka_ps_inv'] = measures_ps_inv['cka']

        # l2
        out_data['l2_ps_inv'] = measures_ps_inv['l2']

        out_data['model_results']['ps_inv'] = {'loss': ps_inv_loss, 'acc': ps_inv_acc}

        out_data['hits'] = {}
        out_data['hits']['m2_frank'] = get_hits_overlap(m2_hits, frank_hits)
        out_data['hits']['m2_ps_inv'] = get_hits_overlap(m2_hits, ps_inv_hits)
        out_data['hits']['frank_ps_inv'] = get_hits_overlap(frank_hits, ps_inv_hits)

        # Similarities
        out_data['m2_sim'] = {'ps_inv': ps_inv_sim, 'after': frank_sim, 'm1': m1_m2_rel_results}

    # Git info
    repo = git.Repo('./')
    out_data['git'] = {'branch': repo.active_branch.name, 'commit': repo.head.commit.hexsha}

    # Save the command what the user used for running
    out_data['runner_code'] = ' '.join(['python'] + sys.argv)

    return out_data


def _extract_trans_diff(data):
    """Get difference of trained and untrained stitching matrices"""
    before = data['before']
    after = data['after']
    w = after['w'] - before['w']
    b = after['b'] - before['b']
    return {'w': w, 'b': b}


def _extract_trans_w_and_b(model):
    w_b_dict = model.transform.get_param_dict()
    return w_b_dict


def save(conf, trainer, trans_m_before_train):
    # Extract data which should be saved
    data = create_data(conf, trainer, trans_m_before_train)

    # Save pickle
    filename = '{}-{}.pkl'.format(conf.front_layer, conf.end_layer)
    matrix_file = os.path.join(conf.out_dir, filename)
    os.makedirs(conf.out_dir, exist_ok=True)
    with open(matrix_file, 'wb') as f:
        pickle.dump(data, f)


def set_random_seeds(seed=0):
    """Given a seed it sets the run deterministic."""
    import torch
    import numpy as np
    import random
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def run(conf):
    """Trains a stitching, evaluates to performance of stitiching and
       then saves the results.

    Args:
        conf (Namespace): Arguments given by argparser.
    """

    # Retrieve save folder
    now = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_name = f"{conf.run_name}" if conf.run_name is not None else f"{now}"
    conf.out_dir = os.path.join(conf.out_dir, run_name)

    # Frankenstein model definition
    model_trainer = FrankStepHandler.from_arg_config(conf)

    # Save transformation matrix before training
    trans_m_before_train = _extract_trans_w_and_b(model_trainer.model)

    # Define dataset
    datasets = get_datasets(conf.dataset)

    # In order to prevent large log files, scale log frequency
    # with the number of epochs
    save_frequency = conf.save_frequency * conf.epochs

    # Train
    trainer = Trainer(
        datasets,
        model_trainer,
        batch_size=conf.batch_size,
        n_workers=4,
        drop_last=True,
        save_folder=conf.out_dir,
    )
    trainer.train(conf.epochs, save_frequency, freeze_bn=True)

    # Save
    save(conf, trainer, trans_m_before_train)
    print('Done.')


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = _parse_args(args)
    set_random_seeds(args.seed)

    logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
    logger = logging.getLogger("trans")
    logger.setLevel(logging.DEBUG if args.debug else logging.WARNING)

    run(args)


if __name__ == '__main__':
    main()
