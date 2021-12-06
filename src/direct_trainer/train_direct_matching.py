"""
This file is the entry point of direct matching methods trained by optimization.
"""
import argparse
import os
import pickle
from os import path

from losses import get_loss
from src.models import load_from_path
from src.models.frank.frankenstein import FrankeinsteinNet
from src.direct_trainer.direct_trainer import DirectTrainer


def _parse_args(args):
    parser = argparse.ArgumentParser(description='Simple settings.')
    parser.add_argument('pickle', nargs="+", help='Path to pickle')
    parser.add_argument('--init_model', type=str,
                        default='ps_inv_model',
                        choices=["frank_model", "ps_inv_model", "front_model", "end_model"],
                        help="Which model's transformation matrix to use as init")
    parser.add_argument('--train_on', type=str,
                        default='val',
                        choices=["train", "val"],
                        help="Which split to use for training, val or train")
    parser.add_argument('--loss', type=str, help="Choose loss from direct_trainer.losses")
    parser.add_argument('--epoch', type=int,
                        default=100,
                        help='Number of epochs to train')
    parser.add_argument('--batch', type=int,
                        default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float,
                        default=0.0001,
                        help='Learning rate')
    parser.add_argument('--dev', type=int,
                        default=0,
                        help='If true save results to temp folder')
    parser.add_argument('--out_dir', type=str,
                        default=None,
                        help='Output folder for result pickle')

    args = parser.parse_args(args)
    return args


def main(args=None):
    args = _parse_args(args)

    for pkl in args.pickle:
        print(f"Processing: {pkl}")
        data_dict = pickle.load(open(pkl, "br"))
        front_model = load_from_path(data_dict['params']['front_model'])
        end_model = load_from_path(data_dict['params']['end_model'])

        frank_model = FrankeinsteinNet.from_data_dict(data_dict, "after")
        ps_inv_model = FrankeinsteinNet.from_data_dict(data_dict, 'ps_inv')

        device = "cuda"
        frank_model.to(device)
        ps_inv_model.to(device)
        front_model.to(device)
        end_model.to(device)

        data_name = data_dict['params']['dataset']

        pickle_dir, pickle_filename = path.split(pkl)
        pickle_filename = pickle_filename.replace(" ", "_").replace(":", "")
        out_file_name = "direct_matching_" + pickle_filename
        if args.out_dir is not None:
            os.makedirs(args.out_dir, exist_ok=True)
            out_file = path.join(args.out_dir, out_file_name)
        else:
            out_file = None

        comparator = DirectTrainer(frank_model=frank_model,
                                   ps_inv_model=ps_inv_model,
                                   front_model=front_model,
                                   end_model=end_model,
                                   init_model=args.init_model,
                                   loss=get_loss(args.loss),
                                   batch_size=args.batch,
                                   epoch=args.epoch,
                                   lr=args.lr,
                                   out_file=out_file,
                                   original_data_dict=data_dict)
        comparator(dataset_str=data_name, train_on=args.train_on)


if __name__ == '__main__':
    main()
