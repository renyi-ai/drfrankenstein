import pickle

import torch
from dotmap import DotMap
from tabulate import tabulate
from torch.utils.data import DataLoader

from src.comparators.comparator_base import ComparatorBaseClass
from src.dataset import get_datasets
from src.models.frank.frankenstein import FrankeinsteinNet


class ActSignComparator(ComparatorBaseClass):

    def __init__(self, frank_model, ps_inv_model) -> None:
        models = {
            "frank_model": frank_model,
            "ps_inv_model": ps_inv_model,
        }
        super().__init__(models=DotMap(models))

    def register_hooks(self):
        layer_name = f"bn{self.models.frank_model.end_layer_name[-1]}"
        self.hook_store.register_activation_saver(
            self.models.ps_inv_model.get_layer(layer_name), f"ps_inv")
        self.hook_store.register_activation_saver(
            self.models.frank_model.get_layer(layer_name), f"frank")

    def compute(self):
        frank_acts = torch.cat(
            self.hook_store.get_and_clear_cache(f"frank"), dim=0)
        ps_inv_acts = torch.cat(
            self.hook_store.get_and_clear_cache(f"ps_inv"), dim=0)
        activation_diff = torch.abs(frank_acts - ps_inv_acts)

        below_dists = ['Below']
        above_dists = ['Above']
        above_counts = ['Above 0 count']
        below_counts = ['Below 0 count']
        headers = ['Threshold computed on']

        thresholds = torch.tensor(0).reshape([-1, 1, 1, 1])

        act_mask = frank_acts > thresholds
        above_counts.append(act_mask.sum())
        below_counts.append((~act_mask).sum())
        dist_below = activation_diff[~act_mask].mean().item()
        dist_above = activation_diff[act_mask].mean().item()
        below_dists.append(dist_below)
        above_dists.append(dist_above)
        headers.append("Frank")

        act_mask = ps_inv_acts > thresholds
        above_counts.append(act_mask.sum())
        below_counts.append((~act_mask).sum())
        dist_below = activation_diff[~act_mask].mean().item()
        dist_above = activation_diff[act_mask].mean().item()
        below_dists.append(dist_below)
        above_dists.append(dist_above)
        headers.append("Ps_inv")

        print(tabulate([below_counts, above_counts, above_dists, below_dists],
                       headers=headers))

    def __call__(self, data_loader):
        self.register_hooks()
        self.iterate_through_data(data_loader)
        self.compute()
        self.hook_store.clear_store()


def cli_main(args=None):
    import argparse
    import torch.utils.data as data_utils
    from src.utils import config

    parser = argparse.ArgumentParser(description='Simple settings.')
    parser.add_argument('pickle', help='Path to pickle')

    args = parser.parse_args(args)

    data_dict = pickle.load(open(args.pickle, "br"))

    frank_model = FrankeinsteinNet.from_data_dict(data_dict, "after")
    ps_inv_model = FrankeinsteinNet.from_data_dict(data_dict, 'ps_inv')

    data_name = data_dict['params']['dataset']

    dataset = get_datasets(data_name)['val']

    if config.debug:
        dataset = data_utils.Subset(dataset, torch.arange(151))

    data_loader = DataLoader(dataset,
                             batch_size=50,
                             num_workers=0 if config.debug else 4,
                             shuffle=True,
                             drop_last=True)

    comparator = ActSignComparator(frank_model=frank_model,
                                   ps_inv_model=ps_inv_model)
    comparator(data_loader)


if __name__ == '__main__':
    cli_main()
