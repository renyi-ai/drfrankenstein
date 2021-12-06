import os
import pickle
from copy import deepcopy
from datetime import datetime
from os import path

import numpy as np
import torch
from dotmap import DotMap
from torch.utils.data import DataLoader

from src.comparators.comparator_base import ComparatorBaseClass
from src.comparators.compare_functions.ps_inv import rearrange_activations
from src.dataset import get_datasets
from src.models import load_from_path
from src.models.frank.frankenstein import FrankeinsteinNet


class SVDComparator(ComparatorBaseClass):

    def __init__(self, frank_model, ps_inv_model, front_model, end_model,
                 out_file, original_data_dict,
                 verbose=False):
        """

        Args:
            frank_model:
            ps_inv_model:
            front_model:
            end_model:
            backward_mode: mean or max
        """
        models = {
            "frank_model": frank_model,
            "ps_inv_model": ps_inv_model,
            "front_model": front_model,
            "end_model": end_model
        }
        super().__init__(models=DotMap(models))
        self.out_file = out_file
        self.original_data_dict = original_data_dict
        self.verbose = verbose
        self.model = deepcopy(ps_inv_model)
        self.model.prepare_models()
        self.transform = self.model.transform

    def register_hooks(self):
        front_layer_name = self.models["frank_model"].front_layer_name
        end_layer_name = self.models["frank_model"].end_layer_name

        front_layer = self.models.front_model.get_layer(front_layer_name)
        end_layer = self.models.end_model.get_layer(end_layer_name)

        ps_inv_transform = self.models.ps_inv_model.transform
        frank_transform = self.models.frank_model.transform

        self.hook_store.register_activation_saver(front_layer, f"front_act",
                                                  model_name="front_model")
        self.hook_store.register_activation_saver(end_layer, f"end_act",
                                                  model_name="end_model")

        self.hook_store.register_activation_saver(ps_inv_transform,
                                                  f"ps_inv_act",
                                                  model_name="ps_inv_model")
        self.hook_store.register_activation_saver(frank_transform, f"frank_act",
                                                  model_name="frank_model")

    def _get_data(self):
        frank_acts = self.hook_store.get_and_clear_cache(f"frank_act")
        ps_inv_acts = self.hook_store.get_and_clear_cache(f"ps_inv_act")

        front_acts = self.hook_store.get_and_clear_cache(f"front_act")
        end_acts = self.hook_store.get_and_clear_cache(f"end_act")

        frank_acts, ps_inv_acts = torch.cat(frank_acts), torch.cat(ps_inv_acts)
        front_acts, end_acts = torch.cat(front_acts), torch.cat(end_acts)
        return frank_acts, ps_inv_acts, front_acts, end_acts

    def __call__(self, data_loader, str_dataset):
        # if find params is None the function will just use a single input for weighting

        self.register_hooks()
        self.iterate_through_data(data_loader)
        frank_acts, ps_inv_acts, front_acts, end_acts = self._get_data()

        diff = frank_acts - ps_inv_acts
        diff = rearrange_activations(diff)
        # u, s, v = np.linalg.svd(x_after - x_before, full_matrices=False)
        ps_inv_acts = rearrange_activations(ps_inv_acts)
        up, sp, vp = np.linalg.svd(ps_inv_acts, full_matrices=False)
        V = torch.tensor(vp)
        cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
        V = V.unsqueeze(0).repeat(diff.shape[0], 1, 1)
        diff = diff.unsqueeze(-1).repeat(1, 1, V.shape[-1])
        sim = cos_sim(diff, V)
        sim = sim.mean(0)
        metrics = {"cos_sim": sim, "S": sp}

        self.hook_store.clear_store()
        self.save_results(metrics)

    def save_results(self, metrics):
        if self.out_file is None:
            return
        out_dict = metrics
        orig_dict = deepcopy(self.original_data_dict)
        del orig_dict["trans_fit"]
        del orig_dict["trans_m"]
        out_dict["original"] = orig_dict

        with open(self.out_file, 'wb') as handle:
            pickle.dump(out_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


def cli_main(args=None):
    import argparse
    import torch.utils.data as data_utils
    from src.utils import config

    parser = argparse.ArgumentParser(description='Simple settings.')
    parser.add_argument('pickle', nargs="+", help='Path to pickle')
    parser.add_argument('--out_dir', type=str,
                        default=None,
                        help='Output folder for result pickle')

    args = parser.parse_args(args)

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

        dataset = get_datasets(data_name)['val']

        if config.debug:
            dataset = data_utils.Subset(dataset, torch.arange(151))

        data_loader = DataLoader(dataset,
                                 batch_size=2048,
                                 num_workers=0 if config.debug else 4,
                                 shuffle=False,
                                 drop_last=False)

        pickle_dir, pickle_filename = path.split(pkl)
        now = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        out_file_name = path.splitext(pickle_filename)[0] + "_" + now + ".pkl"
        if args.out_dir is None:
            out_file = None
        else:
            if not os.path.exists(args.out_dir):
                os.mkdir(args.out_dir)
            out_file = path.join(args.out_dir, out_file_name)

        comparator = SVDComparator(frank_model=frank_model,
                                   ps_inv_model=ps_inv_model,
                                   front_model=front_model,
                                   end_model=end_model,
                                   out_file=out_file,
                                   original_data_dict=data_dict,
                                   verbose=True)

        comparator(data_loader, str_dataset=data_name)
        print(f"Result saved to {out_file}")


if __name__ == '__main__':
    cli_main()
