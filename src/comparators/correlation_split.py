import sys

import numpy as np
import torch

if './' not in sys.path:
    sys.path.append('./')

from src.comparators.activation_comparator import ActivationComparator
from src.comparators.compare_functions import correlation, l2


class CorrelationSplitComparator(ActivationComparator):

    def __call__(self, dataset_name: str, percentiles=(97, 98, 99), batch_size: int = 200,
                 group_at: float = float('inf'), dataset_type: str = 'val', **kwargs):
        self.percentiles = percentiles
        return super().__call__(dataset_name, ['corr'], batch_size, group_at, dataset_type=dataset_type)

    def _calculate_measures(self, measure_names):
        results = {}
        activations = [
            torch.cat(self.hook_store.get_and_clear_cache(key), dim=0).numpy()
            for key in self.models.keys()
        ]
        if len(activations[0]) == 0:
            return None

        correlation_m = np.abs(correlation(*activations))

        for p in self.percentiles:
            threshold = np.percentile(correlation_m, p)
            pixel_mask = correlation_m > threshold
            channel_mask = np.any(pixel_mask, axis=1)
            above = self._get_distance(activations, channel_mask)
            below = self._get_distance(activations, ~channel_mask)
            results[p] = {'below': below, 'above': above}

        return {'corr_split': results}

    def _get_distance(self, activations, channel_mask):
        act1 = activations[0][:, channel_mask]
        act2 = activations[1][:, channel_mask]
        return l2(act1, act2)


def cli_main(args=None):
    import argparse
    import pickle
    from src.models.frank import FrankeinsteinNet

    parser = argparse.ArgumentParser(description='Simple settings.')
    parser.add_argument('pickle', help='Path to pickle')
    parser.add_argument('mode', help='Before, after, ps_inv')
    args = parser.parse_args(args)

    data_dict = pickle.load(open(args.pickle, "br"))

    frank_model = FrankeinsteinNet.from_data_dict(data_dict, args.mode)
    comparator = CorrelationSplitComparator.from_frank_model(frank_model, m1='frank', m2='end')

    results = comparator(data_dict['params']['dataset'])
    print(results)


if __name__ == '__main__':
    cli_main()
