import sys

import numpy as np
import torch

if './' not in sys.path:
    sys.path.append('./')

from src.comparators.hook_store import HookStore
from src.comparators.activation_comparator import ActivationComparator
from src.comparators.compare_functions import tsne


class TSNEComparator(ActivationComparator):

    def __init__(self, frank_model):
        self.models = {"frank": frank_model}
        self.hook_store = HookStore()
        self.layer_names = ['transform']
        self.measurements = dict()
        self.labels = []

    def __call__(self, dataset_name: str, batch_size: int = 200, group_at: float = float('inf'),
                 dataset_type: str = 'val', **kwargs):
        return super().__call__(dataset_name, [None],
                                batch_size, group_at, dataset_type=dataset_type)

    def _step(self, model, inputs, targets):
        inputs = inputs.to(model.device)
        model(inputs)
        new_labels = list(targets.detach().cpu().numpy())
        self.labels.extend(new_labels)

    def _calculate_measures(self, measure_names=None):
        # Retreive activations
        activations = [torch.cat(
            self.hook_store.get_and_clear_cache(key),
            dim=0).numpy() for key in self.models.keys()]
        if len(activations[0]) == 0:
            return None

        # Calculate TSNE
        print(f" Calculating measure: TSNE...")
        X = np.array(activations[0])
        X = X.reshape(X.shape[0], -1)
        tsne_results = tsne(X, no_dims=2)

        # Compose results
        results = {}
        results['x'] = np.array(tsne_results)
        results['y'] = np.array(self.labels)
        results['count'] = len(activations[0])

        self.labels = []

        return results


def main(args=None):
    import argparse
    import pickle
    import pandas as pd
    from matplotlib import pyplot as plt
    import seaborn as sns
    from src.utils import config
    from src.models.frank import FrankeinsteinNet

    parser = argparse.ArgumentParser(description='Simple settings.')
    parser.add_argument('pickle', help='Path to pickle')
    parser.add_argument('mode', help='Before, after, ps_inv')
    args = parser.parse_args(args)

    data_dict = pickle.load(open(args.pickle, "br"))

    frank_model = FrankeinsteinNet.from_data_dict(data_dict, args.mode)
    frank_model.to(config.device)
    dataset_name = data_dict['params']['dataset']

    comparator = TSNEComparator(frank_model)
    results = comparator(dataset_name)

    tsne_df = pd.DataFrame(results['x'])
    tsne_df['y'] = results['y']

    sns.FacetGrid(tsne_df, hue="y", size=6).map(plt.scatter, 0, 1).add_legend()
    plt.show()


if __name__ == '__main__':
    main()
