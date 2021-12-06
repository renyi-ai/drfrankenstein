from functools import partial
from typing import List, Dict, Any

import numpy as np
import torch
from dotmap import DotMap

from src.comparators import get_comparator_function
from src.comparators.comparator_base import ComparatorBaseClass
from src.dataset.utils import _get_data_loader
from src.utils import config


class ActivationComparator(ComparatorBaseClass):
    """ This class compares two models' activations at the desires neurons. """

    def __init__(self, model1, model2, layer_name1, layer_name2):
        models = {
            "model1": model1.to(config.device),
            "model2": model2.to(config.device),
        }
        super().__init__(models=DotMap(models))
        self.layer_names = [layer_name1, layer_name2]
        self.measurements = dict()

    @classmethod
    def from_frank_model(cls, frank_model, m1='front', m2='end'):
        models = {
            'front': frank_model.front_model,
            'end': frank_model.end_model,
            'frank': frank_model
        }
        layers = {
            'front': frank_model.front_layer_name,
            'end': frank_model.end_layer_name,
            'frank': 'transform'
        }
        return cls(models[m1], models[m2], layers[m1], layers[m2])

    # ==============================================================
    # PUBLIC FUNCTIONS
    # ==============================================================

    def __call__(self,
                 dataset_name: str,
                 measure_names: List[str],
                 batch_size: int = 200,
                 group_at: float = float('inf'),
                 stop_at: float = float('inf'),
                 dataset_type: str = 'val') -> Dict[str, Dict[str, Any]]:
        """Running the desired measurements on the given dataset.

        Args:
            dataset_name (str): 
                Name of dataset (e.g.: 'mnist')

            measure_names (List[str]):
                Measurements needed. (e.g.: ['cka'])

            batch_size (int, optional):
                Batch size of data_loader.
                Defaults to 200.

            group_at (float, optional):
                Average results over a group of number of images.
                E.g. if set to 2000, measurements calculated in groups of 2000
                images and then averaged. This saves a lot of RAM.
                Defaults to float('inf').

            stop_at (float, optional):
                Max iteration number.
                Defaults to float('inf')

            dataset_type (str, optional):
                Run it on 'train' or 'val' dataset.
                Defaults to 'val'.

        Returns:
            Dict[str, Dict[str, Any]]:
                A dictionary of results. E.g.:
                {
                    'cka'    : {'value' : 0.72},
                    'mse'    : {'value' : 0.32},
                    'ps_inv' : {'w' : torch.tensor, 'b' : torch.tensor}
                }
        """
        # Register hooks, put models to eval mode
        self._prepare_models()
        # Final measurements

        data_loader = _get_data_loader(dataset_name,
                                       dataset_type,
                                       batch_size=batch_size,
                                       seed=0)
        accumulate_fn = partial(self._accumulate, measure_names)
        self.iterate_through_data(data_loader,
                                  group_at=group_at,
                                  stop_at=stop_at,
                                  accumulate_callback=accumulate_fn,
                                  step_callback=self._step)

        # Remove hooks used for this specific comparison
        self.hook_store.clear_store()

        # Get mean values for each measurement groups.
        # Note: If group_at=infinity, the mean is calculated over
        # a single, large activation batch, so it's equivalent of
        # simply removing one dimension over the calculated measures
        # (e.g. measurements['cka']['value'] = np.mean([0.73]) --> 0.73)
        measurements = self._mean_values(self.measurements)
        return measurements

    # ==============================================================
    # PRIVATE FUNCTIONS
    # ==============================================================

    def _step(self, model, inputs, targets):
        inputs = inputs.to(model.device)
        model(inputs)

    def _accumulate(self, measure_names):
        new_values = self._calculate_measures(measure_names)
        if new_values is not None:
            self._update_measurements(self.measurements, new_values)

    def _calculate_measures(self, measure_names):
        results = {}
        activations = [torch.cat(
            self.hook_store.get_and_clear_cache(key),
            dim=0) for key in self.models.keys()]

        any_numpy = not np.all(['torch' in m for m in measure_names])
        if any_numpy:
            activations = [a.numpy() for a in activations]
        else:
            activations = [a.to(config.device) for a in activations]

        if len(activations[0]) == 0:
            return None
        for measure_name in measure_names:
            print(f" Calculating measure: {measure_name} ...")
            measure_function = get_comparator_function(measure_name)
            measurements = measure_function(*activations)

            if not any_numpy:
                measurements = measurements.detach().cpu().numpy()

            results[measure_name] = measurements
        results['count'] = len(activations[0])
        return results

    def _update_measurements(self, orig, new):
        ''' Updating results by new group measurements '''
        # Loop throught measures such as cka, ps_inv, mse, etc.
        for new_name, new_item in new.items():

            is_dict = isinstance(new_item, dict)
            exist = new_name in orig

            if is_dict:
                orig[new_name] = {} if not exist else orig[new_name]
                orig[new_name] = self._update_measurements(orig[new_name],
                                                           new_item)
            else:
                orig[new_name] = [] if not exist else orig[new_name]
                orig[new_name].append(new_item)

        return orig

    def _mean_values(self, measurements):
        '''
        Get mean values over a measurement group. E.g. if given the following
        dict:
            results = {
                'cka'    : [0.5, 0.6, 0.7],
                'ps_inv' : {'w' : [tensorA1, tensorA2, tensorA3],
                            'b' : [tensorB1, tensorB2, tensorB3]
            }
        then it will return:
            results = {
                'cka'    : 0.6,
                'ps_inv' : {'w' : tensorA,
                            'b' : tensorB
            }
        where tensorA has the same shape as [tensorA1, tensorA2, tensorA3],
        representing the mean of the tensors, and the same applies for tensorB.
        If the input lists has only one element, then the mean will equal to
        that one element, so this method also works when we do not apply
        groupping on activation comparisons.

        Args:
            measurements (Dict[str : Dict[str : Any]]): The measurements taken.

        Returns:
            Dict[str : Dict[str : Any]]: The mean values.
        '''
        counts = None
        if 'counts' in measurements.keys():
            counts = [measurements['counts']]

        for name, item in measurements.items():
            if name == 'counts':
                continue
            is_dict = isinstance(item, dict)
            if is_dict:
                # dict
                measurements[name] = self._mean_values(item)
            else:
                # list
                measurements[name] = np.average(item, axis=0, weights=counts)
        return measurements

    def _prepare_models(self):
        ''' Prepares models for the comparison of some layers '''
        for model in self.models.values():
            model.eval()
        self._install_hooks()

    def _install_hooks(self):
        ''' For each model it registers an activation save at the desired layers '''
        for model_key, lname in zip(self.models.keys(), self.layer_names):
            layer = self.models[model_key].get_layer(lname)
            self.hook_store.register_activation_saver(layer, model_key)
