import numpy as np
import torch

from src.comparators import get_comparator_function
from src.comparators.activation_comparator import ActivationComparator


class LabeledActivationComparator(ActivationComparator):
    """ This class compares two models' activations at the desires neurons. """

    def __init__(self, model1, model2, layer_name1, layer_name2):
        super().__init__(model1, model2, layer_name1, layer_name2)
        # naming hax
        model1.idx = 'model1'
        model2.idx = 'model2'
        self.hits = {model: np.array([], dtype=bool) for model in list(self.models.keys())}

    def _step(self, model, inputs, targets):
        inputs = inputs.to(model.device)
        out = model(inputs)
        pred = torch.argmax(out, 1)  # TODO: make it multilabel compatible
        hits = np.array(np.array(targets) == np.array(pred), dtype=bool)
        self.hits[model.idx] = np.append(self.hits[model.idx], hits)
        # print(self.hits)

    # ==============================================================
    # PRIVATE FUNCTIONS
    # ==============================================================

    def _get_hit_masks(self, m1_hits, m2_hits):
        hit_masks = {}
        L = len(m1_hits)

        hit_masks['rr'] = (m1_hits == m2_hits) & m1_hits
        hit_masks['rw'] = (m1_hits != m2_hits) & m1_hits
        hit_masks['wr'] = (m1_hits != m2_hits) & m2_hits
        hit_masks['ww'] = (m1_hits == m2_hits) & np.logical_not(m1_hits)

        return hit_masks

    def _calculate_measures(self, measure_names):
        results = {}
        activations = [torch.cat(
            self.hook_store.get_and_clear_cache(key),
            dim=0).numpy() for key in self.models.keys()]
        if len(activations[0]) == 0:
            return None
        model_names = list(self.models.keys())

        hits = self.hits
        hit_masks = self._get_hit_masks(hits[model_names[0]], hits[model_names[1]])

        for measure_name in measure_names:
            labeled_activations = {mode: [activations[0][hit_masks[mode]], activations[1][hit_masks[mode]]] for mode in
                                   hit_masks}
            measure_function = get_comparator_function(measure_name)
            labeled_measurements = {label: measure_function(*labeled_activations[label]) for label in
                                    labeled_activations}
            # measurements = measure_function(*activations)

            results[measure_name] = labeled_measurements
        results['frank_acc'] = hits[model_names[0]].sum() / len(hits[model_names[0]])
        results['count'] = len(activations[0])
        # Purge hits
        self.hits = {model: np.array([], dtype=bool) for model in list(self.models.keys())}

        return results
