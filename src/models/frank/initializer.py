import numpy as np
import torch

from src.comparators import ActivationComparator


def identity_init(shape):
    assert len(shape) == 2
    assert shape[0] == shape[1]
    return torch.eye(shape[0]), torch.zeros(shape[0])


def ones_w_zeros_b_init(shape):
    assert len(shape) == 2
    assert shape[0] == shape[1]
    n = shape[0]
    return torch.ones((n, n)), torch.zeros(n)


def random_permutation_mask_init(shape):
    fake_corr = np.random.random(shape)
    max_indices = fake_corr.argmax(axis=1)
    all_indices = np.arange(shape[1])
    mask = max_indices[:, None] == all_indices
    assert shape[0] == mask.shape[1]
    assert shape[1] == mask.shape[0]
    return mask


def permutation_init(shape):
    assert len(shape) == 2
    assert shape[0] == shape[1]
    n = shape[0]
    p = torch.randperm(n)
    w = torch.zeros((n, n))
    for i in range(n):
        w[i, p[i]] = 1
    b = torch.zeros(n)
    return w, b


class PsInvInit:
    def __init__(self,
                 front_model,
                 end_model,
                 front_layer_name,
                 end_layer_name,
                 dataset_name,
                 dataset_type,
                 flatten=False):

        if dataset_name == 'celeba':
            # Inception
            batch_size = 50
            group_at = 2000
            stop_at = 5
        elif 'tiny10' in front_model.name.lower():
            # Tiny10
            batch_size = 1000
            group_at = float('inf')
            stop_at = float('inf')
        else:
            # Resnets
            batch_size = 500
            group_at = 5000
            stop_at = float('inf')

        act_comparator = ActivationComparator(front_model, end_model,
                                              front_layer_name, end_layer_name)
        results = act_comparator(dataset_name, ['ps_inv'], batch_size, group_at, stop_at,
                                 dataset_type)['ps_inv']
        self.w, self.b = results['w'], results['b']

    def __call__(self, shape):
        assert shape[0] == self.w.shape[1]
        assert shape[1] == self.w.shape[0]
        return self.w, self.b


class SemiMatchMaskInit:
    def __init__(self,
                 front_model,
                 end_model,
                 front_layer_name,
                 end_layer_name,
                 dataset_name,
                 dataset_type,
                 flatten=False):

        if dataset_name == 'celeba':
            batch_size = 50
            group_at = 2000
        else:
            batch_size = 200
            group_at = 2000

        act_comparator = ActivationComparator(front_model, end_model,
                                              front_layer_name, end_layer_name)
        self.corr = act_comparator(dataset_name, ['corr'], batch_size, group_at,
                                   dataset_type)['corr']

    def __call__(self, shape):
        max_indices = self.corr.argmax(axis=1)
        all_indices = np.arange(self.corr.shape[1])
        mask = max_indices[:, None] == all_indices
        assert shape[0] == mask.shape[1]
        assert shape[1] == mask.shape[0]
        return mask


class AbsSemiMatchMaskInit(SemiMatchMaskInit):

    def __init__(self,
                 front_model,
                 end_model,
                 front_layer_name,
                 end_layer_name,
                 dataset_name,
                 dataset_type,
                 flatten=False):
        super().__init__(front_model, end_model, front_layer_name,
                         end_layer_name, dataset_name, dataset_type, flatten)
        self.corr = np.abs(self.corr)
