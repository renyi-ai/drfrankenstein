import numpy as np
import torch
from torch import nn
from torch.nn.utils import prune


class Transform(nn.Module):
    def __init__(self,
                 in_shape,
                 out_shape,
                 init_fn=None,
                 init_weights=None,
                 mask_fn=None,
                 mask_weights=None,
                 flatten=False):
        """Applies a linear transformations between two layers.

        Args:
            in_shape (tuple):
                Shape of model1 activation, excluding batch size.
            out_shape (tuple):
                Shape of model2 activation, excluding batch size.
            init_fn (callable):
                A callable which returns the initialization value of w and b parameters.
                The argument of this callable is the shape of the w parameter
                init_fn and init_weights can't be specified at the same time
            init_weights (tuple):
                Predefined initialization values of w and b parameters
                init_fn and init_weights can't be specified at the same time
            mask_fn (callable):
                A callable which returns the initialization value of a mask.
                The argument of this callable is the shape of the mask.
                mask_fn and mask_weights can't be specified at the same time
            mask_weights (tuple):
                Predefined initialization values of mask
                mask_fn and mask_weights can't be specified at the same time
            flatten (bool, optional):
                Either to flatten both layers and apply pixel2pixel
                transformation or not. If shapes to not meet the conditions of a
                conv1x1 transformation, it is automatically set to True.
                Defaults to False.
        """
        assert init_fn is None or init_weights is None, "init_fn and init_weights can't be specified at the same time"
        assert init_weights is None or len(init_weights.shape) == 2
        super().__init__()

        self.in_shape = in_shape
        self.out_shape = out_shape
        self.init_fn = init_fn
        self.init_weights = init_weights
        self.mask_fn = mask_fn
        self.mask_weights = mask_weights
        self.flatten = flatten

        self.transform, self.shape = self._get_trans_layer()

        self.mask = None
        self.reset_parameters()

    def _flatten_forward(self, x):
        # Shapes
        batch_size = x.shape[0]
        target_shape = (batch_size, *self.out_shape)

        # Forward
        x = x.view(batch_size, -1)
        x = self.transform(x)
        x = x.view(target_shape)

        return x

    def load_trans_matrix(self, param_dict):
        ''' Load in weights and bias and ensure they are in right format '''

        if type(param_dict) is tuple:
            w, b = param_dict
        else:
            w = param_dict["w"]
            b = param_dict["b"]
        # Check if they are tensors
        if not isinstance(w, torch.Tensor):
            w = torch.from_numpy(w)
        if not isinstance(b, torch.Tensor):
            b = torch.from_numpy(b)

        # Check if they are in right shape
        if isinstance(self.transform, nn.Conv2d) and len(w.shape) == 2:
            w = w.view(*w.shape, 1, 1)

        self.transform.weight.data.copy_(w)
        self.transform.bias.data.copy_(b)

    def load_mask(self, mask):
        ''' Load mask and ensure they are in right format '''
        # Check if they are tensors
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask)

        mask = mask.view(*self.transform.weight.shape)
        prune.custom_from_mask(self.transform, 'weight', mask)

        self.mask = mask

    def _conv1x1_forward(self, x):
        return self.transform(x)

    def forward(self, x):
        if self.flatten:
            return self._flatten_forward(x)
        else:
            return self._conv1x1_forward(x)

    def _get_trans_layer(self):
        ''' Return either Pixel2Pixel or Channel2Channel transformation and the corresponding shape'''
        return self._p2p() if self.flatten else self._c2c()

    def _p2p(self):
        ''' Pixel2Pixel transformation '''
        shape = [int(np.prod(self.in_shape)), int(np.prod(self.out_shape))]
        return nn.Linear(shape[0], shape[1]), shape

    def _c2c(self):
        ''' Channel2Channel transformation '''
        shape = [self.in_shape[0], self.out_shape[0]]
        return nn.Conv2d(shape[0], shape[1], kernel_size=1), shape

    def reset_parameters(self):
        # Weight and bias
        if self.init_weights is not None:
            self.load_trans_matrix(self.init_weights[0], self.init_weights[1])
        elif self.init_fn is not None:
            self.load_trans_matrix(self.init_fn(self.shape))

        # Mask
        if self.mask_weights is not None:
            self.load_mask(self.mask_weights)
        elif self.mask_fn is not None:
            self.load_mask(self.mask_fn(self.shape))

    def get_param_dict(self):
        w = self.transform.weight.detach().cpu().numpy()
        b = self.transform.bias.detach().cpu().numpy()
        return {'w': w, 'b': b}
