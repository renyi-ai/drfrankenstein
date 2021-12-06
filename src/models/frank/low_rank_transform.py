import torch
from torch import nn


class LowRankTransform(nn.Module):

    def __init__(self,
                 in_shape,
                 out_shape,
                 init_fn=None,
                 init_weights=None,
                 flatten=False,
                 rank=32):
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
        self.flatten = flatten
        self.rank = rank

        self.shape = self._create_trans_layer()

        self.reset_parameters()

    def _flatten_forward(self, x):
        # Shapes
        batch_size = x.shape[0]
        target_shape = (batch_size, *self.out_shape)

        # Forward
        x = x.view(batch_size, -1)
        M = torch.mm(self.w1, self.w2)
        x = torch.mm(x, M)
        x = x.view(target_shape)
        x = x + self.bias

        return x

    def _conv1x1_forward(self, x):
        x = x.permute(0, 2, 3, 1).unsqueeze(3)

        M = torch.mm(self.w1, self.w2)
        x = torch.matmul(x, M)
        # x = torch.matmul(x, self.w1)

        x = x.squeeze(3).permute(0, 3, 1, 2)
        x = x + self.bias.reshape(1, x.shape[1], 1, 1)
        return x

    def forward(self, x):
        if self.flatten:
            return self._flatten_forward(x)
        else:
            return self._conv1x1_forward(x)

    def _create_trans_layer(self):
        in_features = self.in_shape[0]
        out_features = self.out_shape[0]

        self.w1 = nn.Parameter(torch.Tensor(in_features, self.rank))
        self.w2 = nn.Parameter(torch.Tensor(self.rank, out_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))

    def reset_parameters(self):

        torch.nn.init.orthogonal_(self.w1)
        torch.nn.init.orthogonal_(self.w2)

        # torch.nn.init.kaiming_uniform_(self.w1, a=0, mode='fan_in')
        # torch.nn.init.kaiming_uniform_(self.w2, a=0, mode='fan_in')

        # torch.nn.init.normal_(self.w1)
        # torch.nn.init.normal_(self.w2)
        torch.nn.init.zeros_(self.bias)
        return
        # torch.nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        # if self.bias is not None:
        #     fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.w1)
        #     bound = 1 / math.sqrt(fan_in)
        #     torch.nn.init.uniform_(self.bias, -bound, bound)
        # if self.init_weights is not None:
        #    self.load_trans_matrix(self.init_weights[0], self.init_weights[1])
        # elif self.init_fn is not None:
        #    self.load_trans_matrix(*self.init_fn(self.shape))

    def load_trans_matrix(self, param_dict):
        self.w1.data.copy_(param_dict["w1"])
        self.w2.data.copy_(param_dict["w2"])
        self.bias.data.copy_(param_dict["b"])
        ''' Load in weights and bias and ensure they are in right format '''
        """
        # Check if they are tensors
        if not isinstance(w, torch.Tensor):
            w = torch.from_numpy(w)
        if not isinstance(b, torch.Tensor):
            b = torch.from_numpy(b)
        # Check if they are in right shape
        if isinstance(self.transform, nn.Conv2d) and len(w.shape) == 2:
            w = w.view(*w.shape, 1, 1)
        self.w1.data.copy_(w1)
        self.w2.data.copy_(w2)
        """

    def get_param_dict(self):
        w = torch.mm(self.w1, self.w2)
        w1 = self.w1.detach().cpu().numpy()
        w2 = self.w2.detach().cpu().numpy()
        b = self.bias.detach().cpu().numpy()
        w = w.detach().cpu().numpy()[..., None, None]
        return {'w': w, "w1": w1, "w2": w2, 'b': b}
