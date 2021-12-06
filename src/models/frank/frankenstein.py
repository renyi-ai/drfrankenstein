from typing import Any

from torch import nn

from src.dataset import get_datasets
from src.models.classifiers.model_constructor import load_from_path
from src.models.frank.initializer import identity_init, ones_w_zeros_b_init, permutation_init, \
    random_permutation_mask_init, PsInvInit, SemiMatchMaskInit, AbsSemiMatchMaskInit
from src.models.frank.low_rank_transform import LowRankTransform
from src.models.frank.transform import Transform
from src.utils import config
from src.utils.accessor import rgetattr


class FrankeinsteinNet(nn.Module):
    """ This class is responsible for inserting a linear transformation
        layer between two networks with identical architecture. The user has
        a choice to index at which layer should the transformation happen.
        If the indexed layer is a convolution, the first model is transferred
        to the second one by a conv1x1 layer. If the layer is not a convolution
        but a linear layer, the transformation will be a linear layer too.
    """

    # ==============================================================
    # CONSTRUCTORS
    # ==============================================================

    def __init__(self,
                 front_model_path,
                 end_model_path,
                 front_layer_name,
                 end_layer_name,
                 dataset_name: str,
                 init='random',
                 mask=None,
                 flatten=False,
                 low_rank=None):

        super().__init__()

        # Init variables
        self.front_model_path = front_model_path
        self.end_model_path = end_model_path
        self.front_layer_name = front_layer_name
        self.end_layer_name = end_layer_name
        self.dataset_name = dataset_name
        self.init_type = init
        self.mask_type = mask
        self.flatten = flatten
        self.low_rank = low_rank

        # Derived variables
        self.front_model = load_from_path(self.front_model_path)
        self.end_model = load_from_path(self.end_model_path)
        self.models = [self.front_model, self.end_model]
        self.layer_names = [self.front_layer_name, self.end_layer_name]

        # Prepare models and layers for transformation
        self.transform_input = None  # The tensor passed to transformation
        self.forced_output = None  # The tensor passed from front to end
        self.connection_enabled = False  # Either tensor should be passed or not
        self.last_m2_out = None
        self.prepare_models()

        # Define transformation layer
        self.transform = self._get_transform_layer()

        self.seed = -1  # TODO: trainer.py needs this, should be refactored
        self.name = 'Frank'  # TODO: trainer.py needs this, should be refactored

    @classmethod
    def from_arg_config(cls, conf: Any):
        ''' Construct from bash arguments '''
        return cls(conf.front_model,
                   conf.end_model,
                   conf.front_layer,
                   conf.end_layer,
                   conf.dataset,
                   init=conf.init,
                   mask=conf.mask,
                   flatten=conf.flatten,
                   low_rank=conf.low_rank)

    @classmethod
    def from_data_dict(cls, data_dict: dict, mode: str):
        ''' Construct from saved dictionary of pickle '''

        # Construct frank model
        params = data_dict['params']
        frank_model = cls(
            params['front_model'],
            params['end_model'],
            params['front_layer'],
            params['end_layer'],
            params['dataset'],
            init='random',  # Weights loaded after init
            mask=None,  # Loaded after init 
            flatten=params['flatten'])

        # Load weights
        frank_model.transform.load_trans_matrix(data_dict['trans_m'][mode])

        # Load mask
        if 'mask' in data_dict['trans_m']:
            mask = data_dict['trans_m']['mask']['w']
            frank_model.transform.load_mask(mask)

        return frank_model

    @property
    def device(self):
        ''' Cuda or CPU '''
        return next(self.parameters()).device

    # ==============================================================
    # PUBLIC FUNCTIONS
    # ==============================================================

    def forward(self, orig_input):
        # Enabling overriding activations
        self.connection_enabled = True
        # Get middle activations & save middle activation
        self.front_model(orig_input)
        # Transform the activations and save it
        self.forced_output = self.transform(self.transform_input)
        # Load forced output from hook end run the rest of the model
        out = self.end_model(orig_input)
        # Disabling overriding activations
        self.connection_enabled = False
        return out

    def get_layer(self, name):

        # Possibility of transform layer
        if name == 'transform':
            return self.transform.transform

        # Possibility that layer does not exist in front model
        if name not in self.front_model.forward_order:
            return rgetattr(self.end_layer)

        # Possibility that layer exist in front model
        layer_i = self.front_model.forward_order.index(name)
        stop_i = self.front_model.forward_order.index(self.front_layer_name)
        if layer_i <= stop_i:
            return self.front_model.get_layer(name)
        else:
            return self.end_model.get_layer(name)

    def prepare_models(self):
        self._register_connection()
        for model in self.models:
            model.eval_mode()

    def set_cuda(self):
        self.to(config.device)

    # ==============================================================
    # PRIVATE HELPERS
    # ==============================================================

    def _register_connection(self):
        ''' Register activation save on each neuron '''
        self._register_activation_save()
        self._register_activation_load()

    def _register_activation_save(self):
        def save_activation(module, m_in, m_out):
            if self.connection_enabled:
                self.transform_input = m_out

        connect_layer = self.front_model.get_layer(self.front_layer_name)
        connect_layer.register_forward_hook(save_activation)

    def _register_activation_load(self):
        def override_activation(module, m_in, m_out):
            saved_output_exist = self.forced_output is not None
            should_override = saved_output_exist and self.connection_enabled
            activation = self.forced_output if should_override else m_out
            self.last_m2_out = m_out
            return activation

        connect_layer = self.end_model.get_layer(self.end_layer_name)
        connect_layer.register_forward_hook(override_activation)

    def _get_transform_layer(self):
        # Calculate each layers shape
        front_shape, end_shape = self._determine_trans_shapes()
        if self.low_rank is not None and self.low_rank > 0:
            transform_layer = LowRankTransform(front_shape,
                                               end_shape,
                                               init_fn=self._get_transform_init(),
                                               flatten=self.flatten,
                                               rank=self.low_rank)
        else:
            transform_layer = Transform(front_shape,
                                        end_shape,
                                        init_fn=self._get_transform_init(),
                                        mask_fn=self._get_mask_init(),
                                        flatten=self.flatten)
        return transform_layer

    def _determine_trans_shapes(self):
        # Get input shape
        inp_shape = get_datasets(self.dataset_name)['train'][0][0].shape
        # Ask models to save activation shapes and make a forward pass
        for model in self.models:
            model.register_shape_fw_hooks()
            model.register_order_fw_hooks()
            model.simulate_forward_pass(inp_shape)
        # Save shapes
        front_shape = self.front_model.shapes[self.front_layer_name]['out']
        end_shape = self.end_model.shapes[self.end_layer_name]['out']
        # Remove forward hooks to speed up networks
        for model in self.models:
            model.remove_shape_fw_hooks()
            model.remove_order_fw_hooks()

        return front_shape, end_shape

    def _get_transform_init(self):
        name = self.init_type.lower()
        if name == 'random':
            return None  # this is the default behaviour of Transform
        elif name in ['identity', 'eye']:
            return identity_init
        elif name in ['ones-zeros']:
            return ones_w_zeros_b_init
        elif name in ['perm' or 'permutation']:
            return permutation_init
        elif name in ['ps_inv' or 'pseudo_inverse']:
            return PsInvInit(self.front_model.to(config.device),
                             self.end_model.to(config.device),
                             self.front_layer_name,
                             self.end_layer_name,
                             self.dataset_name,
                             dataset_type="train",
                             flatten=self.flatten)
        else:
            raise ValueError('Initializer {} is unknown.'.format(name))

    def _get_mask_init(self):
        if self.mask_type is None:
            return None  # No mask needed

        name = self.mask_type.lower()
        if name in ['identity']:
            return None
        elif name == 'random-permutation':
            return random_permutation_mask_init
        elif name == 'semi-match':
            return SemiMatchMaskInit(self.front_model,
                                     self.end_model,
                                     self.front_layer_name,
                                     self.end_layer_name,
                                     self.dataset_name,
                                     dataset_type="train",
                                     flatten=self.flatten)
        elif name == 'abs-semi-match':
            return AbsSemiMatchMaskInit(self.front_model,
                                        self.end_model,
                                        self.front_layer_name,
                                        self.end_layer_name,
                                        self.dataset_name,
                                        dataset_type="train",
                                        flatten=self.flatten)
        else:
            raise ValueError(f'Maks initializer {name} is unknown.')
