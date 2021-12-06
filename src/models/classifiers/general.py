import logging
from functools import partial

import torch
import torch.nn as nn
from dotmap import DotMap

from src.utils import config
from src.utils.accessor import rgetattr

logger = logging.getLogger("general_model")


class GeneralNet(nn.Module):
    def __init__(self, n_classes, n_channels_in, model_path=None, seed=None):
        super(GeneralNet, self).__init__()

        # Seed for init
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)

        # Number of output classes and input channels
        self.n_classes = n_classes
        self.n_channels_in = n_channels_in

        self.activations = DotMap()
        self.model_path = model_path
        self.to_transform = []
        self.hooks = {}

    # ==============================================================
    # PROPERTIES
    # ==============================================================

    @property
    def device(self):
        ''' Cuda or CPU '''
        return next(self.parameters()).device

    @property
    def layer_info(self):
        ''' List of modules in features '''
        layer_info = [name for (name, _) in self.named_modules()]
        # layer_info = [l[:-7] for l in layer_info if l.endswith('.weight')]
        layer_info = [str(x) for x in layer_info]
        return layer_info

    @property
    def layers(self):
        ''' Name of available conv layers (features only) '''
        return [name for (name, _) in self.to_transform]

    @property
    def name(self):
        ''' Name of the model '''
        return str(self.__class__.__name__)

    # ==============================================================
    # PUBLIC
    # ==============================================================

    def set_cuda(self):
        device = config.device
        self.to(device)

    def eval_mode(self):
        self.freeze()
        self.eval()

    # def stop_at(self, layer_name):
    #     if 'activation' not in self.hooks:
    #         self.hooks['activation'] = []
    #     module = self.get_layer(layer_name)
    #     args = partial(self._save_activation_no_numpy, layer_name)
    #     hook = module.register_forward_hook(args)
    #     self.hooks['activation'].append(hook)

    # def start_at(self, layer_name):
    #     self.forced_input = None
    #     self.hooks['forced_output'] = []
    #     module = self.get_layer(layer_name)
    #     hook = module.register_forward_hook(self._override_activation)
    #     self.hooks['forced_output'].append(hook)

    def set_middle_activation(self, activation):
        self.forced_input = activation

    def get_layer(self, key):
        return rgetattr(self, key)

    def freeze(self):
        ''' Freeze entire network '''
        for param in self.parameters():
            param.requires_grad = False

    def register_shape_fw_hooks(self):
        ''' For each layer available to transform, get transformation 
            matrices' size (which is the number of channels of the prev conv
            layer)
        '''
        self.shapes = {}
        self.hooks['shapes'] = []
        for name, module in self.named_modules():
            args = partial(self._save_shape, name)
            hook = module.register_forward_hook(args)
            self.hooks['shapes'].append(hook)

    def simulate_forward_pass(self, shape):
        dump_input = torch.zeros(shape).unsqueeze_(0)
        dump_input = torch.repeat_interleave(dump_input, 3, dim=0)
        self(dump_input.to(self.device))

    def register_freeze_fw_hooks(self):
        ''' Used to determine which layers are frozen '''
        self.is_frozen = {}
        for module in self.modules():
            module.register_forward_hook(self._is_frozen)

    # def register_activation_fw_hooks(self):
    #     ''' Register activation save on each neuron '''
    #     self.hooks['activation'] = []
    #     for name, module in self.named_modules():
    #         args = partial(self._save_activation, name)
    #         hook = module.register_forward_hook(args)
    #         self.hooks['activation'].append(hook)

    def register_order_fw_hooks(self):
        ''' Register forward call order'''
        self.hooks['forward_order'] = []
        self.forward_order = []
        for name, module in self.named_modules():
            args = partial(self._save_name, name)
            hook = module.register_forward_hook(args)
            self.hooks['forward_order'].append(hook)

    def remove_all_hooks(self):
        if hasattr(self, 'hooks') and isinstance(self.hooks, dict):
            keys_to_remove = [key for key, _ in self.hooks.items()]
            for key in keys_to_remove:
                self._remove_hooks(key)

    def remove_activation_fw_hooks(self):
        self._remove_hooks('activation')

    def remove_shape_fw_hooks(self):
        self._remove_hooks('shape')

    def remove_forced_input_fw_hooks(self):
        self._remove_hooks('forced_input')

    def remove_order_fw_hooks(self):
        self._remove_hooks('forward_order')

    # ==============================================================
    # PRIVATE HELPERS
    # ==============================================================

    def _finish_init(self):
        ''' This function is called after initialization '''
        if self.model_path is not None:
            self.load_state_dict(torch.load(self.model_path,
                                            map_location=self.device),
                                 strict=False)

    def _save_shape(self, name, module, inp, out):
        ''' Save shape of indexed modul '''
        inp = inp[0]  # Assumes multiple inputs

        # Overriding CatLayer input
        if not isinstance(inp, tuple) and not isinstance(inp, list):
            inp_shape = inp.shape[1:]
        else:
            inp_shape = None
        out_shape = out.shape[1:]
        self.shapes[name] = {'in': inp_shape, 'out': out_shape}

    def _is_frozen(self, module, inp, out):
        ''' Save if indexed modul is frozen '''
        for name, parameter in module.named_parameters():
            self.is_frozen[name] = parameter.requires_grad

    def _save_name(self, name, module, m_in, m_out):
        if name not in self.forward_order:
            self.forward_order.append(name)

    # def _save_activation(self, name, module, m_in, m_out):
    #     self.activations[name] = m_out.detach().cpu().numpy()

    def _save_activation_no_numpy(self, name, module, m_in, m_out):
        self.activations[name] = m_out

    def _override_activation(self, module, m_in, m_out):
        if self.forced_input is None:
            act = m_out
        else:
            act = self.forced_input
            self.forced_input = None
        return act

    def _remove_hooks(self, key):
        if key in self.hooks:
            for hook in self.hooks[key]:
                hook.remove()
            del self.hooks[key]
