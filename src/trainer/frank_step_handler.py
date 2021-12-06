import torch

from src.models.frank.low_rank_transform import LowRankTransform
from src.models.frank.soft_losses import SoftBCEWithLogitsLoss, SoftCrossEntropyLoss
from src.trainer.model_step_handler import ModelStepHandler
from src.utils import config


class FrankStepHandler(ModelStepHandler):
    def __init__(self,
                 model,
                 optimizer_name='sgd',
                 lr=1e-4,
                 multilabel=False,
                 weight_decay=0.,
                 l1=0.,
                 cka_reg=0,
                 temperature=1.,
                 target_type='hard'):

        # New parameters
        # (these might be needed for parent constructor, e.g. at loss func)
        self.l1 = l1
        self.cka_reg = cka_reg
        self.temperature = temperature
        self.target_type = target_type
        self.transform_history = {}

        # Default parameters
        super().__init__(model, optimizer_name, lr, multilabel, weight_decay)

    @classmethod
    def from_arg_config(cls, conf):
        # Frankenstein model setup
        from src.models.frank.frankenstein import FrankeinsteinNet
        multilabel = conf.dataset == 'celeba'
        model = FrankeinsteinNet.from_arg_config(conf)

        return cls(model,
                   optimizer_name=conf.optimizer,
                   lr=conf.lr,
                   multilabel=multilabel,
                   weight_decay=conf.weight_decay,
                   l1=conf.l1,
                   cka_reg=conf.cka_reg,
                   temperature=conf.temperature,
                   target_type=conf.target_type)

    @classmethod
    def from_data_dict(cls, data_dict, mode):
        from src.models.frank.frankenstein import FrankeinsteinNet
        params = data_dict['params']
        multilabel = params['dataset'] == 'celeba'
        model = FrankeinsteinNet.from_data_dict(data_dict, mode)
        model.to(config.device)
        cka_reg = 0 if 'cka_reg' not in params else params['cka_reg']
        return cls(model,
                   optimizer_name=params['optimizer'],
                   lr=params['lr'],
                   multilabel=multilabel,
                   weight_decay=params['weight_decay'],
                   l1=params['l1'],
                   cka_reg=cka_reg,
                   temperature=params['temperature'],
                   target_type=params['target_type'])

    def _get_loss(self):
        ''' Determines the loss function '''
        is_hard = self.target_type == 'hard'

        # Get base loss
        if is_hard and self.multilabel:
            catregorical_loss = torch.nn.BCEWithLogitsLoss()
        if is_hard and not self.multilabel:
            catregorical_loss = torch.nn.CrossEntropyLoss()
        if not is_hard and self.multilabel:
            catregorical_loss = SoftBCEWithLogitsLoss(self.temperature)
        if not is_hard and not self.multilabel:
            catregorical_loss = SoftCrossEntropyLoss(self.temperature)

        if type(self.model.transform) is LowRankTransform:
            assert self.l1 == 0, "L1 regularization is not supported with LowRankTransform"
            l1_layer = None
        else:
            l1_layer = self.model.transform.transform

        def mixed_loss(outputs, targets, inputs):
            targets_list = self._get_soft_labels(inputs, targets)
            loss = 0
            for targets in targets_list:
                loss += catregorical_loss(outputs, targets)
            loss /= len(targets_list)

            if l1_layer is not None:
                loss += torch.mean(torch.abs(l1_layer.weight)) * self.l1

            if self.cka_reg > 0:
                from src.comparators.compare_functions.cka_torch import cka
                cka_val = cka(self.model.last_m2_out, self.model.forced_output)
                loss += cka_val * self.cka_reg

            return loss

        return mixed_loss

    def save(self, n_iter_ran, save_folder=None):
        from copy import deepcopy
        self.transform_history[n_iter_ran] = deepcopy(self.model.transform.state_dict())
        return

    def append_and_save_log(self, epoch_data, save_folder):
        self.stats = self.stats.append(epoch_data, ignore_index=True)

    def _get_soft_labels(self, inputs, targets):
        ''' Generates labels by the original models' outputs '''
        front_model = self.model.front_model
        end_model = self.model.end_model
        models = [front_model, end_model]

        if self.target_type == 'hard':
            return [targets]
        elif self.target_type == 'soft_1':
            with torch.no_grad():
                outputs = front_model(inputs)
            return [outputs]
        elif self.target_type == 'soft_2':
            with torch.no_grad():
                outputs = end_model(inputs)
            return [outputs]
        elif self.target_type == 'soft_12':
            with torch.no_grad():
                outputs = [model(inputs) for model in models]
            return outputs
        elif self.target_type == 'soft_1_plus_2':
            with torch.no_grad():
                sum_of_acts = front_model(inputs) + end_model(inputs)
                outputs = sum_of_acts / 2
            return [outputs]
