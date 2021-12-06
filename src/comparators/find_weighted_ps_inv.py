import math
import os
import pickle
from copy import deepcopy
from datetime import datetime
from functools import partial
from os import path

import numpy as np
import torch
from dotmap import DotMap
from torch import Tensor
from torch.nn import functional as F, Parameter
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.comparators.comparator_base import ComparatorBaseClass
from src.dataset import get_datasets
from src.models import load_from_path
from src.models.frank.frankenstein import FrankeinsteinNet
from src.models.frank.soft_losses import SoftCrossEntropyLoss
from src.trainer import FrankStepHandler
from src.utils import config
from src.utils.eval_values import eval_net


class Multiplicative(torch.nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features, out_features):
        super(Multiplicative, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input: Tensor) -> Tensor:
        x = self.weight * input
        return torch.prod(x, -1)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )

    def describe(self, **args):
        all_w_names = args["all_w_names"]
        parameters = {}
        for i, name in enumerate(all_w_names):
            parameters[name] = self.weight[:, i].item()
        return parameters


class Linear(torch.nn.Linear):
    def describe(self, **args):
        all_w_names = args["all_w_names"]
        parameters = {}
        for i, name in enumerate(all_w_names):
            parameters[name] = self.weight[:, i].item()
        return parameters


class ReluSegments(torch.nn.Module):

    def __init__(self, segment_num=100, learned_beta=True):
        super().__init__()
        self.segment_num = segment_num
        self.learned_beta = learned_beta
        self.alpha = Parameter(torch.Tensor(segment_num))

        if learned_beta:
            self.beta = Parameter(torch.Tensor(segment_num))
        self.register_buffer("step", torch.arange(0, segment_num))
        self.relu = torch.nn.ReLU(inplace=True)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.uniform(self.alpha)
        if self.learned_beta:
            torch.nn.init.ones_(self.beta)
            with torch.no_grad():
                self.beta /= self.segment_num

    def forward(self, x):
        assert len(x.shape) == 2
        assert x.shape[1] == 1
        # x = x.repeat(1, self.segment_num)
        x = x.expand(x.shape[0], self.segment_num)
        if self.learned_beta:
            beta = self.beta * self.step
        else:
            beta = self.step
        x = self.alpha * self.relu(x - beta)
        return x.sum(-1)

    def describe(self, **args):
        alpha = self.alpha.detach().cpu().numpy()
        if self.learned_beta:
            beta = self.beta * self.step
        else:
            beta = self.step
        beta = beta.detach().cpu().numpy()
        return {
            "alpha": list(alpha),
            "beta": list(beta),
            # "formula": f"sum( {[a, b for a, b in zip(alpha, beta)]} )"
        }


class FindWeightedPsInv(ComparatorBaseClass):

    def __init__(self, frank_model, ps_inv_model, front_model, end_model,
                 out_file, original_data_dict, backward_mode="mean",
                 verbose=False):
        """

        Args:
            frank_model:
            ps_inv_model:
            front_model:
            end_model:
            backward_mode: mean or max
        """
        models = {
            "frank_model": frank_model,
            "ps_inv_model": ps_inv_model,
            "front_model": front_model,
            "end_model": end_model
        }
        super().__init__(models=DotMap(models))
        self.backward_mode = backward_mode
        self.out_file = out_file
        self.original_data_dict = original_data_dict
        self.verbose = verbose
        self.model = deepcopy(ps_inv_model)
        self.model.prepare_models()
        self.transform = self.model.transform

    def step_callback(self, model, inputs, targets):
        inputs = inputs.to(model.device)
        targets = targets.to(model.device)
        outputs = model(inputs)
        if self.backward_mode == "mean":
            # outputs = torch.softmax(outputs, dim=1)
            outputs.mean().backward()
        elif self.backward_mode == "max":
            # outputs = torch.softmax(outputs, dim=1)
            outputs.max(dim=-1)[0].mean().backward()
        elif self.backward_mode == "hard_loss":
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
        elif self.backward_mode == "soft2_loss":
            active_model = self.hook_store.active_model_name
            self.hook_store.active_model_name = ""
            targets = self.models.end_model(inputs).detach()
            self.hook_store.active_model_name = active_model
            loss = SoftCrossEntropyLoss()(outputs, targets)
            loss.backward()
        else:
            raise NotImplementedError

        model.zero_grad()

    def register_hooks(self):
        front_layer_name = self.models["frank_model"].front_layer_name
        end_layer_name = self.models["frank_model"].end_layer_name

        front_layer = self.models.front_model.get_layer(front_layer_name)
        end_layer = self.models.end_model.get_layer(end_layer_name)

        ps_inv_transform = self.models.ps_inv_model.transform
        frank_transform = self.models.frank_model.transform

        self.hook_store.register_activation_saver(front_layer, f"front_act",
                                                  model_name="front_model")
        self.hook_store.register_activation_saver(end_layer, f"end_act",
                                                  model_name="end_model")
        self.hook_store.register_gradient_saver(front_layer, f"front_grad",
                                                model_name="front_model")
        self.hook_store.register_gradient_saver(end_layer, f"end_grad",
                                                model_name="end_model")

        self.hook_store.register_activation_saver(ps_inv_transform,
                                                  f"ps_inv_act",
                                                  model_name="ps_inv_model")
        self.hook_store.register_activation_saver(frank_transform, f"frank_act",
                                                  model_name="frank_model")
        self.hook_store.register_gradient_saver(ps_inv_transform,
                                                f"ps_inv_grad",
                                                model_name="ps_inv_model")
        self.hook_store.register_gradient_saver(frank_transform, f"frank_grad",
                                                model_name="frank_model")

    def _get_data(self):
        frank_acts = self.hook_store.get_and_clear_cache(f"frank_act")
        ps_inv_acts = self.hook_store.get_and_clear_cache(f"ps_inv_act")
        frank_grads = self.hook_store.get_and_clear_cache(f"frank_grad")
        ps_inv_grads = self.hook_store.get_and_clear_cache(f"ps_inv_grad")

        front_acts = self.hook_store.get_and_clear_cache(f"front_act")
        end_acts = self.hook_store.get_and_clear_cache(f"end_act")
        front_grads = self.hook_store.get_and_clear_cache(f"front_grad")
        end_grads = self.hook_store.get_and_clear_cache(f"end_grad")

        frank_acts, ps_inv_acts = torch.cat(frank_acts), torch.cat(ps_inv_acts)
        frank_grads, ps_inv_grads = torch.cat(frank_grads), torch.cat(
            ps_inv_grads)
        front_grads, end_grads = torch.cat(front_grads), torch.cat(end_grads)
        front_acts, end_acts = torch.cat(front_acts), torch.cat(end_acts)
        return frank_acts, ps_inv_acts, front_grads, end_grads, \
               front_acts, end_acts, frank_grads, ps_inv_grads

    def __call__(self, data_loader, str_dataset, find_params=None, input_filter=None,
                 train_params=None):
        # if find params is None the function will just use a single input for weighting
        assert find_params is not None or (input_filter is not None and len(input_filter) == 1)
        if train_params is None:
            train_params = {"epoch": 200, "lr": 0.0001, "batch_size": 64}

        self.register_hooks()
        self.iterate_through_data(data_loader, step_callback=self.step_callback)
        frank_acts, ps_inv_acts, front_grads, end_grads, \
        front_acts, end_acts, frank_grads, ps_inv_grads = self._get_data()

        acc, loss = evaluate(self.model, str_dataset)
        print(f"Acc before: {acc} soft_2 loss: {loss}")
        metrics = {"before_acc": acc, "before_soft2_loss": loss,
                   "find_params": find_params, "train_params": train_params,
                   "str_dataset": str_dataset, "input_filter": input_filter}
        acc, loss = evaluate(self.models.frank_model, str_dataset)
        print(f"Acc franki: {acc} soft_2 loss: {loss}")

        inputs, additional_logs = self.get_inputs(input_filter, end_acts, end_grads, front_acts, front_grads)
        for key in additional_logs.keys():
            metrics[key] = additional_logs[key]

        common_fns = {
            # "**2": lambda x: (normalize_weights(x)) ** 2,
            # "**3": lambda x: (normalize_weights(x)) ** 3,
            # "**4": lambda x: (normalize_weights(x)) ** 4,
        }
        in_features = len(inputs) * (len(common_fns) + 1)

        if find_params is not None:
            mode = find_params["mode"]
        else:
            mode = None

        if mode == "multiplicative":
            module = Multiplicative(in_features, 1)
        elif mode == "linear":
            module = Linear(in_features, 1, bias=False)
            module.weight.data.copy_(torch.ones_like(module.weight))
        elif mode == "relu_segments":
            module = ReluSegments(segment_num=10)
        elif mode is None:
            module = None
        else:
            raise NotImplementedError

        if find_params is None:
            # bs, e, l = None, 1, 0
            pass
        else:
            bs, e, l = find_params["batch_size"], find_params["epoch"], find_params["lr"]
            weights, parameters = find_weighting(module, inputs, common_fns,
                                                 end_acts, frank_acts, ps_inv_acts,
                                                 normalize_weights,
                                                 batch_size=bs, epoch=e, lr=l
                                                 )
            print(f"Mode is {mode}")
            # print(json.dumps(parameters, indent=4))
            print(parameters)

        # ps_inv_params = ps_inv(front_acts, end_acts)

        if "running_mean" in input_filter:
            sorted, indices = torch.sort(end_acts.flatten())
            sorted_diff = ((frank_acts.flatten() - end_acts.flatten()) ** 2)[indices]
            sorted_ps_inv_diff = ((ps_inv_acts.flatten() - end_acts.flatten()) ** 2)[indices]
            window_size = len(sorted) // 100
            unfolded_diff = sorted_diff.unfold(0, window_size, window_size // 2)
            unfolded_ps_inv_diff = sorted_ps_inv_diff.unfold(0, window_size, window_size // 2)
            unfolded_end_acts = sorted.unfold(0, window_size, window_size // 2)
            metrics["diff_means"] = unfolded_diff.mean(-1).cpu()
            metrics["diff_stds"] = unfolded_diff.std(-1).cpu()
            metrics["ps_inv_diff_means"] = unfolded_ps_inv_diff.mean(-1).cpu()
            metrics["ps_inv_diff_stds"] = unfolded_ps_inv_diff.std(-1).cpu()
            metrics["end_act_means"] = unfolded_end_acts.mean(-1).cpu()
            metrics["end_act_stds"] = unfolded_end_acts.std(-1).cpu()

            sorted, indices = torch.sort(end_grads.flatten())
            sorted_diff = ((frank_acts.flatten() - end_acts.flatten()) ** 2)[indices]
            sorted_ps_inv_diff = ((ps_inv_acts.flatten() - end_acts.flatten()) ** 2)[indices]
            sorted_end_acts = end_acts.flatten()[indices]
            window_size = len(sorted) // 100
            unfolded_diff = sorted_diff.unfold(0, window_size, window_size // 2)
            unfolded_ps_inv_diff = sorted_ps_inv_diff.unfold(0, window_size, window_size // 2)
            unfolded_end_grads = sorted.unfold(0, window_size, window_size // 2)
            unfolded_end_acts = sorted_end_acts.unfold(0, window_size, window_size // 2)
            metrics["grad_sorted_diff_means"] = unfolded_diff.mean(-1).cpu()
            metrics["grad_sorted_diff_stds"] = unfolded_diff.std(-1).cpu()
            metrics["grad_sorted_ps_inv_diff_means"] = unfolded_ps_inv_diff.mean(-1).cpu()
            metrics["grad_sorted_ps_inv_diff_stds"] = unfolded_ps_inv_diff.std(-1).cpu()
            metrics["end_grad_means"] = unfolded_end_grads.mean(-1).cpu()
            metrics["end_grad_stds"] = unfolded_end_grads.std(-1).cpu()
            metrics["grad_sorted_end_grad_means"] = unfolded_end_acts.mean(-1).cpu()
            metrics["grad_sorted_end_grad_stds"] = unfolded_end_acts.std(-1).cpu()

            training_stat = {}

        if not (len(input_filter) == 1 and input_filter[0] == "running_mean"):
            print("training transform")
            if find_params is None:
                assert len(inputs) == 1
                weights = list(inputs.values())[0]
            training_stat = train_transform(self.transform, front_acts, end_acts,
                                            weights, lr=train_params["lr"], l1=train_params["l1"],
                                            epoch=train_params["epoch"], batch_size=train_params["batch_size"])

            acc, loss = evaluate(self.model, str_dataset)
            print(f"Acc after : {acc} soft_2 loss: {loss}")
            metrics["after_acc"] = acc
            metrics["after_soft2_loss"] = loss
            metrics["trans_m"] = {"w": self.transform.transform.weight,
                                  "b": self.transform.transform.bias}

        self.hook_store.clear_store()
        self.save_results(metrics, training_stat)

    def get_inputs(self, input_filter, end_acts, end_grads, front_acts, front_grads):
        inputs_fns = {
            "norm_end_acts": lambda ea, eg, fa, fg: normalize_weights(ea),
            "end_acts": lambda ea, eg, fa, fg: ea,
            "end_acts_2": lambda ea, eg, fa, fg: (normalize_weights(ea)) ** 2,
            "end_acts_4": lambda ea, eg, fa, fg: (normalize_weights(ea)) ** 4,
            "end_acts_6": lambda ea, eg, fa, fg: (normalize_weights(ea)) ** 6,
            "end_acts_8": lambda ea, eg, fa, fg: (normalize_weights(ea)) ** 8,
            "end_grads": lambda ea, eg, fa, fg: eg,
            "end_grad_mask": lambda ea, eg, fa, fg: (self.relu_weighter(eg.abs(), type="hard", bn=False)),
            "end_grad_mask_soft": lambda ea, eg, fa, fg: (eg.abs() > 0).to(dtype=torch.float) * 0.5 + 0.5,
            "end_acts_m_grads": lambda ea, eg, fa, fg: ea * eg,
            "front_acts": lambda ea, eg, fa, fg: fa,
            "front_grads": lambda ea, eg, fa, fg: fg,
            "front_acts_m_grads": lambda ea, eg, fa, fg: fa * fg,
            "end_relu_soft": lambda ea, eg, fa, fg: self.relu_weighter(ea, type="soft", bn=False),
            "end_relu_hard": lambda ea, eg, fa, fg: self.relu_weighter(ea, type="hard", bn=False),
            "end_bn_relu_soft": lambda ea, eg, fa, fg: self.relu_weighter(ea, type="soft", bn=True),
            "end_bn_relu_hard": lambda ea, eg, fa, fg: self.relu_weighter(ea, type="soft", bn=True),
            "end_relu_soft_2": lambda ea, eg, fa, fg: self.relu_weighter(ea, type="soft", bn=False) ** 2,
            "end_relu_soft_4": lambda ea, eg, fa, fg: self.relu_weighter(ea, type="soft", bn=False) ** 4,
            "end_relu_soft_6": lambda ea, eg, fa, fg: self.relu_weighter(ea, type="soft", bn=False) ** 6,
            "end_relu_soft_8": lambda ea, eg, fa, fg: self.relu_weighter(ea, type="soft", bn=False) ** 8,
            "online_relu_soft": lambda ea, eg, fa, fg: partial(self.relu_online_weighter, type="soft", bn=False),
            "online_relu_hard": lambda ea, eg, fa, fg: partial(self.relu_online_weighter, type="hard", bn=False),
            "ones": lambda ea, eg, fa, fg: torch.ones_like(ea),
        }
        valid_keys = inputs_fns.keys() if input_filter is None else input_filter
        inputs = {}
        logs = {}
        for key in valid_keys:
            if key in inputs_fns.keys():
                inputs[key] = inputs_fns[key](end_acts, end_grads, front_acts, front_grads)

        for key in input_filter:
            parts = key.split("_")
            if parts[0] in ["percentile", "softpercentile", "onlineth"]:
                per = int(parts[-1])
                type = parts[1]
                if type == "soft":
                    assert len(parts) == 4
                    exp = int(parts[2])
                else:
                    assert len(parts) == 3
                threshold = np.percentile(end_acts, per)
                print(threshold)
                if "thresholds" not in logs.keys():
                    logs["thresholds"] = []
                logs["thresholds"].append(threshold)
                mask = (end_acts >= threshold).to(dtype=torch.float32)
                if parts[0] == "onlineth":
                    inputs[key] = partial(self.relu_online_weighter, type=type, bn=False, shift=threshold)
                else:
                    if "softpercentile" == parts[0]:
                        mask *= 0.9
                        mask += 0.1
                    if type == "hard":
                        inputs[key] = mask
                    elif type == "soft":
                        inputs[key] = (mask * normalize_weights(end_acts)) ** exp
        return inputs, logs

    def save_results(self, metrics, training):
        if self.out_file is None:
            return
        out_dict = metrics
        out_dict["training"] = training
        orig_dict = deepcopy(self.original_data_dict)
        del orig_dict["trans_fit"]
        del orig_dict["trans_m"]
        out_dict["original"] = orig_dict

        with open(self.out_file, 'wb') as handle:
            pickle.dump(out_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def relu_weighter(self, x, type="soft", bn=False):
        x = x.to(config.device)
        if bn:
            bn_layer_name = f"bn{self.models.frank_model.end_layer_name[-1]}"
            bn_layer = self.models.frank_model.end_model.get_layer(
                bn_layer_name)
            x = bn_layer(x)
        relu_x = torch.relu(x)
        if type == "soft":
            weights = relu_x
        elif type == "hard":
            weights = (relu_x > 0).to(dtype=torch.float)
        return weights

    def relu_online_weighter(self, inputs, outputs, targets, type="soft", bn=False, shift=0):
        inputs = inputs - shift
        outputs = outputs - shift
        if bn:
            bn_layer_name = f"bn{self.models.frank_model.end_layer_name[-1]}"
            bn_layer = self.models.frank_model.end_model.get_layer(bn_layer_name)
            inputs = bn_layer(inputs)
            outputs = bn_layer(outputs)
        relu_inputs = torch.relu(inputs)
        relu_outputs = torch.relu(outputs)
        weights = (relu_inputs - relu_outputs).abs()
        if type == "soft":
            weights = normalize_weights(weights)
        elif type == "hard":
            weights = (weights > 0).to(dtype=torch.float)
        return weights


def normalize_weights(weights, one_mean=False):
    weights = weights - weights.min()
    weights = weights / weights.max()
    if one_mean:
        weights = weights + (1 - weights.mean())
    return weights


def find_weighting(module, inputs, common_fns,
                   end_acts, frank_acts, ps_inv_acts,
                   normalize_fn,
                   batch_size=None, epoch=100, lr=0.0001):
    in_keys, ins = list(inputs.keys()), list(inputs.values())

    module.to(config.device)
    weighted_mse = lambda o, t, w: ((o - t) ** 2 * w)
    non_weighted_mse = lambda o, t: ((o - t) ** 2)
    optimizer = torch.optim.Adam(module.parameters(), lr=lr)

    w_inputs = [i.to("cpu") for i in ins if type(i) is Tensor]
    all_w_inputs = []
    all_w_names = []
    for c_name, c_fn in common_fns.items():
        all_w_inputs.extend(c_fn(i) for i in w_inputs)
        all_w_names.extend(f"({name}){c_name}" for name in in_keys)
    all_w_inputs.extend(w_inputs)
    all_w_names.extend(in_keys)

    feature_num = len(all_w_inputs)
    all_w_inputs = [i.to("cpu") for i in all_w_inputs]
    all_w_inputs = torch.stack(all_w_inputs, dim=-1)
    dataset_tensors = [end_acts, frank_acts, ps_inv_acts, all_w_inputs]
    dataset = TensorDataset(*dataset_tensors)

    batch_size = len(dataset) if batch_size is None else batch_size
    data_loader = DataLoader(dataset, batch_size, shuffle=True,
                             pin_memory=True,
                             num_workers=0 if config.debug else 2)
    sum_loss = .0
    sum_mse = .0
    sum_p_w_mse = .0
    out_metrics = {
        "weighted_mse": [],
        "mse": []
    }
    for epoch_num in range(epoch):
        losses = []
        mses = []
        p_w_mses = []
        tqdm_data_iter = tqdm(enumerate(data_loader),
                              ncols=150,
                              total=len(data_loader),
                              leave=False)
        for i, batch in tqdm_data_iter:
            batch_iter = iter(batch)
            e_acts = next(batch_iter).to(config.device)
            f_acts = next(batch_iter).to(config.device)
            p_acts = next(batch_iter).to(config.device)
            w_in = next(batch_iter).to(config.device)

            optimizer.zero_grad()

            w_in = w_in.view(-1, feature_num)
            weights = module(w_in)
            weights = weights.view_as(e_acts)
            weights = normalize_fn(weights, one_mean=True)

            w_mse = weighted_mse(e_acts, f_acts, weights)
            p_w_mse = weighted_mse(e_acts, p_acts, weights)
            mse = non_weighted_mse(e_acts, f_acts)
            loss = w_mse.mean() - p_w_mse.mean()
            loss.backward()
            losses.extend(w_mse.detach().cpu())
            p_w_mses.extend(p_w_mse.detach().cpu())
            mses.extend(mse.detach().cpu())
            if epoch_num > 1:
                optimizer.step()
            tqdm_data_iter.set_description(
                f"find_weighting epoch: {epoch_num}. mse: {sum_mse:.6}, weighted mse: {sum_loss:.6}, p w mse: {sum_p_w_mse:.6}")
            tqdm_data_iter.refresh()  # to show immediately the update
        sum_loss = torch.stack(losses).mean()
        sum_p_w_mse = torch.stack(p_w_mses).mean()
        out_metrics["weighted_mse"].append(sum_loss)
        sum_mse = torch.stack(mses).mean()
        print(
            f"find_weighting epoch: {epoch_num}. mse: {sum_mse:.6}, weighted mse: {sum_loss:.6}, p w mse: {sum_p_w_mse:.6}")
        out_metrics["mse"].append(sum_mse)
    weights = all_w_inputs.to(config.device).view(-1, feature_num)
    weights = module(weights).view_as(end_acts)
    weights = normalize_weights(weights)
    parameters = module.describe(all_w_names=all_w_names)
    return weights, parameters


def train_transform(module, all_inputs, all_targets, all_weights,
                    batch_size=None, epoch=100,
                    lr=0.001, l1=0):
    module.to(config.device)
    weighted_mse = lambda o, t, w: ((o - t) ** 2 * w)
    non_weighted_mse = lambda o, t: ((o - t) ** 2)
    optimizer = torch.optim.Adam(module.parameters(), lr=lr)
    dataset_tensors = [all_inputs, all_targets]
    if not callable(all_weights):
        all_weights = all_weights.detach().cpu()
        dataset_tensors.append(all_weights)

    dataset = TensorDataset(*dataset_tensors)
    batch_size = len(dataset) if batch_size is None else batch_size
    data_loader = DataLoader(dataset, batch_size, shuffle=True,
                             pin_memory=True,
                             num_workers=0 if config.debug else 2)
    sum_loss = .0
    sum_mse = .0
    out_metrics = {
        "weighted_mse": [],
        "mse": []
    }
    for epoch_num in range(epoch):
        losses = []
        mses = []
        tqdm_data_iter = tqdm(enumerate(data_loader),
                              desc=f"train_transform epoch: {epoch_num}. mse: {sum_mse:.4}, weighted mse: {sum_loss:.4}",
                              total=len(data_loader),
                              leave=False)
        for i, batch in tqdm_data_iter:

            batch_iter = iter(batch)
            inputs = next(batch_iter).to(config.device)
            targets = next(batch_iter).to(config.device)

            optimizer.zero_grad()
            outputs = module(inputs)

            if not callable(all_weights):
                weights = next(batch_iter).to(config.device)
            else:
                weights = all_weights(inputs, outputs, targets)
            loss = weighted_mse(outputs, targets, weights)
            mse = non_weighted_mse(outputs, targets)
            loss_mean = loss.mean()
            if l1 > 0:
                loss_mean += torch.mean(torch.abs(module.transform.weight)) * l1
            loss_mean.backward()
            losses.extend(loss.detach().cpu())
            mses.extend(mse.detach().cpu())
            if epoch_num > 1:
                optimizer.step()
            tqdm_data_iter.set_description(
                f"train_transform epoch: {epoch_num}. mse: {sum_mse:.6}, weighted mse: {sum_loss:.6}")
            tqdm_data_iter.refresh()  # to show immediately the update
        sum_loss = torch.stack(losses).mean()
        out_metrics["weighted_mse"].append(sum_loss)
        sum_mse = torch.stack(mses).mean()
        print(f"train_transform epoch: {epoch_num}. mse: {sum_mse:.6}, weighted mse: {sum_loss:.6}")
        out_metrics["mse"].append(sum_mse)
    return out_metrics


def evaluate(frank, str_dataset):
    frank_trainer = FrankStepHandler(frank, target_type='soft_2')
    loss, acc, hits = eval_net(frank_trainer, str_dataset)
    return acc, loss


def cli_main(args=None):
    import argparse
    import torch.utils.data as data_utils
    from src.utils import config

    parser = argparse.ArgumentParser(description='Simple settings.')
    parser.add_argument('pickle', nargs="+", help='Path to pickle')
    parser.add_argument('--backward_mode', type=str,
                        choices=["max", "mean", "hard_loss", "soft2_loss"],
                        default="mean",
                        help='How to backward on the output. '
                             'max: only backward on the maximum value of '
                             'the output (which is the predicted class)'
                             'mean: use all')
    parser.add_argument('--epoch', type=int,
                        default=200,
                        help='Number of epochs to train')
    parser.add_argument('--batch', type=int,
                        default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float,
                        default=0.001,
                        help='Learning rate')
    parser.add_argument('--l1', type=float,
                        default=0.0,
                        help='L1 regularization')
    parser.add_argument('--find', type=int,
                        default=0,
                        help='If true find params will also run')
    parser.add_argument('--find_model', type=str,
                        choices=["linear", "multiplicative", "relu_segments"],
                        default="linear")
    parser.add_argument('--input_filter', type=str, nargs="+", default=None)
    parser.add_argument('--dev', type=int,
                        default=0,
                        help='If true save results to temp folder')
    parser.add_argument('--out_dir', type=str,
                        default=None,
                        help='Output folder for result pickle')

    args = parser.parse_args(args)

    for pkl in args.pickle:
        print(f"Processing: {pkl}")
        data_dict = pickle.load(open(pkl, "br"))
        front_model = load_from_path(data_dict['params']['front_model'])
        end_model = load_from_path(data_dict['params']['end_model'])

        frank_model = FrankeinsteinNet.from_data_dict(data_dict, "after")
        ps_inv_model = FrankeinsteinNet.from_data_dict(data_dict, 'ps_inv')

        device = "cuda"
        frank_model.to(device)
        ps_inv_model.to(device)
        front_model.to(device)
        end_model.to(device)

        data_name = data_dict['params']['dataset']

        dataset = get_datasets(data_name)['val']

        if config.debug:
            dataset = data_utils.Subset(dataset, torch.arange(151))

        data_loader = DataLoader(dataset,
                                 batch_size=2048,
                                 num_workers=0 if config.debug else 4,
                                 shuffle=False,
                                 drop_last=False)

        pickle_dir, pickle_filename = path.split(pkl)
        now = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        out_file_name = path.splitext(pickle_filename)[0] + "_" + now + ".pkl"
        if args.out_dir is None:
            out_file = None
        else:
            if not os.path.exists(args.out_dir):
                os.mkdir(args.out_dir)
            out_file = path.join(args.out_dir, out_file_name)

        comparator = FindWeightedPsInv(frank_model=frank_model,
                                       ps_inv_model=ps_inv_model,
                                       front_model=front_model,
                                       end_model=end_model,
                                       backward_mode=args.backward_mode,
                                       out_file=out_file,
                                       original_data_dict=data_dict,
                                       verbose=True)
        train_params = {
            "epoch": args.epoch,
            "batch_size": args.batch,
            "lr": args.lr,
            "l1": args.l1
        }
        if args.find:
            find_params = {
                "mode": args.find_model,
                "epoch": 1000,
                "batch_size": None,
                "lr": 0.001
            }
        else:
            find_params = None
        comparator(data_loader, str_dataset=data_name, find_params=find_params,
                   train_params=train_params, input_filter=args.input_filter)
        print(f"Result saved to {out_file}")


if __name__ == '__main__':
    cli_main()
