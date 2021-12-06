import math
import pickle
from os import path

import matplotlib.pyplot as plt
import numpy as np
import torch
from dotmap import DotMap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import pearsonr
from tabulate import tabulate
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.comparators.comparator_base import ComparatorBaseClass
from src.dataset import get_datasets
from src.models import load_from_path
from src.models.frank.frankenstein import FrankeinsteinNet
from src.models.frank.soft_losses import SoftCrossEntropyLoss


def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values - average) ** 2, weights=weights)
    return average, math.sqrt(variance)


def plot_2d_matrix(m, title):
    fig = plt.figure()
    fig.suptitle(title, fontsize=16)
    plt.imshow(m)
    plt.colorbar()
    plt.show()


def add_subplot(fig, pos, data, alpha, title, color):
    ax = fig.add_subplot(*pos)
    ax.set_title(title, fontsize=5)
    # ax.title.set_position([0.3, 1.05])
    if alpha is None:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax.imshow(data, interpolation='none', alpha=alpha, cmap=color)
    if alpha is None:
        fig.colorbar(im, cax=cax, orientation='vertical')


def plot_2d_matrices(data, title):
    plt.rc('xtick', labelsize=3)
    plt.rc('ytick', labelsize=3)
    fig = plt.figure(dpi=600)
    fig.suptitle(title, fontsize=8)

    for i, data_dict in enumerate(data, start=1):
        data = data_dict["data"]
        alpha = data_dict["alpha"] if "alpha" in data_dict else None
        mean, std = weighted_avg_and_std(data, weights=alpha)
        title = data_dict["title"] + f"\nmean: {mean:.2e}, std: {std:.2e}"
        if alpha is not None:
            corr, p_value = compute_w_corr(data, alpha)
            title += f"\ncorr: {corr:.2} p: {p_value:.2}"
            for per in [50]:
                alpha_threshold = np.percentile(alpha, per)
                mask = alpha > alpha_threshold
                data_treshold = data.mean()
                count = (data[mask] > data_treshold).to(dtype=torch.int).sum()
                title += f"\nhv in top {100 - per}%: {count}/{len(data[mask])}={count / len(data[mask]):.2}"
        color = data_dict["color"] if "color" in data_dict else "cool"
        add_subplot(fig, (3, 4, i), data, alpha, title, color=color)

    plt.subplots_adjust(wspace=0.1, hspace=0.8, top=0.85)
    plt.show()


def compute_w_corr(data, alpha):
    normalize = lambda x: (x - x.mean()) / (x.std() + 1e-010)
    data = normalize(data)
    alpha = normalize(alpha)
    corr, p_value = pearsonr(data.reshape(-1), alpha.reshape(-1))
    return corr, p_value


def compute_corr(frank_acts, ps_inv_acts, front_grads, end_grads, end_acts,
                 verbose=False):
    result = dict()
    normalize = lambda x: (x - x.mean()) / (x.std() + 1e-010)

    frank_act_diff = (end_acts - frank_acts).abs()
    ps_inv_act_diff = (end_acts - ps_inv_acts).abs()

    signed_act_diff = frank_act_diff - ps_inv_act_diff  # negative sign means improvement

    activation_diff = signed_act_diff
    # activation_diff = (frank_acts - ps_inv_acts)**2
    activation_diff = normalize(activation_diff)
    tabulate_headers = ['Grad Model'] + ["front", "end"]
    corrlations = ['Correlation']
    p_values = ['p-value']
    for name, grads in zip(("front", "end"), (front_grads, end_grads)):
        grads = torch.abs(grads)
        grads = normalize(grads)
        corr, p_value = pearsonr(activation_diff.view(-1),
                                 grads.view(-1))
        corrlations.append(corr)
        p_values.append(p_value)
        result[name] = {"corr": corr,
                        "p_value": p_value}
    if verbose:
        print(tabulate([corrlations, p_values], headers=tabulate_headers))
    return result


def compute_threshold_diff(frank_acts, ps_inv_acts, front_grads,
                           end_grads, percentiles, verbose=False):
    activation_diff = (frank_acts - ps_inv_acts) ** 2
    tabulate_headers = ['Percentile'] + [p for p in percentiles]
    tabulate_data = []

    for grad_model in ["front", "end"]:
        for threshold_mode in ["per sample", "flattened"]:
            description = [f"Threshold computed on {grad_model} model's "
                           f"{threshold_mode} grads"]
            below_dists = ['Below diff']
            above_dists = ['Above diff']
            threshold_means = ['Mean threshold']
            grads = front_grads if grad_model == "front" else end_grads
            grads = torch.abs(grads)

            threshold_axis = [-1, -2, -3] if threshold_mode == "per sample" else None
            for p in percentiles:
                thresholds = np.percentile(grads, p, axis=threshold_axis)
                grad_mask = grads > torch.tensor(
                    thresholds.reshape([-1, 1, 1, 1]))
                dist_below = activation_diff[~grad_mask].mean().item()
                dist_above = activation_diff[grad_mask].mean().item()
                below_dists.append(dist_below)
                above_dists.append(dist_above)
                threshold_means.append(thresholds.mean().item())
            tabulate_data.extend([description, threshold_means,
                                  above_dists, below_dists])
    if verbose:
        print(tabulate(tabulate_data, headers=tabulate_headers))
    result = {"tabulate_header": tabulate_headers,
              "tabulate_data": tabulate_data}
    return result


class GradWeightedComparator(ComparatorBaseClass):

    def __init__(self, frank_model, ps_inv_model, front_model, end_model,
                 backward_mode="mean", measurements=("threshold_diff", "corr"),
                 out_file=None, verbose=False):
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
        self.measurements = measurements
        self.out_file = out_file
        self.verbose = verbose

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

        self.hook_store.register_activation_saver(front_layer, f"front_act", model_name="front_model")
        self.hook_store.register_activation_saver(end_layer, f"end_act", model_name="end_model")
        self.hook_store.register_gradient_saver(front_layer, f"front_grad", model_name="front_model")
        self.hook_store.register_gradient_saver(end_layer, f"end_grad", model_name="end_model")

        self.hook_store.register_activation_saver(ps_inv_transform, f"ps_inv_act", model_name="ps_inv_model")
        self.hook_store.register_activation_saver(frank_transform, f"frank_act", model_name="frank_model")
        self.hook_store.register_gradient_saver(ps_inv_transform, f"ps_inv_grad", model_name="ps_inv_model")
        self.hook_store.register_gradient_saver(frank_transform, f"frank_grad", model_name="frank_model")

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
        frank_grads, ps_inv_grads = torch.cat(frank_grads), torch.cat(ps_inv_grads)
        front_grads, end_grads = torch.cat(front_grads), torch.cat(end_grads)
        front_acts, end_acts = torch.cat(front_acts), torch.cat(end_acts)
        return frank_acts, ps_inv_acts, front_grads, end_grads, \
               front_acts, end_acts, frank_grads, ps_inv_grads

    def __call__(self, data_loader,
                 percentiles=(1, 3, 5, 10, 20, 30, 40, 50,
                              60, 70, 80, 90, 95, 97, 99)):
        self.register_hooks()
        self.iterate_through_data(data_loader, step_callback=self.step_callback)
        frank_acts, ps_inv_acts, front_grads, end_grads, \
        front_acts, end_acts, frank_grads, ps_inv_grads = self._get_data()

        if "w_grad_diff" in self.measurements:
            self.compute_parameter_grad_diff(front_acts, end_acts, front_grads, end_grads, ps_inv_acts, ps_inv_grads,
                                             frank_acts, frank_grads)

        if "corr" in self.measurements:
            compute_corr(frank_acts, ps_inv_acts, front_grads, end_grads, end_acts,
                         verbose=self.verbose)
        if "threshold_diff" in self.measurements:
            compute_threshold_diff(frank_acts, ps_inv_acts, front_grads,
                                   end_grads, percentiles, verbose=self.verbose)
        self.hook_store.clear_store()

    def compute_parameter_grad_diff(self, front_acts, end_acts, front_grads, end_grads, ps_inv_acts, ps_inv_grads,
                                    frank_acts, frank_grads):

        # front_ch_grads = front_grads # front_acts * front_grads
        front_ch_grads = front_acts * front_grads
        # end_ch_grads = end_grads # end_acts * end_grads
        end_ch_grads = end_acts * end_grads
        # ps_inv_ch_grads = ps_inv_grads # ps_inv_acts * ps_inv_grads
        ps_inv_ch_grads = ps_inv_acts * ps_inv_grads

        accumulate_ch = lambda x: x.mean((0, 2, 3)).view(-1, 1)

        front_grads_per_ch = accumulate_ch(front_ch_grads.abs())
        end_grads_per_ch = accumulate_ch(end_ch_grads.abs())
        ps_inv_ch_grads = accumulate_ch(ps_inv_ch_grads.abs())

        grad_matrix = end_grads_per_ch.mm(front_grads_per_ch.T)

        frank_trans_w = self.models.frank_model.transform.transform.weight
        frank_trans_w = frank_trans_w.squeeze().detach().cpu()
        ps_inv_trans_w = self.models.ps_inv_model.transform.transform.weight
        ps_inv_trans_w = ps_inv_trans_w.squeeze().detach().cpu()

        frank_act_diff = (end_acts - frank_acts).abs()
        ps_inv_act_diff = (end_acts - ps_inv_acts).abs()

        signed_act_diff = ps_inv_act_diff - frank_act_diff  # negative sign means improvement

        n = end_grads_per_ch.shape[0]
        signed_act_diff_per_ch = accumulate_ch(signed_act_diff).repeat(1, n)
        act_diff_sign_per_ch = signed_act_diff_per_ch.sign()
        w_diff_orig = (frank_trans_w - ps_inv_trans_w).abs()
        # w_diff = (frank_trans_w - ps_inv_trans_w).abs() * signed_act_diff_per_ch
        w_diff = ((frank_trans_w - ps_inv_trans_w) ** 2).sqrt()

        n = front_grads_per_ch.shape[0]
        front_ch_grad_matrix = front_grads_per_ch.repeat(1, n).T

        n = end_grads_per_ch.shape[0]
        end_ch_grad_matrix = end_grads_per_ch.repeat(1, n)

        n = ps_inv_ch_grads.shape[0]
        ps_inv_ch_grad_matrix = ps_inv_ch_grads.repeat(1, n)

        # w_diff_low = (w_diff < w_diff.mean()).type(torch.float32)
        # w_diff_high = (w_diff > w_diff.mean()).type(torch.float32)

        w_diff += w_diff.min()
        w_diff_high = w_diff / w_diff.max()
        w_diff_low = -w_diff_high + 1

        fig_title = self.out_file.split("/")
        fig_title = f"{fig_title[-3]}   {fig_title[-1][:-31]}"

        plot_data = [
            {"data": frank_trans_w,
             "title": "Frank W",
             "color": "viridis"},
            {"data": ps_inv_trans_w,
             "title": "Ps inv W",
             "color": "viridis"},

            {"data": w_diff,
             "title": "W diff",
             "color": "viridis"},
            {"data": grad_matrix,
             "title": "Mixed Grad"},

            {"data": grad_matrix,
             "title": "Mixed Grad high diff",
             "alpha": w_diff_high},
            {"data": grad_matrix,
             "title": "Mixed Grad low diff",
             "alpha": w_diff_low},

            {"data": front_ch_grad_matrix,
             "title": "Front Grad high diff",
             "alpha": w_diff_high},
            {"data": front_ch_grad_matrix,
             "title": "Front Grad low diff",
             "alpha": w_diff_low},
            {"data": end_ch_grad_matrix,
             "title": "End Grad high diff",
             "alpha": w_diff_high},
            {"data": end_ch_grad_matrix,
             "title": "End Grad low diff",
             "alpha": w_diff_low},
            {"data": ps_inv_ch_grad_matrix,
             "title": "PsInv Grad high diff",
             "alpha": w_diff_high},
            {"data": ps_inv_ch_grad_matrix,
             "title": "PsInv Grad low diff",
             "alpha": w_diff_low},
        ]
        plot_2d_matrices(plot_data, fig_title)

    def save(self):
        pass


def cli_main(args=None):
    import argparse
    import torch.utils.data as data_utils
    from src.utils import config

    parser = argparse.ArgumentParser(description='Simple settings.')
    parser.add_argument('pickle', nargs="+", help='Path to pickle')
    parser.add_argument('--backward_mode', type=str, choices=["max", "mean", "hard_loss", "soft2_loss"],
                        default="mean",
                        help='How to backward on the output. '
                             'max: only backward on the maximum value of '
                             'the output (which is the predicted class)'
                             'mean: use all')
    parser.add_argument('--measurements', nargs="+",
                        type=str, choices=["threshold_diff", "corr", "w_grad_diff"],
                        default="w_grad_diff",
                        help='Which measurement to calculate')

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
                                 shuffle=True,
                                 drop_last=False)

        pickle_dir, pickle_filename = path.split(pkl)
        out_file_name = path.splitext(pickle_filename)[0] + ".json"
        out_dir = path.join(path.split(pickle_dir)[0], 'grad_weighting')
        out_file = path.join(out_dir, out_file_name)

        comparator = GradWeightedComparator(frank_model=frank_model,
                                            ps_inv_model=ps_inv_model,
                                            front_model=front_model,
                                            end_model=end_model,
                                            measurements=args.measurements,
                                            backward_mode=args.backward_mode,
                                            out_file=out_file,
                                            verbose=True)
        comparator(data_loader)


if __name__ == '__main__':
    cli_main()
