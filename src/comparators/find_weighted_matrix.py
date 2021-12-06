import pickle
from copy import deepcopy

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


class DiagDot(torch.nn.Module):
    def __init__(self, feature_num, only_diag=False):
        super(DiagDot, self).__init__()
        self.feature_num = feature_num
        self.weight = Parameter(torch.Tensor(feature_num))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)

    @property
    def M(self):
        m = torch.diag(self.weight)
        return m

    def forward(self, input: Tensor) -> Tensor:
        x = (input.matmul(self.M) * input).sum(-1)
        return x


class TransformedDot(torch.nn.Module):
    def __init__(self, feature_num, only_diag=False):
        super(TransformedDot, self).__init__()
        self.feature_num = feature_num
        self.tril_indices = torch.tril_indices(feature_num, feature_num)
        self.tril = Parameter(torch.Tensor(self.tril_indices.shape[1]))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.zeros_(self.tril)
        with torch.no_grad():
            self.tril[self.tril_indices[0] == self.tril_indices[1]] = 1

    @property
    def M(self):
        m = torch.zeros((self.feature_num, self.feature_num), device=self.tril.device, dtype=self.tril.dtype)
        m[self.tril_indices[0], self.tril_indices[1]] = self.tril
        return m.matmul(m.T)

    def forward(self, input: Tensor) -> Tensor:
        x = (input.matmul(self.M) * input).sum(-1)
        return x


class FindWeightedMatrix(ComparatorBaseClass):

    def __init__(self, frank_model, ps_inv_model, front_model, end_model,
                 original_data_dict, backward_mode="mean",
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

    def __call__(self, data_loader, str_dataset, epoch,
                 batch_size=None, lr=0.0001):
        self.register_hooks()
        self.iterate_through_data(data_loader, step_callback=self.step_callback)
        frank_acts, ps_inv_acts, front_grads, end_grads, \
        front_acts, end_acts, frank_grads, ps_inv_grads = self._get_data()

        acc, loss = evaluate(self.model, str_dataset)
        print(f"Acc before: {acc} soft_2 loss: {loss}")
        metrics = {
            "before_acc": acc,
            "before_soft2_loss": loss
        }

        rearrange = lambda x: x.permute(0, 2, 3, 1).reshape(-1, end_acts.shape[1])
        f_diff = rearrange(frank_acts - end_acts)
        p_diff = rearrange(ps_inv_acts - end_acts)
        f_smaller = (f_diff < p_diff).to(dtype=int).sum()
        print(f"Franki diff smaller: {f_smaller} / {f_diff.shape[0] * f_diff.shape[1]}")

        f_diff = (f_diff ** 2).sum(dim=0).sqrt()
        p_diff = (p_diff ** 2).sum(dim=0).sqrt()
        f_smaller = (f_diff < p_diff).to(dtype=int).sum()
        print(f"Grouped Franki diff smaller: {f_smaller} / {f_diff.shape[0]}")

        # p_diff = p_diff.unsqueeze(-1) * p_diff.unsqueeze(-2)
        # f_diff = f_diff.unsqueeze(-1) * f_diff.unsqueeze(-2)
        # f_diff = f_diff.sum(dim=0)
        # p_diff = p_diff.sum(dim=0)
        # f_smaller = (f_diff < p_diff).to(dtype=int).sum()
        # print(f"Grouped Franki diff smaller: {f_smaller} / {f_diff.shape[0] * f_diff.shape[1]}")

        w_matrix = find_weighting(end_acts, frank_acts, ps_inv_acts,
                                  batch_size=batch_size, epoch=epoch, lr=lr
                                  )
        print(w_matrix)
        w_matrix = w_matrix.detach()
        w_matrix.requires_grad = True
        print("training transform")
        _ = train_transform(self.transform, front_acts, end_acts,
                            w_matrix, lr=0.0001, epoch=100)

        acc, loss = evaluate(self.model, str_dataset)
        print(f"Acc after : {acc} soft_2 loss: {loss}")

        self.hook_store.clear_store()


def find_weighting(end_acts, frank_acts, ps_inv_acts,
                   batch_size=None, epoch=100, lr=0.0001):
    frank_acts = torch.rand_like(frank_acts)
    feature_num = end_acts.shape[1]
    rearrange = lambda x: x.permute(0, 2, 3, 1).reshape(-1, feature_num)
    transformed_dot = TransformedDot(feature_num)
    # transformed_dot = DiagDot(feature_num)
    transformed_dot.to(config.device)
    optimizer = torch.optim.Adam(transformed_dot.parameters(), lr=lr)

    dataset_tensors = [end_acts, frank_acts, ps_inv_acts]
    dataset = TensorDataset(*dataset_tensors)

    batch_size = len(dataset) if batch_size is None else batch_size
    data_loader = DataLoader(dataset, batch_size, shuffle=True,
                             pin_memory=True,
                             num_workers=0 if config.debug else 2)
    franki_sum_t_dot = .0
    franki_sum_dot = .0
    ps_inv_sum_t_dot = .0
    ps_inv_sum_dot = .0
    sum_reg = .0
    out_metrics = {
        "franki_t_dot": [],
        "franki_dot": [],
        "ps_inv_dot": [],
        "ps_inv_t_dot": [],
    }
    for epoch_num in range(epoch):
        franki_t_dots = []
        franki_dots = []
        ps_inv_t_dots = []
        ps_inv_dots = []
        regs = []
        tqdm_data_iter = tqdm(enumerate(data_loader),
                              ncols=150,
                              total=len(data_loader),
                              leave=False)
        for i, batch in tqdm_data_iter:
            batch_iter = iter(batch)
            e_acts = next(batch_iter).to(config.device)
            f_acts = next(batch_iter).to(config.device)
            p_acts = next(batch_iter).to(config.device)

            optimizer.zero_grad()

            f_diff = rearrange(f_acts - e_acts)
            p_diff = rearrange(p_acts - e_acts)

            franki_t_dot = transformed_dot(f_diff)
            franki_dot = (f_diff * f_diff).sum(1)

            ps_inv_t_dot = transformed_dot(p_diff)
            ps_inv_dot = (p_diff * p_diff).sum(1)

            loss = (franki_t_dot / feature_num).mean()
            loss -= (ps_inv_t_dot / feature_num).mean()

            # ortho_reg = (transformed_dot.M * transformed_dot.M.T - torch.eye(transformed_dot.M.shape[0], device=transformed_dot.M.device)).abs().sum()
            # frob_reg = torch.clamp(math.sqrt(feature_num) - torch.norm(transformed_dot.M), 0)
            # areg = torch.clamp((1/(transformed_dot.M + 1e-8)).sum() - feature_num, 0)
            # nuc_reg = torch.clamp(torch.norm(torch.eye(feature_num), p="nuc") - torch.norm(transformed_dot.M, p="nuc"), 0)
            # trace_reg = torch.clamp(feature_num - torch.trace(transformed_dot.M), 0)
            # trace_reg2 = torch.clamp(torch.trace(1/(transformed_dot.M + 1e-8)) - feature_num, 0)

            reg = torch.Tensor(0)
            # loss += reg
            loss.backward()
            regs.append(reg.detach().cpu())
            franki_t_dots.extend(franki_t_dot.detach().cpu())
            franki_dots.extend(franki_dot.detach().cpu())
            ps_inv_t_dots.extend(ps_inv_t_dot.detach().cpu())
            ps_inv_dots.extend(ps_inv_dot.detach().cpu())
            if epoch_num > 1:
                optimizer.step()
            tqdm_data_iter.set_description(
                f"find_weighting epoch: {epoch_num}. "
                f"frank dot: {franki_sum_dot:.4}, frank t_dot: {franki_sum_t_dot:.4}, "
                f"ps dot: {ps_inv_sum_dot:.4}, ps t_dot: {ps_inv_sum_t_dot:.4}, "
                f"reg term: {sum_reg:.4}, loss: {loss:.4}"
            )
            tqdm_data_iter.refresh()  # to show immediately the update
        franki_sum_t_dot = torch.stack(franki_t_dots).mean()
        franki_sum_dot = torch.stack(franki_dots).mean()
        ps_inv_sum_t_dot = torch.stack(ps_inv_t_dots).mean()
        ps_inv_sum_dot = torch.stack(ps_inv_dots).mean()
        out_metrics["franki_t_dot"].append(franki_sum_t_dot)
        out_metrics["franki_dot"].append(franki_sum_dot)
        out_metrics["ps_inv_t_dot"].append(ps_inv_sum_t_dot)
        out_metrics["ps_inv_dot"].append(ps_inv_sum_dot)
        sum_reg = torch.stack(regs).mean()
        print(
            f"find_weighting epoch: {epoch_num}. "
            f"frank dot: {franki_sum_dot:.4}, frank t_dot: {franki_sum_t_dot:.4}, "
            f"ps dot: {ps_inv_sum_dot:.4}, ps t_dot: {ps_inv_sum_t_dot:.4}, "
            f"reg term: {sum_reg:.4}"
        )

    return transformed_dot.M


def train_transform(module, all_inputs, all_targets, w_matrix,
                    batch_size=None, epoch=100,
                    lr=0.0001):
    feature_num = all_inputs.shape[1]
    rearrange = lambda x: x.permute(0, 2, 3, 1).reshape(-1, feature_num)
    module.to(config.device)
    optimizer = torch.optim.Adam(module.parameters(), lr=lr)
    l2_fn = lambda o, t: ((o - t) ** 2)
    dataset_tensors = [all_inputs, all_targets]
    dataset = TensorDataset(*dataset_tensors)
    batch_size = len(dataset) if batch_size is None else batch_size
    data_loader = DataLoader(dataset, batch_size, shuffle=True,
                             pin_memory=True,
                             num_workers=0 if config.debug else 2)
    sum_t_dots = .0
    sum_dots = .0
    sum_l2 = .0
    out_metrics = {
        "t_dot": [],
        "dot": [],
        "l2": [],
    }
    for epoch_num in range(epoch):
        t_dots = []
        l2s = []
        dots = []
        tqdm_data_iter = tqdm(enumerate(data_loader),
                              ncols=150,
                              total=len(data_loader),
                              leave=False)
        for i, batch in tqdm_data_iter:

            batch_iter = iter(batch)
            inputs = next(batch_iter).to(config.device)
            targets = next(batch_iter).to(config.device)

            optimizer.zero_grad()
            outputs = module(inputs)

            outputs = rearrange(outputs)
            targets = rearrange(targets)
            diff = outputs - targets

            transformed_dot_product = (diff.matmul(w_matrix) * diff).sum(-1)
            dot_product = (diff * diff).sum(1)
            loss = transformed_dot_product.mean()
            loss.mean().backward()
            t_dots.extend(transformed_dot_product.detach().cpu())
            dots.extend(dot_product.detach().cpu())
            l2 = l2_fn(outputs, targets).sum(1)
            l2s.extend(l2.detach().cpu())
            if epoch_num > 1:
                optimizer.step()
            tqdm_data_iter.set_description(
                f"train_transform epoch: {epoch_num}. l2: {sum_l2:.6}, dots: {sum_dots:.6} t_dots: {sum_t_dots:.6}")
            tqdm_data_iter.refresh()  # to show immediately the update
        sum_t_dots = torch.stack(t_dots).mean()
        out_metrics["t_dot"].append(sum_t_dots)
        sum_dots = torch.stack(dots).mean()
        out_metrics["dot"].append(sum_dots)
        sum_l2 = torch.stack(l2s).mean()
        out_metrics["l2"].append(sum_l2)
        print(f"train_transform epoch: {epoch_num}. l2: {sum_l2:.6}, dots: {sum_dots:.6} t_dots: {sum_t_dots:.6}")
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
                        default=300,
                        help='Number of epochs to train')
    parser.add_argument('--batch', type=int,
                        default=1024,
                        help='Batch size')
    parser.add_argument('--lr', type=float,
                        default=0.01,
                        help='Learning rate')
    parser.add_argument('--dev', type=int,
                        default=0,
                        help='If true save results to temp folder')

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

        comparator = FindWeightedMatrix(frank_model=frank_model,
                                        ps_inv_model=ps_inv_model,
                                        front_model=front_model,
                                        end_model=end_model,
                                        backward_mode=args.backward_mode,
                                        original_data_dict=data_dict,
                                        verbose=True)
        comparator(data_loader, str_dataset=data_name,
                   epoch=args.epoch, batch_size=args.batch, lr=args.lr)


if __name__ == '__main__':
    cli_main()
