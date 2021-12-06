import pickle
from copy import deepcopy

import torch
from dotmap import DotMap
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.comparators.comparator_base import ComparatorBaseClass
from src.dataset import get_datasets
from src.trainer import FrankStepHandler
from src.utils import config
from src.utils.eval_values import eval_net


class DirectTrainer(ComparatorBaseClass):

    def __init__(self, frank_model, ps_inv_model, front_model, end_model, loss, init_model="ps_inv_model",
                 batch_size=None, epoch=100, lr=0.001, out_file=None, original_data_dict=None):
        models = {
            "frank_model": frank_model,
            "ps_inv_model": ps_inv_model,
            "front_model": front_model,
            "end_model": end_model
        }
        super().__init__(models=DotMap(models))
        self.model = deepcopy(models[init_model])
        self.model.prepare_models()
        self.transform = self.model.transform
        self.batch_size = batch_size
        self.epoch = epoch
        self.lr = lr
        self.loss = loss
        self.out_file = out_file
        self.original_data_dict = original_data_dict

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
        front_acts = self.hook_store.get_and_clear_cache(f"front_act")
        end_acts = self.hook_store.get_and_clear_cache(f"end_act")

        frank_acts, ps_inv_acts = torch.cat(frank_acts), torch.cat(ps_inv_acts)
        front_acts, end_acts = torch.cat(front_acts), torch.cat(end_acts)
        return frank_acts, ps_inv_acts, front_acts, end_acts

    def get_activations(self, data_loader):
        self.register_hooks()
        self.iterate_through_data(data_loader)
        frank_acts, ps_inv_acts, front_acts, end_acts = self._get_data()

        return frank_acts, ps_inv_acts, front_acts, end_acts

    def train_transform(self, module, all_inputs, all_targets):
        module.to(config.device)
        optimizer = torch.optim.Adam(module.parameters(), lr=self.lr)
        dataset_tensors = [all_inputs, all_targets]

        dataset = TensorDataset(*dataset_tensors)
        batch_size = len(dataset) if self.batch_size is None else self.batch_size
        data_loader = DataLoader(dataset, batch_size, shuffle=True,
                                 pin_memory=True,
                                 num_workers=0 if config.debug else 2)
        sum_loss = .0
        out_metrics = {
            "loss": [],
        }
        for epoch_num in range(self.epoch):
            losses = []
            tqdm_data_iter = tqdm(enumerate(data_loader),
                                  ncols=150,
                                  total=len(data_loader),
                                  leave=True)
            for i, batch in tqdm_data_iter:
                batch_iter = iter(batch)
                inputs = next(batch_iter).to(config.device)
                targets = next(batch_iter).to(config.device)

                optimizer.zero_grad()
                outputs = module(inputs)

                loss = self.loss(outputs, targets)
                loss.mean().backward()
                losses.extend(loss.detach().cpu())
                optimizer.step()
                tqdm_data_iter.set_description(
                    f"train_transform epoch: {epoch_num}. loss: {sum_loss:.6}")
                tqdm_data_iter.refresh()
            sum_loss = torch.stack(losses).mean()
            out_metrics["loss"].append(sum_loss)
        return out_metrics

    def get_dataloader(self, dataset_str, split):
        dataset = get_datasets(dataset_str)[split]

        if config.debug:
            dataset = torch.utils.data.Subset(dataset, torch.arange(151))

        data_loader = DataLoader(dataset,
                                 batch_size=2048,
                                 num_workers=0 if config.debug else 4,
                                 shuffle=False,
                                 drop_last=False)
        return data_loader

    def __call__(self, dataset_str, train_on="val"):
        data_loader = self.get_dataloader(dataset_str, train_on)

        frank_acts, ps_inv_acts, front_acts, end_acts = self.get_activations(data_loader)
        acc, cross_entropy = self.evaluate(self.model, dataset_str)
        print(f"Before accuracy: {acc} cross_entropy: {cross_entropy}")
        metrics = {
            "before_acc": acc,
            "before_cross_entropy": cross_entropy,
            "dataset": dataset_str,
            "train_on": train_on
        }
        train_metrics = self.train_transform(self.transform, front_acts, end_acts)
        metrics["train_metrics"] = train_metrics
        acc, cross_entropy = self.evaluate(self.model, dataset_str)
        print(f"After accuracy: {acc} cross_entropy: {cross_entropy}")
        metrics["after_acc"] = acc
        metrics["after_cross_entropy"] = cross_entropy
        self.save_results(metrics)

    def evaluate(self, frank, str_dataset):
        frank_trainer = FrankStepHandler(frank, target_type='soft_2')
        cross_entropy, acc, hits = eval_net(frank_trainer, str_dataset)
        return acc, cross_entropy

    def save_results(self, metrics):
        if self.out_file is None:
            return
        metrics["batch_size"] = self.batch_size
        metrics["epoch"] = self.epoch
        metrics["lr"] = self.lr
        orig_dict = deepcopy(self.original_data_dict)
        del orig_dict["trans_fit"]
        del orig_dict["trans_m"]
        metrics["original"] = orig_dict
        with open(self.out_file, 'wb') as handle:
            pickle.dump(metrics, handle, protocol=pickle.HIGHEST_PROTOCOL)
